"""Bridge to the ABC logic-synthesis tool for Boolean-semantic reasoning
(equivalence checking, constant provability, positive symmetry) that the
earlier, purely structural stages of this codebase (graph.py, analysis.py,
transform.py) deliberately never attempt. Everything downstream trusts this
module as ground truth for "did my transform preserve functionality" -- treat
it as safety-critical.

Empirical ABC behavior (confirmed by direct testing against the resolved
binary before this module was written -- do not re-derive, just rely on it):

  1. `read_verilog` cannot parse named-port gate instances (a `dff` instance
     with `.RN(...)`/`.SN(...)`/etc. connections fails to parse). It parses
     positional-only instances fine (this codebase's writer.py already emits
     every non-DFF primitive positionally). Consequently no Verilog handed to
     ABC by this module may ever contain a `dff` instance -- DFF boundaries
     are always turned into ports/ties first (see `extract_combinational_view`).

  2. Equivalence checking is `cec fileA fileB` (two file paths, no prior
     `read_*` needed; `_run_cec` passes two `.blif` paths -- see finding 4 --
     but `cec` itself is format-agnostic, confirmed against a `.v`/`.blif`
     mix too). Confirmed verbatim output patterns:
       - equivalent:     "Networks are equivalent after structural hashing.  Time = ..."
       - not equivalent: "Networks are NOT EQUIVALENT.  Time = ..." followed by
                          a counterexample block (INPUT:/OUTPUT: line, a
                          "Verification failed for at least N outputs" line,
                          per-output value lines, an "Input pattern:" line).
     `cec` matches PIs/POs by NAME, not textual port-list order. ABC's own
     process exit code is ALWAYS 0 regardless of outcome (equivalent, not
     equivalent, or parse/miter failure) -- never trust returncode, always
     parse stdout text. Failure patterns seen: "Miter computation has failed"
     (PI/PO count or name mismatch) and "Reading network from file has
     failed" (unreadable file). A third, non-verdict outcome is "Networks are
     undecided (SAT solver timed out)." (verbatim, exit code 0): ABC gave no
     answer either way; this raises `CecUndecidedError` (an
     `ABCInconclusiveError`). A subprocess timeout raises `ABCTimeoutError`
     (also an `ABCInconclusiveError`). This module raises plain
     `ABCBridgeError` if stdout matches none of the known patterns, rather
     than silently guessing.
     Unused/extra PI ports declared on both sides are harmless as long as
     both sides declare the identical PI (and PO) name *sets* -- this module
     guarantees that by construction (it builds both files) and additionally
     pre-flight-checks it in Python before ever shelling out to ABC.

  3. Undriven/floating internal wires produce a harmless stdout warning
     ("Warning: Constant-0 drivers added to N non-driven nets...") -- not an
     error, never treated as one here.

  4. `read_verilog` asserts and crashes ABC (SIGABRT, "Assertion failed:
     nMsb < 128 ... Ver_ParseInsertsSuffix") on any bus whose declared MSB
     bit index reaches 128 -- real in this project's own corpus (final's
     widest buses run 192-4096 bits). `cec`/synthesis therefore never hand
     ABC Verilog: both go through `write_blif` (below) and `read_blif`/
     `cec a.blif b.blif` instead, which has no such limit. Confirmed
     empirically (scratchpad, 2026-09-30) that the PI/PO name tokens ABC
     itself writes back out are byte-for-byte identical whether the input
     was Verilog-then-strashed or BLIF-then-strashed (`n5[199]` etc., same
     "name"/"name[bit]" shape `netbit_token`/`parse_net` already expect) --
     so `abc_synth.py`'s BLIF-token-based read-back (`parse_blif`,
     `_token_resolver`) needed no changes for this switch. Also confirmed:
     an undriven net referenced by `read_blif` (never a `.names`/`.gate`
     output column) gets the identical harmless constant-0-tie treatment as
     an undriven net under `read_verilog` (point 3) -- so `write_blif` never
     needs to emit anything explicit for a floating net, only omit it from
     every output column, exactly as `write_verilog` already implicitly did.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Literal, Optional

from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import (
    Const,
    DFF_PIN_ORDER,
    Design,
    Direction,
    Gate,
    GateType,
    NetBit,
    ONE_INPUT_GATES,
    OUTPUT_PIN,
    Pin,
    Port,
    Signal,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIND_ABC_SCRIPT = os.path.join(REPO_ROOT, "scripts", "find_abc.sh")

# Generous default for a single ABC invocation; a pathological/hung ABC run
# raises ABCBridgeError (via subprocess.TimeoutExpired) rather than hanging
# silently. Callers of the public functions below may override per call.
DEFAULT_ABC_TIMEOUT = 120.0
# Separate, larger budget for the post-synthesis equivalence check (`cec`).
# Measured `cec` on final_release_100/test062: 49.5 s (this Q-boundary
# model, 2026-09-30); 121 s (run_corpus, idle) and 257 s (CPU contended) under
# earlier models. n=3, different code and different CPU load each time.
DEFAULT_VERIFY_TIMEOUT = 600.0
_RESOLVE_TIMEOUT = 30.0

DffQMode = Literal["free_pi", "const_zero"]


class ABCBridgeError(Exception):
    """Raised whenever this module cannot proceed with confidence: the ABC
    binary can't be resolved/invoked, ABC crashes (nonzero exit), ABC's stdout
    matches none of the known equivalent/not-equivalent/undecided/failure
    patterns, ABC itself reports it could not build the miter (e.g. mismatched
    PI/PO sets -- caught pre-flight in Python instead, see `verify_equivalence`),
    or an internal invariant of this module is violated (e.g. a net reported
    equivalent to both constant 0 and constant 1). `ABCInconclusiveError`
    (and its subclasses `ABCTimeoutError`, `CecUndecidedError`) means "ABC gave
    no verdict" -- the only outcomes callers may degrade to "undetermined";
    everything else must stay loud."""


class ABCInconclusiveError(ABCBridgeError):
    """ABC did not give a verdict (timed out, or reported the miter
    undecided). Never to be mapped onto `EquivResult(equivalent=False)`:
    "could not decide" is not "not equivalent"."""


class ABCTimeoutError(ABCInconclusiveError):
    """The ABC subprocess exceeded its time budget."""


class CecUndecidedError(ABCInconclusiveError):
    """`cec` itself reported the networks undecided (e.g. "Networks are
    undecided (SAT solver timed out).")."""


_abc_path: Optional[str] = None


def _resolve_abc() -> str:
    global _abc_path
    if _abc_path is None:
        try:
            result = subprocess.run(
                ["bash", FIND_ABC_SCRIPT], capture_output=True, text=True, timeout=_RESOLVE_TIMEOUT
            )
        except subprocess.TimeoutExpired as exc:
            raise ABCBridgeError(f"timed out resolving ABC binary via {FIND_ABC_SCRIPT}") from exc
        if result.returncode != 0:
            raise ABCBridgeError(
                f"failed to resolve ABC binary via {FIND_ABC_SCRIPT}: {result.stderr.strip()}"
            )
        _abc_path = result.stdout.strip()
    return _abc_path


def _script_command(script: str) -> str:
    """The ABC command name only (e.g. `cec`) -- the rest of `script` is
    temp-file paths, kept out of the timeout / nonzero-exit messages (those
    end up in user-facing responses via the optimize paths). This does not
    cover ABC's stdout: its first line echoes the full command line, paths
    included, and messages built from stdout still carry it."""
    words = script.split()
    return words[0] if words else ""


def _run_abc(script: str, timeout: float) -> str:
    abc_path = _resolve_abc()
    try:
        result = subprocess.run([abc_path, "-c", script], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ABCTimeoutError(f"ABC invocation timed out after {timeout}s running {_script_command(script)!r}") from exc
    if result.returncode != 0:
        # A nonzero exit is ABC crashing (e.g. an internal assert/SIGABRT),
        # distinct from the "always 0" guarantee documented above for the
        # graceful equivalent/not-equivalent/failure-pattern outcomes of
        # `cec` -- this module's only caller of `_run_abc` (`_run_cec`) never
        # relied on reading `stdout` when the process crashed, so raising
        # here instead of returning is safe.
        stderr_tail = result.stderr.strip()[-500:]
        raise ABCBridgeError(
            f"ABC exited with code {result.returncode} running {_script_command(script)!r}: {stderr_tail}"
        )
    return result.stdout


def _parse_cec_output(stdout: str) -> "EquivResult":
    if "Networks are equivalent" in stdout:
        return EquivResult(True, stdout.strip())
    if "Networks are NOT EQUIVALENT" in stdout:
        return EquivResult(False, stdout.strip())
    if "Miter computation has failed" in stdout or "Reading network from file has failed" in stdout:
        raise ABCBridgeError(f"ABC could not compare the two networks: {stdout.strip()}")
    if "undecided" in stdout.lower():
        raise CecUndecidedError(f"ABC `cec` could not reach a verdict: {stdout.strip()}")
    raise ABCBridgeError(
        f"unrecognized ABC `cec` output (matched neither the equivalent nor the "
        f"not-equivalent pattern) -- failing loudly instead of guessing: {stdout.strip()}"
    )


# Single-output-column SOP covers for the fixed primitive gate set, ALWAYS in
# onset form (every listed row's output value column is "1") -- deliberately
# avoids the offset-listing BLIF convention (rows of "0", everything else
# implicitly 1) entirely, so there is exactly one row-emission rule for every
# gate type instead of two, and no reader-dependent "what's the default for
# an unlisted combination" question to get wrong. Verified byte-for-byte
# equivalent (via ABC `cec`) against ABC's own `read_verilog` semantics for
# every one of these primitives, including the ones (NAND/NOR/XNOR) whose
# naive rendering would be an offset cover (scratchpad, 2026-09-30).
_TWO_INPUT_BLIF_COVERS: dict[GateType, tuple[str, ...]] = {
    GateType.AND: ("11 1",),
    GateType.OR: ("1- 1", "-1 1"),
    GateType.NAND: ("0- 1", "-0 1"),
    GateType.NOR: ("00 1",),
    GateType.XOR: ("10 1", "01 1"),
    GateType.XNOR: ("11 1", "00 1"),
}
_ONE_INPUT_BLIF_COVERS: dict[GateType, tuple[str, ...]] = {
    GateType.NOT: ("0 1",),
    GateType.BUF: ("1 1",),
}


def _blif_token(value: Pin, const_names: dict[Const, str], design: Design, unconnected_counter: list[int]) -> str:
    if value is None:
        # Mirrors writer.py's `_render_pin(None) -> ""` (an unconnected
        # positional pin becomes an empty Verilog port-connection slot, i.e.
        # ABC sees a brand-new floating net there): allocate a fresh,
        # otherwise-unreferenced BLIF token so it is likewise never driven --
        # same constant-0-tie outcome as any other undriven net (module
        # docstring, finding 4). Not known to be exercised by any real gate
        # in this codebase's corpus today; handled defensively for parity
        # with write_verilog rather than assumed unreachable.
        unconnected_counter[0] += 1
        name = f"__blif_unconnected_{unconnected_counter[0]}__"
        while name in design.signals:
            unconnected_counter[0] += 1
            name = f"__blif_unconnected_{unconnected_counter[0]}__"
        return name
    if isinstance(value, Const):
        if value not in const_names:
            base = "c0" if value == Const.ZERO else "c1"
            name = base
            suffix = 0
            while name in design.signals or name in const_names.values():
                suffix += 1
                name = f"{base}_{suffix}"
            const_names[value] = name
        return const_names[value]
    return value.name if value.bit is None else f"{value.name}[{value.bit}]"


def write_blif(design: Design, path: str) -> None:
    """Write `design` (must already be free of `dff` instances, e.g. via
    `extract_combinational_view` -- raises `ABCBridgeError` if one is found)
    as a BLIF netlist for ABC's `read_blif`, sidestepping `read_verilog`'s
    >=128-bit-bus-index assertion crash entirely (module docstring, finding
    4). PI/PO port token spelling ("name" / "name[bit]") is exactly what
    `write_verilog` + ABC's own `read_verilog` already produced -- confirmed
    empirically identical round-tripped back out of ABC (see finding 4) --
    so nothing downstream of this module's callers (abc_synth.py's
    `parse_blif`/`_token_resolver`) needed to change.

    Every primitive gate becomes one `.names` line (see `_TWO_INPUT_BLIF_COVERS`/
    `_ONE_INPUT_BLIF_COVERS`); a `Const.ZERO`/`Const.ONE` pin value is routed
    through one shared, lazily-declared constant net per value (`.names c0`
    with no cover row = constant 0, `.names c1` / `1` = constant 1 --
    confirmed against ABC, scratchpad 2026-09-30), collision-checked against
    `design`'s own signal names since a real net could plausibly already be
    named "c0". A floating net (referenced by a gate pin or a primary output,
    but driven by nothing -- neither a PI nor any gate's `O`) gets an
    explicit `.names <net>` (no-cover = constant 0) tie emitted for it below,
    rather than relying on ABC's own identical implicit default (confirmed
    empirically, finding 4) -- explicit here is not required for correctness
    against the ABC version this was tested against, but does not depend on
    that default continuing to hold in a future ABC version either.

    No line-wrapping (`\\`-continuation) is used for long `.inputs`/`.outputs`
    lines: confirmed ABC's `read_blif` parses a single unwrapped line of
    20,000 whitespace-separated tokens (~590KB) without issue (scratchpad,
    2026-09-30) -- comfortably past this project's largest corpus design
    (test051, ~213k gates) -- so the continuation machinery the module
    docstring speculated might be needed is not implemented.
    """
    const_names: dict[Const, str] = {}
    unconnected_counter = [0]

    def token(value: Pin) -> str:
        return _blif_token(value, const_names, design, unconnected_counter)

    # A port name is a whole Signal, possibly multi-bit -- must be expanded to
    # one "name[bit]" token per bit here, exactly as ABC's own `read_verilog`
    # already flattens a `input [msb:lsb] name;` declaration into per-bit PIs
    # (confirmed: writing just the bare port name once, not one token per bit,
    # silently produced a network with far fewer PIs than gates actually
    # reference -- every bit past bit 0 then reads back as an *undriven
    # internal* net instead of the PI bit it should be, which is exactly
    # `optimize_gate_count`'s "collapsed to near-constant" false gate-count
    # win caught by `tests/test_abc_synth.py`'s test18/and_not regression
    # test during this fix's own verification).
    def port_tokens(direction: Direction) -> list[str]:
        toks: list[str] = []
        for p in design.ports:
            if p.direction != direction:
                continue
            for nb in design.signals[p.name].bits():
                toks.append(nb.name if nb.bit is None else f"{nb.name}[{nb.bit}]")
        return toks

    pi_tokens = port_tokens(Direction.INPUT)
    po_tokens = port_tokens(Direction.OUTPUT)

    lines: list[str] = [f".model {design.module_name}"]
    lines.append(".inputs " + " ".join(pi_tokens))
    lines.append(".outputs " + " ".join(po_tokens))

    driven_tokens: set[str] = set(pi_tokens)
    referenced_tokens: set[str] = set(po_tokens)
    gate_lines: list[str] = []
    for g in design.gates:
        if g.gate_type == GateType.DFF:
            raise ABCBridgeError(
                f"write_blif: dff instance {g.inst_name!r} reached the ABC BLIF writer -- "
                "callers must pass a Design already run through extract_combinational_view"
            )
        o_tok = token(g.pins.get("O"))
        driven_tokens.add(o_tok)
        if g.gate_type in ONE_INPUT_GATES:
            i0_tok = token(g.pins.get("I0"))
            referenced_tokens.add(i0_tok)
            gate_lines.append(f".names {i0_tok} {o_tok}")
            gate_lines.extend(_ONE_INPUT_BLIF_COVERS[g.gate_type])
        else:
            i0_tok = token(g.pins.get("I0"))
            i1_tok = token(g.pins.get("I1"))
            referenced_tokens.add(i0_tok)
            referenced_tokens.add(i1_tok)
            gate_lines.append(f".names {i0_tok} {i1_tok} {o_tok}")
            gate_lines.extend(_TWO_INPUT_BLIF_COVERS[g.gate_type])
    lines.extend(gate_lines)

    for value, name in const_names.items():
        driven_tokens.add(name)
        lines.append(f".names {name}")
        if value == Const.ONE:
            lines.append("1")

    # Explicit constant-0 tie for every net referenced (as a gate input or a
    # primary output) but driven by nothing (not a PI, not any gate's O, not
    # one of the const nets just declared above) -- see docstring. Sorted so
    # output is deterministic byte-for-byte, not dependent on dict/set
    # iteration order.
    for undriven_tok in sorted(referenced_tokens - driven_tokens):
        lines.append(f".names {undriven_tok}")

    lines.append(".end")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _run_cec(design_a: Design, design_b: Design, timeout: float) -> "EquivResult":
    with tempfile.TemporaryDirectory(prefix="abc_bridge_") as tmpdir:
        path_a = os.path.join(tmpdir, "a.blif")
        path_b = os.path.join(tmpdir, "b.blif")
        write_blif(design_a, path_a)
        write_blif(design_b, path_b)
        stdout = _run_abc(f'cec "{path_a}" "{path_b}"', timeout=timeout)
    return _parse_cec_output(stdout)


# ----------------------------------------------------------------------
# Capability 1: sequential -> combinational DFF-boundary extraction
# ----------------------------------------------------------------------


def _is_declared_bit(sig: Signal, nb: NetBit) -> bool:
    """O(1) "is `nb` a declared bit of `sig`". Deliberately NOT
    `nb in sig.bits()`: that materialises the whole bus per call, which made
    extraction quadratic on the 4096-bit buses in the final corpus."""
    if sig.msb is None or sig.lsb is None:
        return nb.bit is None
    return nb.bit is not None and min(sig.msb, sig.lsb) <= nb.bit <= max(sig.msb, sig.lsb)


def extract_combinational_view(
    design: Design,
    dff_q_mode: DffQMode,
    promoted_q_source: Optional[dict[str, NetBit]] = None,
    *,
    tap_control_pins: bool = False,
    skip_control_taps: frozenset = frozenset(),
) -> Design:
    """Return a new, purely-combinational `Design` (never mutates `design`):
    every non-DFF gate is copied unchanged, and every DFF instance is dropped
    after its Q/D boundary is turned into ports or ties so the result never
    contains a `dff` instance (see module docstring, point 1, for why that
    matters for everything downstream that hands this to ABC).

    `dff_q_mode`:
      - "free_pi": every DFF gets one new primary INPUT `__dff_Q__<inst>`
        (a free variable) and a synthesized BUF re-drives the DFF's original
        Q net from it. Used for equivalence/symmetry checking, where a
        flop's stored value must range over both 0 and 1.
      - "const_zero": every DFF's Q net is tied to Const.ZERO via a
        synthesized BUF gate instead -- used only by
        `is_constant` (via the cone-restriction helper below) to ask "is
        this net constant when every flop happens to hold 0".

    `tap_control_pins` (default False; only `verify_equivalence` turns it on):
    when True, every DFF's control pins (`DFF_PIN_ORDER` minus D and Q, i.e.
    RN/SN/CK) are ALSO exposed as primary OUTPUTs `__dff_<PIN>__<inst>`, one
    BUF tap each, exactly like the D tap (None = unconnected is skipped;
    constants are tapped; a name collision raises `ABCBridgeError`). Without
    it the equivalence boundary saw only Q and D, so rewiring SN/RN/CK was
    judged "equivalent". It is off for every other caller on purpose:
    `abc_synth._splice_whole_design` does not handle `__dff_<PIN>__` tokens,
    and the query-style callers (`is_constant`, `check_implication`,
    `check_symmetry`, `are_equivalent`, `signal_pair_search`) ask about
    combinational functions, which the extra POs do not belong to.

    `skip_control_taps`: a set of `(inst_name, pin)` pairs that are NOT tapped
    even when `tap_control_pins` is on. `verify_equivalence` passes the same
    set for both designs so their PO name sets stay equal.

    `promoted_q_source`, if given (mutated in place; only meaningful for
    "free_pi"), is populated with one entry per DFF: the PI's signal name
    `__dff_Q__<inst>` in the returned Design -> that DFF's original Q
    NetBit in `design`. abc_synth.py's depth optimizers need this to wire
    newly-synthesized gates (which reference the view's PI names) back into
    `design`'s own namespace.

    Q side is identified by DFF *instance* name, symmetric with the D side
    below; both are kept in sync across renames by the same
    `Session.mirror_rename`. The original Q net is NOT renamed, split or
    promoted, and no consumer is rewired. Rewiring consumers or renaming Q
    nets (earlier designs) required every entry point to map names back:
    a D-tap reading a Q net lost its value (F1), extraction cost O(gates x
    DFFs) (F2), the query entry points (`are_equivalent`, `check_symmetry`,
    `find_pair_for_op`) needed a reverse map (F3), and a true Q-pin swap
    was invisible because Q was keyed by net name while D was keyed by
    instance (F4). Re-driving the untouched net through a BUF is the same
    shape `const_zero` already used.

    Sibling bits of a Q bus that no DFF drives and nothing else drives are
    left floating and are tied to 0 by `write_blif`, the existing
    convention (zero of them were read by any gate in the corpus).

    Malformed Q wiring raises `ABCBridgeError`: two DFFs on one Q net, a Q
    net also driven by a gate, a Q net that is a primary input or not a
    declared bit, or a name collision with `__dff_Q__<inst>`.

    Every DFF's D-pin value is exposed as a new primary OUTPUT in BOTH modes
    (uniformity: `is_constant`'s cone-restriction step discards whichever POs
    it doesn't need anyway), via a BUF tap driving a canonical fresh output
    net named `__dff_D__<dff instance name>` -- NOT under the D net's own
    name. Keying the boundary on the DFF *instance* instead of the *net*
    makes before/after-transform equivalence checks robust to rewires that
    change which net feeds a D pin (buffer insertion, double-inverter
    collapse):

    with net-name keying those produced spurious "PO name sets differ"
    errors on designs that were in fact equivalent.

    Correctness here does NOT come from "no transform renames a DFF
    instance" -- that claim is false: `router._h_rename_gate` /
    `llm/tools_schema.rename_gate` rename gate instances generically, DFFs
    included, and nothing stops a request from targeting one. What
    actually keeps `verify_equivalence`'s two designs' PO name sets in
    sync across such a rename is `Session.mirror_rename` (`session.py`),
    called from every rename call site immediately after a successful
    rename, which relabels `original_snapshot`'s copy of the same
    instance the same way `current_design`'s was just relabeled -- see
    that method's docstring, and `experiments/snapshot_collision_2026-09-03/`,
    for the collision case this has to handle when the freed name was
    already stale in the snapshot. One exception to that sync, spelled out
    there and repeated here so this file is not read as an unconditional
    guarantee: when the name being renamed onto already labels a DFF in the
    snapshot, `mirror_rename` skips rather than relabels, and the PO name
    sets would then diverge exactly as described above. That case is
    argued -- and, via `tests/test_snapshot_rename_collision.py`, executed
    -- to be unreachable today, not proven impossible.

    Residual limitation, accepted: a transform that removed a DFF outright
    would change the boundary set and fail the name-set pre-check in
    `verify_equivalence` (honestly, as an error -- not as a wrong verdict),
    because there is no rename to mirror in that case, only a
    disappearance. Stated in the conditional because no transform in this
    codebase does that today -- all six gate-removal sites exempt DFFs, and
    `tests/test_snapshot_rename_collision.py` runs them to say so rather
    than asserting it in prose. The Q side is keyed by instance name too and
    relies on the same `mirror_rename` sync.
    """
    new_design = Design(module_name=design.module_name)
    for name, sig in design.signals.items():
        new_design.signals[name] = Signal(name=sig.name, msb=sig.msb, lsb=sig.lsb, direction=sig.direction)
    new_design.ports = [Port(name=p.name, direction=p.direction) for p in design.ports]

    dff_gates = [g for g in design.gates if g.gate_type == GateType.DFF]
    for g in design.gates:
        if g.gate_type == GateType.DFF:
            continue
        new_design.add_gate(Gate(inst_name=g.inst_name, gate_type=g.gate_type, pins=dict(g.pins)))

    # Q side: the original Q net keeps its name, direction and port entry.
    # It is simply re-driven by one BUF whose source is a per-instance
    # primary input (free_pi) or Const.ZERO (const_zero). Nothing is rewired
    # and no bus is split or promoted, so every original net in the view
    # keeps its name and meaning (see the docstring for why).
    q_owner: dict[NetBit, str] = {}
    for g in dff_gates:
        q = g.pins.get("Q")
        if not isinstance(q, NetBit):
            continue
        if q in q_owner:
            raise ABCBridgeError(
                f"DFFs {q_owner[q]!r} and {g.inst_name!r} both drive {q}"
            )
        if q in new_design.net_driver:
            raise ABCBridgeError(
                f"DFF {g.inst_name!r} Q net {q} is also driven by a combinational gate"
            )
        sig = new_design.signals.get(q.name)
        if sig is None or not _is_declared_bit(sig, q):
            raise ABCBridgeError(
                f"DFF {g.inst_name!r} Q net {q} is not a declared bit of any signal"
            )
        if sig.direction == Direction.INPUT:
            raise ABCBridgeError(
                f"DFF {g.inst_name!r} Q net {q} is a primary input"
            )
        q_owner[q] = g.inst_name
        if dff_q_mode == "free_pi":
            pi_name = f"__dff_Q__{g.inst_name}"
            if pi_name in new_design.signals:
                raise ABCBridgeError(
                    f"canonical DFF Q name {pi_name!r} collides with an existing signal"
                )
            new_design.signals[pi_name] = Signal(name=pi_name, msb=None, lsb=None, direction=Direction.INPUT)
            new_design.ports.append(Port(name=pi_name, direction=Direction.INPUT))
            src: NetBit | Const = NetBit(pi_name, None)
            if promoted_q_source is not None:
                promoted_q_source[pi_name] = q
        else:
            src = Const.ZERO
        new_design.add_gate(
            Gate(
                inst_name=new_design.fresh_gate_name(),
                gate_type=GateType.BUF,
                pins={"O": q, "I0": src},
            )
        )

    # THEN the D side: one canonical BUF tap per DFF instance (see the
    # docstring). Unconditional and uniform -- because the tap drives its
    # own fresh output net, there is no INPUT/OUTPUT direction conflict to
    # special-case even when the D net is itself a PI, another DFF's Q
    # (direct DFF-to-DFF chain), or a shared net feeding several D pins
    # (each instance simply gets its own tap). A D pin tied to a constant
    # is tapped too (both sides of a comparison tap it identically); only a
    # genuinely unconnected D pin (None) has nothing to observe and is
    # skipped.
    for g in dff_gates:
        d = g.pins.get("D")
        if d is None:
            continue
        out_name = f"__dff_D__{g.inst_name}"
        if out_name in new_design.signals:
            raise ABCBridgeError(
                f"canonical DFF D-tap name {out_name!r} collides with an existing signal"
            )
        new_design.signals[out_name] = Signal(name=out_name, msb=None, lsb=None, direction=Direction.OUTPUT)
        new_design.ports.append(Port(name=out_name, direction=Direction.OUTPUT))
        new_design.add_gate(
            Gate(
                inst_name=new_design.fresh_gate_name(),
                gate_type=GateType.BUF,
                pins={"O": NetBit(out_name, None), "I0": d},
            )
        )

    if tap_control_pins:
        # Control pins (RN/SN/CK): same canonical per-instance BUF tap as D,
        # except the (inst, pin) pairs in `skip_control_taps` (identical
        # Const / identical PI on both sides, see `verify_equivalence`).
        # Derived from DFF_PIN_ORDER so a pin added there is covered too.
        for pin in (p for p in DFF_PIN_ORDER if p not in ("D", "Q")):
            for g in dff_gates:
                val = g.pins.get(pin)
                if val is None or (g.inst_name, pin) in skip_control_taps:
                    continue
                out_name = f"__dff_{pin}__{g.inst_name}"
                if out_name in new_design.signals:
                    raise ABCBridgeError(
                        f"canonical DFF {pin}-tap name {out_name!r} collides with an existing signal"
                    )
                new_design.signals[out_name] = Signal(name=out_name, msb=None, lsb=None, direction=Direction.OUTPUT)
                new_design.ports.append(Port(name=out_name, direction=Direction.OUTPUT))
                new_design.add_gate(
                    Gate(
                        inst_name=new_design.fresh_gate_name(),
                        gate_type=GateType.BUF,
                        pins={"O": NetBit(out_name, None), "I0": val},
                    )
                )

    return new_design


# ----------------------------------------------------------------------
# Cone restriction (internal helper backing is_constant/check_symmetry)
# ----------------------------------------------------------------------


def _restrict_to_fanin_cone(comb_design: Design, target: NetBit, output_signal_name: str) -> Design:
    """Deep-copy just `target`'s fanin cone (target's own driving gate, if
    any, plus everything upstream of it) out of `comb_design` into a fresh
    `Design` with exactly one output port (`output_signal_name`) carrying
    `target`'s value. Makes is_constant/check_symmetry tractable on
    100k-gate designs instead of running ABC over the whole netlist per query.

    PI-inclusion strategy (judgment call): copies `comb_design`'s WHOLE
    primary-input port list unchanged, rather than computing exactly which
    PIs the selected gate subset references. Simpler, and callers build
    matching reference Designs directly off of this same PI list, which is
    trivial when it is just "all of comb_design's PIs" -- extra unused PI
    ports are harmless (confirmed empirically, see module docstring).
    """
    graph = NetlistGraph(comb_design)
    cone_gate_names = graph.backward_reachable_gates(target)

    new_design = Design(module_name=comb_design.module_name)
    for p in comb_design.ports:
        if p.direction != Direction.INPUT:
            continue
        sig = comb_design.signals[p.name]
        new_design.signals[p.name] = Signal(name=p.name, msb=sig.msb, lsb=sig.lsb, direction=Direction.INPUT)
        new_design.ports.append(Port(name=p.name, direction=Direction.INPUT))

    for g in comb_design.gates:
        if g.inst_name not in cone_gate_names:
            continue
        new_gate = Gate(inst_name=g.inst_name, gate_type=g.gate_type, pins=dict(g.pins))
        new_design.add_gate(new_gate)
        for val in new_gate.pins.values():
            if isinstance(val, NetBit) and val.name not in new_design.signals:
                orig_sig = comb_design.signals[val.name]
                new_design.signals[val.name] = Signal(
                    name=val.name, msb=orig_sig.msb, lsb=orig_sig.lsb, direction=Direction.INTERNAL
                )

    if target.name not in new_design.signals:
        # target has no driving gate in the cone (a direct PI/pseudo-PI
        # passthrough) -- still register its signal so the BUF tie below has
        # somewhere valid to read from.
        orig_sig = comb_design.signals[target.name]
        new_design.signals[target.name] = Signal(
            name=target.name, msb=orig_sig.msb, lsb=orig_sig.lsb, direction=Direction.INTERNAL
        )

    if output_signal_name in new_design.signals:
        raise ABCBridgeError(
            f"requested cone output name {output_signal_name!r} collides with an "
            "existing net in the fanin cone"
        )
    new_design.signals[output_signal_name] = Signal(
        name=output_signal_name, msb=None, lsb=None, direction=Direction.OUTPUT
    )
    new_design.ports.append(Port(name=output_signal_name, direction=Direction.OUTPUT))
    # Always synthesize a fresh BUF tying `target` to the new output net,
    # rather than conditionally reusing target's own driving gate's output
    # pin as the port: target may be a bit-select of a wider bus, or already
    # an INPUT (a PI/pseudo-PI), neither of which can be renamed/repurposed
    # as a differently-named PO in this IR (a PO's identity is a whole
    # Signal's name). A uniform extra BUF sidesteps that and is functionally
    # free -- this is also exactly what's needed for the "target has zero
    # gates in the cone" case, generalized to always apply.
    new_design.add_gate(
        Gate(
            inst_name=new_design.fresh_gate_name(),
            gate_type=GateType.BUF,
            pins={"O": NetBit(output_signal_name, None), "I0": target},
        )
    )
    return new_design


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EquivResult:
    equivalent: bool
    detail: str  # raw relevant ABC stdout: the counterexample block, or the equivalent-confirmation line(s)


_DFF_LEGEND = (
    "(__dff_Q__<name> / __dff_D__<name> are flip-flop <name>'s current state (Q) "
    "and next-state input (D); __dff_RN__/__dff_SN__/__dff_CK__<name> are its "
    "reset, set and clock pins.)"
)


def _with_dff_legend(result: EquivResult) -> EquivResult:
    """Append a one-line legend to a NON-equivalent result whose detail names a
    synthetic DFF boundary PI, so the user can read `__dff_Q__r1`. The line
    must not contain "Input pattern:" (`property_check._INPUT_PATTERN_RE`
    matches on it)."""
    if result.equivalent or not any(
        tag in result.detail
        for tag in ("__dff_Q__", "__dff_D__", *(f"__dff_{p}__" for p in DFF_PIN_ORDER if p not in ("D", "Q")))
    ):
        return result
    return EquivResult(False, result.detail + "\n" + _DFF_LEGEND)


def _identical_control_pins(design_a: Design, design_b: Design) -> frozenset:
    """(inst, pin) pairs for DFF control pins that are the same `Const` or the
    same primary-input NetBit on both sides (see `verify_equivalence`).
    Instances present on only one side are never skipped, so the PO name-set
    check still sees them. If either side has a duplicated DFF instance name,
    nothing is skipped: (inst, pin) is then not a unique key, and skipping
    would swallow the tap-name collision `ABCBridgeError` that would otherwise
    be raised, letting `verify_equivalence` return a wrong EQ (A: two `r1`
    with SN=a/b, B: two `r1` with SN=a/a, no D/Q)."""
    def is_pi(design: Design, nb: NetBit) -> bool:
        sig = design.signals.get(nb.name)
        return sig is not None and sig.direction == Direction.INPUT

    names_a = [g.inst_name for g in design_a.gates if g.gate_type == GateType.DFF]
    names_b = [g.inst_name for g in design_b.gates if g.gate_type == GateType.DFF]
    if len(set(names_a)) != len(names_a) or len(set(names_b)) != len(names_b):
        return frozenset()
    dffs_b = {g.inst_name: g for g in design_b.gates if g.gate_type == GateType.DFF}
    skip = set()
    for ga in design_a.gates:
        if ga.gate_type != GateType.DFF or ga.inst_name not in dffs_b:
            continue
        gb = dffs_b[ga.inst_name]
        for pin in (p for p in DFF_PIN_ORDER if p not in ("D", "Q")):
            va, vb = ga.pins.get(pin), gb.pins.get(pin)
            if va is None or va != vb:
                continue
            # is_pi(design_b, ...) is redundant on normally parsed designs (the
            # PI-set precheck in verify_equivalence rejects a PI-name mismatch
            # first); kept as defence in depth.
            if isinstance(va, Const) or (isinstance(va, NetBit) and is_pi(design_a, va) and is_pi(design_b, vb)):
                skip.add((ga.inst_name, pin))
    return frozenset(skip)


def verify_equivalence(
    design_a: Design,
    design_b: Design,
    signals: Optional[list[str]] = None,
    timeout: float = DEFAULT_VERIFY_TIMEOUT,
) -> EquivResult:
    """Whole-design (or, if `signals` given, per-named-output-cone) Boolean
    equivalence check via ABC `cec`, across the DFF boundary (both designs
    are first passed through `extract_combinational_view(..., "free_pi",
    tap_control_pins=True)`). The boundary therefore includes each DFF's
    RN/SN/CK pins (as `__dff_<PIN>__<inst>` outputs), so rewiring a control
    pin is reported as NOT equivalent -- but only when `signals=None` (the
    per-signal path compares just the named cones). This is on only here: the
    other callers ask about combinational functions and keep the default off.
    A control pin that is unconnected on one side and connected on the other
    changes that side's PO name set, so it raises `ABCBridgeError` (see
    below) rather than returning "not equivalent".

    Skip rule: a (DFF instance, pin) present on both sides whose values are
    identical and are either the same `Const` or the same primary-input net
    is not tapped. Same constant or same PI means the same function by
    construction, so comparing them only costs time (the corpus's control
    pins are almost all constants or PIs). Any NetBit that is not an INPUT-
    direction signal is tapped even when the name matches (a net driven by a
    gate, an undriven net, a DFF Q net), because the logic driving it may have
    changed.

    Raises `ABCBridgeError` if the two designs' post-extraction PI or PO name
    *sets* don't match -- this codebase's transforms never rename ports, and
    DFF renames are mirrored into the snapshot by `Session.mirror_rename`, so
    a mismatch here means something is genuinely wrong upstream, not a normal
    code path. Cheaper and more testable to catch in Python
    than to parse ABC's own mismatch error text for this particular case.

    `signals`, if given, restricts the comparison to just those named
    outputs' fanin cones (one `cec` call per bit, via `_restrict_to_fanin_cone`
    on each side) instead of the whole design -- useful when only a specific
    output is of interest on a huge design.
    """
    skip = _identical_control_pins(design_a, design_b)
    comb_a = extract_combinational_view(design_a, "free_pi", tap_control_pins=True, skip_control_taps=skip)
    comb_b = extract_combinational_view(design_b, "free_pi", tap_control_pins=True, skip_control_taps=skip)

    pi_a = {p.name for p in comb_a.ports if p.direction == Direction.INPUT}
    pi_b = {p.name for p in comb_b.ports if p.direction == Direction.INPUT}
    po_a = {p.name for p in comb_a.ports if p.direction == Direction.OUTPUT}
    po_b = {p.name for p in comb_b.ports if p.direction == Direction.OUTPUT}
    if pi_a != pi_b:
        raise ABCBridgeError(
            "primary input name sets differ between design_a and design_b after "
            f"DFF-boundary extraction: only-in-a={sorted(pi_a - pi_b)}, only-in-b={sorted(pi_b - pi_a)}"
        )
    if po_a != po_b:
        raise ABCBridgeError(
            "primary output name sets differ between design_a and design_b after "
            f"DFF-boundary extraction: only-in-a={sorted(po_a - po_b)}, only-in-b={sorted(po_b - po_a)}"
        )

    if signals is None:
        return _with_dff_legend(_run_cec(comb_a, comb_b, timeout=timeout))

    details = []
    for idx, sig_name in enumerate(signals):
        sig = comb_a.signals[sig_name]
        # Go through Signal.bits() rather than rebuilding the range by hand.
        # The hand-rolled `range(msb, lsb - 1, -1)` this replaces is the same
        # expression that made `Signal.bits()` return [] for an ascending
        # declaration (`wire [0:7] x`) -- and here an empty bit list means the
        # per-signal loop below simply never runs for that signal, so the
        # function reports the two designs EQUIVALENT having compared nothing.
        # Reachable only via the `signals=` argument, which no caller passes
        # today; fixed anyway so the family has one implementation, not two.
        bits = [nb.bit for nb in sig.bits()]
        for bit in bits:
            nb = NetBit(sig_name, bit)
            out_name = f"__verify_eq_out_{idx}_{bit if bit is not None else 0}"
            cone_a = _restrict_to_fanin_cone(comb_a, nb, out_name)
            cone_b = _restrict_to_fanin_cone(comb_b, nb, out_name)
            result = _run_cec(cone_a, cone_b, timeout=timeout)
            if not result.equivalent:
                return _with_dff_legend(result)
            details.append(result.detail)
    return EquivResult(True, "\n".join(details))


def _const_reference_design(pi_ports: list[Port], pi_signals: dict[str, Signal], out_name: str, value: Const) -> Design:
    ref = Design(module_name="abc_bridge_const_ref")
    for p in pi_ports:
        sig = pi_signals[p.name]
        ref.signals[p.name] = Signal(name=p.name, msb=sig.msb, lsb=sig.lsb, direction=Direction.INPUT)
        ref.ports.append(Port(name=p.name, direction=Direction.INPUT))
    ref.signals[out_name] = Signal(name=out_name, msb=None, lsb=None, direction=Direction.OUTPUT)
    ref.ports.append(Port(name=out_name, direction=Direction.OUTPUT))
    ref.add_gate(
        Gate(inst_name=ref.fresh_gate_name(), gate_type=GateType.BUF, pins={"O": NetBit(out_name, None), "I0": value})
    )
    return ref


def is_constant(design: Design, net: NetBit, timeout: float = DEFAULT_ABC_TIMEOUT) -> Optional[Const]:
    """Whether `net` is provably constant across every reachable state (every
    flop tied to 0, per `extract_combinational_view`'s "const_zero" mode) and
    every PI assignment. Returns Const.ZERO/Const.ONE if provably so, else
    None. Two `cec` calls (against a Const.ZERO and a Const.ONE reference
    design sharing the cone's exact PI list) on just `net`'s fanin cone, not
    the whole design.
    """
    comb = extract_combinational_view(design, "const_zero")
    cone = _restrict_to_fanin_cone(comb, net, "is_constant_out")
    pi_ports = [p for p in cone.ports if p.direction == Direction.INPUT]
    out_name = next(p.name for p in cone.ports if p.direction == Direction.OUTPUT)

    zero_ref = _const_reference_design(pi_ports, cone.signals, out_name, Const.ZERO)
    one_ref = _const_reference_design(pi_ports, cone.signals, out_name, Const.ONE)

    zero_result = verify_equivalence(cone, zero_ref, timeout=timeout)
    one_result = verify_equivalence(cone, one_ref, timeout=timeout)
    if zero_result.equivalent and one_result.equivalent:
        raise ABCBridgeError(
            "internal invariant violated: net reported equivalent to BOTH constant 0 "
            "and constant 1 -- this should be logically impossible"
        )
    if zero_result.equivalent:
        return Const.ZERO
    if one_result.equivalent:
        return Const.ONE
    return None


def check_implication(
    design: Design,
    net: NetBit,
    promoted_q_source: Optional[dict[str, NetBit]] = None,
    timeout: float = DEFAULT_VERIFY_TIMEOUT,
) -> EquivResult:
    """Whether `net` is provably constant-1 (true across every reachable
    flop state -- every DFF Q free per `extract_combinational_view`'s
    "free_pi" mode -- and every PI assignment). Same underlying check as
    `is_constant`'s constant-1 half, but returns the full `EquivResult`
    (preserving ABC's counterexample text when the property does NOT hold)
    instead of collapsing the answer to `Optional[Const]`. Used by
    property-verification callers (netlist_agent/property_check.py's
    "asserted only when ..." handler) that need a concrete counterexample
    assignment when a property fails, not just a yes/no answer; `is_constant`
    itself is left untouched (its existing callers only ever want the
    yes/no/which-constant answer).

    `promoted_q_source`, if given, is populated exactly as
    `extract_combinational_view` populates it -- lets a caller distinguish a
    counterexample's PI names that are genuine primary inputs of `design`
    from ones that are free-running DFF-Q promotions (whose "any value"
    freedom is a real over-approximation of reachable states; see that
    function's docstring).
    """
    comb = extract_combinational_view(design, "free_pi", promoted_q_source)
    cone = _restrict_to_fanin_cone(comb, net, "check_implication_out")
    pi_ports = [p for p in cone.ports if p.direction == Direction.INPUT]
    out_name = next(p.name for p in cone.ports if p.direction == Direction.OUTPUT)

    one_ref = _const_reference_design(pi_ports, cone.signals, out_name, Const.ONE)
    return verify_equivalence(cone, one_ref, timeout=timeout)


def _swap_consumers(comb: Design, a: NetBit, b: NetBit) -> Design:
    """Copy of `comb` with every gate INPUT pin whose value equals `a`
    rewritten to `b` and vice versa. Output (driver) pins are left alone, so
    swapping two internal nets exchanges what their readers see, not what
    drives them.
    """

    def _swap_val(v: Pin) -> Pin:
        if v == a:
            return b
        if v == b:
            return a
        return v

    swapped = Design(module_name=comb.module_name)
    for name, sig in comb.signals.items():
        swapped.signals[name] = Signal(name=sig.name, msb=sig.msb, lsb=sig.lsb, direction=sig.direction)
    swapped.ports = [Port(name=p.name, direction=p.direction) for p in comb.ports]
    for g in comb.gates:
        out_pin = OUTPUT_PIN[g.gate_type]
        new_pins = {pin: (val if pin == out_pin else _swap_val(val)) for pin, val in g.pins.items()}
        swapped.add_gate(Gate(inst_name=g.inst_name, gate_type=g.gate_type, pins=new_pins))
    return swapped


def _build_xor_miter(cone_a: Design, cone_b: Design, suffix_a: str, suffix_b: str) -> Design:
    """Combine two structurally-identical-PI-list cones into one Design,
    XOR-ing their respective single outputs into one miter output net. Every
    non-PI net/gate name is suffixed per side (`suffix_a`/`suffix_b`) to avoid
    collisions -- both cones were built by copying the same source signal
    namespace (or two cones of the very same source design), so their
    internal names can otherwise collide/alias onto the same wires if merged
    unrenamed. Shared by `check_symmetry` (original vs. input-swapped cone of
    one signal) and `are_equivalent` (two different signals' cones).
    """
    miter = Design(module_name="abc_bridge_xor_miter")
    pi_names = {p.name for p in cone_a.ports if p.direction == Direction.INPUT}
    for name in pi_names:
        sig = cone_a.signals[name]
        miter.signals[name] = Signal(name=name, msb=sig.msb, lsb=sig.lsb, direction=Direction.INPUT)
        miter.ports.append(Port(name=name, direction=Direction.INPUT))

    out_names: dict[str, str] = {}

    def _copy_gates(src: Design, suffix: str) -> None:
        def _rename(val: Pin) -> Pin:
            if isinstance(val, NetBit) and val.name not in pi_names:
                new_name = f"{val.name}{suffix}"
                if new_name not in miter.signals:
                    orig_sig = src.signals[val.name]
                    miter.signals[new_name] = Signal(
                        name=new_name, msb=orig_sig.msb, lsb=orig_sig.lsb, direction=Direction.INTERNAL
                    )
                return NetBit(new_name, val.bit)
            return val

        for g in src.gates:
            new_pins = {pin: _rename(v) for pin, v in g.pins.items()}
            miter.add_gate(Gate(inst_name=f"{g.inst_name}{suffix}", gate_type=g.gate_type, pins=new_pins))

        out_port = next(p.name for p in src.ports if p.direction == Direction.OUTPUT)
        out_names[suffix] = f"{out_port}{suffix}"

    _copy_gates(cone_a, suffix_a)
    _copy_gates(cone_b, suffix_b)

    miter.signals["sym_miter_out"] = Signal(name="sym_miter_out", msb=None, lsb=None, direction=Direction.OUTPUT)
    miter.ports.append(Port(name="sym_miter_out", direction=Direction.OUTPUT))
    miter.add_gate(
        Gate(
            inst_name=miter.fresh_gate_name(),
            gate_type=GateType.XOR,
            pins={
                "O": NetBit("sym_miter_out", None),
                "I0": NetBit(out_names[suffix_a], None),
                "I1": NetBit(out_names[suffix_b], None),
            },
        )
    )
    return miter


def check_symmetry(
    design: Design, output_net: NetBit, input_a: NetBit, input_b: NetBit, timeout: float = DEFAULT_ABC_TIMEOUT
) -> bool:
    """Positive symmetry only (per contest clarification -- no complemented/
    negative symmetry): whether swapping `input_a` <-> `input_b` everywhere
    in `output_net`'s fanin cone never changes `output_net`'s value, for any
    PI/flop-state assignment.

    Builds `output_net`'s fanin cone (free_pi mode) twice -- original wiring,
    and with input_a/input_b swapped across every gate -- XORs the two
    cones' outputs into one miter, and asks `is_constant`-style (cec against
    a Const.ZERO reference) whether that miter is always 0.

    The vacuous case (neither input actually appears in the cone) needs no
    special-casing: swapping a net that's not referenced anywhere is a
    no-op, so the two cone copies end up structurally identical and the
    miter trivially proves constant-0 -- symmetric by definition.

    If an input is an internal net, "swap" means exchanging the two wires'
    readers. When each is in the other's fanin cone the swap creates a
    combinational loop, and ABC fails loudly (`ABCBridgeError`).
    """
    comb = extract_combinational_view(design, "free_pi")
    cone_orig = _restrict_to_fanin_cone(comb, output_net, "sym_orig_out")
    cone_swap = _restrict_to_fanin_cone(_swap_consumers(comb, input_a, input_b), output_net, "sym_orig_out")
    miter = _build_xor_miter(cone_orig, cone_swap, "_orig", "_swap")

    pi_ports = [p for p in miter.ports if p.direction == Direction.INPUT]
    zero_ref = _const_reference_design(pi_ports, miter.signals, "sym_miter_out", Const.ZERO)

    result = verify_equivalence(miter, zero_ref, timeout=timeout)
    return result.equivalent


def are_equivalent(design: Design, net_a: NetBit, net_b: NetBit, timeout: float = DEFAULT_ABC_TIMEOUT) -> bool:
    """Whether two internal signals of the SAME design compute the identical
    Boolean function of the primary inputs (and free-running flop state), for
    every reachable assignment -- e.g. "are internal signals n1035 and n1029
    functionally equivalent". `verify_equivalence` compares two whole/cone
    designs against each other; `is_constant` compares one signal against a
    fixed 0/1; neither answers "these two named nets within one design".

    Built the same way `check_symmetry` builds its miter -- two fanin cones
    (free_pi mode) XOR'd together and checked against a constant-0 reference
    via `verify_equivalence` -- just without the input-swapping step: there is
    only one cone per net here, not an original-vs-swapped variant of one.
    """
    comb = extract_combinational_view(design, "free_pi")
    cone_a = _restrict_to_fanin_cone(comb, net_a, "eq_a_out")
    cone_b = _restrict_to_fanin_cone(comb, net_b, "eq_b_out")
    miter = _build_xor_miter(cone_a, cone_b, "_a", "_b")

    pi_ports = [p for p in miter.ports if p.direction == Direction.INPUT]
    zero_ref = _const_reference_design(pi_ports, miter.signals, "sym_miter_out", Const.ZERO)

    result = verify_equivalence(miter, zero_ref, timeout=timeout)
    return result.equivalent
