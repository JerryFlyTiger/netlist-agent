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

  2. Equivalence checking is `cec fileA.v fileB.v` (two file paths, no prior
     `read_verilog` needed). Confirmed verbatim output patterns:
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
     failed" (unreadable file). This module raises `ABCBridgeError` if stdout
     matches none of the known patterns, rather than silently guessing.
     Unused/extra PI ports declared on both sides are harmless as long as
     both sides declare the identical PI (and PO) name *sets* -- this module
     guarantees that by construction (it builds both files) and additionally
     pre-flight-checks it in Python before ever shelling out to ABC.

  3. Undriven/floating internal wires produce a harmless stdout warning
     ("Warning: Constant-0 drivers added to N non-driven nets...") -- not an
     error, never treated as one here.
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
    Design,
    Direction,
    Gate,
    GateType,
    NetBit,
    Pin,
    Port,
    Signal,
)
from netlist_agent.writer import write_verilog

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIND_ABC_SCRIPT = os.path.join(REPO_ROOT, "scripts", "find_abc.sh")

# Generous default for a single ABC invocation; a pathological/hung ABC run
# raises ABCBridgeError (via subprocess.TimeoutExpired) rather than hanging
# silently. Callers of the public functions below may override per call.
DEFAULT_ABC_TIMEOUT = 120.0
_RESOLVE_TIMEOUT = 30.0

DffQMode = Literal["free_pi", "const_zero"]


class ABCBridgeError(Exception):
    """Raised whenever this module cannot proceed with confidence: the ABC
    binary can't be resolved/invoked (including a timeout), ABC's stdout
    matches none of the known equivalent/not-equivalent/failure patterns, ABC
    itself reports it could not build the miter (e.g. mismatched PI/PO sets
    -- caught pre-flight in Python instead, see `verify_equivalence`), or an
    internal invariant of this module is violated (e.g. a net reported
    equivalent to both constant 0 and constant 1)."""


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


def _run_abc(script: str, timeout: float) -> str:
    abc_path = _resolve_abc()
    try:
        result = subprocess.run([abc_path, "-c", script], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ABCBridgeError(f"ABC invocation timed out after {timeout}s running {script!r}") from exc
    return result.stdout


def _parse_cec_output(stdout: str) -> "EquivResult":
    if "Networks are equivalent" in stdout:
        return EquivResult(True, stdout.strip())
    if "Networks are NOT EQUIVALENT" in stdout:
        return EquivResult(False, stdout.strip())
    if "Miter computation has failed" in stdout or "Reading network from file has failed" in stdout:
        raise ABCBridgeError(f"ABC could not compare the two networks: {stdout.strip()}")
    raise ABCBridgeError(
        f"unrecognized ABC `cec` output (matched neither the equivalent nor the "
        f"not-equivalent pattern) -- failing loudly instead of guessing: {stdout.strip()}"
    )


def _run_cec(design_a: Design, design_b: Design, timeout: float) -> "EquivResult":
    with tempfile.TemporaryDirectory(prefix="abc_bridge_") as tmpdir:
        path_a = os.path.join(tmpdir, "a.v")
        path_b = os.path.join(tmpdir, "b.v")
        write_verilog(design_a, path_a)
        write_verilog(design_b, path_b)
        stdout = _run_abc(f'cec "{path_a}" "{path_b}"', timeout=timeout)
    return _parse_cec_output(stdout)


# ----------------------------------------------------------------------
# Capability 1: sequential -> combinational DFF-boundary extraction
# ----------------------------------------------------------------------


def _set_or_add_port(design: Design, name: str, direction: Direction) -> None:
    for p in design.ports:
        if p.name == name:
            p.direction = direction
            return
    design.ports.append(Port(name=name, direction=direction))


def _split_bit_to_fresh_input(design: Design, nb: NetBit) -> NetBit:
    """Split one bit of a bus off into its own fresh single-bit primary
    INPUT: every gate pin currently referencing `nb` is rewired to the fresh
    net-bit instead, and the fresh net is added as a new input port. `nb`'s
    original signal (and every one of its *other* bits) is left completely
    untouched -- confirmed to occur for real (test39): a DFF's Q pin can be a
    bit-select of a wider bus whose other bits are independently driven by
    ordinary combinational gates, so whole-Signal-granularity Direction
    promotion (the only kind this IR's Port/Signal model supports directly)
    would leave those other bits simultaneously a primary-input bit and
    gate-driven -- an invalid multiply-driven net. Since `nb` itself was
    driven only by the (already-excluded) DFF, nothing in `design` drives it
    after the split; ABC tolerates an undriven net gracefully (a harmless
    "Constant-0 drivers added" warning), same as it already does for
    ordinary floating ports in these testcases.
    """
    fresh = design.fresh_net("t_dffq_split_")
    for gate in design.gates:
        for pin_name, value in list(gate.pins.items()):
            if value == nb:
                design.rewire_pin(gate, pin_name, fresh)
    design.signals[fresh.name].direction = Direction.INPUT
    design.ports.append(Port(name=fresh.name, direction=Direction.INPUT))
    return fresh


def extract_combinational_view(
    design: Design, dff_q_mode: DffQMode, promoted_q_source: Optional[dict[str, NetBit]] = None
) -> Design:
    """Return a new, purely-combinational `Design` (never mutates `design`):
    every non-DFF gate is copied unchanged, and every DFF instance is dropped
    after its Q/D boundary is turned into ports or ties so the result never
    contains a `dff` instance (see module docstring, point 1, for why that
    matters for everything downstream that hands this to ABC).

    `dff_q_mode`:
      - "free_pi": every DFF's Q net becomes a genuine new primary INPUT
        (free variable). Used for equivalence/symmetry checking, where a
        flop's stored value must range over both 0 and 1.
      - "const_zero": every DFF's Q net is tied to Const.ZERO via a
        synthesized BUF gate instead of becoming a port -- used only by
        `is_constant` (via the cone-restriction helper below) to ask "is
        this net constant when every flop happens to hold 0".

    `promoted_q_source`, if given (mutated in place; only meaningful for
    "free_pi"), is populated with one entry per promoted DFF-Q primary
    input: the PI's SIGNAL NAME in the returned Design -> the original Q
    NetBit in `design` it stands for. For an ordinary promotion this is a
    trivial same-name entry (`nb.name -> nb`); for the `_split_bit_to_fresh_input`
    corner it is the only way to recover which original net-bit a fresh
    split name (e.g. "t_dffq_split_0") actually represents, since that name
    exists nowhere in `design` itself. abc_synth.py's whole-design/cone depth
    optimizers need this to correctly wire newly-synthesized gates (which
    reference the extracted view's PI names) back into `design`'s own
    namespace.

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
    than asserting it in prose. The Q side stays keyed by net name: Q nets are
    driver-side and none of the existing transforms rewire or rename them.
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

    # Dedupe by net-bit identity (name+bit), not just by signal name: two
    # distinct DFF Q/D pins may share a signal *name* while addressing
    # different bits of the same bus (only matters for const_zero's
    # per-net-bit BUF ties below; port/direction promotion operates at
    # Signal granularity regardless, since Port has no per-bit direction).
    q_netbits: dict[NetBit, None] = {}
    for g in dff_gates:
        q = g.pins.get("Q")
        if isinstance(q, NetBit):
            q_netbits.setdefault(q, None)

    # Deterministic promotion order: ALL Q promotions first, forcing INPUT
    # direction and overwriting any pre-existing (e.g. OUTPUT) Port entry of
    # the same name -- INPUT status always wins. This resolves the "DFF.Q
    # wired straight to a PO" case (the Q net was already a declared output)
    # by keeping it INPUT rather than leaving a conflicting output port.
    if dff_q_mode == "free_pi":
        seen_q_names: set[str] = set()
        for nb in q_netbits:
            if nb.name in seen_q_names:
                continue
            sig = new_design.signals[nb.name]
            # Direction lives on the whole Signal in this IR, so promoting nb
            # to INPUT at Signal granularity is only safe when no *other* bit
            # of the same bus is independently driven by an ordinary gate --
            # otherwise that sibling would end up simultaneously a
            # primary-input bit and gate-driven (an invalid multiply-driven
            # net). When that conflict exists, split just this one bit off
            # into its own fresh single-bit input instead (see
            # `_split_bit_to_fresh_input`) rather than promoting -- or
            # refusing to promote -- the whole bus; every other bit of `sig`
            # (including any other DFF's Q sharing this same bus, which is
            # not itself in `net_driver` and so never trips this check) is
            # left exactly as it was.
            conflict = any(
                other_nb != nb and other_nb in new_design.net_driver for other_nb in sig.bits()
            )
            if conflict:
                fresh = _split_bit_to_fresh_input(new_design, nb)
                if promoted_q_source is not None:
                    promoted_q_source[fresh.name] = nb
                continue
            seen_q_names.add(nb.name)
            sig.direction = Direction.INPUT
            _set_or_add_port(new_design, nb.name, Direction.INPUT)
            if promoted_q_source is not None:
                promoted_q_source[nb.name] = nb
    else:
        for nb in q_netbits:
            new_design.add_gate(
                Gate(
                    inst_name=new_design.fresh_gate_name(),
                    gate_type=GateType.BUF,
                    pins={"O": nb, "I0": Const.ZERO},
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


def verify_equivalence(
    design_a: Design,
    design_b: Design,
    signals: Optional[list[str]] = None,
    timeout: float = DEFAULT_ABC_TIMEOUT,
) -> EquivResult:
    """Whole-design (or, if `signals` given, per-named-output-cone) Boolean
    equivalence check via ABC `cec`, across the DFF boundary (both designs
    are first passed through `extract_combinational_view(..., "free_pi")`).

    Raises `ABCBridgeError` if the two designs' post-extraction PI or PO name
    *sets* don't match -- this codebase's transforms never rename ports or
    DFFs, so a mismatch here means something is genuinely wrong upstream,
    not a normal code path. Cheaper and more testable to catch in Python
    than to parse ABC's own mismatch error text for this particular case.

    `signals`, if given, restricts the comparison to just those named
    outputs' fanin cones (one `cec` call per bit, via `_restrict_to_fanin_cone`
    on each side) instead of the whole design -- useful when only a specific
    output is of interest on a huge design.
    """
    comb_a = extract_combinational_view(design_a, "free_pi")
    comb_b = extract_combinational_view(design_b, "free_pi")

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
        return _run_cec(comb_a, comb_b, timeout=timeout)

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
                return result
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
    timeout: float = DEFAULT_ABC_TIMEOUT,
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


def _swap_in_cone(cone: Design, a: NetBit, b: NetBit) -> Design:
    """Copy of `cone` with every gate pin whose value equals `a` rewritten to
    `b` and vice versa (a full swap across every gate in the cone). Building
    the swap on the already-cone-restricted design rather than on the whole
    comb design first is equivalent (swapping a leaf PI value never changes
    which gates are backward-reachable) and cheaper.
    """

    def _swap_val(v: Pin) -> Pin:
        if v == a:
            return b
        if v == b:
            return a
        return v

    swapped = Design(module_name=cone.module_name)
    for name, sig in cone.signals.items():
        swapped.signals[name] = Signal(name=sig.name, msb=sig.msb, lsb=sig.lsb, direction=sig.direction)
    swapped.ports = [Port(name=p.name, direction=p.direction) for p in cone.ports]
    for g in cone.gates:
        new_pins = {pin: _swap_val(val) for pin, val in g.pins.items()}
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
    """
    comb = extract_combinational_view(design, "free_pi")
    cone_orig = _restrict_to_fanin_cone(comb, output_net, "sym_orig_out")
    cone_swap = _swap_in_cone(cone_orig, input_a, input_b)
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
