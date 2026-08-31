"""ABC-backed logic restructuring for depth (critical-path) optimization.

Where abc_bridge.py answers Boolean-semantic YES/NO questions (equivalence,
constant-ness, symmetry) by shelling out to ABC's `cec`, this module asks ABC
to actually REWRITE a design's combinational logic to reduce depth (this
project's own `NetlistGraph.max_design_depth`/`depth_to_sink` gate-level
metric, not ABC's internal AIG level count -- the two do not correspond 1:1,
so every result here is re-measured with this project's own yardstick after
splicing back, never trusted from ABC's own stdout).

Empirical findings that shaped this module (confirmed by hand against the
resolved ABC binary before writing the production code below -- see
abc_bridge.py's module docstring for the same discipline applied to stage 4):

  1. Technology-mapping to a small SIS-genlib library is exactly the basis
     enforcement mechanism it looks like: `read_genlib lib.genlib; map` maps
     the strashed AIG onto ONLY the gates declared in the library, so "the
     restructured cone may only use NAND and NOT" falls straight out of which
     GATE lines the library contains -- no post-hoc gate-type filtering
     needed. Gate/pin names in the genlib are ours to choose; short synthetic
     names (`and2`, `not1`, `buf1`, ...) keyed back to GateType by a fixed
     table make the BLIF read-back trivial.

  2. A genlib WITHOUT an explicit 0-input constant-generator gate
     (`GATE zero 0 O=CONST0;` / `GATE one 0 O=CONST1;`) makes `map` segfault
     (not error cleanly) the moment the strashed network contains an actual
     constant node -- which happens for real, even on ordinarily-behaved
     designs: ABC ties any floating/undriven net to constant 0 during
     `strash` (a harmless warning elsewhere in this codebase, e.g.
     abc_bridge.py's DFF-boundary extraction deliberately leaves some nets
     floating), and that tie alone is enough to crash a const-gate-less
     `map`. Every genlib generated here therefore always includes both
     constant gates, unconditionally.

  3. Depth-oriented pre-map script: `strash; balance; dch` beats every other
     sequence tried (`rewrite`/`refactor` rounds, `dch` followed by more
     `balance`/`rewrite`, resyn2-style scripts) on real corpus files, judged
     by the FINAL MAPPED depth (`map`'s reported `lev`, not the AIG's own
     level count): test22's combinational view maps to depth 47 (vs. 48-49
     for balance/rewrite-only scripts), and test28's (an AND/NOT-only,
     48k-gate design after an earlier `remap_to_basis` step) maps to depth
     104 (vs. 125-130 for scripts that add more balance/rewrite AFTER dch --
     empirically that makes this particular design's mapped depth WORSE, not
     better, so the script stops at plain `dch`).

  4. The mapped BLIF's `.gate` lines are read back via a small
     hand-written parser (not a general BLIF library): `.inputs`/`.outputs`
     wrap onto continuation lines ending in a backslash (confirmed on a
     31-PI synthetic design); unused PIs still appear in `.inputs` (an
     "unused_pi" port with zero fanout was still declared); gate name
     tokens are whitespace-padded for column alignment when multiple gate
     names differ in length, so splitting on whitespace (not fixed columns)
     is required. With both constant gates and a `buf1` (pure identity)
     gate always present in the library, ABC was never observed to fall
     back to a raw `.names` truth-table line for a constant or feedthrough
     PO in any of the cases tried (including a deliberately-redundant
     double-buffer/double-inverter chain, which mapped straight to a single
     `buf1`) -- this module treats a `.names` line as a hard parse error
     instead of silently guessing at a truth table it was never designed to
     interpret.
"""

from __future__ import annotations

import copy
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from netlist_agent.abc_bridge import (
    ABCBridgeError,
    DEFAULT_ABC_TIMEOUT,
    _restrict_to_fanin_cone,
    _resolve_abc,
    extract_combinational_view,
    verify_equivalence,
)
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import Const, Design, Gate, GateType, NetBit, OUTPUT_PIN, Pin
from netlist_agent.netref import netbit_token, parse_net
from netlist_agent.transform import remove_dangling_gates
from netlist_agent.writer import write_verilog

# ----------------------------------------------------------------------
# Genlib generation (empirical finding 1 & 2 above)
# ----------------------------------------------------------------------

# genlib gate name -> (GateType it represents, Boolean formula body, pin phase)
_GENLIB_GATE_INFO: dict[str, tuple[GateType, str, str]] = {
    "and2": (GateType.AND, "a*b", "NONINV"),
    "or2": (GateType.OR, "a+b", "NONINV"),
    "nand2": (GateType.NAND, "!(a*b)", "INV"),
    "nor2": (GateType.NOR, "!(a+b)", "INV"),
    "xor2": (GateType.XOR, "a^b", "UNKNOWN"),
    "xnor2": (GateType.XNOR, "!(a^b)", "UNKNOWN"),
    "not1": (GateType.NOT, "!a", "INV"),
    "buf1": (GateType.BUF, "a", "NONINV"),
}
_TWO_INPUT_GENLIB_NAMES = frozenset({"and2", "or2", "nand2", "nor2", "xor2", "xnor2"})
_ONE_INPUT_GENLIB_NAMES = frozenset({"not1", "buf1"})
# genlib gate name -> the constant it ties its output to (0-input gates,
# handled outside _GENLIB_GATE_INFO since they carry no GateType of their own
# -- reconstructed as a BUF-tied-to-Const, matching how the rest of this
# codebase already represents a constant-driven net, e.g.
# abc_bridge._const_reference_design).
_CONST_GENLIB_NAMES: dict[str, Const] = {"zero": Const.ZERO, "one": Const.ONE}

# Basis name -> which 2-input genlib gates that basis allows (NOT and BUF are
# implicitly always available -- NOT is a member of every basis this project
# recognizes, matching transform.BASES' own convention; BUF is a free wire).
# Naming matches transform.BASES exactly (and extends it with "and_or_not"),
# so a basis string means the same thing in both modules.
BASIS_GATE_NAMES: dict[str, frozenset[str]] = {
    "and_not": frozenset({"and2"}),
    "nand_not": frozenset({"nand2"}),
    "nor_not": frozenset({"nor2"}),
    "and_or_not": frozenset({"and2", "or2"}),
}
_FULL_BASIS_NAMES = _TWO_INPUT_GENLIB_NAMES


def _validate_basis(basis: Optional[str]) -> None:
    if basis is not None and basis not in BASIS_GATE_NAMES:
        raise ValueError(f"unsupported basis {basis!r}; choose one of {sorted(BASIS_GATE_NAMES)} or None")


def allowed_gate_types(basis: Optional[str]) -> frozenset[GateType]:
    """The GateTypes a resynthesized cone/design is allowed to contain under
    `basis` (None = unrestricted: any of the 6 two-input primitives plus
    NOT/BUF). Exposed publicly so tests can assert basis compliance without
    hand-duplicating this module's own basis table."""
    _validate_basis(basis)
    names = _FULL_BASIS_NAMES if basis is None else BASIS_GATE_NAMES[basis]
    return frozenset({_GENLIB_GATE_INFO[n][0] for n in names} | {GateType.NOT, GateType.BUF})


def _genlib_text(basis: Optional[str]) -> str:
    _validate_basis(basis)
    two_input_names = _FULL_BASIS_NAMES if basis is None else BASIS_GATE_NAMES[basis]
    lines = ["GATE zero   0  O=CONST0;", "GATE one    0  O=CONST1;"]
    for name in sorted(two_input_names) + ["not1", "buf1"]:
        gate_type, formula, phase = _GENLIB_GATE_INFO[name]
        # Unit delay throughout (block delay 1, zero fanout-dependent delay):
        # this is what makes ABC's own reported `lev` a meaningful per-gate
        # depth count in the first place, matching this project's own
        # gate-level depth metric closely enough to guide the optimizer --
        # the actual acceptance test is still re-measuring with our own
        # analysis.py after splicing back, never trusting ABC's number.
        lines.append(f"GATE {name:<6} 1  O={formula};   PIN * {phase} 1 999 1 0 1 0")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# BLIF read-back (empirical finding 4 above)
# ----------------------------------------------------------------------


class _SynthError(Exception):
    """Any failure of the ABC round trip below (process failure/crash,
    timeout, or a mapped BLIF this module cannot make sense of) -- always
    caught by optimize_depth/optimize_cone_depth and degraded to the
    verified no-op path, per this module's acceptance-gating contract."""


@dataclass(frozen=True)
class _BlifGate:
    name: str
    pins: dict[str, str]


@dataclass(frozen=True)
class _BlifNetlist:
    inputs: list[str]
    outputs: list[str]
    gates: list[_BlifGate]


def _read_logical_lines(text: str) -> list[str]:
    """Join BLIF's backslash-continued lines into one logical line each
    (confirmed empirically: ABC wraps long `.inputs`/`.outputs` lines this
    way). Blank lines and '#' comment lines are dropped."""
    logical: list[str] = []
    pending: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending.append(line[:-1].strip())
        else:
            pending.append(line)
            logical.append(" ".join(pending))
            pending = []
    if pending:
        logical.append(" ".join(pending))
    return logical


def parse_blif(text: str) -> _BlifNetlist:
    """Parse a mapped BLIF netlist (as written by this module's own
    `write_blif` invocation) into inputs/outputs/gates. Deliberately narrow:
    only handles the directives this module's own ABC script ever produces
    (`.model`/`.inputs`/`.outputs`/`.gate`/`.end`); a `.names` line is
    treated as a hard error rather than a truth table this module was never
    designed to interpret (see module docstring, finding 4)."""
    inputs: list[str] = []
    outputs: list[str] = []
    gates: list[_BlifGate] = []
    for line in _read_logical_lines(text):
        tokens = line.split()
        if not tokens:
            continue
        directive = tokens[0]
        if directive == ".model":
            continue
        elif directive == ".inputs":
            inputs.extend(tokens[1:])
        elif directive == ".outputs":
            outputs.extend(tokens[1:])
        elif directive == ".gate":
            if len(tokens) < 2:
                raise _SynthError(f"malformed .gate line: {line!r}")
            pins: dict[str, str] = {}
            for tok in tokens[2:]:
                if "=" not in tok:
                    raise _SynthError(f"malformed gate pin token {tok!r} in line: {line!r}")
                pin, _, net = tok.partition("=")
                pins[pin] = net
            gates.append(_BlifGate(tokens[1], pins))
        elif directive == ".names":
            raise _SynthError(
                "unexpected '.names' line in ABC-mapped BLIF -- this module's genlibs always "
                f"provide zero/one/buf1 gates so a truth-table fallback should never be needed: {line!r}"
            )
        elif directive == ".end":
            break
        # Anything else (.default_input_arrival, etc.) is silently ignored.
    return _BlifNetlist(inputs=inputs, outputs=outputs, gates=gates)


# ----------------------------------------------------------------------
# The ABC round trip itself
# ----------------------------------------------------------------------

# Pre-map optimization script (empirical finding 3 above; also the sole
# depth-oriented member of _AREA_CANDIDATE_SCRIPTS below).
_OPT_SCRIPT = "strash; balance; dch"

# Gate-count (area) candidate scripts, in preference order (see the
# measurement writeup this module's docstring links nowhere but this task's
# own scratchpad -- summary: `strash; balance; dch` is already the best-or-
# near-best gate count in 6/8 measured design/mode combos AND the best depth
# in 8/8, so it stays candidate #1; `strash; dc2; resub; dc2` is the only
# script that reliably beats it on raw gate count, but at a depth cost that
# grows with design size (seen up to +24%), so it is only ever a candidate,
# never a replacement -- optimize_gate_count/optimize_cone_gate_count below
# accept it only when it independently passes every acceptance gate,
# including any caller-supplied hard depth budget).
_AREA_CANDIDATE_SCRIPTS: tuple[str, ...] = (_OPT_SCRIPT, "strash; dc2; resub; dc2")


def _run_abc_synthesis(view: Design, basis: Optional[str], timeout: float, opt_script: str = _OPT_SCRIPT) -> _BlifNetlist:
    genlib_text = _genlib_text(basis)
    with tempfile.TemporaryDirectory(prefix="abc_synth_") as tmpdir:
        v_path = os.path.join(tmpdir, "in.v")
        lib_path = os.path.join(tmpdir, "basis.genlib")
        blif_path = os.path.join(tmpdir, "out.blif")
        write_verilog(view, v_path)
        with open(lib_path, "w") as f:
            f.write(genlib_text)
        script = (
            f'read_verilog "{v_path}"; {opt_script}; '
            f'read_genlib "{lib_path}"; map; write_blif "{blif_path}"'
        )
        try:
            abc_path = _resolve_abc()
        except ABCBridgeError as exc:
            raise _SynthError(str(exc)) from exc
        try:
            result = subprocess.run([abc_path, "-c", script], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise _SynthError(f"ABC synthesis timed out after {timeout}s") from exc
        if result.returncode != 0:
            # Covers both a clean ABC-reported error and a hard crash (e.g. a
            # segfault, seen for real during development before finding 2
            # above was understood) -- either way, treated identically as a
            # failed synthesis attempt, never propagated as a crash.
            detail = (result.stderr or result.stdout or "").strip()
            raise _SynthError(f"ABC synthesis process exited with code {result.returncode}: {detail}")
        if not os.path.exists(blif_path) or os.path.getsize(blif_path) == 0:
            raise _SynthError(f"ABC synthesis produced no BLIF output: {result.stdout.strip()}")
        with open(blif_path) as f:
            blif_text = f.read()
    return parse_blif(blif_text)


# ----------------------------------------------------------------------
# Splicing the synthesized gates into the working Design (by duplication,
# never in-place mutation of the original cone/design -- see module docstring
# of optimize_depth/optimize_cone_depth below for the rationale).
# ----------------------------------------------------------------------


def _token_resolver(
    work: Design, pi_tokens: frozenset[str], promoted_q_source: dict[str, NetBit], net_prefix: str
) -> Callable[[str], NetBit]:
    """Builds a memoized token -> NetBit resolver for `work`'s namespace: a
    PI token resolves to the REAL net it stands for (a promoted/split DFF-Q
    net via `promoted_q_source`, or an ordinary PI parsed by name); every
    other token (an ABC-invented internal wire, or one of our own PO/D-tap
    output names) gets a brand-new net freshly allocated in `work` -- this is
    the "duplication" half of the splice-by-duplication approach: nothing
    here ever reuses an existing internal net name from the pre-optimization
    cone/design.
    """
    cache: dict[str, NetBit] = {}

    def resolve(token: str) -> NetBit:
        if token in cache:
            return cache[token]
        if token in pi_tokens:
            nb = promoted_q_source.get(token)
            if nb is None:
                nb = parse_net(token)
        else:
            nb = work.fresh_net(net_prefix)
        cache[token] = nb
        return nb

    return resolve


def _add_synth_gates(
    work: Design, blif: _BlifNetlist, resolve: Callable[[str], NetBit], gate_prefix: str
) -> dict[str, NetBit]:
    """Add every BLIF gate to `work` as a brand-new Gate (fresh instance
    name via `work.fresh_gate_name`), translating genlib gate names back to
    real GateTypes (or a Const tie, for the 0-input constant gates). Returns
    token -> NetBit for each of `blif.outputs`, which the caller splices into
    the real design's PO/D-pin/target wiring.
    """
    for g in blif.gates:
        out_token = g.pins.get("O")
        if out_token is None:
            raise _SynthError(f"BLIF gate {g.name!r} has no O pin: {g.pins!r}")
        out_nb = resolve(out_token)

        const_value = _CONST_GENLIB_NAMES.get(g.name)
        if const_value is not None:
            pins: dict[str, Pin] = {"O": out_nb, "I0": const_value}
            work.add_gate(Gate(inst_name=work.fresh_gate_name(gate_prefix), gate_type=GateType.BUF, pins=pins))
            continue

        info = _GENLIB_GATE_INFO.get(g.name)
        if info is None:
            raise _SynthError(f"unrecognized mapped gate {g.name!r} in ABC output BLIF")
        gate_type = info[0]
        if g.name in _ONE_INPUT_GENLIB_NAMES:
            in_token = g.pins.get("a")
            if in_token is None:
                raise _SynthError(f"1-input gate {g.name!r} is missing pin 'a': {g.pins!r}")
            pins = {"O": out_nb, "I0": resolve(in_token)}
        else:
            a_token, b_token = g.pins.get("a"), g.pins.get("b")
            if a_token is None or b_token is None:
                raise _SynthError(f"2-input gate {g.name!r} is missing pin(s) 'a'/'b': {g.pins!r}")
            pins = {"O": out_nb, "I0": resolve(a_token), "I1": resolve(b_token)}
        work.add_gate(Gate(inst_name=work.fresh_gate_name(gate_prefix), gate_type=gate_type, pins=pins))

    return {token: resolve(token) for token in blif.outputs}


_DFF_D_TAP_PREFIX = "__dff_D__"  # must match abc_bridge.extract_combinational_view's own convention


def _retap_output(work: Design, real_out: NetBit, fresh_source: NetBit) -> None:
    """Make `real_out` (a real PO/target net-bit whose identity is pinned to
    its net name) carry `fresh_source`'s value instead of whatever used to
    drive it: rewire the OLD driver's output pin (if any) off to a fresh dead
    net first -- never deleting that old gate outright, since it may still
    have other fanout elsewhere (matching the same why-not-just-delete
    reasoning transform.py's own PO-tie special cases document) -- then add
    one fresh BUF tying `real_out` to `fresh_source`. `remove_dangling_gates`
    (run once by the caller after all retapping is done) sweeps away
    whatever of the old driver's own now-unreachable fanin cone this leaves
    behind; a gate any part of that cone still shares with live logic
    elsewhere survives automatically, same as any other dangling-gate sweep
    in this codebase.
    """
    old_driver = work.net_driver.get(real_out)
    if old_driver is not None:
        dead = work.fresh_net("t_depth_opt_dead_")
        work.rewire_pin(old_driver, OUTPUT_PIN[old_driver.gate_type], dead)
    work.add_gate(
        Gate(inst_name=work.fresh_gate_name("t_depth_opt_tap_"), gate_type=GateType.BUF, pins={"O": real_out, "I0": fresh_source})
    )


def _splice_whole_design(work: Design, blif: _BlifNetlist, promoted_q_source: dict[str, NetBit]) -> None:
    pi_tokens = frozenset(blif.inputs)
    resolve = _token_resolver(work, pi_tokens, promoted_q_source, "t_depth_opt_net_")
    out_map = _add_synth_gates(work, blif, resolve, "t_depth_opt_gate_")

    dff_gates = [g for g in work.gates if g.gate_type == GateType.DFF]
    dff_by_inst = {g.inst_name: g for g in dff_gates}
    # A DFF's Q bit can appear as one of `blif.outputs` without being a real
    # sink to retap: extract_combinational_view's free_pi promotion keeps a
    # promoted (or split) Q bit's ORIGINAL bus declared as an OUTPUT port
    # (Port direction lives at whole-signal granularity, so promoting one
    # bit can't clear it), yet that bit's true driver -- the DFF itself --
    # was dropped from `comb` entirely; the result is a floating, ABC-tied-
    # to-constant-0 PO bit in the extracted view that is NOT the real net's
    # value at all (test39-style corner: a Q bit sharing a bus with
    # combinationally-driven sibling bits). Retapping it here would
    # literally disconnect the real DFF's Q pin (its output pin IS this
    # net-bit's registered driver in `work`, per `net_driver`) and replace
    # it with ABC's meaningless constant guess -- so any such token is
    # skipped outright; the DFF's Q value is a source, never a sink, and is
    # correctly left completely untouched.
    dff_q_bits = {g.pins["Q"] for g in dff_gates if isinstance(g.pins.get("Q"), NetBit)}
    for token in blif.outputs:
        fresh_nb = out_map[token]
        if token.startswith(_DFF_D_TAP_PREFIX):
            inst = token[len(_DFF_D_TAP_PREFIX):]
            dff_gate = dff_by_inst.get(inst)
            if dff_gate is None:
                raise _SynthError(f"D-tap output {token!r} names an unknown DFF instance {inst!r}")
            # A gate's D pin, unlike a PO, is not pinned to a fixed net name
            # -- a plain rewire (no dead-net/BUF dance) suffices.
            work.rewire_pin(dff_gate, "D", fresh_nb)
            continue
        po_nb = parse_net(token)
        if po_nb in dff_q_bits:
            continue
        _retap_output(work, po_nb, fresh_nb)


def _splice_cone(
    work: Design, target: NetBit, blif: _BlifNetlist, out_token: str, promoted_q_source: dict[str, NetBit]
) -> None:
    pi_tokens = frozenset(blif.inputs)
    resolve = _token_resolver(work, pi_tokens, promoted_q_source, "t_depth_opt_net_")
    out_map = _add_synth_gates(work, blif, resolve, "t_depth_opt_gate_")
    _retap_output(work, target, out_map[out_token])


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DepthOptResult:
    design: Design
    changed: bool
    depth_before: int
    depth_after: int
    note: str


_CONE_OUT_TOKEN = "__depth_opt_cone_out__"


def optimize_depth(design: Design, basis: Optional[str] = None, timeout: float = DEFAULT_ABC_TIMEOUT) -> DepthOptResult:
    """Whole-design depth optimization: re-synthesize the ENTIRE
    combinational logic (every non-DFF gate) via ABC, honoring `basis` if
    given (see `BASIS_GATE_NAMES`; None allows any of the 6 two-input
    primitives plus NOT/BUF), then splice the result back in place of the
    original combinational gates.

    Never mutates `design` in place: all work happens on a `copy.deepcopy`,
    committed (returned as `result.design`) only if ABC synthesis succeeds,
    the spliced result verifies functionally equivalent to the untouched
    `design` (via `verify_equivalence`, across the same DFF boundary
    convention used everywhere else in this codebase), AND depth genuinely
    improved. Every failure mode -- ABC crashing/erroring/timing out, a BLIF
    this module can't parse, or (should it ever happen) a failed
    equivalence check -- degrades to `changed=False` with `design` (the
    original, untouched object) returned unchanged, never raises out of this
    function and never leaves `design` partially mutated.

    Accept-or-reject rule (judgment call, documented per the project's
    acceptance-gating contract): accept if the resynthesized depth is
    STRICTLY lower, or equal with a lower total gate count (a tie-breaker
    favoring the smaller of two equal-depth designs); otherwise reject and
    report the original as already optimal.
    """
    _validate_basis(basis)
    depth_before = NetlistGraph(design).max_design_depth()
    if depth_before == 0:
        return DepthOptResult(design, False, 0, 0, "Design has zero combinational depth already; nothing to optimize.")

    work = copy.deepcopy(design)
    try:
        promoted_q_source: dict[str, NetBit] = {}
        comb = extract_combinational_view(work, "free_pi", promoted_q_source)
        blif = _run_abc_synthesis(comb, basis, timeout)
        _splice_whole_design(work, blif, promoted_q_source)
        remove_dangling_gates(work)
    except (ABCBridgeError, _SynthError) as exc:
        return DepthOptResult(
            design, False, depth_before, depth_before, f"ABC could not improve depth ({exc}); kept the original design."
        )

    eq = verify_equivalence(design, work, timeout=timeout)
    if not eq.equivalent:
        return DepthOptResult(
            design,
            False,
            depth_before,
            depth_before,
            f"Equivalence check failed after resynthesis (this indicates a bug, not a normal outcome) -- "
            f"kept the original design untouched. Detail: {eq.detail}",
        )

    depth_after = NetlistGraph(work).max_design_depth()
    if depth_after < depth_before or (depth_after == depth_before and len(work.gates) < len(design.gates)):
        return DepthOptResult(work, True, depth_before, depth_after, f"Reduced maximum logic depth from {depth_before} to {depth_after}.")
    return DepthOptResult(
        design,
        False,
        depth_before,
        depth_before,
        f"Depth {depth_before} is already optimal under the given constraints; design unchanged.",
    )


def optimize_cone_depth(
    design: Design, target: NetBit, basis: Optional[str] = None, timeout: float = DEFAULT_ABC_TIMEOUT
) -> DepthOptResult:
    """Cone-restricted depth optimization: re-synthesize only `target`'s
    fanin cone via ABC, honoring `basis` if given, leaving every other gate
    in `design` completely untouched.

    Splice-by-duplication (never deletes the old cone up front): a cone gate
    may still drive logic OUTSIDE the cone via some other net, so the old
    cone is never removed directly. Instead the newly-synthesized logic is
    added as entirely new gates computing `target` from the same PI/DFF.Q
    support, `target`'s own old driver (if any) is rewired off to a dead net,
    and a fresh BUF drives `target` from the new logic -- then
    `remove_dangling_gates` sweeps whatever of the old cone became
    unreachable as a result (a gate the old cone shared with live logic
    elsewhere survives automatically, since it's still reachable from
    there).

    Same never-mutates-`design`-in-place, always-verify, same accept/reject
    rule as `optimize_depth` (see its docstring) -- restricted to a single
    net-bit's cone depth (`NetlistGraph.depth_to_sink`) rather than the whole
    design's.
    """
    _validate_basis(basis)
    depth_before = NetlistGraph(design).depth_to_sink(target)
    if depth_before == 0:
        return DepthOptResult(design, False, 0, 0, f"Depth of the cone of {netbit_token(target)} is already 0; nothing to optimize.")

    work = copy.deepcopy(design)
    try:
        promoted_q_source: dict[str, NetBit] = {}
        # `target` itself is guaranteed to keep its original name/identity in
        # `comb`: extract_combinational_view only ever promotes/splits a net
        # that is some DFF's Q pin, and depth_before > 0 (checked above)
        # already implies target's own driver is a real combinational gate,
        # not a DFF -- so target is never a candidate for that promotion.
        comb = extract_combinational_view(work, "free_pi", promoted_q_source)
        cone = _restrict_to_fanin_cone(comb, target, _CONE_OUT_TOKEN)
        blif = _run_abc_synthesis(cone, basis, timeout)
        _splice_cone(work, target, blif, _CONE_OUT_TOKEN, promoted_q_source)
        remove_dangling_gates(work)
    except (ABCBridgeError, _SynthError) as exc:
        return DepthOptResult(
            design, False, depth_before, depth_before, f"ABC could not improve depth ({exc}); kept the original design."
        )

    eq = verify_equivalence(design, work, timeout=timeout)
    if not eq.equivalent:
        return DepthOptResult(
            design,
            False,
            depth_before,
            depth_before,
            f"Equivalence check failed after resynthesis (this indicates a bug, not a normal outcome) -- "
            f"kept the original design untouched. Detail: {eq.detail}",
        )

    depth_after = NetlistGraph(work).depth_to_sink(target)
    if depth_after < depth_before or (depth_after == depth_before and len(work.gates) < len(design.gates)):
        return DepthOptResult(
            work,
            True,
            depth_before,
            depth_after,
            f"Reduced the depth of the cone of {netbit_token(target)} from {depth_before} to {depth_after}.",
        )
    return DepthOptResult(
        design,
        False,
        depth_before,
        depth_before,
        f"Depth {depth_before} is already optimal under the given constraints; design unchanged.",
    )


# ----------------------------------------------------------------------
# Gate-count (area) optimization: a distinct dual-objective sibling of
# optimize_depth/optimize_cone_depth above, not a variant of them.
#
# Rationale (see this task's measurement writeup): no single pre-map script
# is both the best area optimizer AND safe to swap in unconditionally --
# `_OPT_SCRIPT` (`strash; balance; dch`) is already the best-or-near-best
# gate count in most measured cases and the best depth in every one, while
# the one script that reliably beats it on raw gate count
# (`strash; dc2; resub; dc2`) does so at a depth cost that grows with design
# size (up to +24% observed). So both are tried as independent candidates
# (`_AREA_CANDIDATE_SCRIPTS`) and the one with the SMALLEST resulting gate
# count is accepted, but ONLY from among candidates that independently pass
# every one of these gates:
#   1. the ABC round trip succeeds and the mapped BLIF parses,
#   2. the spliced result verifies functionally equivalent to the original
#      (`verify_equivalence`, same as every other splice in this module),
#   3. if the caller supplied a `max_depth` (a HARD constraint, e.g. from a
#      request like "...so that the maximum depth is <= 5 and the gate count
#      is minimized" -- violating it is not a partial win, it is a rejected
#      candidate), the resulting depth does not exceed it,
#   4. the resulting gate count is strictly less than the original (no
#      improvement is not an improvement, even if every other gate passes).
# If no candidate survives all four gates, the original design is returned
# completely untouched (changed=False) -- this is the expected, honest
# outcome under a restrictive basis (observed to inflate gate count by
# multiple times in the measurement writeup), not a bug.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AreaOptResult:
    design: Design
    changed: bool
    gates_before: int
    gates_after: int
    depth_before: int
    depth_after: int
    note: str


def optimize_gate_count(
    design: Design, basis: Optional[str] = None, max_depth: Optional[int] = None, timeout: float = DEFAULT_ABC_TIMEOUT
) -> AreaOptResult:
    """Whole-design gate-count optimization: try every script in
    `_AREA_CANDIDATE_SCRIPTS` against the entire combinational logic (same
    `extract_combinational_view`/`_splice_whole_design` round trip
    `optimize_depth` uses), and commit whichever verified-equivalent
    candidate has the smallest gate count among those honoring `max_depth`
    (if given -- a hard ceiling on `NetlistGraph.max_design_depth()` after
    splicing, not a target to approach). Never mutates `design` in place;
    see this module's optimize_depth for the same never-mutates/always-
    verify discipline, shared verbatim here.
    """
    _validate_basis(basis)
    gates_before = len(design.gates)
    depth_before = NetlistGraph(design).max_design_depth()
    if depth_before == 0:
        return AreaOptResult(
            design, False, gates_before, gates_before, 0, 0, "Design has zero combinational depth already; nothing to optimize."
        )

    best_work: Optional[Design] = None
    best_gates: Optional[int] = None
    best_depth: Optional[int] = None
    for candidate_script in _AREA_CANDIDATE_SCRIPTS:
        work = copy.deepcopy(design)
        try:
            promoted_q_source: dict[str, NetBit] = {}
            comb = extract_combinational_view(work, "free_pi", promoted_q_source)
            blif = _run_abc_synthesis(comb, basis, timeout, candidate_script)
            _splice_whole_design(work, blif, promoted_q_source)
            remove_dangling_gates(work)
        except (ABCBridgeError, _SynthError):
            continue
        eq = verify_equivalence(design, work, timeout=timeout)
        if not eq.equivalent:
            continue
        gates_after = len(work.gates)
        depth_after = NetlistGraph(work).max_design_depth()
        if max_depth is not None and depth_after > max_depth:
            continue
        if gates_after >= gates_before:
            continue
        if best_gates is None or gates_after < best_gates:
            best_work, best_gates, best_depth = work, gates_after, depth_after

    if best_work is None or best_gates is None or best_depth is None:
        note = "No candidate restructuring reduced the gate count below the original"
        note += " while honoring the maximum-depth constraint" if max_depth is not None else ""
        note += "; design left unchanged."
        return AreaOptResult(design, False, gates_before, gates_before, depth_before, depth_before, note)
    return AreaOptResult(
        best_work, True, gates_before, best_gates, depth_before, best_depth, f"Reduced gate count from {gates_before} to {best_gates}."
    )


def optimize_cone_gate_count(
    design: Design,
    target: NetBit,
    basis: Optional[str] = None,
    max_depth: Optional[int] = None,
    timeout: float = DEFAULT_ABC_TIMEOUT,
) -> AreaOptResult:
    """Cone-restricted gate-count optimization: the `optimize_cone_depth`
    counterpart of `optimize_gate_count` above -- restricted to `target`'s
    fanin cone (via `_restrict_to_fanin_cone`/`_splice_cone`, same splice-by-
    duplication discipline `optimize_cone_depth` uses), `max_depth` (if
    given) is checked against `NetlistGraph.depth_to_sink(target)` after
    splicing, not the whole design's depth. Every other gate in `design` is
    left completely untouched, same as `optimize_cone_depth`.
    """
    _validate_basis(basis)
    gates_before = len(design.gates)
    depth_before = NetlistGraph(design).depth_to_sink(target)
    if depth_before == 0:
        return AreaOptResult(
            design,
            False,
            gates_before,
            gates_before,
            0,
            0,
            f"Depth of the cone of {netbit_token(target)} is already 0; nothing to optimize.",
        )

    best_work: Optional[Design] = None
    best_gates: Optional[int] = None
    best_depth: Optional[int] = None
    for candidate_script in _AREA_CANDIDATE_SCRIPTS:
        work = copy.deepcopy(design)
        try:
            promoted_q_source: dict[str, NetBit] = {}
            comb = extract_combinational_view(work, "free_pi", promoted_q_source)
            cone = _restrict_to_fanin_cone(comb, target, _CONE_OUT_TOKEN)
            blif = _run_abc_synthesis(cone, basis, timeout, candidate_script)
            _splice_cone(work, target, blif, _CONE_OUT_TOKEN, promoted_q_source)
            remove_dangling_gates(work)
        except (ABCBridgeError, _SynthError):
            continue
        eq = verify_equivalence(design, work, timeout=timeout)
        if not eq.equivalent:
            continue
        gates_after = len(work.gates)
        depth_after = NetlistGraph(work).depth_to_sink(target)
        if max_depth is not None and depth_after > max_depth:
            continue
        if gates_after >= gates_before:
            continue
        if best_gates is None or gates_after < best_gates:
            best_work, best_gates, best_depth = work, gates_after, depth_after

    if best_work is None or best_gates is None or best_depth is None:
        note = "No candidate restructuring reduced the gate count below the original"
        note += " while honoring the maximum-depth constraint" if max_depth is not None else ""
        note += "; design left unchanged."
        return AreaOptResult(design, False, gates_before, gates_before, depth_before, depth_before, note)
    return AreaOptResult(
        best_work, True, gates_before, best_gates, depth_before, best_depth, f"Reduced gate count from {gates_before} to {best_gates}."
    )
