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

  3. Pre-map scripts, judged by the FINAL MAPPED depth (`map`'s reported
     `lev`, not the AIG's own level count). `strash; balance; dch` beat or
     tied the `rewrite`/`refactor` rounds, `dch` followed by more
     `balance`/`rewrite` and resyn2-style scripts on real corpus files
     (basis `and_not`, re-checked 2026-11-15: test22's combinational view
     maps to depth 47 vs. 47-49, the tie being dch followed by more
     balance/rewrite; test28 after remap to AND/NOT, 48k gates, to 102 vs.
     130-131), so it remains the area path's candidate #1. For DEPTH it is
     not always the best: batch 34 measured SOP-balancing chains
     (`if -g -K 6`) at test22 depth 13 (basis unrestricted; vs. 41
     unoptimized, 24 with `_OPT_SCRIPT`; SPEC_34 G3), so the depth paths try
     several candidate scripts (`_DEPTH_SMALL_SCRIPTS` /
     `_DEPTH_LARGE_SCRIPTS`) and keep the best.

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

from netlist_agent import deadline
from netlist_agent.abc_bridge import (
    ABCBridgeError,
    ABCInconclusiveError,
    DEFAULT_ABC_TIMEOUT,
    DEFAULT_VERIFY_TIMEOUT,
    _is_declared_bit,
    _os_error_detail,
    _restrict_to_fanin_cone,
    _resolve_abc,
    _scrub_abc_output,
    extract_combinational_view,
    verify_equivalence,
    write_blif,
)
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, OUTPUT_PIN, Pin
from netlist_agent.netref import netbit_token, parse_net
from netlist_agent.transform import count_dangling_gates, remove_dangling_gates

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


class _SynthTimeout(_SynthError):
    """The ABC subprocess hit its timeout (as opposed to crashing or erroring).
    A subclass so every existing `except _SynthError` still catches it; the
    depth paths record it as that candidate's `incomplete` reason and go on
    to the next candidate script."""


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

# Quick depth script, the first candidate of the large-design list (batch 30
# introduced it as the retry after `_OPT_SCRIPT` timed out; batch 34b-1 made it
# candidate #1 for views above `_DEPTH_LARGE_THRESHOLD`).
# Measured on a 213k-gate design: `strash; balance` takes 2.7s and reaches the
# same depth (48) as the full script, which took 65s at load 10 and hit the
# 120s cap in another run at load 20-48. The two are separate measurements
# taken at different times, so attributing the timeout to load is an inference.
_FAST_OPT_SCRIPT = "strash; balance"

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

# Depth candidate scripts (batch 34b-1, D6). Source: FINDINGS.md of the batch-34
# experiment (13 scripts measured): test051 (213k gates) cannot finish the full
# script in 120s but `&sopb` x2 (no dch) reaches 18-25s; on small designs the
# `if -g -K 6` (SOP balancing) chain x4 was best of 13; test064 is only best
# with the current script and test062 only with the XOR one, so both stay.
_DEPTH_LARGE_THRESHOLD = 150_000  # gates in the view ABC gets
_DEPTH_SOP_ROUNDS_SMALL = 4
_DEPTH_SOP_ROUNDS_LONG = 8


def _sop_script(rounds: int) -> str:
    return "strash; " + "if -g -K 6; strash; " * rounds + "balance; dch"


_DEPTH_SMALL_SCRIPTS: tuple[str, ...] = (
    _OPT_SCRIPT,
    _sop_script(_DEPTH_SOP_ROUNDS_SMALL),
    "strash; balance; dch; if -x -K 6; strash; balance",
    _sop_script(_DEPTH_SOP_ROUNDS_LONG),
)
_DEPTH_LARGE_SCRIPTS: tuple[str, ...] = (
    _FAST_OPT_SCRIPT,
    "strash; &get -n; &st; &sopb; &sopb; &put; balance",
)

# Budget for candidates 2.. when the request has a time budget (D10): keep
# `reserve = 0.35 * budget` (about 95s of 270s) for cec and Python-side work,
# and do not start a candidate with less than 10s of the remainder. Source:
# FINDINGS test051 measurement -- cec 66s, Python about 23s.
# The same constant also guards the post-cec rebuild of a runner-up, but there
# it is compared against the raw `remaining()` (no reserve subtracted).
_DEPTH_RESERVE_FRACTION = 0.35
_DEPTH_MIN_CANDIDATE_SECONDS = 10.0


def depth_change_sentence(subject: str, depth_before: int, depth_after: int, gates_before: int, gates_after: int) -> str:
    """The accepted-result sentence shared by every depth-optimization reply
    (whole/cone `note` and the router responses). An accepted result may keep
    the depth and only save gates; "from X to X" would read as a bug there."""
    if depth_after < depth_before:
        return f"Reduced {subject} from {depth_before} to {depth_after}."
    return (
        f"{subject[:1].upper()}{subject[1:]} stays at {depth_before}; "
        f"the resynthesized design has fewer gates ({gates_before} to {gates_after})."
    )


def _verify_failure_prefix(exc: ABCBridgeError) -> str:
    """Wording for a `verify_equivalence` error: a timeout / undecided miter
    (`ABCInconclusiveError`) gave no verdict, so it must not be phrased like a
    failed check, which reads as "not equivalent"."""
    if isinstance(exc, ABCInconclusiveError):
        return "equivalence check was inconclusive"
    return "equivalence check failed"


def _run_abc_synthesis(view: Design, basis: Optional[str], timeout: float, opt_script: str = _OPT_SCRIPT) -> _BlifNetlist:
    # Clamp to the request's remaining time (batch 21b). Checked first so a spent
    # budget skips the temp dir and the BLIF write; checked again after the
    # resolve below, which can use up time on a cold cache.
    t = deadline.spawn_timeout(timeout)
    if t is None:
        raise _SynthError(
            f"ABC synthesis was not started: this request had less than "
            f"{deadline.MIN_SPAWN_REMAINING:g}s of its time budget left"
        )
    genlib_text = _genlib_text(basis)
    # Every OSError below (disk full, unwritable temp dir, exec failure) becomes
    # `_SynthError` so the optimize paths report a `failure` instead of letting
    # it escape as an "Internal error" with a temp path in it; the message is
    # built by `_os_error_detail` and carries no absolute path.
    try:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="abc_synth_", ignore_cleanup_errors=True)
    except OSError as exc:
        raise _SynthError(f"could not create ABC temp directory: {_os_error_detail(exc)}") from exc
    with tmp_ctx as tmpdir:
        in_blif_path = os.path.join(tmpdir, "in.blif")
        lib_path = os.path.join(tmpdir, "basis.genlib")
        blif_path = os.path.join(tmpdir, "out.blif")
        # `read_blif`, not `read_verilog`: the latter asserts and crashes ABC
        # on any bus whose MSB reaches 128 (abc_bridge.py module docstring,
        # finding 4) -- real on this project's own corpus. `strash` (the
        # first stage of every `opt_script` below) accepts either input form
        # identically, so nothing downstream (opt_script, map, write_blif,
        # this function's own `parse_blif` read-back) needed to change.
        try:
            write_blif(view, in_blif_path)
        except OSError as exc:
            raise _SynthError(f"could not write ABC input file: {_os_error_detail(exc)}") from exc
        try:
            with open(lib_path, "w") as f:
                f.write(genlib_text)
        except OSError as exc:
            raise _SynthError(f"could not write ABC library file: {_os_error_detail(exc)}") from exc
        script = (
            f'read_blif "{in_blif_path}"; {opt_script}; '
            f'read_genlib "{lib_path}"; map; write_blif "{blif_path}"'
        )
        try:
            abc_path = _resolve_abc()
        except ABCBridgeError as exc:
            raise _SynthError(str(exc)) from exc
        t = deadline.spawn_timeout(timeout)
        if t is None:
            raise _SynthError(
                f"ABC synthesis was not started: this request had less than "
                f"{deadline.MIN_SPAWN_REMAINING:g}s of its time budget left"
            )
        try:
            result = subprocess.run([abc_path, "-c", script], capture_output=True, text=True, timeout=t)
        except subprocess.TimeoutExpired as exc:
            if t < timeout:
                raise _SynthTimeout(
                    f"ABC synthesis timed out after {t:.1f}s (the time left in this request's budget)"
                ) from exc
            raise _SynthTimeout(f"ABC synthesis timed out after {timeout}s") from exc
        except OSError as exc:
            raise _SynthError(f"could not run ABC: {_os_error_detail(exc)}") from exc
        if result.returncode != 0:
            # Covers both a clean ABC-reported error and a hard crash (e.g. a
            # segfault, seen for real during development before finding 2
            # above was understood) -- either way, treated identically as a
            # failed synthesis attempt, never propagated as a crash.
            detail = _scrub_abc_output(result.stderr or "", tmpdir).strip()
            if not detail:
                detail = _scrub_abc_output(result.stdout or "", tmpdir).strip()
            detail = detail[-500:]
            suffix = f": {detail}" if detail else ""
            raise _SynthError(f"ABC synthesis process exited with code {result.returncode}{suffix}")
        if not os.path.exists(blif_path) or os.path.getsize(blif_path) == 0:
            # rc == 0 with no output: ABC reports most errors (e.g. a missing
            # genlib) on stderr, with stdout holding only the echo line.
            detail = _scrub_abc_output(result.stderr or "", tmpdir).strip()
            if not detail:
                detail = _scrub_abc_output(result.stdout or "", tmpdir).strip()
            detail = detail[-500:]
            suffix = f": {detail}" if detail else ""
            raise _SynthError(f"ABC synthesis produced no BLIF output{suffix}")
        try:
            with open(blif_path) as f:
                blif_text = f.read()
        except OSError as exc:
            raise _SynthError(f"could not read ABC output file: {_os_error_detail(exc)}") from exc
    return parse_blif(blif_text)


# ----------------------------------------------------------------------
# Splicing the synthesized gates into the working Design (by duplication,
# never in-place mutation of the original cone/design -- see module docstring
# of optimize_depth/optimize_cone_depth below for the rationale).
# ----------------------------------------------------------------------


def _token_resolver(
    work: Design,
    pi_tokens: frozenset[str],
    promoted_q_source: dict[str, NetBit],
    net_prefix: str,
    direct: Optional[dict[str, NetBit]] = None,
) -> Callable[[str], NetBit]:
    """Builds a memoized token -> NetBit resolver for `work`'s namespace: a
    PI token resolves to the REAL net it stands for (a DFF's Q net via
    `promoted_q_source`, or an ordinary PI parsed by name and required to be
    a declared bit of an INPUT signal of `work`); every
    other token (an ABC-invented internal wire, or one of our own PO/D-tap
    output names, except those listed in `direct`, see below) gets a
    brand-new net freshly allocated in `work` -- this is
    the "duplication" half of the splice-by-duplication approach: apart from
    the `direct` nets, nothing here ever reuses an existing internal net name
    from the pre-optimization cone/design.

    `direct` maps output tokens that the caller has decided to wire straight
    onto a real net (a PO bit / cone target) to that net: such a token
    resolves to the real net instead of a fresh one, so the synthesized gate
    that drives it drives the real net itself (no tap BUF, no orphan net).
    """
    cache: dict[str, NetBit] = dict(direct) if direct else {}

    def resolve(token: str) -> NetBit:
        if token in cache:
            return cache[token]
        if token in pi_tokens:
            nb = promoted_q_source.get(token)
            if nb is None:
                nb = parse_net(token)
                sig = work.signals.get(nb.name)
                if sig is None or sig.direction != Direction.INPUT or not _is_declared_bit(sig, nb):
                    raise _SynthError(f"ABC PI token {token!r} maps to no primary input of the design")
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


def _can_wire_directly(work: Design, real_out: NetBit, token: str, pi_tokens: frozenset[str], gate_out_tokens: frozenset[str]) -> bool:
    """True when the synthesized gate driving `token` may drive `real_out`
    itself instead of going through a tap BUF: `real_out` has a driver that is
    not a DFF (a DFF's Q pin is a source, never retapped), `token` is the O
    pin of some BLIF gate, and it is not a primary-input token."""
    old_driver = work.net_driver.get(real_out)
    if old_driver is None or old_driver.gate_type == GateType.DFF:
        return False
    return token in gate_out_tokens and token not in pi_tokens


def _detach_old_driver(work: Design, real_out: NetBit) -> None:
    """Rewire `real_out`'s old driver off to a fresh dead net (first half of
    `_retap_output`); the net is then free for a synthesized gate to drive."""
    old_driver = work.net_driver.get(real_out)
    if old_driver is not None:
        dead = work.fresh_net("t_depth_opt_dead_")
        work.rewire_pin(old_driver, OUTPUT_PIN[old_driver.gate_type], dead)


def _splice_whole_design(work: Design, blif: _BlifNetlist, promoted_q_source: dict[str, NetBit]) -> None:
    pi_tokens = frozenset(blif.inputs)
    dff_gates = [g for g in work.gates if g.gate_type == GateType.DFF]
    dff_by_inst = {g.inst_name: g for g in dff_gates}
    # A DFF's Q net can appear as one of `blif.outputs`: in the extracted
    # view it is a PO (or an internal net) re-driven by the synthesized BUF
    # from the `__dff_Q__<inst>` PI. It is a source in `work`, never a sink
    # to retap -- retapping would disconnect the real DFF's Q pin -- so any
    # such token is skipped.
    dff_q_bits = {g.pins["Q"] for g in dff_gates if isinstance(g.pins.get("Q"), NetBit)}
    # Decide BEFORE adding gates which POs the synthesized gate drives
    # directly (no tap BUF): the resolver then hands those tokens their real
    # net, so BLIF readers of the token read the real net too.
    gate_out_tokens = frozenset(g.pins["O"] for g in blif.gates if "O" in g.pins)
    direct: dict[str, NetBit] = {}
    direct_bits: set[NetBit] = set()
    for token in blif.outputs:
        if token.startswith(_DFF_D_TAP_PREFIX):
            continue
        po_nb = parse_net(token)
        if po_nb in dff_q_bits or po_nb in direct_bits:
            continue
        if _can_wire_directly(work, po_nb, token, pi_tokens, gate_out_tokens):
            direct[token] = po_nb
            direct_bits.add(po_nb)
    for po_nb in direct.values():
        _detach_old_driver(work, po_nb)
    resolve = _token_resolver(work, pi_tokens, promoted_q_source, "t_depth_opt_net_", direct)
    out_map = _add_synth_gates(work, blif, resolve, "t_depth_opt_gate_")
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
        if token in direct:
            continue
        po_nb = parse_net(token)
        if po_nb in dff_q_bits:
            continue
        # A PO bit nothing drives in the original design (floating) stays
        # floating: ABC ties it to constant 0 in the view, and retapping
        # would invent a constant driver the original never had (F6).
        if po_nb not in work.net_driver:
            continue
        _retap_output(work, po_nb, fresh_nb)


def _splice_cone(
    work: Design, target: NetBit, blif: _BlifNetlist, out_token: str, promoted_q_source: dict[str, NetBit]
) -> None:
    pi_tokens = frozenset(blif.inputs)
    gate_out_tokens = frozenset(g.pins["O"] for g in blif.gates if "O" in g.pins)
    direct: dict[str, NetBit] = {}
    if _can_wire_directly(work, target, out_token, pi_tokens, gate_out_tokens):
        _detach_old_driver(work, target)
        direct[out_token] = target
    resolve = _token_resolver(work, pi_tokens, promoted_q_source, "t_depth_opt_net_", direct)
    out_map = _add_synth_gates(work, blif, resolve, "t_depth_opt_gate_")
    if out_token not in direct:
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
    # Non-None: optimization did NOT complete (every candidate script failed or
    # was skipped, or the winner's post-synthesis equivalence check raised or
    # found a mismatch) -- a short, human-readable reason, never derived from
    # `note` (router reads this field, not `note`, to decide how to phrase
    # failure). None says nothing about whether anything ran: it is also the
    # value of the depth-0 early return, where ABC never ran. It only means
    # "no failure was recorded". Some candidate completed and none was
    # accepted => None (see `incomplete`).
    failure: Optional[str] = None
    # One reason per candidate script that did not run to completion (synthesis
    # error/timeout or skipped for lack of time budget, in script order; then
    # the reasons from ranking: an inconclusive equivalence check, a failed
    # rebuild, or a skipped rebuild). Filled on success too:
    # it says "a lower depth may exist". When the equivalence check finds a
    # mismatch, the last entry is the `failure` itself.
    # Always len(incomplete) <= candidates.
    incomplete: tuple[str, ...] = ()
    # Number of candidate scripts in the list for this design size.
    candidates: int = 1


_CONE_OUT_TOKEN = "__depth_opt_cone_out__"


def _depth_candidate_scripts(gate_count: int) -> tuple[str, ...]:
    """The candidate list for a view of `gate_count` gates (D6)."""
    if gate_count > _DEPTH_LARGE_THRESHOLD:
        return _DEPTH_LARGE_SCRIPTS
    return _DEPTH_SMALL_SCRIPTS


@dataclass
class _Accepted:
    """A candidate that passed the accept gate (not yet equivalence-checked)."""

    depth: int
    gates: int
    order: int
    work: Optional[Design]  # only the current best keeps its Design (the loop body also holds the one being built)
    blif: _BlifNetlist


def _optimize_depth_candidates(
    design: Design,
    basis: Optional[str],
    timeout: float,
    verify_timeout: float,
    depth_before: int,
    baseline: int,
    subject: str,
    prepare: Callable[[], tuple[Design, Design, dict[str, NetBit]]],
    splice: Callable[[Design, _BlifNetlist, dict[str, NetBit]], None],
    measure: Callable[[Design], int],
) -> DepthOptResult:
    """Shared body of `optimize_depth` / `optimize_cone_depth` (D6-D12).

    `prepare()` deep-copies `design` and returns `(work, view, promoted_q_source)`
    where `view` is what ABC gets (whole comb view, or the restricted cone)."""

    def failure_result(failure: str, incomplete: tuple[str, ...], n: int, note: Optional[str] = None) -> DepthOptResult:
        return DepthOptResult(
            design,
            False,
            depth_before,
            depth_before,
            note or f"ABC could not improve depth ({failure}); kept the original design.",
            failure=failure,
            incomplete=incomplete,
            candidates=n,
        )

    try:
        first = prepare()
    except (ABCBridgeError, _SynthError) as exc:
        return failure_result(str(exc), (str(exc),), 1)
    scripts = _depth_candidate_scripts(len(first[1].gates))
    n = len(scripts)

    incomplete: list[str] = []
    completed = 0
    accepted: list[_Accepted] = []  # in candidate order
    best: Optional[_Accepted] = None
    out_of_time = False
    for order, script in enumerate(scripts):
        cand_timeout = timeout
        if order >= 1:
            left = deadline.remaining()
            if left is not None:
                avail = left - _DEPTH_RESERVE_FRACTION * (deadline.current_budget() or 0.0)
                if out_of_time or avail < _DEPTH_MIN_CANDIDATE_SECONDS:
                    out_of_time = True
                    incomplete.append(
                        "skipped: not enough of this request's time budget was left for another candidate"
                    )
                    continue
                cand_timeout = min(timeout, avail)
        try:
            if order == 0:
                work, view, promoted_q_source = first
                first = None  # drop the extra reference: it must not outlive candidate 0
            else:
                work = view = promoted_q_source = None  # free the previous candidate before building the next
                work, view, promoted_q_source = prepare()
            blif = _run_abc_synthesis(view, basis, cand_timeout, script)
            splice(work, blif, promoted_q_source)
            remove_dangling_gates(work)
        except (ABCBridgeError, _SynthError) as exc:
            incomplete.append(str(exc))
            continue
        completed += 1
        depth_after = measure(work)
        if depth_after < depth_before or (depth_after == depth_before and len(work.gates) < baseline):
            cand = _Accepted(depth_after, len(work.gates), order, work, blif)
            accepted.append(cand)
            if best is None or (cand.depth, cand.gates, cand.order) < (best.depth, best.gates, best.order):
                if best is not None:
                    best.work = None
                best = cand
            else:
                cand.work = None

    work = view = promoted_q_source = None
    # Rank the accepted candidates; cec only the first, then (undecided) the next.
    ranked = sorted(accepted, key=lambda c: (c.depth, c.gates, c.order))
    cec_reasons: list[str] = []
    rebuild_out_of_time = False
    for cand in ranked:
        work = cand.work
        if work is None:
            if not rebuild_out_of_time:
                left = deadline.remaining()
                rebuild_out_of_time = left is not None and left < _DEPTH_MIN_CANDIDATE_SECONDS
            if rebuild_out_of_time:
                reason = "skipped: not enough of this request's time budget was left to rebuild and verify another candidate"
                incomplete.append(reason)
                continue
            try:
                work, view, promoted_q_source = prepare()
                splice(work, cand.blif, promoted_q_source)
                remove_dangling_gates(work)
            except (ABCBridgeError, _SynthError) as exc:
                reason = f"could not rebuild a candidate result: {exc}"
                incomplete.append(reason)
                cec_reasons.append(reason)
                continue
        try:
            eq = verify_equivalence(design, work, timeout=verify_timeout)
        except ABCBridgeError as exc:
            # "Inconclusive" (timeout / undecided) is not "failed": ABC gave no
            # verdict, which must not read as a mismatch. Any other bridge error
            # stays loud under the old prefix.
            reason = f"{_verify_failure_prefix(exc)}: {exc}"
            incomplete.append(reason)
            cec_reasons.append(reason)
            continue
        if not eq.equivalent:
            # Not equivalent means the splice or ABC has a bug: never swap in
            # the next candidate to hide it.
            failure = f"equivalence check found the resynthesized design not equivalent to the original: {eq.detail}"
            return failure_result(
                failure,
                tuple(incomplete) + (failure,),
                n,
                note=(
                    f"Equivalence check failed after resynthesis (this indicates a bug, not a normal outcome) -- "
                    f"kept the original design untouched. Detail: {eq.detail}"
                ),
            )
        note = depth_change_sentence(subject, depth_before, cand.depth, len(design.gates), len(work.gates))
        if incomplete:
            note += (
                f" {len(incomplete)} of {n} candidate scripts did not finish ({incomplete[0]}), "
                f"so a lower depth may exist."
            )
        return DepthOptResult(
            work, True, depth_before, cand.depth, note, incomplete=tuple(incomplete), candidates=n
        )

    inc = tuple(incomplete)
    if cec_reasons:
        # Every accepted candidate's equivalence check was undecided.
        return failure_result(cec_reasons[0], inc, n)
    if completed == 0:
        failure = incomplete[0]
        if len(incomplete) > 1:
            failure += f"; and {len(incomplete) - 1} more candidate(s) did not complete"
        return failure_result(failure, inc, n)
    if incomplete:
        note = (
            f"Depth {depth_before} was not reduced by the candidate scripts that completed, but "
            f"{len(incomplete)} of {n} could not be completed ({incomplete[0]}), so a lower depth is "
            f"not ruled out; design unchanged."
        )
        return DepthOptResult(design, False, depth_before, depth_before, note, incomplete=inc, candidates=n)
    return DepthOptResult(
        design,
        False,
        depth_before,
        depth_before,
        f"Depth {depth_before} is already optimal under the given constraints; design unchanged.",
        candidates=n,
    )


def optimize_depth(
    design: Design,
    basis: Optional[str] = None,
    timeout: float = DEFAULT_ABC_TIMEOUT,
    verify_timeout: float = DEFAULT_VERIFY_TIMEOUT,
) -> DepthOptResult:
    """Whole-design depth optimization: re-synthesize the ENTIRE
    combinational logic (every non-DFF gate) via ABC, honoring `basis` if
    given (see `BASIS_GATE_NAMES`; None allows any of the 6 two-input
    primitives plus NOT/BUF), then splice the result back in place of the
    original combinational gates.

    Several candidate scripts are tried (`_DEPTH_SMALL_SCRIPTS`, or
    `_DEPTH_LARGE_SCRIPTS` above `_DEPTH_LARGE_THRESHOLD` gates), each as a
    full round trip on its own deep copy. Candidates passing the accept gate
    are ranked by `(depth, gate count, script order)`; only the first is
    equivalence-checked. An undecided check falls back to the next (rebuilt
    from its BLIF, no second ABC run); a NOT-equivalent verdict stops at once
    and returns the original (it means a bug).

    Never mutates `design` in place: all work happens on `copy.deepcopy`s,
    committed (returned as `result.design`) only if a candidate verifies
    functionally equivalent to the untouched `design` (via
    `verify_equivalence`, across the same DFF boundary convention used
    everywhere else in this codebase) AND depth genuinely improved. Every
    failure mode degrades to `changed=False` with `design` (the original,
    untouched object) returned unchanged, never raises out of this function
    and never leaves `design` partially mutated.

    Accept rule: a candidate is accepted if its depth is STRICTLY lower, or
    equal with a lower total gate count (a tie-breaker favoring the smaller of
    two equal-depth designs). The gate-count comparison baseline is the
    original gate count minus the gates that were already dangling (those are
    swept and must not count as a gain). "Already optimal" is only claimed
    when every candidate ran to completion; otherwise `incomplete` says which
    did not and the note says a lower depth is not ruled out.

    `verify_timeout` bounds each `cec` equivalence check only; it is
    separate from `timeout` (ABC synthesis), default `DEFAULT_VERIFY_TIMEOUT`.
    """
    _validate_basis(basis)
    baseline = len(design.gates) - count_dangling_gates(design)
    depth_before = NetlistGraph(design).max_design_depth()
    if depth_before == 0:
        return DepthOptResult(design, False, 0, 0, "Design has zero combinational depth already; nothing to optimize.")

    def prepare() -> tuple[Design, Design, dict[str, NetBit]]:
        work = copy.deepcopy(design)
        promoted_q_source: dict[str, NetBit] = {}
        comb = extract_combinational_view(work, "free_pi", promoted_q_source)
        return work, comb, promoted_q_source

    return _optimize_depth_candidates(
        design,
        basis,
        timeout,
        verify_timeout,
        depth_before,
        baseline,
        "maximum logic depth",
        prepare,
        _splice_whole_design,
        lambda w: NetlistGraph(w).max_design_depth(),
    )


def optimize_cone_depth(
    design: Design,
    target: NetBit,
    basis: Optional[str] = None,
    timeout: float = DEFAULT_ABC_TIMEOUT,
    verify_timeout: float = DEFAULT_VERIFY_TIMEOUT,
) -> DepthOptResult:
    """Cone-restricted depth optimization: re-synthesize only `target`'s
    fanin cone via ABC, honoring `basis` if given, leaving every other gate
    in `design` completely untouched (gates that were already dangling in
    `design` are swept too, but do not count toward an improvement).

    Splice-by-duplication (never deletes the old cone up front): a cone gate
    may still drive logic OUTSIDE the cone via some other net, so the old
    cone is never removed directly. Instead the newly-synthesized logic is
    added as entirely new gates computing `target` from the same PI/DFF.Q
    support, `target`'s own old driver (if any) is rewired off to a dead net,
    and the new logic drives `target` directly (a fresh tap BUF is used only
    when `target` does not qualify for direct wiring) -- then
    `remove_dangling_gates` sweeps whatever of the old cone became
    unreachable as a result (a gate the old cone shared with live logic
    elsewhere survives automatically, since it's still reachable from
    there).

    Same never-mutates-`design`-in-place, always-verify, same multi-candidate
    and accept/reject rule as `optimize_depth` (see its docstring, including
    that the gate-count baseline excludes gates that were already dangling)
    -- restricted to a single net-bit's cone depth
    (`NetlistGraph.depth_to_sink`) rather than the whole design's.

    `verify_timeout` bounds each `cec` equivalence check only; it is
    separate from `timeout` (ABC synthesis), default `DEFAULT_VERIFY_TIMEOUT`.
    """
    _validate_basis(basis)
    baseline = len(design.gates) - count_dangling_gates(design)
    depth_before = NetlistGraph(design).depth_to_sink(target)
    if depth_before == 0:
        return DepthOptResult(design, False, 0, 0, f"Depth of the cone of {netbit_token(target)} is already 0; nothing to optimize.")

    def prepare() -> tuple[Design, Design, dict[str, NetBit]]:
        work = copy.deepcopy(design)
        promoted_q_source: dict[str, NetBit] = {}
        # Every original net (Q nets included) keeps its name in `comb`:
        # a Q net is just re-driven by a BUF from its `__dff_Q__<inst>` PI,
        # so `target` needs no mapping.
        comb = extract_combinational_view(work, "free_pi", promoted_q_source)
        cone = _restrict_to_fanin_cone(comb, target, _CONE_OUT_TOKEN)
        return work, cone, promoted_q_source

    return _optimize_depth_candidates(
        design,
        basis,
        timeout,
        verify_timeout,
        depth_before,
        baseline,
        f"the depth of the cone of {netbit_token(target)}",
        prepare,
        lambda w, b, p: _splice_cone(w, target, b, _CONE_OUT_TOKEN, p),
        lambda w: NetlistGraph(w).depth_to_sink(target),
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
    # Same contract as `DepthOptResult.failure`, adapted for multi-candidate
    # search: None as long as AT LEAST ONE candidate script ran to
    # completion (ABC succeeded and verified equivalent), even if none of
    # them beat the original gate count or honored `max_depth` -- that is
    # not a failure (but see `incomplete`: if it is non-empty, "nothing
    # better found" does not rule out a smaller result). Only non-None
    # when EVERY candidate failed to complete (= `incomplete[0]`).
    failure: Optional[str] = None
    # One reason per candidate that did NOT run to completion (synthesis or
    # verification error/timeout, or a "not equivalent" verdict), in script
    # order, same text `failure` would carry. Filled on success too. Unlike
    # `failure`, non-empty even when another candidate completed: it tells the
    # presentation layer that "nothing smaller" was not established by every
    # candidate. Relation: some candidate completed => failure is None;
    # candidates > 0 and none completed => failure == incomplete[0]. The
    # depth-0 early return has failure None with candidates == 0 and
    # incomplete == (). Always len(incomplete) <= candidates.
    incomplete: tuple[str, ...] = ()
    # Number of candidate scripts actually tried (0 on the depth-0 early return).
    candidates: int = 0


def optimize_gate_count(
    design: Design,
    basis: Optional[str] = None,
    max_depth: Optional[int] = None,
    timeout: float = DEFAULT_ABC_TIMEOUT,
    verify_timeout: float = DEFAULT_VERIFY_TIMEOUT,
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

    `verify_timeout` bounds the final `cec` equivalence check only; it is
    separate from `timeout` (ABC synthesis), default `DEFAULT_VERIFY_TIMEOUT`.
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
    any_completed = False
    incomplete: list[str] = []
    for candidate_script in _AREA_CANDIDATE_SCRIPTS:
        work = copy.deepcopy(design)
        try:
            promoted_q_source: dict[str, NetBit] = {}
            comb = extract_combinational_view(work, "free_pi", promoted_q_source)
            blif = _run_abc_synthesis(comb, basis, timeout, candidate_script)
            _splice_whole_design(work, blif, promoted_q_source)
            remove_dangling_gates(work)
        except (ABCBridgeError, _SynthError) as exc:
            incomplete.append(str(exc))
            continue
        try:
            eq = verify_equivalence(design, work, timeout=verify_timeout)
        except ABCBridgeError as exc:
            incomplete.append(f"{_verify_failure_prefix(exc)}: {exc}")
            continue
        if not eq.equivalent:
            incomplete.append(f"equivalence check found the resynthesized design not equivalent to the original: {eq.detail}")
            continue
        # This candidate ran to completion (ABC succeeded, equivalence
        # verified) -- it counts as a completed optimization attempt even if
        # it goes on to lose to `max_depth`/gate-count below.
        any_completed = True
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
        failure = None if any_completed else incomplete[0]
        return AreaOptResult(
            design,
            False,
            gates_before,
            gates_before,
            depth_before,
            depth_before,
            note,
            failure=failure,
            incomplete=tuple(incomplete),
            candidates=len(_AREA_CANDIDATE_SCRIPTS),
        )
    return AreaOptResult(
        best_work,
        True,
        gates_before,
        best_gates,
        depth_before,
        best_depth,
        f"Reduced gate count from {gates_before} to {best_gates}.",
        incomplete=tuple(incomplete),
        candidates=len(_AREA_CANDIDATE_SCRIPTS),
    )


def optimize_cone_gate_count(
    design: Design,
    target: NetBit,
    basis: Optional[str] = None,
    max_depth: Optional[int] = None,
    timeout: float = DEFAULT_ABC_TIMEOUT,
    verify_timeout: float = DEFAULT_VERIFY_TIMEOUT,
) -> AreaOptResult:
    """Cone-restricted gate-count optimization: the `optimize_cone_depth`
    counterpart of `optimize_gate_count` above -- restricted to `target`'s
    fanin cone (via `_restrict_to_fanin_cone`/`_splice_cone`, same splice-by-
    duplication discipline `optimize_cone_depth` uses), `max_depth` (if
    given) is checked against `NetlistGraph.depth_to_sink(target)` after
    splicing, not the whole design's depth. Every other gate in `design` is
    left completely untouched, same as `optimize_cone_depth` (gates that were
    already dangling are swept too, but do not count toward an improvement).

    `verify_timeout` bounds the final `cec` equivalence check only; it is
    separate from `timeout` (ABC synthesis), default `DEFAULT_VERIFY_TIMEOUT`.
    """
    _validate_basis(basis)
    gates_before = len(design.gates)
    baseline = gates_before - count_dangling_gates(design)
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
    any_completed = False
    incomplete: list[str] = []
    for candidate_script in _AREA_CANDIDATE_SCRIPTS:
        work = copy.deepcopy(design)
        try:
            promoted_q_source: dict[str, NetBit] = {}
            comb = extract_combinational_view(work, "free_pi", promoted_q_source)
            cone = _restrict_to_fanin_cone(comb, target, _CONE_OUT_TOKEN)
            blif = _run_abc_synthesis(cone, basis, timeout, candidate_script)
            _splice_cone(work, target, blif, _CONE_OUT_TOKEN, promoted_q_source)
            remove_dangling_gates(work)
        except (ABCBridgeError, _SynthError) as exc:
            incomplete.append(str(exc))
            continue
        try:
            eq = verify_equivalence(design, work, timeout=verify_timeout)
        except ABCBridgeError as exc:
            incomplete.append(f"{_verify_failure_prefix(exc)}: {exc}")
            continue
        if not eq.equivalent:
            incomplete.append(f"equivalence check found the resynthesized design not equivalent to the original: {eq.detail}")
            continue
        any_completed = True
        gates_after = len(work.gates)
        depth_after = NetlistGraph(work).depth_to_sink(target)
        if max_depth is not None and depth_after > max_depth:
            continue
        if gates_after >= baseline:
            continue
        if best_gates is None or gates_after < best_gates:
            best_work, best_gates, best_depth = work, gates_after, depth_after

    if best_work is None or best_gates is None or best_depth is None:
        note = "No candidate restructuring reduced the gate count below the original"
        note += " while honoring the maximum-depth constraint" if max_depth is not None else ""
        note += "; design left unchanged."
        failure = None if any_completed else incomplete[0]
        return AreaOptResult(
            design,
            False,
            gates_before,
            gates_before,
            depth_before,
            depth_before,
            note,
            failure=failure,
            incomplete=tuple(incomplete),
            candidates=len(_AREA_CANDIDATE_SCRIPTS),
        )
    return AreaOptResult(
        best_work,
        True,
        gates_before,
        best_gates,
        depth_before,
        best_depth,
        f"Reduced gate count from {gates_before} to {best_gates}.",
        incomplete=tuple(incomplete),
        candidates=len(_AREA_CANDIDATE_SCRIPTS),
    )
