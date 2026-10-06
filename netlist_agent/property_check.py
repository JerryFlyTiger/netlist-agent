"""Property verification with counterexamples: "for output X, verify that it
is asserted only when <condition>, and provide a counterexample if this is
not true" (spec section 4.2's own example).

Semantics: "X is asserted only when COND" means X==1 implies COND==1, i.e.
the property P = ~X | COND must be constant-1 across every input and every
register state. This module builds P as a throwaway splice of NOT/AND/OR
gates on a deep copy of the design, hands it to `abc_bridge.check_implication`
(free_pi DFF-boundary mode -- see that function's docstring), and parses
ABC's own counterexample text when the property does not hold.

Cost of free_pi (must be surfaced to the user, not silently swallowed): a
DFF's Q output is treated as a free variable, so a counterexample may pin a
register to a state that is not actually reachable from reset. See
`check_asserted_only_when`'s `caveat` field.

Supported condition grammar (bounded on purpose -- outside this shape,
`parse_condition` raises `ValueError` so the caller can fall back to the LLM
router instead of guessing): one or more "<net> is <0|1|high|low>" literals
joined by "and"/"or", with an optional leading "both".
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Optional

from netlist_agent.abc_bridge import DEFAULT_ABC_TIMEOUT, check_implication, extract_combinational_view
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import OUTPUT_PIN, Design, Direction, Gate, GateType, NetBit
from netlist_agent.netref import netbit_token, resolve_bit

_NET = r"\w+(?:\[\d+\])?"


def free_pi_caveat(subject: str) -> str:
    """Shared free_pi-boundary caveat wording (see this module's own "Cost
    of free_pi" docstring paragraph above): `subject` is the noun this
    particular result is warning about (e.g. "counterexample", "function")
    -- everything else is reused VERBATIM so this module and
    boolean_function.py (F5: it independently hand-wrote its own differently
    -worded caveat, which this shared helper replaces) never drift into two
    differently-worded descriptions of the exact same free_pi
    approximation."""
    return (
        f"This {subject} assumes at least one flip-flop can hold an arbitrary state "
        "(its Q output was treated as a free input); reachability of that state from reset "
        "was not checked."
    )

_COND_ATOM_RE = re.compile(rf"({_NET})\s+is\s+(0|1|high|low)", re.IGNORECASE)
_COND_OP_RE = re.compile(r"\b(and|or)\b", re.IGNORECASE)
_BOTH_RE = re.compile(r"\bboth\b", re.IGNORECASE)
# `[ \t]*`, not `\s*`: an EMPTY "Input pattern:" is followed by a newline, and
# `\s*` would swallow it and read the NEXT line (e.g. "Time = 0.00 sec") as
# the pattern's content.
_INPUT_PATTERN_RE = re.compile(r"Input pattern:[ \t]*(.*)")
# ABC's own "INPUT: a = 1'h1, n0 = 4'ha.  OUTPUT: ..." line: one item per PI
# (a bus PI is ONE word `name = W'hHEX`, not per-bit), items joined by ", ".
_INPUT_LINE_RE = re.compile(r"INPUT: (.*?)\.  OUTPUT:", re.S)
_INPUT_ITEM_RE = re.compile(r"(\S+) = (\d+)'h([0-9a-fA-F]+)")
# ABC names individual bus bits like "n0[7]" (confirmed by real output on a
# multi-bit-PI design, e.g. `Input pattern:  n0[7]=0 n0[0]=0 ...`), not just
# bare single-bit names like the module docstring's own worked example
# ("a=1 b=0") -- the bracket suffix must be captured too, or a bus-PI
# design's counterexample line fails to parse at all (see
# `parse_counterexample`'s docstring: that must be a loud error, and it was,
# until this pattern was widened to match this real case).
_PI_ASSIGN_RE = re.compile(r"(\w+(?:\[\d+\])?)=([01])")

_VALUE_MAP = {"0": 0, "1": 1, "low": 0, "high": 1}


@dataclass(frozen=True)
class CondLiteral:
    token: str
    value: int  # 0 or 1


@dataclass(frozen=True)
class ConditionExpr:
    literals: list[CondLiteral]
    ops: list[str]  # "and"/"or", one fewer entry than literals (left-associative)


def parse_condition(text: str) -> ConditionExpr:
    """Parse a bounded condition: one or more "<net> is <0|1|high|low>"
    literals joined by "and"/"or" (an optional leading "both" is stripped
    first). Raises `ValueError` on anything outside this shape -- this is a
    deliberate scope boundary, not a bug to be "fixed" by loosening it (see
    module docstring)."""
    cleaned = _BOTH_RE.sub(" ", text)
    parts = _COND_OP_RE.split(cleaned)
    # re.split() with one capturing group interleaves [atom, op, atom, op, ...].
    atom_strs, ops = parts[0::2], [s.lower() for s in parts[1::2]]
    literals: list[CondLiteral] = []
    for atom in atom_strs:
        m = _COND_ATOM_RE.search(atom)
        if not m:
            raise ValueError(f"cannot parse condition literal {atom.strip()!r} (from {text!r})")
        literals.append(CondLiteral(m.group(1), _VALUE_MAP[m.group(2).lower()]))
    if not literals:
        raise ValueError(f"no condition literals found in {text!r}")
    return ConditionExpr(literals, ops)


class EmptyInputPatternError(ValueError):
    """ABC printed an "Input pattern:" line with no assignments on it."""


def parse_counterexample(detail: str) -> dict[str, int]:
    """Parse ABC `cec`'s "Input pattern:  a=1 b=0" line (see abc_bridge.py's
    module docstring for the confirmed verbatim format) into a {pi_name:
    0/1} dict. Raises `ValueError` -- never silently returns an empty/partial
    result -- if the line is missing or has no parseable assignments, so a
    format this hasn't been tested against fails loudly instead of being
    guessed at.

    A line that is present but whitespace-only raises the subclass
    `EmptyInputPatternError` (a line with non-assignment content raises plain
    `ValueError`). That is a real ABC output, not a format
    surprise: "Input pattern:" lists only the failing output's AIG support
    in Network1. When the property net strashes down to a single PI literal
    (possibly inverted) or a constant, the AIG has no AND node, the support
    is empty, and the line is printed empty. The "INPUT:" line of the same
    output still carries a full counterexample (see `parse_input_line`)."""
    m = _INPUT_PATTERN_RE.search(detail)
    if not m:
        raise ValueError(f"no 'Input pattern:' line found in ABC output: {detail!r}")
    content = m.group(1)
    assignments = {name: int(val) for name, val in _PI_ASSIGN_RE.findall(content)}
    if not assignments:
        if content.strip():
            # Something is on the line but nothing parses: an unknown format,
            # not ABC's genuinely empty pattern. Fail loudly as before.
            raise ValueError(f"'Input pattern:' line had unparseable content: {detail!r}")
        raise EmptyInputPatternError(f"'Input pattern:' line had no parseable assignments: {detail!r}")
    return assignments


def parse_input_line(detail: str) -> dict[str, int]:
    """Parse ABC's "INPUT: name = W'hHEX, ...  .  OUTPUT:" line into
    {bit_token: 0/1}. A W>1 bus PI is decoded as bit i = (HEX >> i) & 1 under
    key "name[i]". A W==1 item gets BOTH keys "name" and "name[0]": ABC prints
    a 1-bit PI as `name = 1'h1` whether it is a scalar or a `[0:0]` bus, so the
    index cannot be recovered from the line; the cone restriction in
    `check_asserted_only_when` picks whichever one the design actually uses.
    Raises `ValueError` on a missing line, on ANY item not shaped
    `name = W'hHEX` (never a partial result; an empty line `INPUT: .` is one
    empty item and fails this way), or on a value that does not fit W bits."""
    m = _INPUT_LINE_RE.search(detail)
    if not m:
        raise ValueError(f"no 'INPUT:' line found in ABC output: {detail!r}")
    out: dict[str, int] = {}
    for item in m.group(1).split(", "):
        im = _INPUT_ITEM_RE.fullmatch(item.strip())
        if not im:
            raise ValueError(f"unparseable item {item!r} in ABC 'INPUT:' line: {detail!r}")
        name, width, v = im.group(1), int(im.group(2)), int(im.group(3), 16)
        if width < 1 or v >= 2**width:
            raise ValueError(f"value {item!r} does not fit its width in ABC 'INPUT:' line: {detail!r}")
        if width == 1:
            out[name] = v
            out[f"{name}[0]"] = v
        else:
            for i in range(width):
                out[f"{name}[{i}]"] = (v >> i) & 1
    return out


def _fanin_cone_pi_tokens(work: Design, prop_net: NetBit) -> set[str]:
    """PI bit tokens read by `prop_net`'s structural fanin cone, in the SAME
    combinational view `check_implication` builds (free_pi, so DFF Q's are
    "__dff_Q__<inst>" pseudo-PIs named exactly as ABC prints them)."""
    comb = extract_combinational_view(work, "free_pi", {})

    def _is_pi(nb: object) -> bool:
        return isinstance(nb, NetBit) and comb.signals[nb.name].direction == Direction.INPUT

    tokens: set[str] = set()
    if _is_pi(prop_net):
        tokens.add(netbit_token(prop_net))
    cone = NetlistGraph(comb).backward_reachable_gates(prop_net)
    for g in comb.gates:
        if g.inst_name not in cone:
            continue
        out_pin = OUTPUT_PIN[g.gate_type]
        for pin, val in g.pins.items():
            if pin != out_pin and _is_pi(val):
                tokens.add(netbit_token(val))
    return tokens


def _literal_net(work: Design, lit: CondLiteral) -> NetBit:
    nb = resolve_bit(work, lit.token)
    if lit.value == 1:
        return nb
    out = work.fresh_net("t_propcheck_not_")
    work.add_gate(
        Gate(inst_name=work.fresh_gate_name("t_propcheck_gate_"), gate_type=GateType.NOT, pins={"O": out, "I0": nb})
    )
    return out


def _build_condition_net(work: Design, expr: ConditionExpr) -> NetBit:
    acc = _literal_net(work, expr.literals[0])
    for op, lit in zip(expr.ops, expr.literals[1:]):
        rhs = _literal_net(work, lit)
        out = work.fresh_net("t_propcheck_cond_")
        gate_type = GateType.AND if op == "and" else GateType.OR
        work.add_gate(
            Gate(inst_name=work.fresh_gate_name("t_propcheck_gate_"), gate_type=gate_type, pins={"O": out, "I0": acc, "I1": rhs})
        )
        acc = out
    return acc


@dataclass(frozen=True)
class PropertyResult:
    holds: bool
    detail: str
    assignment: Optional[dict[str, int]] = None
    caveat: Optional[str] = None
    # `assignment == {}` (not None): the property fails under EVERY input and
    # register state (its net is constant 0), so no variable needs pinning.


def check_asserted_only_when(
    design: Design, signal_token: str, condition_text: str, timeout: float = DEFAULT_ABC_TIMEOUT
) -> PropertyResult:
    """"For output <signal_token>, verify that it is asserted only when
    <condition_text>, and provide a counterexample if this is not true."

    Builds P = ~signal | condition on a throwaway deep copy of `design` and
    asks `abc_bridge.check_implication` whether P is constant-1. When it is
    not, `PropertyResult.assignment` is ABC's own counterexample (parsed by
    `parse_counterexample`) and `PropertyResult.caveat`, if set, means the
    counterexample pins at least one free-running DFF-Q pseudo-input to an
    arbitrary value -- reachability of that register state from reset was
    NOT checked (see module docstring). If ABC's "Input pattern:" line is empty
    (the property net is a single PI literal or a constant, so its AIG support
    is empty), the assignment is taken from ABC's full "INPUT:" line restricted
    to the property's structural fanin cone; `assignment == {}` then means the
    property is constant 0, i.e. violated under EVERY input and register
    state. Raises `ValueError` if
    `condition_text` is outside `parse_condition`'s supported grammar, or
    (`netref.NetRefError`, a `ValueError` subclass) if `signal_token` or any
    net reference inside `condition_text` doesn't resolve to exactly one
    net-bit on `design` (unknown signal, missing/invalid bit-select, or a
    bit-select outside the signal's declared range).
    """
    expr = parse_condition(condition_text)
    target = resolve_bit(design, signal_token)

    work = copy.deepcopy(design)
    cond_net = _build_condition_net(work, expr)
    not_target = work.fresh_net("t_propcheck_not_target_")
    work.add_gate(
        Gate(inst_name=work.fresh_gate_name("t_propcheck_gate_"), gate_type=GateType.NOT, pins={"O": not_target, "I0": target})
    )
    prop_net = work.fresh_net("t_propcheck_prop_")
    work.add_gate(
        Gate(
            inst_name=work.fresh_gate_name("t_propcheck_gate_"),
            gate_type=GateType.OR,
            pins={"O": prop_net, "I0": not_target, "I1": cond_net},
        )
    )

    promoted_q_source: dict[str, NetBit] = {}
    result = check_implication(work, prop_net, promoted_q_source, timeout=timeout)
    if result.equivalent:
        return PropertyResult(True, result.detail)

    try:
        assignment = parse_counterexample(result.detail)
    except EmptyInputPatternError:
        # Property net is a single PI literal or a constant: ABC's "Input
        # pattern:" is empty. Take the full "INPUT:" line, restricted to the
        # property's own fanin cone; a cone token it lacks means the decode
        # rule is wrong, so fail loudly rather than guess.
        cone = _fanin_cone_pi_tokens(work, prop_net)
        if cone:
            full = parse_input_line(result.detail)
            missing = sorted(t for t in cone if t not in full)
            if missing:
                raise ValueError(f"cone PI(s) {missing} missing from ABC 'INPUT:' line: {result.detail!r}")
            assignment = {k: full[k] for k in sorted(cone)}
        else:
            # Constant-0 property. A design with no PI at all prints a
            # genuinely EMPTY "INPUT: ." line (real ABC output), which
            # `parse_input_line` rightly rejects, so it is not consulted here.
            assignment = {}
    caveat = None
    # `assignment` keys are the counterexample's PI tokens. A DFF's Q state is
    # a synthetic per-instance PI "__dff_Q__<inst>" (see
    # `extract_combinational_view`); `promoted_q_source` maps each such token
    # to the original Q net-bit it drives. The caveat check stays BEFORE the
    # mapping (the synthetic keys are what mark a DFF-state counterexample),
    # then the keys are rewritten to the design's own net names.
    if any(name in promoted_q_source for name in assignment):
        caveat = free_pi_caveat("counterexample")
    assignment = {
        (netbit_token(promoted_q_source[k]) if k in promoted_q_source else k): v for k, v in assignment.items()
    }
    return PropertyResult(False, result.detail, assignment, caveat)
