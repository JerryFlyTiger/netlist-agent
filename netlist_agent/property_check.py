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

from netlist_agent.abc_bridge import DEFAULT_ABC_TIMEOUT, check_implication
from netlist_agent.ir import Design, Gate, GateType, NetBit
from netlist_agent.netref import resolve_bit, signal_name_only

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
_INPUT_PATTERN_RE = re.compile(r"Input pattern:\s*(.*)")
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


def parse_counterexample(detail: str) -> dict[str, int]:
    """Parse ABC `cec`'s "Input pattern:  a=1 b=0" line (see abc_bridge.py's
    module docstring for the confirmed verbatim format) into a {pi_name:
    0/1} dict. Raises `ValueError` -- never silently returns an empty/partial
    result -- if the line is missing or has no parseable assignments, so a
    format this hasn't been tested against fails loudly instead of being
    guessed at."""
    m = _INPUT_PATTERN_RE.search(detail)
    if not m:
        raise ValueError(f"no 'Input pattern:' line found in ABC output: {detail!r}")
    assignments = {name: int(val) for name, val in _PI_ASSIGN_RE.findall(m.group(1))}
    if not assignments:
        raise ValueError(f"'Input pattern:' line had no parseable assignments: {detail!r}")
    return assignments


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
    NOT checked (see module docstring). Raises `ValueError` if
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

    assignment = parse_counterexample(result.detail)
    caveat = None
    # `assignment` keys may be per-bit-of-a-bus tokens like "n0[7]" (ABC bit-
    # blasts bus PIs in its own "Input pattern:" line -- confirmed by real
    # output, see `_PI_ASSIGN_RE`), while `promoted_q_source` is keyed by
    # whole-SIGNAL name (no bracket suffix -- promotion happens at Signal
    # granularity, see `extract_combinational_view`'s docstring) -- strip the
    # bit-select before checking membership, or a promoted-DFF-Q bus PI's
    # caveat would be silently missed.
    if any(signal_name_only(name) in promoted_q_source for name in assignment):
        caveat = free_pi_caveat("counterexample")
    return PropertyResult(False, result.detail, assignment, caveat)
