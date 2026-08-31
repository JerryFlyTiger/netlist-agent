"""Unit tests for netlist_agent/property_check.py: condition-clause parsing,
ABC counterexample parsing, and the end-to-end "X asserted only when COND"
implication check (both a property that HOLDS and one that does NOT, the
latter's counterexample independently re-simulated via tests/sim.py rather
than trusted from the handler's own answer).
"""

from __future__ import annotations

import pytest

from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, Port, Signal
from netlist_agent.property_check import (
    CondLiteral,
    ConditionExpr,
    check_asserted_only_when,
    parse_condition,
    parse_counterexample,
)
from tests.sim import simulate

# ----------------------------------------------------------------------
# parse_condition
# ----------------------------------------------------------------------


def test_parse_condition_both_and() -> None:
    expr = parse_condition("both req is 1 and busy is 0")
    assert expr == ConditionExpr(
        literals=[CondLiteral("req", 1), CondLiteral("busy", 0)], ops=["and"]
    )


def test_parse_condition_or() -> None:
    expr = parse_condition("x is 0 or y is 1")
    assert expr == ConditionExpr(literals=[CondLiteral("x", 0), CondLiteral("y", 1)], ops=["or"])


def test_parse_condition_high_low_single_literal() -> None:
    expr = parse_condition("a is high")
    assert expr == ConditionExpr(literals=[CondLiteral("a", 1)], ops=[])
    expr2 = parse_condition("a is low")
    assert expr2 == ConditionExpr(literals=[CondLiteral("a", 0)], ops=[])


def test_parse_condition_out_of_scope_raises() -> None:
    with pytest.raises(ValueError):
        parse_condition("the FSM is in state IDLE")


def test_parse_condition_bit_select_literal() -> None:
    expr = parse_condition("n6[3] is 1")
    assert expr == ConditionExpr(literals=[CondLiteral("n6[3]", 1)], ops=[])


# ----------------------------------------------------------------------
# parse_counterexample
# ----------------------------------------------------------------------

_REAL_ABC_NOT_EQUIV_OUTPUT = (
    '======== ABC command line "cec ..." \n'
    "Networks are NOT EQUIVALENT.  Time =     0.01 sec\n"
    "INPUT: a = 1'h1, b = 1'h0.  OUTPUT: prop_out = 1'h0 (a), prop_out = 1'h1 (b).\n"
    "Verification failed for at least 1 outputs:  prop_out\n"
    "Output prop_out: Value in Network1 = 0. Value in Network2 = 1.\n"
    "Input pattern:  a=1 b=0"
)


def test_parse_counterexample_real_abc_output() -> None:
    assert parse_counterexample(_REAL_ABC_NOT_EQUIV_OUTPUT) == {"a": 1, "b": 0}


def test_parse_counterexample_missing_line_raises() -> None:
    with pytest.raises(ValueError):
        parse_counterexample("Networks are equivalent after structural hashing.  Time = 0.00 sec")


def test_parse_counterexample_bus_bit_names() -> None:
    # ABC bit-blasts bus PIs in its own "Input pattern:" line -- confirmed by
    # a real run against Alpha_Testcase/test01 (a multi-bit-input design).
    detail = "Verification failed.\nInput pattern:  n0[7]=0 n0[0]=0 n2[1]=0 n0[1]=1 n2[0]=1"
    assert parse_counterexample(detail) == {
        "n0[7]": 0,
        "n0[0]": 0,
        "n2[1]": 0,
        "n0[1]": 1,
        "n2[0]": 1,
    }


# ----------------------------------------------------------------------
# check_asserted_only_when: end-to-end, both directions
# ----------------------------------------------------------------------


def _pi(design: Design, name: str) -> None:
    design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    design.ports.append(Port(name=name, direction=Direction.INPUT))


def _po(design: Design, name: str) -> None:
    design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports.append(Port(name=name, direction=Direction.OUTPUT))


def _build_holds_design() -> Design:
    """done = req AND ~busy -- exactly satisfies "done asserted only when
    req is 1 and busy is 0"."""
    design = Design(module_name="holds")
    _pi(design, "req")
    _pi(design, "busy")
    _po(design, "done")
    not_busy = design.fresh_net("t_")
    design.add_gate(Gate("g_not_busy", GateType.NOT, {"O": not_busy, "I0": NetBit("busy")}))
    design.add_gate(Gate("g_done", GateType.AND, {"O": NetBit("done"), "I0": NetBit("req"), "I1": not_busy}))
    return design


def _build_violates_design() -> Design:
    """done = req (ignores busy entirely) -- violates the property whenever
    req=1 and busy=1."""
    design = Design(module_name="violates")
    _pi(design, "req")
    _pi(design, "busy")
    _po(design, "done")
    design.add_gate(Gate("g_done", GateType.BUF, {"O": NetBit("done"), "I0": NetBit("req")}))
    return design


def test_check_asserted_only_when_holds() -> None:
    design = _build_holds_design()
    result = check_asserted_only_when(design, "done", "both req is 1 and busy is 0")
    assert result.holds is True
    assert result.assignment is None


def test_check_asserted_only_when_violates_with_verified_counterexample() -> None:
    design = _build_violates_design()
    result = check_asserted_only_when(design, "done", "both req is 1 and busy is 0")
    assert result.holds is False
    assert result.assignment is not None
    assert result.caveat is None  # no DFFs in this design, so no free-pi promotion

    # Independently re-derive the violation from the reported assignment via
    # the plain simulator, rather than trusting the handler's own verdict.
    values = simulate(
        design,
        inputs={NetBit("req"): result.assignment["req"], NetBit("busy"): result.assignment["busy"]},
    )
    done_val = values[NetBit("done")]
    property_holds_here = (done_val == 0) or (
        result.assignment["req"] == 1 and result.assignment["busy"] == 0
    )
    assert done_val == 1
    assert not property_holds_here


def test_check_asserted_only_when_dff_counterexample_carries_caveat() -> None:
    """`busy` is a DFF Q output (not a genuine PI); a counterexample that
    pins it must carry the reachability caveat (free_pi over-approximation
    -- see property_check.py's module docstring)."""
    design = Design(module_name="seq")
    _pi(design, "req")
    _pi(design, "clk")
    _pi(design, "rn")
    _po(design, "done")
    design.signals["busy"] = Signal(name="busy", msb=None, lsb=None, direction=Direction.INTERNAL)
    design.add_gate(Gate("g_done", GateType.BUF, {"O": NetBit("done"), "I0": NetBit("req")}))
    design.add_gate(
        Gate(
            "dff0",
            GateType.DFF,
            {"RN": NetBit("rn"), "SN": Const.ONE, "CK": NetBit("clk"), "D": Const.ZERO, "Q": NetBit("busy")},
        )
    )
    design.build_indices()

    result = check_asserted_only_when(design, "done", "both req is 1 and busy is 0")
    assert result.holds is False
    assert result.assignment is not None
    assert "busy" in result.assignment
    assert result.caveat is not None
