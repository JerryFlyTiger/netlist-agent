"""Unit tests for netlist_agent/property_check.py: condition-clause parsing,
ABC counterexample parsing, and the end-to-end "X asserted only when COND"
implication check (both a property that HOLDS and one that does NOT, the
latter's counterexample independently re-simulated via tests/sim.py rather
than trusted from the handler's own answer).
"""

from __future__ import annotations

import pytest

from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, Port, Signal
from netlist_agent import property_check
from netlist_agent.property_check import (
    CondLiteral,
    ConditionExpr,
    EmptyInputPatternError,
    check_asserted_only_when,
    parse_condition,
    parse_counterexample,
    parse_input_line,
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


def test_parse_counterexample_keeps_dotted_and_dollar_names_verbatim() -> None:
    detail = "Verification failed.\nInput pattern:  a=0 __dff_Q__u1.r_reg=1 x$y=0 b.x[3]=1"
    assert parse_counterexample(detail) == {"a": 0, "__dff_Q__u1.r_reg": 1, "x$y": 0, "b.x[3]": 1}


@pytest.mark.parametrize(
    "bad",
    ["a=0 b=2", "a=0 junk", "a=0 =1", "a=0 b=", "a:0", "a=0 b=1x", "a=0 b=01", "a=0 b=10", "a=0=1"],
)
def test_parse_counterexample_any_bad_token_is_plain_valueerror(bad: str) -> None:
    with pytest.raises(ValueError) as ei:
        parse_counterexample(f"Verification failed.\nInput pattern:  {bad}")
    assert not isinstance(ei.value, EmptyInputPatternError)


def test_parse_counterexample_duplicate_key_raises() -> None:
    with pytest.raises(ValueError, match="twice") as ei:
        parse_counterexample("Verification failed.\nInput pattern:  a=0 a=1")
    assert not isinstance(ei.value, EmptyInputPatternError)


def test_parse_counterexample_whitespace_only_is_still_empty_pattern_error() -> None:
    with pytest.raises(EmptyInputPatternError):
        parse_counterexample("Verification failed.\nInput pattern:   \t \nTime = 0.00 sec")


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
    # the synthetic per-instance PI is mapped back to the design's own net name
    assert "busy" in result.assignment
    assert not any("__dff_Q__" in k for k in result.assignment)
    assert result.caveat is not None


# ----------------------------------------------------------------------
# F-H: empty "Input pattern:" (property net is a single PI literal or a
# constant). ABC output below is captured from REAL runs via a spy on
# `property_check.check_implication`, never hand-written (except where a
# test says "synthetic").
# ----------------------------------------------------------------------

@pytest.fixture
def abc_details(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collects `result.detail` of every real `check_implication` call."""
    details: list[str] = []
    orig = property_check.check_implication

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        r = orig(*args, **kwargs)
        details.append(r.detail)
        return r

    monkeypatch.setattr(property_check, "check_implication", spy)
    return details


def _buf_pi_design(n_pi: int) -> Design:
    """`n_pi` PIs p0..; done = p0 (single-PI case names it `req` via n_pi==1)."""
    design = Design(module_name="bufpi")
    names = ["req"] if n_pi == 1 else [f"p{i}" for i in range(n_pi)]
    for n in names:
        _pi(design, n)
    _po(design, "done")
    design.add_gate(Gate("g_done", GateType.BUF, {"O": NetBit("done"), "I0": NetBit(names[0])}))
    return design


def _const_done_design(with_pi: bool) -> Design:
    design = Design(module_name="constdone")
    if with_pi:
        _pi(design, "req")
    _po(design, "done")
    design.add_gate(Gate("g_done", GateType.BUF, {"O": NetBit("done"), "I0": Const.ONE}))
    return design


def test_parse_counterexample_real_empty_pattern_at_end_of_output(abc_details: list[str]) -> None:
    # Kills: reverting to the old behaviour (plain ValueError, no subclass) --
    # callers distinguish "empty but real" from "format unknown".
    check_asserted_only_when(_buf_pi_design(1), "done", "req is 0")
    assert abc_details[-1].rstrip().endswith("Input pattern:")
    with pytest.raises(EmptyInputPatternError):
        parse_counterexample(abc_details[-1])


def test_parse_counterexample_real_empty_pattern_followed_by_time_line(abc_details: list[str]) -> None:
    check_asserted_only_when(_const_done_design(with_pi=False), "done", "done is 0")
    assert "Input pattern: \nTime =" in abc_details[-1]
    with pytest.raises(EmptyInputPatternError):
        parse_counterexample(abc_details[-1])


def test_parse_counterexample_empty_pattern_does_not_read_next_line() -> None:
    # SYNTHETIC string. Kills: `[ \t]*` -> `\s*` in _INPUT_PATTERN_RE, which
    # would swallow the newline and return {"x": 1} from the NEXT line.
    with pytest.raises(EmptyInputPatternError):
        parse_counterexample("Input pattern: \nx=1 is on the next line")


def test_parse_input_line_one_bit_items() -> None:
    detail = "foo\nINPUT: a = 1'h1, b = 1'h0, c = 1'h1.  OUTPUT: o = 1'h0 (a).\n"
    assert parse_input_line(detail) == {"a": 1, "a[0]": 1, "b": 0, "b[0]": 0, "c": 1, "c[0]": 1}


def test_parse_input_line_bus_bit_order() -> None:
    # 0xa = 0b1010 is asymmetric: kills bit order reversed / off-by-one.
    got = parse_input_line("INPUT: n0 = 4'ha.  OUTPUT: o = 1'h0 (a).")
    assert got == {"n0[0]": 0, "n0[1]": 1, "n0[2]": 0, "n0[3]": 1}


def test_parse_input_line_rejects_any_malformed_item() -> None:
    # Kills: skipping unparseable items (silent partial result).
    with pytest.raises(ValueError):
        parse_input_line("INPUT: a = 1'h1, garbage, b = 1'h0.  OUTPUT: o = 1'h0 (a).")


def test_parse_input_line_rejects_value_wider_than_width() -> None:
    # Kills: dropping the `v >= 2**W` check.
    with pytest.raises(ValueError):
        parse_input_line("INPUT: n0 = 4'h1f.  OUTPUT: o = 1'h0 (a).")
    # exactly 2**W (boundary): kills `>=` -> `>`.
    with pytest.raises(ValueError):
        parse_input_line("INPUT: n0 = 4'h10.  OUTPUT: o = 1'h0 (a).")


def test_parse_input_line_missing_line_raises() -> None:
    with pytest.raises(ValueError):
        parse_input_line("Networks are equivalent.  Time = 0.00 sec")


def test_parse_input_line_empty_line_raises() -> None:
    # Real shape of a no-PI design's line ("INPUT: .  OUTPUT:"). Kills:
    # returning {} for it.
    with pytest.raises(ValueError):
        parse_input_line("INPUT: .  OUTPUT: o = 1'h0 (a).")


def _bus(design: Design, name: str, msb: int, lsb: int, direction: Direction = Direction.INPUT) -> None:
    design.signals[name] = Signal(name=name, msb=msb, lsb=lsb, direction=direction)
    design.ports.append(Port(name=name, direction=direction))


def test_parse_input_line_agrees_with_real_nonempty_input_pattern(abc_details: list[str]) -> None:
    # Hand-built bus-PI design (a, b are [7:0]); done = a[0] & a[1] & b[2].
    # Kills: wrong bus decoding, which would disagree with ABC's own per-bit
    # (non-empty) pattern.
    design = Design(module_name="busagree")
    _bus(design, "a", 7, 0)
    _bus(design, "b", 7, 0)
    _po(design, "done")
    design.signals["t"] = Signal(name="t", msb=None, lsb=None, direction=Direction.INTERNAL)
    design.add_gate(Gate("g0", GateType.AND, {"O": NetBit("t"), "I0": NetBit("a", 0), "I1": NetBit("a", 1)}))
    design.add_gate(Gate("g1", GateType.AND, {"O": NetBit("done"), "I0": NetBit("t"), "I1": NetBit("b", 2)}))
    design.build_indices()
    result = check_asserted_only_when(design, "done", "b[5] is 1")
    assert result.holds is False
    detail = abc_details[-1]
    pattern = parse_counterexample(detail)  # non-empty path
    assert pattern
    full = parse_input_line(detail)
    # really exercised the bus decoding (a W==1 item also yields "name[0]",
    # so "any key with [" would hold even for an all-scalar design)
    assert "a[7]" in full and "b[7]" in full
    for k, v in pattern.items():
        assert full[k] == v


def test_parse_counterexample_garbage_content_is_plain_valueerror() -> None:
    # Kills: raising EmptyInputPatternError for any no-assignment line (would
    # route an unknown format into the INPUT:-line fallback instead of
    # failing loudly).
    with pytest.raises(ValueError) as ei:
        parse_counterexample("Input pattern:  ???garbage")
    assert not isinstance(ei.value, EmptyInputPatternError)


def test_parse_input_line_zero_width_raises() -> None:
    # Kills: deleting the `width < 1` check.
    with pytest.raises(ValueError):
        parse_input_line("INPUT: a = 1'h1, z = 0'h0.  OUTPUT: x")


def test_parse_input_line_one_bit_item_has_both_keys() -> None:
    # Kills: dropping the `name[0]` alias (a [0:0] bus PI is printed unindexed).
    assert parse_input_line("INPUT: n = 1'h1.  OUTPUT: x") == {"n": 1, "n[0]": 1}


def _fake_equiv(detail: str):  # type: ignore[no-untyped-def]
    from netlist_agent.abc_bridge import EquivResult

    def fake(*args, **kwargs):  # type: ignore[no-untyped-def]
        return EquivResult(False, detail)

    return fake


def test_empty_pattern_cone_token_missing_from_input_line_raises_valueerror(monkeypatch: pytest.MonkeyPatch) -> None:
    # Kills: deleting `if missing: raise` (would be a KeyError on full[k]).
    monkeypatch.setattr(
        property_check,
        "check_implication",
        _fake_equiv("NOT EQUIVALENT.  INPUT: other = 1'h0.  OUTPUT: o = 1'h0 (a).\nInput pattern: \n"),
    )
    with pytest.raises(ValueError, match="missing"):
        check_asserted_only_when(_buf_pi_design(1), "done", "req is 0")


def test_missing_input_pattern_line_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Kills: widening `except EmptyInputPatternError` to `except ValueError`
    # (a format with no "Input pattern:" line at all must stay a loud failure).
    monkeypatch.setattr(
        property_check,
        "check_implication",
        _fake_equiv("NOT EQUIVALENT.  INPUT: req = 1'h1.  OUTPUT: o = 1'h0 (a).\n"),
    )
    with pytest.raises(ValueError):
        check_asserted_only_when(_buf_pi_design(1), "done", "req is 0")


def test_check_asserted_only_when_single_bit_bus_pi(abc_details: list[str]) -> None:
    # `input [0:0] n; done = n[0]`: ABC prints `n = 1'h1` (no index), so the
    # cone token "n[0]" is only found through parse_input_line's alias.
    # Kills: removing the `name[0]` alias (-> "cone PI(s) ['n[0]'] missing").
    design = Design(module_name="bus1")
    _bus(design, "n", 0, 0)
    _po(design, "done")
    design.add_gate(Gate("g_done", GateType.BUF, {"O": NetBit("done"), "I0": NetBit("n", 0)}))
    design.build_indices()
    result = check_asserted_only_when(design, "done", "n[0] is 0")
    assert result.holds is False
    with pytest.raises(EmptyInputPatternError):  # really took the empty path
        parse_counterexample(abc_details[-1])
    assert result.assignment == {"n[0]": 1}
    values = simulate(design, inputs={NetBit("n", 0): 1})
    assert values[NetBit("done")] == 1


def _assert_violation(design: Design, sig: str, cond_net: str, cond_val: int, assignment: dict[str, int]) -> None:
    """Independent re-simulation: with `assignment` (unlisted PIs = 0, which
    must not matter), `sig` is 1 while `cond_net` != cond_val."""
    inputs = {
        NetBit(p.name): assignment.get(p.name, 0) for p in design.ports if p.direction == Direction.INPUT
    }
    values = simulate(design, inputs=inputs)
    assert values[NetBit(sig)] == 1
    assert values.get(NetBit(cond_net), inputs.get(NetBit(cond_net))) != cond_val


def test_check_asserted_only_when_single_pi_literal(abc_details: list[str]) -> None:
    design = _buf_pi_design(1)
    result = check_asserted_only_when(design, "done", "req is 0")
    assert result.holds is False
    with pytest.raises(EmptyInputPatternError):  # really took the empty path
        parse_counterexample(abc_details[-1])
    assert result.assignment == {"req": 1}
    _assert_violation(design, "done", "req", 0, result.assignment)


def test_check_asserted_only_when_cone_restricts_to_property_inputs(abc_details: list[str]) -> None:
    # Kills: returning ALL of INPUT:'s PIs instead of the cone's.
    design = _buf_pi_design(3)
    result = check_asserted_only_when(design, "done", "p0 is 0")
    assert result.holds is False
    with pytest.raises(EmptyInputPatternError):
        parse_counterexample(abc_details[-1])
    assert {k for k in parse_input_line(abc_details[-1]) if "[" not in k} == {"p0", "p1", "p2"}
    assert result.assignment == {"p0": 1}
    _assert_violation(design, "done", "p0", 0, result.assignment)


def test_check_asserted_only_when_constant_property_has_empty_assignment(abc_details: list[str]) -> None:
    # No-PI design: real ABC prints an empty "INPUT: ." line, so this also
    # kills consulting parse_input_line when the cone is empty.
    design = _const_done_design(with_pi=False)
    result = check_asserted_only_when(design, "done", "done is 0")
    assert result.holds is False
    assert "INPUT: .  OUTPUT:" in abc_details[-1]
    assert result.assignment == {}
    assert result.caveat is None


def test_check_asserted_only_when_constant_property_with_unrelated_pi() -> None:
    # PI exists but the property cone is empty: still `{}`, not {"req": ...}.
    design = _const_done_design(with_pi=True)
    result = check_asserted_only_when(design, "done", "done is 0")
    assert result.holds is False
    assert result.assignment == {}


def test_check_asserted_only_when_dff_single_literal_maps_name_and_caveat(abc_details: list[str]) -> None:
    # done = BUF(busy), busy = DFF.Q; "busy is 0" shrinks to one inverted
    # literal. Kills: skipping the __dff_Q__ -> net-name mapping or the
    # caveat on the empty-pattern path.
    design = Design(module_name="seqlit")
    _pi(design, "clk")
    _pi(design, "rn")
    _po(design, "done")
    design.signals["busy"] = Signal(name="busy", msb=None, lsb=None, direction=Direction.INTERNAL)
    design.add_gate(Gate("g_done", GateType.BUF, {"O": NetBit("done"), "I0": NetBit("busy")}))
    design.add_gate(
        Gate(
            "dff0",
            GateType.DFF,
            {"RN": NetBit("rn"), "SN": Const.ONE, "CK": NetBit("clk"), "D": Const.ZERO, "Q": NetBit("busy")},
        )
    )
    design.build_indices()
    result = check_asserted_only_when(design, "done", "busy is 0")
    assert result.holds is False
    with pytest.raises(EmptyInputPatternError):
        parse_counterexample(abc_details[-1])
    assert result.assignment == {"busy": 1}
    assert result.caveat is not None
    assert not any("__dff_Q__" in k for k in result.assignment)
    values = simulate(design, inputs={NetBit("clk"): 0, NetBit("rn"): 1}, dff_q={NetBit("busy"): 1})
    assert values[NetBit("done")] == 1


# ----------------------------------------------------------------------
# Batch 10: dotted DFF instance name (bypassing the rename entry guard) must
# still give the right key and keep the free_pi caveat.
# ----------------------------------------------------------------------

_DOT_NETLIST = """module top(clk, a, b, y);
  input clk, a, b;
  output y;
  wire q, n1;
  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(q));
  and g1(n1, q, b);
  buf g2(y, n1);
endmodule
"""


def test_dotted_dff_instance_name_keeps_key_and_caveat(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from netlist_agent.parser import parse_verilog

    path = tmp_path / "dot.v"
    path.write_text(_DOT_NETLIST)
    design = parse_verilog(str(path))
    gate = next(g for g in design.gates if g.inst_name == "r1")
    gate.inst_name = "u1.r_reg"  # bypasses the entry-point name guard on purpose
    design._gate_index = {}

    result = check_asserted_only_when(design, "y", "a is 1")
    assert result.holds is False
    assert result.assignment == {"a": 0, "b": 1, "q": 1}
    assert not any("__dff_Q__" in k for k in result.assignment)
    assert result.caveat is not None
