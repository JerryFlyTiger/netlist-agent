"""Unit tests for netlist_agent/boolean_function.py.

Two kinds of fixtures:
  * Real corpus files (Alpha_Testcase/testcase/test{31,34,35,37,39}) -- these
    pin down the exact numbers the spec's own cross-validation (via a
    scratch probe script) already confirmed against the corpus's
    ground_truth.json: support size, PI/DFF.Q composition, and truth-table
    onset counts. If any of these five numbers drifts, the derivation logic
    itself is wrong -- these are NOT adjustable expectations.
  * Small hand-built synthetic Designs (`_pi`/`_po`/`_internal` helpers,
    mirroring test_property_check.py's own style) covering cases the real
    corpus doesn't happen to exercise: a pure-PI combinational function, a
    combinational function whose support mixes PI and DFF.Q, a DFF-driven
    target whose D-cone support mixes PI and DFF.Q, and a support size that
    exceeds the exhaustive-simulation cap.
"""

from __future__ import annotations

import os

import pytest

from netlist_agent.boolean_function import SUPPORT_EXHAUSTIVE_CAP, derive_boolean_function
from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, Port, Signal
from netlist_agent.parser import parse_verilog
from netlist_agent.property_check import free_pi_caveat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTCASE_DIR = os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase")


def _load(case: str) -> Design:
    _p = os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase")
    if not os.path.isdir(_p):
        pytest.skip("Alpha_Testcase corpus not present -- put the released testcases under Alpha_Testcase/testcase/testNN/ to enable this test")
    d = parse_verilog(os.path.join(TESTCASE_DIR, case, f"{case}.v"))
    d.build_indices()
    return d


# ----------------------------------------------------------------------
# Real corpus cases -- the five numbers in the spec's own table, confirmed
# to match ground_truth.json's support size / PI-Q composition / onset count.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,net,cone_gates,support,pi,dffq,onset,total",
    [
        ("test31", "n16", 49, 15, 0, 15, 3136, 32768),
        ("test34", "n30", 54, 19, 11, 8, 235445, 524288),
        ("test35", "n25", 5, 6, 0, 6, 63, 64),
        ("test37", "n8", 3, 3, 0, 3, 1, 8),
        ("test39", "n12", 14, 9, 0, 9, 95, 512),
    ],
)
def test_corpus_cases_match_ground_truth_facts(
    case: str, net: str, cone_gates: int, support: int, pi: int, dffq: int, onset: int, total: int
) -> None:
    design = _load(case)
    result = derive_boolean_function(design, net)
    assert result.cone_gate_count == cone_gates
    assert len(result.support) == support
    assert len(result.support_pi) == pi
    assert len(result.support_dffq) == dffq
    assert len(result.support_other) == 0
    assert result.onset_count == onset
    assert result.total_count == total
    assert result.truncated is False
    assert result.verified is True  # self-check: rendered expression == exhaustive simulation


def test_bit_select_out_of_declared_range_raises(caplog=None) -> None:
    """F2: n47 is declared `[3:0]` in test31 -- n47[99] previously silently
    fell into the `support_other` "floating free variable" bucket and
    reported a bogus, self-"verified" tautology instead of erroring."""
    design = _load("test31")
    with pytest.raises(ValueError, match=r"n47\[99\].*out of range"):
        derive_boolean_function(design, "n47[99]")


def test_scalar_signal_with_bit_select_raises() -> None:
    design = _load("test31")
    with pytest.raises(ValueError, match="scalar"):
        derive_boolean_function(design, "n16[0]")


def test_multi_bit_signal_without_bit_select_raises() -> None:
    design = _load("test31")
    with pytest.raises(ValueError, match="bit-select is required"):
        derive_boolean_function(design, "n47")


def test_test31_case1_dff_driven_all_q_support_wording() -> None:
    """n16 is directly a DFF's Q -- no PI equation of n16 itself exists at
    all, regardless of the D-cone's own support composition (here: 0 PI)."""
    design = _load("test31")
    result = derive_boolean_function(design, "n16")
    assert result.is_dff_q is True
    assert result.dff_inst == "g1027"
    assert result.expressible_in_pis_only is False
    assert "cannot be expressed as a combinational function of the primary inputs" in result.explanation
    assert "registered" in result.explanation.lower() or "sequential" in result.explanation.lower()
    # Case-1 wording must not claim a PI-only equation of the TARGET exists.
    assert f"{result.target} = " not in result.explanation.split("Expression:")[0]


def test_test34_case1_mixed_pi_and_q_support_named_explicitly() -> None:
    design = _load("test34")
    result = derive_boolean_function(design, "n30")
    assert result.is_dff_q is True
    assert set(result.support_pi) == {
        "n2", "n20[0]", "n20[1]", "n20[2]", "n20[3]", "n20[4]", "n20[5]", "n20[6]", "n20[7]", "n20[8]", "n20[9]"
    }
    assert set(result.support_dffq) == {"n20559", "n20560", "n20561", "n30", "n42", "n63[5]", "n63[6]", "n63[7]"}
    # Both buckets must be individually named in the explanation, not lumped.
    assert "11 primary input(s)" in result.explanation
    assert "8 register (DFF.Q) output(s)" in result.explanation
    assert result.caveat is not None


def test_test35_case2_combinational_with_all_q_support_wording() -> None:
    """n25 IS a combinational function (5 gates), but its support is 6
    DFF.Q nets, zero PI -- distinct wording from the case-1 (Q-driven)
    message: this one says the function itself exists, just not PI-only."""
    design = _load("test35")
    result = derive_boolean_function(design, "n25")
    assert result.is_dff_q is False
    assert result.expressible_in_pis_only is False
    assert "is a combinational function of 5 gate(s)" in result.explanation
    assert "cannot be written using only primary-input names" in result.explanation
    assert result.caveat is not None
    # Ground truth: n25=0 only at exactly one minterm (n990=n987=1, rest 0);
    # the offset is small enough (1 <= MINTERM_LIST_LIMIT) to be listed.
    assert result.offset_minterms is not None
    assert len(result.offset_minterms) == 1


def test_test37_case1_single_cube_next_state_not_a_false_positive() -> None:
    """The spec's own trap: test37's next-state function is a single cube
    (n13[2] & ~n13[1] & ~n13[0]) -- verified here alongside the genuinely
    multi-cube test31/34/35/39 cases so a renderer that only handles the
    single-cube shape correctly would still be caught by the others."""
    design = _load("test37")
    result = derive_boolean_function(design, "n8")
    assert result.onset_count == 1
    assert result.onset_minterms == [{"n13[0]": 0, "n13[1]": 0, "n13[2]": 1}]


def test_test35_rendered_expression_pinned() -> None:
    """F1: pin down the exact rendered (inline, 5-gate) expression string --
    display is the single source of truth for eval verification, but a
    display-only regression could still slip past self-verification if the
    corruption happens to be a coincidental structural no-op; pinning the
    literal string closes that gap."""
    design = _load("test35")
    result = derive_boolean_function(design, "n25")
    assert result.expression_lines == ["n25 = (~(n990 & ~(n989 | n988)) | ~(n987 & ~(n986 | n985)))"]


def test_test37_rendered_expression_pinned() -> None:
    """F1/F8: test37's next-state function is a NOT gate wired directly onto
    a NAND output -- the rendered `~~` double negation must be folded away
    (F8), and the resulting string is pinned exactly."""
    design = _load("test37")
    result = derive_boolean_function(design, "n8")
    assert result.expression_lines == ["n1168 = (n13[2] & ~(n13[1] | n13[0]))"]


# ----------------------------------------------------------------------
# Synthetic fixtures (see module docstring)
# ----------------------------------------------------------------------


def _pi(design: Design, name: str) -> None:
    design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    design.ports.append(Port(name=name, direction=Direction.INPUT))


def _po(design: Design, name: str) -> None:
    design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports.append(Port(name=name, direction=Direction.OUTPUT))


def _internal(design: Design, name: str) -> None:
    design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)


def _dff(design: Design, inst: str, d, q: str, clk: str = "clk", rn: str = "rn") -> None:
    design.add_gate(
        Gate(inst, GateType.DFF, {"RN": NetBit(rn), "SN": Const.ONE, "CK": NetBit(clk), "D": d, "Q": NetBit(q)})
    )


def _build_pi_only_design() -> Design:
    """out = a & b -- case 3: combinational, support entirely PI."""
    design = Design(module_name="pi_only")
    _pi(design, "a")
    _pi(design, "b")
    _po(design, "out")
    design.add_gate(Gate("g0", GateType.AND, {"O": NetBit("out"), "I0": NetBit("a"), "I1": NetBit("b")}))
    return design


def test_case3_pure_pi_combinational() -> None:
    design = _build_pi_only_design()
    result = derive_boolean_function(design, "out")
    assert result.is_dff_q is False
    assert result.expressible_in_pis_only is True
    assert result.support_pi == ["a", "b"]
    assert result.support_dffq == []
    assert result.caveat is None
    assert result.onset_count == 1  # only a=1,b=1
    assert result.total_count == 4
    assert result.verified is True
    assert "Yes, out is a combinational function" in result.explanation


def _build_mixed_pi_q_design() -> Design:
    """out = a & q -- case 2: combinational, support mixes PI and DFF.Q."""
    design = Design(module_name="mixed")
    _pi(design, "a")
    _pi(design, "clk")
    _pi(design, "rn")
    _po(design, "out")
    _internal(design, "q")
    design.add_gate(Gate("g0", GateType.AND, {"O": NetBit("out"), "I0": NetBit("a"), "I1": NetBit("q")}))
    _dff(design, "dff0", Const.ZERO, "q")
    design.build_indices()
    return design


def test_case2_mixed_pi_and_q_support_correctly_split() -> None:
    """Confirms the PI/Q split is genuinely computed, not accidentally
    always-empty or always-everything -- support has exactly 1 PI ('a') and
    1 DFF.Q ('q'), never conflated."""
    design = _build_mixed_pi_q_design()
    result = derive_boolean_function(design, "out")
    assert result.is_dff_q is False
    assert result.support_pi == ["a"]
    assert result.support_dffq == ["q"]
    assert result.expressible_in_pis_only is False
    assert result.caveat is not None
    assert "cannot be written using only primary-input names" in result.explanation


def _build_all_q_multi_cube_design() -> Design:
    """out = (q0 & q1) | (~q2 & q3) -- case 2, support all-DFF.Q, genuinely
    multi-cube (not the single-cube shape the spec warns can be gamed)."""
    design = Design(module_name="allq")
    _pi(design, "clk")
    _pi(design, "rn")
    _po(design, "out")
    for name in ("q0", "q1", "q2", "q3"):
        _internal(design, name)
    _internal(design, "not_q2")
    _internal(design, "and_a")
    _internal(design, "and_b")
    design.add_gate(Gate("g_not", GateType.NOT, {"O": NetBit("not_q2"), "I0": NetBit("q2")}))
    design.add_gate(Gate("g_and_a", GateType.AND, {"O": NetBit("and_a"), "I0": NetBit("q0"), "I1": NetBit("q1")}))
    design.add_gate(Gate("g_and_b", GateType.AND, {"O": NetBit("and_b"), "I0": NetBit("not_q2"), "I1": NetBit("q3")}))
    design.add_gate(Gate("g_out", GateType.OR, {"O": NetBit("out"), "I0": NetBit("and_a"), "I1": NetBit("and_b")}))
    for i, name in enumerate(("q0", "q1", "q2", "q3")):
        _dff(design, f"dff{i}", Const.ZERO, name)
    design.build_indices()
    return design


def test_case2_all_q_multi_cube_onset_and_self_verification() -> None:
    design = _build_all_q_multi_cube_design()
    result = derive_boolean_function(design, "out")
    assert result.support_pi == []
    assert set(result.support_dffq) == {"q0", "q1", "q2", "q3"}
    # (q0&q1)|(~q2&q3): true for 4 rows where q0=q1=1 (q2,q3 free: 4 combos)
    # plus rows where q2=0,q3=1 and NOT(q0=q1=1) already counted -> compute
    # via brute force instead of hand-deriving the overlap to avoid a
    # transcription bug in the test itself.
    total = 16
    onset = sum(
        1
        for q0 in (0, 1)
        for q1 in (0, 1)
        for q2 in (0, 1)
        for q3 in (0, 1)
        if (q0 & q1) | ((1 - q2) & q3)
    )
    assert result.onset_count == onset
    assert result.total_count == total
    assert result.verified is True
    assert onset > 1  # genuinely multi-cube, not the single-minterm trap


def _build_case1_allq_design() -> Design:
    """target is a DFF.Q; D = qa | qb (both other DFF.Q, zero PI) -- the
    test31/test39 shape (D-cone support entirely register state)."""
    design = Design(module_name="case1_allq")
    _pi(design, "clk")
    _pi(design, "rn")
    _po(design, "target")
    _internal(design, "qa")
    _internal(design, "qb")
    _internal(design, "dnet")
    design.add_gate(Gate("g0", GateType.OR, {"O": NetBit("dnet"), "I0": NetBit("qa"), "I1": NetBit("qb")}))
    _dff(design, "dff_target", NetBit("dnet"), "target")
    _dff(design, "dff_a", Const.ZERO, "qa")
    _dff(design, "dff_b", Const.ZERO, "qb")
    design.build_indices()
    return design


def test_case1_dff_driven_target_all_q_d_support() -> None:
    design = _build_case1_allq_design()
    result = derive_boolean_function(design, "target")
    assert result.is_dff_q is True
    assert result.dff_inst == "dff_target"
    assert result.root == "dnet"
    assert result.support_pi == []
    assert set(result.support_dffq) == {"qa", "qb"}
    assert result.expressible_in_pis_only is False
    assert result.verified is True


def _build_case1_mixed_design() -> Design:
    """target is a DFF.Q; D = sel & qc (1 PI, 1 DFF.Q) -- the test34 shape
    (D-cone support mixes PI and register state)."""
    design = Design(module_name="case1_mixed")
    _pi(design, "clk")
    _pi(design, "rn")
    _pi(design, "sel")
    _po(design, "target")
    _internal(design, "qc")
    _internal(design, "dnet")
    design.add_gate(Gate("g0", GateType.AND, {"O": NetBit("dnet"), "I0": NetBit("sel"), "I1": NetBit("qc")}))
    _dff(design, "dff_target", NetBit("dnet"), "target")
    _dff(design, "dff_c", Const.ZERO, "qc")
    design.build_indices()
    return design


def test_case1_dff_driven_target_mixed_pi_and_q_d_support() -> None:
    design = _build_case1_mixed_design()
    result = derive_boolean_function(design, "target")
    assert result.is_dff_q is True
    assert result.support_pi == ["sel"]
    assert result.support_dffq == ["qc"]
    assert result.caveat is not None
    assert result.verified is True


def _build_case1_pi_only_d_design() -> Design:
    """target is a DFF.Q; D = a & b, both primary inputs -- the F4 trap:
    the D pin's OWN support is entirely PI, but `target` itself is still
    sequential state, never a PI-only combinational function of it."""
    design = Design(module_name="case1_pi_only_d")
    _pi(design, "clk")
    _pi(design, "rn")
    _pi(design, "a")
    _pi(design, "b")
    _po(design, "target")
    _internal(design, "dnet")
    design.add_gate(Gate("g0", GateType.AND, {"O": NetBit("dnet"), "I0": NetBit("a"), "I1": NetBit("b")}))
    _dff(design, "dff_target", NetBit("dnet"), "target")
    design.build_indices()
    return design


def test_f4_dff_driven_target_never_expressible_in_pis_only_even_with_pi_only_d() -> None:
    """F4: `expressible_in_pis_only` is about TARGET, so it must be False
    here despite the D pin's support being 100% PI; `next_state_...` is the
    field that carries the D pin's own (True) PI-only-ness."""
    design = _build_case1_pi_only_d_design()
    result = derive_boolean_function(design, "target")
    assert result.is_dff_q is True
    assert result.support_pi == ["a", "b"]
    assert result.support_dffq == []
    assert result.expressible_in_pis_only is False
    assert result.next_state_expressible_in_pis_only is True
    assert result.caveat is None
    assert result.verified is True


def test_f4_non_dff_target_next_state_field_is_none() -> None:
    """`next_state_expressible_in_pis_only` only applies when `is_dff_q`;
    for a directly-combinational target there is no "next state" concept at
    all (root IS target), so it must stay None."""
    design = _build_pi_only_design()
    result = derive_boolean_function(design, "out")
    assert result.is_dff_q is False
    assert result.next_state_expressible_in_pis_only is None


def test_f6_empty_cone_identity_function() -> None:
    """F6: `cone` empty -- target is directly a true source with zero gates
    (here: a DFF.Q wired straight through to another DFF's D with no
    combinational logic in between). `_fanin_support`'s `if not cone:
    support.add(root)` branch (boolean_function.py:148-149, per the fix
    spec's own line numbers) is otherwise never exercised by any test."""
    design = Design(module_name="empty_cone")
    _pi(design, "clk")
    _pi(design, "rn")
    _internal(design, "qa")
    _po(design, "target")
    _dff(design, "dff_a", Const.ZERO, "qa")
    _dff(design, "dff_target", NetBit("qa"), "target")
    design.build_indices()
    result = derive_boolean_function(design, "target")
    assert result.is_dff_q is True
    assert result.dff_inst == "dff_target"
    assert result.root == "qa"
    assert result.cone_gate_count == 0
    assert result.support == ["qa"]
    assert result.support_dffq == ["qa"]
    assert result.onset_count == 1
    assert result.total_count == 2
    assert result.verified is True


def _build_dangling_d_pin_design() -> Design:
    """out = a & floating, where `floating` is an internal net declared but
    never driven by any gate (F6's `support_other` bucket -- a true source
    that is neither a declared PI nor a DFF.Q)."""
    design = Design(module_name="dangling_d")
    _pi(design, "a")
    _po(design, "out")
    _internal(design, "floating")
    design.add_gate(Gate("g0", GateType.AND, {"O": NetBit("out"), "I0": NetBit("a"), "I1": NetBit("floating")}))
    design.build_indices()
    return design


def test_f6_support_other_bucket_via_undriven_internal_net() -> None:
    design = _build_dangling_d_pin_design()
    result = derive_boolean_function(design, "out")
    assert result.support_pi == ["a"]
    assert result.support_dffq == []
    assert result.support_other == ["floating"]
    assert result.expressible_in_pis_only is False
    assert "other undriven, non-primary-input net(s)" in result.explanation
    assert result.verified is True


def test_f7_support_exactly_at_cap_boundary_still_computes_truth_table() -> None:
    """F7: the existing truncation test uses cap+1; pin the `>` (not `>=`)
    boundary at exactly SUPPORT_EXHAUSTIVE_CAP so a `>` -> `>=` regression
    would turn this red."""
    n = SUPPORT_EXHAUSTIVE_CAP
    design = Design(module_name="at_cap")
    names = [f"a{i}" for i in range(n)]
    for name in names:
        _pi(design, name)
    _po(design, "out")
    prev = NetBit(names[0])
    for i in range(1, n):
        out_name = f"x{i}" if i < n - 1 else "out"
        if out_name != "out":
            _internal(design, out_name)
        design.add_gate(Gate(f"g{i}", GateType.XOR, {"O": NetBit(out_name), "I0": prev, "I1": NetBit(names[i])}))
        prev = NetBit(out_name)
    design.build_indices()
    result = derive_boolean_function(design, "out")
    assert len(result.support) == SUPPORT_EXHAUSTIVE_CAP
    assert result.truncated is False
    assert result.onset_count is not None
    assert result.verified is True


def _build_all_ops_structural_design() -> Design:
    """A 9-gate chain (> INLINE_CONE_GATE_LIMIT, forcing STRUCTURAL
    rendering) exercising every one of the 8 gate types the structural
    `net = OP(args)` rendering (and its `_STRUCT_OP_EVAL` eval-conversion
    table) supports, at least once each -- none of test31/34/39's real
    structural cones happen to contain an AND or BUF gate, so without this
    fixture `_STRUCT_OP_EVAL["AND"]`/`["BUF"]` are never truly exercised in
    structural mode by anything (confirmed via a manual mutation of the
    `AND` entry, which survived the full suite before this fixture was
    added)."""
    design = Design(module_name="all_ops_struct")
    for name in ("p0", "p1"):
        _pi(design, name)
    _po(design, "out")
    for n in [f"g{i}" for i in range(1, 9)]:
        _internal(design, n)

    def gate(gt: GateType, o: str, *ins: str) -> None:
        pins: dict = {"O": NetBit(o)}
        for i, a in enumerate(ins):
            pins[f"I{i}"] = NetBit(a)
        design.add_gate(Gate(f"i_{o}", gt, pins))

    gate(GateType.AND, "g1", "p0", "p1")
    gate(GateType.OR, "g2", "g1", "p0")
    gate(GateType.XOR, "g3", "g2", "p1")
    gate(GateType.NAND, "g4", "g3", "p0")
    gate(GateType.NOR, "g5", "g4", "p1")
    gate(GateType.XNOR, "g6", "g5", "p0")
    gate(GateType.NOT, "g7", "g6")
    gate(GateType.BUF, "g8", "g7")
    gate(GateType.AND, "out", "g8", "p1")
    design.build_indices()
    return design


def test_all_gate_types_render_and_verify_in_structural_mode() -> None:
    design = _build_all_ops_structural_design()
    result = derive_boolean_function(design, "out")
    assert result.cone_gate_count == 9
    assert result.verified is True
    assert result.expression_lines == [
        "g1 = AND(p0, p1)",
        "g2 = OR(g1, p0)",
        "g3 = XOR(g2, p1)",
        "g4 = NAND(g3, p0)",
        "g5 = NOR(g4, p1)",
        "g6 = XNOR(g5, p0)",
        "g7 = NOT(g6)",
        "g8 = BUF(g7)",
        "out = AND(g8, p1)",
    ]


def _build_oversized_support_design() -> Design:
    """A parity chain over SUPPORT_EXHAUSTIVE_CAP+1 primary inputs -- support
    exceeds the exhaustive-simulation cap, so no truth table may be computed."""
    n = SUPPORT_EXHAUSTIVE_CAP + 1
    design = Design(module_name="oversized")
    names = [f"a{i}" for i in range(n)]
    for name in names:
        _pi(design, name)
    _po(design, "out")
    prev = NetBit(names[0])
    for i in range(1, n):
        out_name = f"x{i}" if i < n - 1 else "out"
        _internal(design, out_name) if out_name != "out" else None
        design.add_gate(Gate(f"g{i}", GateType.XOR, {"O": NetBit(out_name), "I0": prev, "I1": NetBit(names[i])}))
        prev = NetBit(out_name)
    design.build_indices()
    return design


def test_support_cap_exceeded_is_reported_not_silently_truncated() -> None:
    design = _build_oversized_support_design()
    result = derive_boolean_function(design, "out")
    assert len(result.support) == SUPPORT_EXHAUSTIVE_CAP + 1
    assert result.truncated is True
    assert result.onset_count is None
    assert result.total_count is None
    assert result.verified is None
    assert str(SUPPORT_EXHAUSTIVE_CAP) in result.explanation
    assert "not computed" in result.explanation.lower() or "omitted" in result.explanation.lower()


# ----------------------------------------------------------------------
# Second-round fix-spec regression tests (G1-G5)
# ----------------------------------------------------------------------


def test_g1_inline_mode_unconnected_pin_does_not_crash() -> None:
    """G1: an INLINE-mode cone (<=INLINE_CONE_GATE_LIMIT gates) with an
    unconnected pin previously raised `NameError` inside `eval()` -- the
    structural path already mapped the `unconnected` sentinel to `0`, but
    `_inline_display_to_eval` had no equivalent handling at all."""
    design = Design(module_name="inline_unconnected")
    _pi(design, "a")
    _po(design, "out")
    design.add_gate(Gate("g0", GateType.AND, {"O": NetBit("out"), "I0": NetBit("a")}))  # I1 left unconnected
    design.build_indices()
    result = derive_boolean_function(design, "out")
    assert result.expression_lines == ["out = (a & unconnected)"]
    # a & 0 is always 0, regardless of a.
    assert result.onset_count == 0
    assert result.total_count == 2
    assert result.verified is True


def test_g2_inline_mode_real_net_named_mask_does_not_collide_with_constant_sentinel() -> None:
    """G2: a real net literally named `MASK`, combined with a `1'b1`
    constant in the same INLINE-mode cone, previously collided at the text
    level (`_inline_display_to_eval` naively string-replaced `1'b1` with the
    literal text `"MASK"` before tokenizing, indistinguishable from the real
    net's own token) and produced a bogus self-verification failure on a
    perfectly legal design."""
    design = Design(module_name="mask_named_net")
    _pi(design, "MASK")
    _pi(design, "b")
    _po(design, "out")
    design.add_gate(Gate("g0", GateType.AND, {"O": NetBit("mid"), "I0": NetBit("MASK"), "I1": NetBit("b")}))
    design.signals["mid"] = Signal(name="mid", msb=None, lsb=None, direction=Direction.INTERNAL)
    design.add_gate(Gate("g1", GateType.OR, {"O": NetBit("out"), "I0": NetBit("mid"), "I1": Const.ONE}))
    design.build_indices()
    result = derive_boolean_function(design, "out")
    # out = (MASK & b) | 1 -- always 1, regardless of MASK/b.
    assert result.onset_count == result.total_count
    assert result.verified is True


def _build_structural_net_named_unconnected_design() -> Design:
    """A >INLINE_CONE_GATE_LIMIT-gate (structural-mode) cone where one real
    PI is literally named `unconnected`, mirroring
    `_build_all_ops_structural_design`'s 9-gate all-op-types chain but with
    `p0` renamed -- exercises G3 (a real net's own token must win over the
    `unconnected` sentinel meaning)."""
    design = Design(module_name="struct_unconnected_named_net")
    for name in ("unconnected", "p1"):
        _pi(design, name)
    _po(design, "out")
    for n in [f"g{i}" for i in range(1, 9)]:
        _internal(design, n)

    def gate(gt: GateType, o: str, *ins: str) -> None:
        pins: dict = {"O": NetBit(o)}
        for i, a in enumerate(ins):
            pins[f"I{i}"] = NetBit(a)
        design.add_gate(Gate(f"i_{o}", gt, pins))

    gate(GateType.AND, "g1", "unconnected", "p1")
    gate(GateType.OR, "g2", "g1", "unconnected")
    gate(GateType.XOR, "g3", "g2", "p1")
    gate(GateType.NAND, "g4", "g3", "unconnected")
    gate(GateType.NOR, "g5", "g4", "p1")
    gate(GateType.XNOR, "g6", "g5", "unconnected")
    gate(GateType.NOT, "g7", "g6")
    gate(GateType.BUF, "g8", "g7")
    gate(GateType.AND, "out", "g8", "p1")
    design.build_indices()
    return design


def test_g3_structural_mode_real_net_named_unconnected_does_not_collide_with_sentinel() -> None:
    design = _build_structural_net_named_unconnected_design()
    result = derive_boolean_function(design, "out")
    assert result.cone_gate_count == 9
    assert "unconnected" in result.support_pi
    # Previously `unconnected` (the real PI) was misread as a floating pin
    # (forced to 0) on the eval side only, so self-verification against the
    # direct netlist simulation (which uses the real PI's actual value)
    # would fail with AssertionError.
    assert result.verified is True


def test_g4_ascending_declared_signal_error_message_reports_positive_width() -> None:
    """G4: the "bit-select required" error message previously used
    `signal.width` (`msb - lsb + 1`), which goes negative for an ASCENDING
    declaration (`[0:7]`, msb < lsb); the actual bound-checking logic
    (`lo, hi = min/max`) was always correct, only this message's wording was
    wrong. `ir.Signal.width` itself is intentionally left untouched -- only
    the message's own local width computation (`hi - lo + 1`) is fixed."""
    design = Design(module_name="ascending_signal")
    design.signals["n47"] = Signal(name="n47", msb=0, lsb=7, direction=Direction.INTERNAL)
    with pytest.raises(ValueError, match=r"n47 is a 8-bit signal \(declared \[0:7\]\)"):
        derive_boolean_function(design, "n47")


def test_g5_caveat_wording_is_shared_with_property_check_not_independently_authored() -> None:
    """F5's own docstring claims boolean_function.py reuses
    `property_check.free_pi_caveat`'s wording verbatim rather than
    hand-authoring a parallel paraphrase -- but until this test, nothing
    actually asserted the exact text, only `is not None` (G5): a future
    edit that reintroduced an independently-worded caveat in one module
    would go undetected."""
    design = _build_mixed_pi_q_design()
    result = derive_boolean_function(design, "out")
    assert result.caveat == free_pi_caveat("function")
