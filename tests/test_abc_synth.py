"""Unit and integration tests for netlist_agent/abc_synth.py (ABC-backed
depth optimization).

Layout mirrors tests/test_abc_bridge.py (synthetic, hand-inspectable designs)
plus a few real-corpus spot checks (mirroring test_abc_bridge_real_files.py's
style) at the end.
"""

from __future__ import annotations

import copy
import os

import pytest

from netlist_agent import abc_synth as abc_synth_module
from netlist_agent import abc_bridge as abc_bridge_module
from netlist_agent.abc_bridge import ABCBridgeError, EquivResult, verify_equivalence
from netlist_agent.abc_synth import (
    BASIS_GATE_NAMES,
    AreaOptResult,
    DepthOptResult,
    _BlifGate,
    _SynthError,
    allowed_gate_types,
    optimize_cone_depth,
    optimize_cone_gate_count,
    optimize_depth,
    optimize_gate_count,
    parse_blif,
)
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, Port, Signal
from netlist_agent.parser import parse_verilog
from netlist_agent.transform import (
    collapse_double_inverters,
    remap_to_basis,
    remove_dangling_gates,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _nb(name: str, bit: int | None = None) -> NetBit:
    return NetBit(name, bit)


# ----------------------------------------------------------------------
# BLIF reader
# ----------------------------------------------------------------------


def test_parse_blif_basic_gates_and_continuation_lines() -> None:
    text = """
# a comment line, ignored
.model top
.inputs a b c \\
 d e
.outputs y z
.gate and2 a=a b=b O=new_n6
.gate not1  a=c O=y
.gate zero O=z
.end
"""
    blif = parse_blif(text)
    assert blif.inputs == ["a", "b", "c", "d", "e"]
    assert blif.outputs == ["y", "z"]
    assert blif.gates == [
        _BlifGate("and2", {"a": "a", "b": "b", "O": "new_n6"}),
        _BlifGate("not1", {"a": "c", "O": "y"}),
        _BlifGate("zero", {"O": "z"}),
    ]


def test_parse_blif_ignores_blank_lines_and_stops_at_end() -> None:
    text = """
.model top

.inputs a

.outputs y
.gate buf1 a=a O=y
.end
.gate and2 a=a b=a O=should_not_be_parsed
"""
    blif = parse_blif(text)
    assert len(blif.gates) == 1
    assert blif.gates[0].name == "buf1"


def test_parse_blif_names_line_raises() -> None:
    text = """
.model top
.inputs a
.outputs y
.names a y
1 1
.end
"""
    with pytest.raises(_SynthError, match=r"\.names"):
        parse_blif(text)


def test_parse_blif_malformed_gate_line_raises() -> None:
    with pytest.raises(_SynthError):
        parse_blif(".model top\n.gate\n.end\n")
    with pytest.raises(_SynthError):
        parse_blif(".model top\n.gate and2 a_no_equals_sign\n.end\n")


# ----------------------------------------------------------------------
# Basis / genlib helpers
# ----------------------------------------------------------------------


def test_basis_gate_names_cover_the_bases_this_module_promises() -> None:
    assert set(BASIS_GATE_NAMES) == {"and_not", "nand_not", "nor_not", "and_or_not"}


@pytest.mark.parametrize(
    ("basis", "expected"),
    [
        ("and_not", {GateType.AND, GateType.NOT, GateType.BUF}),
        ("nand_not", {GateType.NAND, GateType.NOT, GateType.BUF}),
        ("nor_not", {GateType.NOR, GateType.NOT, GateType.BUF}),
        ("and_or_not", {GateType.AND, GateType.OR, GateType.NOT, GateType.BUF}),
    ],
)
def test_allowed_gate_types_per_basis(basis: str, expected: set) -> None:
    assert allowed_gate_types(basis) == frozenset(expected)


def test_allowed_gate_types_unrestricted() -> None:
    assert allowed_gate_types(None) == frozenset(
        {GateType.AND, GateType.OR, GateType.NAND, GateType.NOR, GateType.XOR, GateType.XNOR, GateType.NOT, GateType.BUF}
    )


def test_allowed_gate_types_rejects_unknown_basis() -> None:
    with pytest.raises(ValueError):
        allowed_gate_types("xor_not")


# ----------------------------------------------------------------------
# Synthetic end-to-end: a deliberately-deep, rebalanceable design
# ----------------------------------------------------------------------


def _build_and_chain(width: int = 8) -> Design:
    """A linear AND chain over `width` inputs (naive depth = width - 1),
    which ABC's balancing should reduce significantly since AND is
    associative/commutative."""
    design = Design(module_name="top")
    names = [f"a{i}" for i in range(width)]
    for n in names:
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port(n, Direction.INPUT) for n in names] + [Port("y", Direction.OUTPUT)]

    prev = _nb(names[0])
    for i in range(1, width):
        last = i == width - 1
        out = _nb("y") if last else design.fresh_net("t_")
        design.add_gate(Gate(f"g{i}", GateType.AND, {"O": out, "I0": prev, "I1": _nb(names[i])}))
        prev = out
    design.build_indices()
    return design


def test_optimize_depth_reduces_depth_on_deep_synthetic_chain() -> None:
    design = _build_and_chain()
    original = copy.deepcopy(design)
    depth_before = NetlistGraph(design).max_design_depth()
    assert depth_before == 7

    result = optimize_depth(design)
    assert result.changed
    assert result.depth_before == 7
    assert result.depth_after < 7

    # design itself must be untouched (never mutated in place).
    assert design.gates == original.gates or len(design.gates) == len(original.gates)
    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail
    assert NetlistGraph(result.design).max_design_depth() == result.depth_after


def test_optimize_depth_honors_and_not_basis() -> None:
    design = _build_and_chain()
    original = copy.deepcopy(design)
    result = optimize_depth(design, basis="and_not")
    assert result.changed
    gate_types = {g.gate_type for g in result.design.gates}
    assert gate_types <= allowed_gate_types("and_not")
    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail


def test_optimize_depth_no_improvement_leaves_design_bit_identical() -> None:
    """A single 2-input gate has depth 1 -- nothing can reduce that further,
    so the design must come back completely unchanged (changed=False, same
    gate list, no ABC-added instances)."""
    design = Design(module_name="top")
    for n in ("a", "b"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port("a", Direction.INPUT), Port("b", Direction.INPUT), Port("y", Direction.OUTPUT)]
    design.add_gate(Gate("g0", GateType.AND, {"O": _nb("y"), "I0": _nb("a"), "I1": _nb("b")}))
    design.build_indices()
    before_gates = list(design.gates)

    result = optimize_depth(design)
    assert not result.changed
    assert result.depth_before == 1
    assert result.depth_after == 1
    assert result.design is design
    assert design.gates == before_gates


def test_optimize_depth_zero_depth_design_is_a_fast_noop() -> None:
    """Depth is measured in #combinational GATES (a BUF counts as one, per
    graph.py's own docstring), so the only genuinely zero-depth case is a
    DFF's Q wired straight to a PO with no gate at all in between."""
    design = Design(module_name="top")
    for n in ("a", "clk", "rn"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port(n, Direction.INPUT) for n in ("a", "clk", "rn")] + [Port("y", Direction.OUTPUT)]
    design.add_gate(
        Gate("dff0", GateType.DFF, {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("a"), "Q": _nb("y")})
    )
    design.build_indices()

    result = optimize_depth(design)
    assert not result.changed
    assert result.depth_before == 0
    assert result.depth_after == 0
    assert result.design is design


# ----------------------------------------------------------------------
# optimize_cone_depth
# ----------------------------------------------------------------------


def _build_two_cone_design(y_width: int = 16) -> Design:
    """Two independent chains feeding two POs: y (a long AND chain we will
    optimize the cone of -- long enough that even the structural overhead of
    a strict NAND/NOT basis can't hide a real balancing win) and z (a short
    chain of OR/XOR gates we leave alone -- used to assert untouched gates
    elsewhere really are untouched)."""
    design = Design(module_name="top")
    names = [f"a{i}" for i in range(y_width)]
    for n in names:
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.signals["z"] = Signal("z", None, None, Direction.OUTPUT)
    design.ports = [Port(n, Direction.INPUT) for n in names] + [Port("y", Direction.OUTPUT), Port("z", Direction.OUTPUT)]

    prev = _nb(names[0])
    for i in range(1, y_width):
        last = i == y_width - 1
        out = _nb("y") if last else design.fresh_net("t_")
        design.add_gate(Gate(f"g{i}", GateType.AND, {"O": out, "I0": prev, "I1": _nb(names[i])}))
        prev = out

    prevz = _nb(names[0])
    for i in range(1, 6):
        last = i == 5
        out = _nb("z") if last else design.fresh_net("u_")
        gt = GateType.OR if i % 2 else GateType.XOR
        design.add_gate(Gate(f"h{i}", gt, {"O": out, "I0": prevz, "I1": _nb(names[i])}))
        prevz = out
    design.build_indices()
    return design


def test_optimize_cone_depth_reduces_only_the_target_cone() -> None:
    design = _build_two_cone_design()
    original = copy.deepcopy(design)
    graph = NetlistGraph(design)
    y_depth_before = graph.depth_to_sink(_nb("y"))
    z_depth_before = graph.depth_to_sink(_nb("z"))
    assert y_depth_before == 15

    result = optimize_cone_depth(design, _nb("y"), basis="nand_not")
    assert result.changed
    assert result.depth_before == y_depth_before
    assert result.depth_after < y_depth_before

    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail

    new_graph = NetlistGraph(result.design)
    assert new_graph.depth_to_sink(_nb("z")) == z_depth_before

    # every gate reachable from z (the untouched cone) must be a gate that
    # was already present, unchanged, in the original design.
    z_cone_names = new_graph.backward_reachable_gates(_nb("z"))
    original_by_name = {g.inst_name: g for g in original.gates}
    for name in z_cone_names:
        new_gate = next(g for g in result.design.gates if g.inst_name == name)
        assert name in original_by_name
        assert original_by_name[name].gate_type == new_gate.gate_type

    # the NEW y-cone must honor the requested basis.
    y_cone_names = new_graph.backward_reachable_gates(_nb("y"))
    y_cone_types = {g.gate_type for g in result.design.gates if g.inst_name in y_cone_names}
    assert y_cone_types <= allowed_gate_types("nand_not")


def test_optimize_cone_depth_zero_depth_target_is_noop() -> None:
    design = _build_two_cone_design()
    # a1 is a plain primary input -- its own "cone" (as a sink) is depth 0.
    result = optimize_cone_depth(design, _nb("a1"), basis="and_not")
    assert not result.changed
    assert result.depth_before == 0
    assert result.design is design


# ----------------------------------------------------------------------
# DFF whole-design splice (D-pin rewiring) + the Q-bus-sharing corner
# ----------------------------------------------------------------------


def _build_dff_design() -> Design:
    """PIs feed an AND chain into a DFF's D pin; the DFF's Q then feeds a
    second AND chain out to a PO -- exercises both the D-tap rewiring and
    (separately, in the test below) the free_pi Q promotion."""
    design = Design(module_name="top")
    names = [f"a{i}" for i in range(8)]
    for n in names:
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    for n in ("clk", "rn"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.signals["q1"] = Signal("q1", None, None, Direction.INTERNAL)
    design.ports = [Port(n, Direction.INPUT) for n in names + ["clk", "rn"]] + [Port("y", Direction.OUTPUT)]

    prev = _nb(names[0])
    for i in range(1, 8):
        t = design.fresh_net("t_")
        design.add_gate(Gate(f"g{i}", GateType.AND, {"O": t, "I0": prev, "I1": _nb(names[i])}))
        prev = t
    design.add_gate(
        Gate(
            "dff0",
            GateType.DFF,
            {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": prev, "Q": _nb("q1")},
        )
    )
    prev2 = _nb("q1")
    for i in range(7):
        last = i == 6
        out = _nb("y") if last else design.fresh_net("u_")
        design.add_gate(Gate(f"h{i}", GateType.AND, {"O": out, "I0": prev2, "I1": _nb(names[i])}))
        prev2 = out
    design.build_indices()
    return design


def test_optimize_depth_rewires_dff_d_pin_correctly() -> None:
    design = _build_dff_design()
    original = copy.deepcopy(design)
    depth_before = NetlistGraph(design).max_design_depth()

    result = optimize_depth(design)
    assert result.changed
    assert result.depth_after < depth_before

    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail

    dff_gates = [g for g in result.design.gates if g.gate_type == GateType.DFF]
    assert len(dff_gates) == 1
    d_value = dff_gates[0].pins["D"]
    assert isinstance(d_value, NetBit)
    # the D pin must now be driven by some gate in the spliced design (not
    # dangling, not still pointing at the old pre-optimization net).
    assert d_value in result.design.net_driver
    # the DFF's own Q net-bit identity is untouched.
    assert dff_gates[0].pins["Q"] == _nb("q1")


def test_optimize_depth_handles_q_bus_sharing_corner() -> None:
    """test39-style corner: one bit of a bus is a DFF's Q output, a sibling
    bit of the SAME bus is combinationally driven. Whole-design optimization
    must leave the DFF's Q bit exactly as-is (still driven by the DFF); the
    Q net is a PO of the extracted view, re-driven by a BUF from the
    `__dff_Q__<inst>` PI, and must not be retapped
    (see abc_synth._splice_whole_design's dff_q_bits guard)."""
    design = Design(module_name="top")
    for n in ("a", "b", "clk", "rn"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["n11"] = Signal("n11", 1, 0, Direction.OUTPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port(n, Direction.INPUT) for n in ("a", "b", "clk", "rn")] + [
        Port("n11", Direction.OUTPUT),
        Port("y", Direction.OUTPUT),
    ]
    design.add_gate(Gate("g0", GateType.AND, {"O": _nb("n11", 1), "I0": _nb("a"), "I1": _nb("b")}))
    design.add_gate(
        Gate(
            "dff0",
            GateType.DFF,
            {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("a"), "Q": _nb("n11", 0)},
        )
    )
    prev = _nb("n11", 1)
    for i in range(6):
        last = i == 5
        out = _nb("y") if last else design.fresh_net("t_")
        design.add_gate(Gate(f"h{i}", GateType.AND, {"O": out, "I0": prev, "I1": _nb("n11", 0)}))
        prev = out
    design.build_indices()
    original = copy.deepcopy(design)
    depth_before = NetlistGraph(design).max_design_depth()

    result = optimize_depth(design)

    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail
    # n11[0] must still be the DFF's own Q output, untouched.
    driver0 = result.design.net_driver.get(_nb("n11", 0))
    assert driver0 is not None and driver0.gate_type == GateType.DFF
    if result.changed:
        assert result.depth_after < depth_before


# ----------------------------------------------------------------------
# Real-corpus spot checks
# ----------------------------------------------------------------------


def _load(case: str) -> Design:
    _p = os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase")
    if not os.path.isdir(_p):
        pytest.skip("Alpha_Testcase corpus not present -- the 2026 CAD Contest at ICCAD publishes them with Problem A; put the released testcases under Alpha_Testcase/testcase/testNN/ to enable this test")
    return parse_verilog(os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase", case, f"{case}.v"))


def test_real_file_test22_whole_design_depth_improves(capsys) -> None:
    original = _load("test22")
    design = _load("test22")
    depth_before = NetlistGraph(design).max_design_depth()

    result = optimize_depth(design)
    print(f"\n[spot check] test22: whole-design depth {depth_before} -> {result.depth_after} (changed={result.changed})")

    assert result.changed
    assert result.depth_after < depth_before
    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail


def test_real_file_test25_cone_of_n11_is_already_optimal() -> None:
    """n11[0] is directly a DFF's Q output in the real test25 corpus file
    (zero combinational gates upstream of it) -- optimize_cone_depth must
    recognize this as already-optimal without needing ABC at all."""
    design = _load("test25")
    result = optimize_cone_depth(design, _nb("n11", 0), basis="nand_not")
    assert not result.changed
    assert result.depth_before == 0
    assert result.depth_after == 0


def test_real_file_test28_and_not_whole_design_after_remap(capsys) -> None:
    """Reproduces test28's actual real prompt sequence up to the depth-
    optimization request (remap to AND/NOT, sweep dangling, collapse double
    inverters), then checks the ABC round trip genuinely improves depth
    while staying within the AND/NOT/BUF/DFF basis."""
    original = _load("test28")
    design = _load("test28")
    remap_to_basis(design, "and_not")
    remove_dangling_gates(design)
    collapse_double_inverters(design)
    depth_before = NetlistGraph(design).max_design_depth()

    result = optimize_depth(design, basis="and_not")
    print(f"\n[spot check] test28 (AND/NOT basis): depth {depth_before} -> {result.depth_after} (changed={result.changed})")

    assert result.changed
    assert result.depth_after < depth_before
    gate_types = {g.gate_type for g in result.design.gates}
    assert gate_types <= (allowed_gate_types("and_not") | {GateType.DFF})

    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail


# ----------------------------------------------------------------------
# Gate-count minimization: the acceptance contract
#
# These assert the *contract* rather than any ABC-version-specific number,
# because the whole point of the multi-candidate search is that which
# candidate wins depends on what ABC produces. test18 is the design where
# the two candidates genuinely disagree -- the smaller-gate-count one is
# also the deeper one -- so a depth cap between them forces the choice to
# flip. On a design where one candidate dominates both axes, the cap can
# never change the answer and the test would prove nothing.
# ----------------------------------------------------------------------


def test_gate_count_optimization_never_returns_a_design_over_the_depth_cap(capsys) -> None:
    """The depth cap is a hard requirement -- violating it forfeits the whole
    testcase under the contest's scoring, so a candidate that saves gates but
    busts the cap must lose to one that doesn't, or to no change at all."""
    free = optimize_gate_count(_load("test18"))
    assert free.changed, "expected a gate-count win on test18 with no cap"
    print(f"\n[spot check] test18 uncapped: {free.gates_before} -> {free.gates_after} gates, depth {free.depth_before} -> {free.depth_after}")

    # Calibrate the cap off whatever ABC just produced, so this holds no
    # matter which script wins on a future ABC build.
    for cap in (free.depth_after - 1, free.depth_after // 2):
        capped = optimize_gate_count(_load("test18"), max_depth=cap)
        print(f"[spot check] test18 cap={cap}: changed={capped.changed} gates={capped.gates_after} depth={capped.depth_after}")
        if capped.changed:
            assert capped.depth_after <= cap
            assert capped.gates_after < capped.gates_before
            assert verify_equivalence(_load("test18"), capped.design, timeout=300).equivalent
        else:
            assert capped.gates_after == capped.gates_before
            assert capped.depth_after == capped.depth_before


def test_gate_count_optimization_reports_no_change_when_it_cannot_shrink(capsys) -> None:
    """Under a restricted basis, decomposing a mixed-gate netlist into two-input
    primitives inflates the gate count, so no candidate can beat the original.
    The honest answer is "left unchanged" -- committing an inflated netlist
    would lose points on the very metric the request asked to minimize."""
    result = optimize_gate_count(_load("test18"), basis="and_not")
    print(f"\n[spot check] test18 and_not: changed={result.changed} {result.gates_before} -> {result.gates_after}")

    assert not result.changed
    assert result.gates_after == result.gates_before
    assert "left unchanged" in result.note


# ----------------------------------------------------------------------
# `failure` field: batch 6b (2026-09-30) -- a failed optimization must be
# distinguishable, in the result object, from a genuine "already optimal"
# outcome. Covers all four public entry points (whole-design and
# cone-restricted, depth and area).
# ----------------------------------------------------------------------

_DEPTH_ENTRY_POINTS = [
    ("optimize_depth", lambda design: optimize_depth(design)),
    ("optimize_cone_depth", lambda design: optimize_cone_depth(design, _nb("y"))),
]

_AREA_ENTRY_POINTS = [
    ("optimize_gate_count", lambda design: optimize_gate_count(design)),
    ("optimize_cone_gate_count", lambda design: optimize_cone_gate_count(design, _nb("y"))),
]


@pytest.mark.parametrize("name,call", _DEPTH_ENTRY_POINTS, ids=[n for n, _ in _DEPTH_ENTRY_POINTS])
def test_depth_opt_synth_failure_sets_failure_field(monkeypatch, name, call) -> None:
    def _boom(*args, **kwargs):
        raise ABCBridgeError("ABC exited with code 2: boom")

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _boom)
    design = _build_and_chain()
    result = call(design)
    assert not result.changed
    assert result.design is design
    assert result.failure is not None
    assert "boom" in result.failure


@pytest.mark.parametrize("name,call", _DEPTH_ENTRY_POINTS, ids=[n for n, _ in _DEPTH_ENTRY_POINTS])
def test_depth_opt_verify_equivalence_exception_is_caught(monkeypatch, name, call) -> None:
    """`verify_equivalence` raising ABCBridgeError must degrade to
    changed=False + failure=... rather than propagate -- the docstring's
    "never raises out of this function" promise, made real."""

    def _boom(*args, **kwargs):
        raise ABCBridgeError("cec timed out")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _boom)
    design = _build_and_chain()
    original = copy.deepcopy(design)
    result = call(design)  # must not raise
    assert not result.changed
    assert result.design is design
    assert design.gates == original.gates
    assert result.failure is not None
    assert "cec timed out" in result.failure


@pytest.mark.parametrize("name,call", _DEPTH_ENTRY_POINTS, ids=[n for n, _ in _DEPTH_ENTRY_POINTS])
def test_depth_opt_not_equivalent_sets_failure_field(monkeypatch, name, call) -> None:
    def _not_equiv(*args, **kwargs):
        return EquivResult(False, "counterexample: a=1 b=0")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _not_equiv)
    design = _build_and_chain()
    result = call(design)
    assert not result.changed
    assert result.design is design
    assert result.failure == (
        "equivalence check found the resynthesized design not equivalent to the original: "
        "counterexample: a=1 b=0"
    )
    assert result.note == (
        "Equivalence check failed after resynthesis (this indicates a bug, not a normal outcome) -- "
        "kept the original design untouched. Detail: counterexample: a=1 b=0"
    )


def test_depth_opt_already_optimal_leaves_failure_none() -> None:
    """The genuine "nothing better found" outcome (from an earlier test in
    this file) must NOT be mistaken for a failure."""
    design = Design(module_name="top")
    for n in ("a", "b"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port("a", Direction.INPUT), Port("b", Direction.INPUT), Port("y", Direction.OUTPUT)]
    design.add_gate(Gate("g0", GateType.AND, {"O": _nb("y"), "I0": _nb("a"), "I1": _nb("b")}))
    design.build_indices()

    result = optimize_depth(design)
    assert not result.changed
    assert result.failure is None


@pytest.mark.parametrize("name,call", _AREA_ENTRY_POINTS, ids=[n for n, _ in _AREA_ENTRY_POINTS])
def test_area_opt_all_candidates_fail_sets_failure_field(monkeypatch, name, call) -> None:
    def _boom(*args, **kwargs):
        raise ABCBridgeError("all candidates broken")

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _boom)
    design = _build_and_chain()
    result = call(design)
    assert not result.changed
    assert result.design is design
    assert result.failure is not None
    assert "all candidates broken" in result.failure


@pytest.mark.parametrize("name,call", _AREA_ENTRY_POINTS, ids=[n for n, _ in _AREA_ENTRY_POINTS])
def test_area_opt_one_candidate_completes_leaves_failure_none_even_without_improvement(monkeypatch, name, call) -> None:
    """First candidate script fails outright, second runs to completion (ABC
    succeeds + verifies equivalent) but doesn't beat the original gate count
    -- this counts as "ran to completion", not a failure, per the module's
    multi-candidate contract."""
    real_run = abc_synth_module._run_abc_synthesis
    calls = {"n": 0}

    def _first_fails_rest_real(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ABCBridgeError("first candidate broken")
        return real_run(view, basis, timeout, opt_script)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _first_fails_rest_real)
    design = _build_and_chain()
    result = call(design)
    assert calls["n"] >= 1
    # Either a real candidate beat the original (changed=True, failure=None
    # per contract) or none did but at least one completed (failure=None
    # anyway) -- either way `failure` must be None once any candidate ran to
    # completion.
    assert result.failure is None


@pytest.mark.parametrize("name,call", _AREA_ENTRY_POINTS, ids=[n for n, _ in _AREA_ENTRY_POINTS])
def test_area_opt_verify_equivalence_exception_all_candidates_sets_failure(monkeypatch, name, call) -> None:
    def _boom(*args, **kwargs):
        raise ABCBridgeError("cec broke")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _boom)
    design = _build_and_chain()
    result = call(design)
    assert not result.changed
    assert result.design is design
    assert result.failure is not None
    assert "cec broke" in result.failure


# ----------------------------------------------------------------------
# `incomplete` / `candidates` fields: batch 13 (2026-10-07) -- a candidate
# that failed while another completed must still be visible in the result,
# so the presentation layer cannot claim "nothing smaller exists".
# ----------------------------------------------------------------------


def _build_single_and() -> Design:
    """Two PIs into one AND to PO: already minimal, nothing can shrink it."""
    design = Design(module_name="top")
    for n in ("a", "b"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port("a", Direction.INPUT), Port("b", Direction.INPUT), Port("y", Direction.OUTPUT)]
    design.add_gate(Gate("g0", GateType.AND, {"O": _nb("y"), "I0": _nb("a"), "I1": _nb("b")}))
    design.build_indices()
    return design


def _first_synth_raises(monkeypatch, message: str = "synthetic") -> None:
    real_run = abc_synth_module._run_abc_synthesis
    calls = {"n": 0}

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ABCBridgeError(message)
        return real_run(view, basis, timeout, opt_script)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)


@pytest.mark.parametrize("name,call", _AREA_ENTRY_POINTS, ids=[n for n, _ in _AREA_ENTRY_POINTS])
def test_area_opt_partial_synth_failure_is_recorded_in_incomplete(monkeypatch, name, call) -> None:
    _first_synth_raises(monkeypatch)
    result = call(_build_single_and())
    assert result.changed is False
    assert result.failure is None
    assert result.incomplete == ("synthetic",)
    assert result.candidates == len(abc_synth_module._AREA_CANDIDATE_SCRIPTS) == 2


@pytest.mark.parametrize("name,call", _AREA_ENTRY_POINTS, ids=[n for n, _ in _AREA_ENTRY_POINTS])
def test_area_opt_all_candidates_fail_failure_is_first_incomplete(monkeypatch, name, call) -> None:
    calls = {"n": 0}

    def _boom(*args, **kwargs):
        calls["n"] += 1
        raise ABCBridgeError(f"synthetic{calls['n']}")

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _boom)
    result = call(_build_single_and())
    assert result.incomplete == ("synthetic1", "synthetic2")
    assert result.failure == "synthetic1"
    assert result.candidates == 2


@pytest.mark.parametrize("name,call", _AREA_ENTRY_POINTS, ids=[n for n, _ in _AREA_ENTRY_POINTS])
def test_area_opt_no_failure_leaves_incomplete_empty(name, call) -> None:
    result = call(_build_single_and())
    assert result.changed is False
    assert result.failure is None
    assert result.incomplete == ()
    assert result.candidates == 2


@pytest.mark.parametrize("name,call", _AREA_ENTRY_POINTS, ids=[n for n, _ in _AREA_ENTRY_POINTS])
def test_area_opt_partial_verify_exception_is_recorded(monkeypatch, name, call) -> None:
    real_verify = abc_synth_module.verify_equivalence
    calls = {"n": 0}

    def _fake(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ABCBridgeError("cec timed out")
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _fake)
    result = call(_build_single_and())
    assert result.failure is None
    assert result.incomplete == ("equivalence check failed: cec timed out",)


@pytest.mark.parametrize("name,call", _AREA_ENTRY_POINTS, ids=[n for n, _ in _AREA_ENTRY_POINTS])
def test_area_opt_partial_not_equivalent_is_recorded(monkeypatch, name, call) -> None:
    real_verify = abc_synth_module.verify_equivalence
    calls = {"n": 0}

    def _fake(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return EquivResult(False, "counterexample: a=1 b=0")
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _fake)
    result = call(_build_single_and())
    assert result.failure is None
    assert len(result.incomplete) == 1
    assert result.incomplete[0].startswith("equivalence check found the resynthesized design not equivalent")
    assert "counterexample: a=1 b=0" in result.incomplete[0]


def test_area_opt_incomplete_is_also_filled_on_success(monkeypatch) -> None:
    _first_synth_raises(monkeypatch)
    result = optimize_gate_count(_load("test18"))
    assert result.changed, "expected the second candidate alone to shrink test18"
    assert result.failure is None
    assert result.incomplete == ("synthetic",)
    assert result.candidates == 2


def _build_shrinkable_cone() -> Design:
    """y = BUF(NOT(NOT(AND(a, b)))): the cone of y shrinks to an AND plus the tap BUF (4 -> 2 gates)."""
    design = Design(module_name="top")
    for n in ("a", "b"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    for n in ("n1", "n2", "n3"):
        design.signals[n] = Signal(n, None, None, Direction.INTERNAL)
    design.ports = [Port("a", Direction.INPUT), Port("b", Direction.INPUT), Port("y", Direction.OUTPUT)]
    design.add_gate(Gate("g0", GateType.AND, {"O": _nb("n1"), "I0": _nb("a"), "I1": _nb("b")}))
    design.add_gate(Gate("g1", GateType.NOT, {"O": _nb("n2"), "I0": _nb("n1")}))
    design.add_gate(Gate("g2", GateType.NOT, {"O": _nb("n3"), "I0": _nb("n2")}))
    design.add_gate(Gate("g3", GateType.BUF, {"O": _nb("y"), "I0": _nb("n3")}))
    design.build_indices()
    return design


def test_area_opt_cone_incomplete_is_also_filled_on_success(monkeypatch) -> None:
    _first_synth_raises(monkeypatch)
    result = optimize_cone_gate_count(_build_shrinkable_cone(), _nb("y"))
    assert result.changed is True
    assert result.failure is None
    assert result.incomplete == ("synthetic",)
    assert result.candidates == 2


def test_area_opt_depth_zero_early_return_has_no_candidates() -> None:
    design = Design(module_name="top")
    design.signals["a"] = Signal("a", None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port("a", Direction.INPUT), Port("y", Direction.OUTPUT)]
    design.build_indices()
    result = optimize_gate_count(design)
    assert result.candidates == 0
    assert result.incomplete == ()


# ----------------------------------------------------------------------
# Batch 15 (F3): temp-dir paths from ABC must not reach _SynthError text
# ----------------------------------------------------------------------


def _patch_synth_abc(monkeypatch, returncode, stdout_fmt, stderr_fmt):
    import re
    import subprocess as subprocess_module
    import types

    def _fake_run(argv, **k):
        d = re.search(r'read_blif "(.*?)/in\.blif"', argv[2]).group(1)
        return types.SimpleNamespace(
            returncode=returncode, stdout=stdout_fmt.format(d=d), stderr=stderr_fmt.format(d=d)
        )

    monkeypatch.setattr(abc_synth_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(subprocess_module, "run", _fake_run)


def test_synth_error_nonzero_exit_stderr_has_no_tmpdir(monkeypatch) -> None:
    _patch_synth_abc(monkeypatch, 1, "", 'Cannot open input file "{d}/basis.genlib". ')
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    msg = str(exc.value)
    assert "code 1" in msg and "Cannot open input file" in msg
    assert "abc_synth_" not in msg


def test_synth_error_nonzero_exit_falls_back_to_scrubbed_stdout(monkeypatch) -> None:
    # Empty stderr: the message falls back to stdout, which carries the echo
    # line and the paths, so the fallback must be scrubbed too.
    out = '======== ABC command line "read_blif "{d}/in.blif"; strash"\nCannot open input file "{d}/in.blif". \n'
    _patch_synth_abc(monkeypatch, 1, out, "")
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    msg = str(exc.value)
    assert "code 1" in msg and 'Cannot open input file "in.blif"' in msg
    assert "abc_synth_" not in msg and "ABC command line" not in msg


def test_synth_error_no_blif_output_stdout_has_no_tmpdir(monkeypatch) -> None:
    out = '======== ABC command line "read_blif "{d}/in.blif"; strash"\nCannot open input file "{d}/in.blif". \n'
    _patch_synth_abc(monkeypatch, 0, out, "")
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    msg = str(exc.value)
    assert "no BLIF output" in msg and "Cannot open input file" in msg
    assert "abc_synth_" not in msg and "ABC command line" not in msg


# ----------------------------------------------------------------------
# Batch 16a: OSErrors in `_run_abc_synthesis` become `_SynthError` (so the
# optimize paths report `failure`), pathless; no-BLIF reads stderr; no
# dangling colon; verify catch separates "inconclusive" from "failed".
# ----------------------------------------------------------------------

_INJECTED_PATH = "/var/folders/zz/leak_dir/abc_synth_xyz/in.blif"


def _assert_synth_oserror(exc_info) -> None:
    assert type(exc_info.value) is _SynthError
    msg = str(exc_info.value)
    assert "No space left on device" in msg
    assert "/var/folders" not in msg and "leak_dir" not in msg and "abc_synth_xyz" not in msg
    assert isinstance(exc_info.value.__cause__, OSError)


def _oserror() -> OSError:
    return OSError(28, "No space left on device", _INJECTED_PATH)


def _inject_synth_oserror(monkeypatch, point: str) -> None:
    """Make exactly one OS-touching step of `_run_abc_synthesis` raise."""
    import builtins
    import subprocess as subprocess_module
    import tempfile as tempfile_module

    def _boom(*a, **k):
        raise _oserror()

    if point == "tempdir":
        monkeypatch.setattr(tempfile_module, "TemporaryDirectory", _boom)
    elif point == "write_blif":
        monkeypatch.setattr(abc_synth_module, "write_blif", _boom)
    elif point == "genlib":
        real_open = builtins.open

        def _open(path, *a, **k):
            if str(path).endswith(".genlib"):
                raise _oserror()
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", _open)
    elif point == "exec":
        monkeypatch.setattr(abc_synth_module, "_resolve_abc", lambda: "/fake/abc")
        monkeypatch.setattr(subprocess_module, "run", _boom)
    elif point == "readback":
        import types

        real_open = builtins.open

        def _fake_run(argv, **k):
            import re

            d = re.search(r'read_blif "(.*?)/in\.blif"', argv[2]).group(1)
            with real_open(os.path.join(d, "out.blif"), "w") as f:
                f.write(".model x\n")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        def _open(path, *a, **k):
            if str(path).endswith("out.blif") and not (a and "w" in a[0]):
                raise _oserror()
            return real_open(path, *a, **k)

        monkeypatch.setattr(abc_synth_module, "_resolve_abc", lambda: "/fake/abc")
        monkeypatch.setattr(subprocess_module, "run", _fake_run)
        monkeypatch.setattr(builtins, "open", _open)
    else:
        raise AssertionError(point)


_SYNTH_OS_POINTS = ["tempdir", "write_blif", "genlib", "exec", "readback"]


@pytest.mark.parametrize("point", _SYNTH_OS_POINTS)
def test_run_abc_synthesis_oserror_is_synth_error(monkeypatch, point) -> None:
    design = _build_and_chain()
    _inject_synth_oserror(monkeypatch, point)
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(design, None, 5.0)
    _assert_synth_oserror(exc)


@pytest.mark.parametrize("point", _SYNTH_OS_POINTS)
def test_optimize_depth_oserror_becomes_failure_without_path(monkeypatch, point) -> None:
    design = _build_and_chain()
    _inject_synth_oserror(monkeypatch, point)
    result = optimize_depth(design)
    assert result.failure is not None
    assert "No space left on device" in result.failure
    assert "/var/folders" not in result.failure and "leak_dir" not in result.failure
    assert "/var/folders" not in result.note
    assert result.design is design and not result.changed


def test_synth_no_blif_output_prefers_stderr(monkeypatch) -> None:
    out = '======== ABC command line "read_blif "{d}/in.blif"; strash"\n'
    _patch_synth_abc(monkeypatch, 0, out, 'Cannot open genlib file "{d}/basis.genlib".\n')
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    msg = str(exc.value)
    assert "no BLIF output" in msg and 'Cannot open genlib file "basis.genlib"' in msg
    assert "abc_synth_" not in msg and "ABC command line" not in msg


def test_synth_no_blif_output_both_empty_has_no_dangling_colon(monkeypatch) -> None:
    _patch_synth_abc(monkeypatch, 0, "", "  \n")
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    assert str(exc.value) == "ABC synthesis produced no BLIF output"


def test_synth_nonzero_exit_both_empty_has_no_dangling_colon(monkeypatch) -> None:
    _patch_synth_abc(monkeypatch, -11, "", "")
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    assert str(exc.value) == "ABC synthesis process exited with code -11"


@pytest.mark.parametrize("rc", [0, 1])
def test_synth_detail_is_truncated_to_last_500_chars(monkeypatch, rc) -> None:
    _patch_synth_abc(monkeypatch, rc, "", "HEAD" + "x" * 600 + "TAIL")
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    msg = str(exc.value)
    assert msg.endswith("TAIL") and "HEAD" not in msg
    assert len(msg) < 600


_ALL_ENTRY_POINTS = _DEPTH_ENTRY_POINTS + _AREA_ENTRY_POINTS


def _failure_text(result) -> str:
    return result.failure if result.failure is not None else result.incomplete[0]


@pytest.mark.parametrize("name,call", _ALL_ENTRY_POINTS, ids=[n for n, _ in _ALL_ENTRY_POINTS])
@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: abc_bridge_module.ABCTimeoutError("ABC invocation timed out after 5.0s running 'cec'"),
        lambda: abc_bridge_module.CecUndecidedError("Networks are undecided."),
    ],
    ids=["timeout", "undecided"],
)
def test_verify_inconclusive_has_inconclusive_prefix(monkeypatch, name, call, exc_factory) -> None:
    def _raise(*args, **kwargs):
        raise exc_factory()

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _raise)
    result = call(_build_and_chain())
    text = _failure_text(result)
    assert text.startswith("equivalence check was inconclusive: ")
    assert "equivalence check failed" not in text
    if name in ("optimize_depth", "optimize_cone_depth"):
        assert "equivalence check was inconclusive" in result.note
        assert "equivalence check failed" not in result.note
    assert not result.changed


@pytest.mark.parametrize("name,call", _ALL_ENTRY_POINTS, ids=[n for n, _ in _ALL_ENTRY_POINTS])
def test_verify_generic_bridge_error_keeps_failed_prefix(monkeypatch, name, call) -> None:
    def _raise(*args, **kwargs):
        raise ABCBridgeError("cec crashed")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _raise)
    result = call(_build_and_chain())
    text = _failure_text(result)
    assert text == "equivalence check failed: cec crashed"
    assert "inconclusive" not in text


# ----------------------------------------------------------------------
# Batch 16 round-1 fixes: F2 (temp-dir cleanup errors), F9 (whitespace-only
# stderr), gaps 1-3.
# ----------------------------------------------------------------------


@pytest.fixture
def _rmdir_fails_for_abc_dirs(monkeypatch):
    """Make `os.rmdir` fail for this module's temp dirs, as a cleanup error
    in `TemporaryDirectory.__exit__` would; remove the leftovers afterwards."""
    import shutil

    real_rmdir = os.rmdir
    left = []

    def _rmdir(path, *a, **k):
        if os.path.basename(str(path)).startswith(("abc_bridge_", "abc_synth_")):
            left.append(str(path))
            raise OSError(16, "Device or resource busy", str(path))
        return real_rmdir(path, *a, **k)

    monkeypatch.setattr(os, "rmdir", _rmdir)
    yield left
    monkeypatch.undo()
    for p in left:
        shutil.rmtree(p, ignore_errors=True)


def test_synth_cleanup_error_does_not_mask_synth_error(monkeypatch, _rmdir_fails_for_abc_dirs) -> None:
    _patch_synth_abc(monkeypatch, 1, "", "boom")
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    assert "boom" in str(exc.value)
    assert _rmdir_fails_for_abc_dirs, "the cleanup failure was never exercised"


def test_synth_cleanup_error_does_not_break_success(_rmdir_fails_for_abc_dirs) -> None:
    blif = abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 60.0)
    assert blif is not None
    assert _rmdir_fails_for_abc_dirs, "the cleanup failure was never exercised"


def test_synth_nonzero_exit_whitespace_stderr_falls_back_to_stdout(monkeypatch) -> None:
    _patch_synth_abc(monkeypatch, 1, "STDOUT-DIAGNOSTIC\n", "  \n")
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    assert "code 1: STDOUT-DIAGNOSTIC" in str(exc.value)


def test_synth_nonzero_exit_prefers_nonblank_stderr(monkeypatch) -> None:
    _patch_synth_abc(monkeypatch, 1, "STDOUT-WORDS\n", "STDERR-WORDS\n")
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    msg = str(exc.value)
    assert "STDERR-WORDS" in msg and "STDOUT-WORDS" not in msg


def test_synth_no_blif_output_nonecho_stdout_loses_to_stderr(monkeypatch) -> None:
    _patch_synth_abc(monkeypatch, 0, "STDOUT-WORDS\n", "STDERR-WORDS\n")
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    msg = str(exc.value)
    assert "STDERR-WORDS" in msg and "STDOUT-WORDS" not in msg


@pytest.mark.parametrize("rc", [0, 1])
def test_synth_detail_scrubs_before_truncating(monkeypatch, rc) -> None:
    # stderr is the temp path, "/in.blif", then 487 chars: 495 + len(tmpdir)
    # in all, so a 500-char tail cut BEFORE the scrub starts inside the temp
    # dir (for any tmpdir longer than 5 chars) and leaves half a path behind.
    # rc=0 is the no-BLIF branch, rc=1 the nonzero-exit branch.
    _patch_synth_abc(monkeypatch, rc, "", "{d}/in.blif" + "y" * 487)
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    msg = str(exc.value)
    assert "in.blif" in msg and "/" not in msg


def test_synth_exec_oserror_cause_is_the_injected_error(monkeypatch) -> None:
    import subprocess as subprocess_module

    injected = OSError(8, "Exec format error", _INJECTED_PATH)

    def _boom(*a, **k):
        raise injected

    monkeypatch.setattr(abc_synth_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(subprocess_module, "run", _boom)
    with pytest.raises(_SynthError) as exc:
        abc_synth_module._run_abc_synthesis(_build_and_chain(), None, 5.0)
    assert exc.value.__cause__ is injected


def test_optimize_depth_locator_failure_is_one_clean_line(monkeypatch) -> None:
    import subprocess as subprocess_module

    root = abc_bridge_module.REPO_ROOT
    stderr = (
        "error: could not find an ABC binary for platform 'darwin-arm64'.\n\n"
        f"Checked, in order:\n  {root}/vendor/darwin-arm64/abc\n  abc on $PATH\n"
    )

    def _run(args, *a, **k):
        assert list(args) == ["bash", abc_bridge_module.FIND_ABC_SCRIPT]
        return subprocess_module.CompletedProcess(args, 1, "", stderr)

    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    monkeypatch.setattr(subprocess_module, "run", _run)
    design = _build_and_chain()
    result = optimize_depth(design)
    assert result.failure is not None
    assert "could not locate the ABC binary" in result.failure
    assert root not in result.failure and "\n" not in result.failure


# ----------------------------------------------------------------------
# Batch 30 -> 34b-1: the single quick-script retry was replaced by a candidate
# list (`_DEPTH_SMALL_SCRIPTS`). A candidate that times out / errors is recorded
# in `incomplete` and the others still run; "already optimal" is never claimed
# while any candidate did not finish; the winner's `cec` result decides.
# (These tests were the batch-30 quick-fallback tests; each was rewritten for
# the new behavior, see the 34b-1 report. The old `quick_fallback` field was
# removed in 34b-2.)
# ----------------------------------------------------------------------

_PRIMARY_TIMEOUT = "ABC synthesis timed out after 120s"
_SMALL = abc_synth_module._DEPTH_SMALL_SCRIPTS
_QUICK_CASES = [
    ("whole", _build_and_chain, lambda d: optimize_depth(d)),
    ("cone", _build_two_cone_design, lambda d: optimize_cone_depth(d, _nb("y"))),
]
_QUICK_IDS = [c[0] for c in _QUICK_CASES]


def _times_out_then(monkeypatch, rest):
    """The first candidate (`_OPT_SCRIPT`) raises `_SynthTimeout`; every later one runs
    `rest(real_run, view, basis, timeout, opt_script)`. Returns the list of `opt_script`s."""
    real_run = abc_synth_module._run_abc_synthesis
    scripts: list[str] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        scripts.append(opt_script)
        if opt_script == abc_synth_module._OPT_SCRIPT:
            raise abc_synth_module._SynthTimeout(_PRIMARY_TIMEOUT)
        return rest(real_run, view, basis, timeout, opt_script)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    return scripts


def _real_rest(real_run, view, basis, timeout, opt_script):
    return real_run(view, basis, timeout, opt_script)


def test_synth_timeout_is_a_synth_error() -> None:
    assert issubclass(abc_synth_module._SynthTimeout, _SynthError)


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_timed_out_first_candidate_improvement_is_committed_with_honest_note(monkeypatch, name, build, call) -> None:
    """was: quick fallback improvement committed. Now the other candidates win and the note says one did not finish."""
    scripts = _times_out_then(monkeypatch, _real_rest)
    design = build()
    result = call(design)
    assert scripts == list(_SMALL)
    assert result.changed is True
    assert result.design is not design
    assert result.failure is None
    assert result.incomplete == (_PRIMARY_TIMEOUT,)
    assert result.candidates == len(_SMALL)
    assert result.depth_after < result.depth_before
    assert result.note.startswith("Reduced ")
    assert result.note.endswith(
        f" 1 of {len(_SMALL)} candidate scripts did not finish ({_PRIMARY_TIMEOUT}), so a lower depth may exist."
    )


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_no_improvement_with_a_timed_out_candidate_is_not_ruled_out_not_a_failure(monkeypatch, name, build, call) -> None:
    """was: quick fallback without improvement is a failure. Now (D12) the other candidates
    completed, so it is no failure, but no optimality claim either."""
    scripts = _times_out_then(monkeypatch, _real_rest)
    design = _build_single_and()
    result = call(design) if name == "whole" else optimize_cone_depth(design, _nb("y"))
    assert len(scripts) == len(_SMALL)
    assert result.changed is False
    assert result.design is design
    assert result.failure is None
    assert "already optimal" not in result.note
    assert "a lower depth is not ruled out" in result.note
    assert f"1 of {len(_SMALL)} could not be completed ({_PRIMARY_TIMEOUT})" in result.note
    assert result.incomplete == (_PRIMARY_TIMEOUT,)


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
@pytest.mark.parametrize(
    "later_exc",
    [abc_synth_module._SynthError("later broke"), abc_synth_module._SynthTimeout("later also timed out")],
    ids=["error", "timeout"],
)
def test_all_candidates_failing_keeps_every_reason(monkeypatch, name, build, call, later_exc) -> None:
    """was: both reasons kept when the quick script also failed."""

    def _later_fails(real_run, view, basis, timeout, opt_script):
        raise later_exc

    _times_out_then(monkeypatch, _later_fails)
    design = build()
    result = call(design)
    assert result.changed is False
    assert result.design is design
    assert result.failure == f"{_PRIMARY_TIMEOUT}; and {len(_SMALL) - 1} more candidate(s) did not complete"
    assert result.incomplete == (_PRIMARY_TIMEOUT,) + (str(later_exc),) * (len(_SMALL) - 1)
    assert len(result.incomplete) <= result.candidates


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_a_non_timeout_synth_error_does_not_stop_the_other_candidates(monkeypatch, name, build, call) -> None:
    """was: a non-timeout error is not retried. Now every candidate is independent: all are tried."""
    scripts: list[str] = []

    def _crash(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        scripts.append(opt_script)
        raise _SynthError("ABC synthesis process exited with code 1")

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _crash)
    design = build()
    result = call(design)
    assert scripts == list(_SMALL)
    assert result.changed is False
    assert result.design is design
    assert result.failure == f"ABC synthesis process exited with code 1; and {len(_SMALL) - 1} more candidate(s) did not complete"


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_all_candidates_complete_keeps_the_old_sentence(monkeypatch, name, build, call) -> None:
    """was: full script success runs once. Now all candidates run; nothing incomplete => the plain sentence."""
    real_run = abc_synth_module._run_abc_synthesis
    scripts: list[str] = []

    def _spy(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        scripts.append(opt_script)
        return real_run(view, basis, timeout, opt_script)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _spy)
    result = call(build())
    assert scripts == list(_SMALL)
    assert result.changed is True
    assert result.incomplete == ()
    if name == "whole":
        assert result.note == f"Reduced maximum logic depth from {result.depth_before} to {result.depth_after}."
    else:
        assert result.note == (
            f"Reduced the depth of the cone of y from {result.depth_before} to {result.depth_after}."
        )


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_winner_must_still_verify_equivalent(monkeypatch, name, build, call) -> None:
    """was: the quick result must still verify equivalent (unchanged assertions)."""
    _times_out_then(monkeypatch, _real_rest)
    monkeypatch.setattr(
        abc_synth_module, "verify_equivalence", lambda *a, **k: EquivResult(False, "counterexample: a=1")
    )
    design = build()
    result = call(design)
    assert result.changed is False
    assert result.design is design
    assert result.failure is not None and "not equivalent" in result.failure


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_verify_inconclusive_for_every_winner_reports_the_cec_reason(monkeypatch, name, build, call) -> None:
    """was: inconclusive keeps the primary reason in `failure`. Now (D12) `failure` is the first cec
    reason and the timed-out candidate's reason lives in `incomplete`."""
    _times_out_then(monkeypatch, _real_rest)

    def _inconclusive(*a, **k):
        raise abc_bridge_module.ABCInconclusiveError("cec gave up")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _inconclusive)
    design = build()
    result = call(design)
    assert result.changed is False
    assert result.design is design
    assert "already optimal" not in result.note
    assert result.failure == "equivalence check was inconclusive: cec gave up"
    assert result.incomplete[0] == _PRIMARY_TIMEOUT
    assert "equivalence check was inconclusive: cec gave up" in result.incomplete[1:]
    assert len(result.incomplete) <= result.candidates


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_not_equivalent_stops_with_the_old_bug_note(monkeypatch, name, build, call) -> None:
    """was: a mismatch carries the primary timeout. Now the note has no quick-fallback context; the
    earlier reasons are in `incomplete`."""
    _times_out_then(monkeypatch, _real_rest)
    monkeypatch.setattr(
        abc_synth_module, "verify_equivalence", lambda *a, **k: EquivResult(False, "counterexample: a=1")
    )
    design = build()
    result = call(design)
    assert result.changed is False
    assert result.design is design
    assert "already optimal" not in result.note
    assert result.note.startswith("Equivalence check failed after resynthesis (this indicates a bug")
    assert result.note.endswith("Detail: counterexample: a=1")
    assert result.failure is not None
    assert "not equivalent" in result.failure and "counterexample: a=1" in result.failure
    assert result.incomplete[0] == _PRIMARY_TIMEOUT
    assert result.incomplete == (_PRIMARY_TIMEOUT, result.failure)  # non-empty prefix: the failure is appended last


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_splice_rejection_is_recorded_per_candidate(monkeypatch, name, build, call) -> None:
    """was: F6, a bogus gate from the quick script is a rejected result. The splice runs inside each
    candidate, so a bogus BLIF is that candidate's failure; the first (timed-out) reason leads."""

    def _bogus(real_run, view, basis, timeout, opt_script):
        real = real_run(view, basis, timeout, opt_script)
        out = real.outputs[0]
        return abc_synth_module._BlifNetlist(
            real.inputs, real.outputs, [abc_synth_module._BlifGate("bogus_gate9", {"O": out})]
        )

    _times_out_then(monkeypatch, _bogus)
    design = build()
    result = call(design)
    assert result.changed is False
    assert result.design is design
    assert result.failure == f"{_PRIMARY_TIMEOUT}; and {len(_SMALL) - 1} more candidate(s) did not complete"
    assert len(result.incomplete) == len(_SMALL)
    assert all("bogus_gate9" in r for r in result.incomplete[1:])


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_every_candidate_gets_the_same_basis_and_timeout(monkeypatch, name, build, call) -> None:
    """was: F5, both ABC calls get the caller's basis and synthesis timeout (no request scope here)."""
    real_run = abc_synth_module._run_abc_synthesis
    calls: list[tuple] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        calls.append((basis, timeout, opt_script))
        if opt_script == abc_synth_module._OPT_SCRIPT:
            raise abc_synth_module._SynthTimeout(_PRIMARY_TIMEOUT)
        return real_run(view, basis, timeout, opt_script)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    design = build()
    if name == "whole":
        optimize_depth(design, basis="and_or_not", timeout=77.0)
    else:
        optimize_cone_depth(design, _nb("y"), basis="and_or_not", timeout=77.0)
    assert calls == [("and_or_not", 77.0, s) for s in _SMALL]


@pytest.mark.parametrize("name,build,call", _QUICK_CASES, ids=_QUICK_IDS)
def test_real_subprocess_some_candidates_time_out_others_win(monkeypatch, tmp_path, name, build, call) -> None:
    """was: spec 7, the full script times out in a real subprocess and the quick one wins. Now a
    wrapper around the real ABC sleeps whenever the command has `if -g` (the two SOP candidates);
    `exec` makes the sleep replace the shell, so the timeout kill hits the sleep itself (no orphan
    holding the stdout pipe past this test). The other two candidates run for real."""
    real_abc = abc_synth_module._resolve_abc()
    wrapper = tmp_path / "abc_wrapper.sh"
    wrapper.write_text(
        "#!/bin/sh\n"
        'case "$2" in\n'
        '  *"if -g"*) exec sleep 60 ;;\n'
        "esac\n"
        f'exec "{real_abc}" "$@"\n'
    )
    wrapper.chmod(0o755)
    monkeypatch.setattr(abc_synth_module, "_resolve_abc", lambda: str(wrapper))
    design = build()
    if name == "whole":
        result = optimize_depth(design, timeout=2.0)
    else:
        result = optimize_cone_depth(design, _nb("y"), timeout=2.0)
    assert result.failure is None, result.failure
    assert result.changed is True
    assert result.incomplete == ("ABC synthesis timed out after 2.0s",) * 2
    assert f"2 of {len(_SMALL)} candidate scripts did not finish" in result.note
    assert result.depth_after < result.depth_before
    # Re-check with the same `verify_equivalence`, here through the real (unwrapped) cec.
    assert verify_equivalence(design, result.design).equivalent


# ----------------------------------------------------------------------
# Batch 34a: synthesized gates drive POs / the cone target directly (no tap BUF)
# ----------------------------------------------------------------------


def _tap_gates(design: Design) -> list[Gate]:
    return [g for g in design.gates if g.inst_name.startswith("t_depth_opt_tap_")]


def _force_retap(monkeypatch) -> None:
    """Disable direct wiring: every PO/target goes through `_retap_output`."""
    monkeypatch.setattr(abc_synth_module, "_can_wire_directly", lambda *a, **k: False)


def test_whole_design_has_no_tap_gate_and_depth_is_one_less_than_tapped(monkeypatch) -> None:
    design = _build_and_chain()
    original = copy.deepcopy(design)
    direct = optimize_depth(design)
    assert direct.changed
    assert _tap_gates(direct.design) == []
    eq = verify_equivalence(original, direct.design)
    assert eq.equivalent, eq.detail

    _force_retap(monkeypatch)
    tapped = optimize_depth(copy.deepcopy(original))
    assert len(_tap_gates(tapped.design)) == 1
    assert direct.depth_after == tapped.depth_after - 1
    assert NetlistGraph(direct.design).max_design_depth() == direct.depth_after


def test_q_bus_corner_has_no_tap_and_stays_equivalent() -> None:
    design = Design(module_name="top")
    for n in ("a", "b", "clk", "rn"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["n11"] = Signal("n11", 1, 0, Direction.OUTPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port(n, Direction.INPUT) for n in ("a", "b", "clk", "rn")] + [
        Port("n11", Direction.OUTPUT),
        Port("y", Direction.OUTPUT),
    ]
    design.add_gate(Gate("g0", GateType.AND, {"O": _nb("n11", 1), "I0": _nb("a"), "I1": _nb("b")}))
    design.add_gate(
        Gate("dff0", GateType.DFF, {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("a"), "Q": _nb("n11", 0)})
    )
    prev = _nb("n11", 1)
    for i in range(6):
        last = i == 5
        out = _nb("y") if last else design.fresh_net("t_")
        design.add_gate(Gate(f"h{i}", GateType.AND, {"O": out, "I0": prev, "I1": _nb("n11", 0)}))
        prev = out
    design.build_indices()
    original = copy.deepcopy(design)

    result = optimize_depth(design)

    assert result.changed, result.failure
    assert result.failure is None
    assert _tap_gates(result.design) == []
    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail
    driver0 = result.design.net_driver.get(_nb("n11", 0))
    assert driver0 is not None and driver0.gate_type == GateType.DFF
    # the PO bit read by an internal gate is still driven (by a non-DFF gate)
    driver1 = result.design.net_driver.get(_nb("n11", 1))
    assert driver1 is not None and driver1.gate_type != GateType.DFF


def test_pi_passthrough_q_po_and_floating_po_keep_their_behavior() -> None:
    design = Design(module_name="top")
    for n in ("a", "b", "c", "clk", "rn"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    for n in ("pt", "q", "fl", "y"):
        design.signals[n] = Signal(n, None, None, Direction.OUTPUT)
    design.ports = [Port(n, Direction.INPUT) for n in ("a", "b", "c", "clk", "rn")] + [
        Port(n, Direction.OUTPUT) for n in ("pt", "q", "fl", "y")
    ]
    design.add_gate(Gate("bp", GateType.BUF, {"O": _nb("pt"), "I0": _nb("a")}))
    design.add_gate(
        Gate("dff0", GateType.DFF, {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("b"), "Q": _nb("q")})
    )
    prev = _nb("a")
    for i, n in enumerate(("b", "c", "a", "b", "c")):
        last = i == 4
        out = _nb("y") if last else design.fresh_net("t_")
        design.add_gate(Gate(f"h{i}", GateType.AND, {"O": out, "I0": prev, "I1": _nb(n)}))
        prev = out
    design.build_indices()
    original = copy.deepcopy(design)

    result = optimize_depth(design)
    # Without this, a rejected candidate (the original design handed back)
    # passes every assertion below.
    assert result.changed, result.failure

    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail
    out = result.design
    # Q stays driven by its DFF; PI passthrough PO still has a driver and
    # keeps its name; the floating PO stays floating.
    assert out.net_driver[_nb("q")].gate_type == GateType.DFF
    assert _nb("pt") in out.net_driver
    assert "pt" in out.signals and out.signals["pt"].direction == Direction.OUTPUT
    assert _nb("fl") not in out.net_driver


def _build_shared_target_design() -> Design:
    """`m` is an internal net at the end of a deep AND chain, read both by the
    chain's cone and (outside it) by `y = NOT m`."""
    design = Design(module_name="top")
    names = [f"a{i}" for i in range(8)]
    for n in names:
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.signals["m"] = Signal("m", None, None, Direction.INTERNAL)
    design.ports = [Port(n, Direction.INPUT) for n in names] + [Port("y", Direction.OUTPUT)]
    prev = _nb(names[0])
    for i in range(1, 8):
        out = _nb("m") if i == 7 else design.fresh_net("t_")
        design.add_gate(Gate(f"g{i}", GateType.AND, {"O": out, "I0": prev, "I1": _nb(names[i])}))
        prev = out
    design.add_gate(Gate("gn", GateType.NOT, {"O": _nb("y"), "I0": _nb("m")}))
    design.build_indices()
    return design


def test_cone_target_is_driven_directly_and_outside_reader_still_reads_it(monkeypatch) -> None:
    design = _build_shared_target_design()
    original = copy.deepcopy(design)
    result = optimize_cone_depth(design, _nb("m"))
    assert result.changed
    assert _tap_gates(result.design) == []
    driver = result.design.net_driver[_nb("m")]
    assert driver.inst_name.startswith("t_depth_opt_gate_")
    readers = [g.inst_name for g in result.design.net_fanout.get(_nb("m"), [])]
    assert "gn" in readers
    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail

    _force_retap(monkeypatch)
    tapped = optimize_cone_depth(copy.deepcopy(original), _nb("m"))
    assert len(_tap_gates(tapped.design)) == 1
    assert result.depth_after == tapped.depth_after - 1


def test_gate_count_optimization_leaves_no_tap_gate(monkeypatch) -> None:
    design = _redundant_design()
    original = copy.deepcopy(design)
    result = optimize_gate_count(design)
    assert isinstance(result, AreaOptResult)
    assert result.changed
    assert _tap_gates(result.design) == []
    assert len(result.design.gates) < len(original.gates)
    eq = verify_equivalence(original, result.design)
    assert eq.equivalent, eq.detail

    _force_retap(monkeypatch)
    tapped = optimize_gate_count(copy.deepcopy(original))
    assert len(_tap_gates(tapped.design)) == 1


def test_depth_change_sentence_wording() -> None:
    from netlist_agent.abc_synth import depth_change_sentence

    assert depth_change_sentence("maximum logic depth", 7, 3, 10, 12) == "Reduced maximum logic depth from 7 to 3."
    same = depth_change_sentence("maximum logic depth", 4, 4, 10, 8)
    assert same == "Maximum logic depth stays at 4; the resynthesized design has fewer gates (10 to 8)."
    cone = depth_change_sentence("the depth of the cone of n11[0]", 2, 2, 9, 7)
    assert cone.startswith("The depth of the cone of n11[0] stays at 2;")


_SAME_DEPTH_BLIF = """
.model top
.inputs a b
.outputs y
.gate and2 a=a b=b O=new_n1
.gate and2 a=new_n1 b=new_n1 O=y
.end
"""


def _redundant_design() -> Design:
    """y = AND(AND(a,b), AND(a,b)): depth 2, 3 gates."""
    design = Design(module_name="top")
    for n in ("a", "b"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port("a", Direction.INPUT), Port("b", Direction.INPUT), Port("y", Direction.OUTPUT)]
    t1, t2 = design.fresh_net("t_"), design.fresh_net("t_")
    design.add_gate(Gate("g1", GateType.AND, {"O": t1, "I0": _nb("a"), "I1": _nb("b")}))
    design.add_gate(Gate("g2", GateType.AND, {"O": t2, "I0": _nb("a"), "I1": _nb("b")}))
    design.add_gate(Gate("g3", GateType.AND, {"O": _nb("y"), "I0": t1, "I1": t2}))
    design.build_indices()
    return design


def test_same_depth_fewer_gates_note_is_not_from_x_to_x(monkeypatch) -> None:
    blif_out = parse_blif(_SAME_DEPTH_BLIF)
    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", lambda view, basis, timeout, opt_script=None: blif_out)
    whole = optimize_depth(_redundant_design())
    assert whole.changed and whole.depth_before == whole.depth_after == 2
    assert whole.note.startswith("Maximum logic depth stays at 2; the resynthesized design has fewer gates (3 to 2)")
    assert "from 2 to 2" not in whole.note

    cone_blif = parse_blif(_SAME_DEPTH_BLIF.replace("O=y", f"O={abc_synth_module._CONE_OUT_TOKEN}").replace(".outputs y", f".outputs {abc_synth_module._CONE_OUT_TOKEN}"))
    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", lambda view, basis, timeout, opt_script=None: cone_blif)
    cone = optimize_cone_depth(_redundant_design(), _nb("y"))
    assert cone.changed and cone.depth_before == cone.depth_after == 2
    assert cone.note.startswith("The depth of the cone of y stays at 2; the resynthesized design has fewer gates (3 to 2)")
    assert "from 2 to 2" not in cone.note


# ----------------------------------------------------------------------
# 34a-fix: gates that were already dangling do not count as an improvement
# ----------------------------------------------------------------------


def _optimal_design_with_dead_gate(with_dead: bool) -> Design:
    """y = AND(a, b): one gate, already optimal for depth (1) and area (1).
    With `with_dead`, also a NOT(c) whose output nobody reads (dangling)."""
    design = Design(module_name="top")
    for n in ("a", "b", "c"):
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port(n, Direction.INPUT) for n in ("a", "b", "c")] + [Port("y", Direction.OUTPUT)]
    design.add_gate(Gate("g1", GateType.AND, {"O": _nb("y"), "I0": _nb("a"), "I1": _nb("b")}))
    if with_dead:
        design.add_gate(Gate("gdead", GateType.NOT, {"O": design.fresh_net("dead_"), "I0": _nb("c")}))
    design.build_indices()
    return design


def test_fix_whole_depth_preexisting_dangling_gate_is_not_an_improvement() -> None:
    control = optimize_depth(_optimal_design_with_dead_gate(False))
    assert control.changed is False
    design = _optimal_design_with_dead_gate(True)
    result = optimize_depth(design)
    assert result.changed is False
    assert result.design is design
    assert len(design.gates) == 2


def test_fix_cone_depth_preexisting_dangling_gate_is_not_an_improvement() -> None:
    control = optimize_cone_depth(_optimal_design_with_dead_gate(False), _nb("y"))
    assert control.changed is False
    design = _optimal_design_with_dead_gate(True)
    result = optimize_cone_depth(design, _nb("y"))
    assert result.changed is False
    assert result.design is design
    assert len(design.gates) == 2


def test_fix_cone_gate_count_preexisting_dangling_gate_is_not_an_improvement() -> None:
    control = optimize_cone_gate_count(_optimal_design_with_dead_gate(False), _nb("y"))
    assert control.changed is False
    design = _optimal_design_with_dead_gate(True)
    result = optimize_cone_gate_count(design, _nb("y"))
    assert result.changed is False
    assert result.design is design
    assert result.gates_before == 2
    assert result.gates_after == 2


def test_fix_whole_gate_count_still_counts_removing_a_dangling_gate_as_an_improvement() -> None:
    """SPEC_34 F1: whole-design gate-count optimization deliberately keeps the
    plain `len(design.gates)` criterion -- dropping dead logic is a legitimate
    area saving there, unlike the depth/cone tie-breaks."""
    result = optimize_gate_count(_optimal_design_with_dead_gate(True))
    assert result.changed is True
    assert result.gates_before == 2 and result.gates_after == 1


def _redundant_design_with_dead_gate(with_dead: bool) -> Design:
    """`_redundant_design()` (3 gates -> 2 after resynthesis, same depth), plus
    optionally one NOT(a) nobody reads. Correct baseline is 3 either way;
    subtracting the dangling gate twice would give 2 and reject `2 < 2`."""
    design = _redundant_design()
    if with_dead:
        design.add_gate(Gate("gdead", GateType.NOT, {"O": design.fresh_net("dead_"), "I0": _nb("a")}))
        design.build_indices()
    return design


def _patch_same_depth_blif(monkeypatch, cone: bool) -> None:
    text = _SAME_DEPTH_BLIF
    if cone:
        tok = abc_synth_module._CONE_OUT_TOKEN
        text = text.replace("O=y", f"O={tok}").replace(".outputs y", f".outputs {tok}")
    blif_out = parse_blif(text)
    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", lambda view, basis, timeout, opt_script=None: blif_out)


def test_fix_whole_depth_real_improvement_with_dangling_gate_is_still_accepted(monkeypatch) -> None:
    _patch_same_depth_blif(monkeypatch, cone=False)
    control = optimize_depth(_redundant_design_with_dead_gate(False))
    assert control.changed is True
    design = _redundant_design_with_dead_gate(True)
    assert len(design.gates) == 4
    result = optimize_depth(design)
    assert result.changed is True
    assert len(result.design.gates) == 2
    # The reply reports the design's actual gate counts (dangling gate included), not the baseline.
    assert "(4 to 2)" in result.note


def test_fix_cone_depth_real_improvement_with_dangling_gate_is_still_accepted(monkeypatch) -> None:
    _patch_same_depth_blif(monkeypatch, cone=True)
    control = optimize_cone_depth(_redundant_design_with_dead_gate(False), _nb("y"))
    assert control.changed is True
    design = _redundant_design_with_dead_gate(True)
    assert len(design.gates) == 4
    result = optimize_cone_depth(design, _nb("y"))
    assert result.changed is True
    assert len(result.design.gates) == 2
    assert "(4 to 2)" in result.note


def test_fix_cone_gate_count_real_improvement_with_dangling_gate_is_still_accepted(monkeypatch) -> None:
    _patch_same_depth_blif(monkeypatch, cone=True)
    control = optimize_cone_gate_count(_redundant_design_with_dead_gate(False), _nb("y"))
    assert control.changed is True
    design = _redundant_design_with_dead_gate(True)
    assert len(design.gates) == 4
    result = optimize_cone_gate_count(design, _nb("y"))
    assert result.changed is True
    assert len(result.design.gates) == 2
    assert result.gates_before == 4 and result.gates_after == 2
    assert result.note == "Reduced gate count from 4 to 2."


def test_fix_count_dangling_gates_counts_and_does_not_mutate() -> None:
    from netlist_agent.transform import count_dangling_gates

    assert count_dangling_gates(_optimal_design_with_dead_gate(False)) == 0
    design = _optimal_design_with_dead_gate(True)
    gate_ids = [id(g) for g in design.gates]
    assert count_dangling_gates(design) == 1
    assert len(design.gates) == 2
    assert [id(g) for g in design.gates] == gate_ids
    assert remove_dangling_gates(design) == 1
    assert count_dangling_gates(design) == 0


def test_fix_count_dangling_gates_counts_a_whole_dead_chain() -> None:
    """A dead chain NOT(c) -> NOT -> NOT -> (unread): only the last gate's
    output is unread, so a shallow "output nobody reads" count would say 1,
    and "unread plus its direct driver" would say 2. Three gates tell both
    apart from the full backward closure."""
    from netlist_agent.transform import count_dangling_gates

    design = _optimal_design_with_dead_gate(False)
    mid0 = design.fresh_net("dead_")
    mid1 = design.fresh_net("dead_")
    design.add_gate(Gate("gdead0", GateType.NOT, {"O": mid0, "I0": _nb("c")}))
    design.add_gate(Gate("gdead1", GateType.NOT, {"O": mid1, "I0": mid0}))
    design.add_gate(Gate("gdead2", GateType.NOT, {"O": design.fresh_net("dead_"), "I0": mid1}))
    design.build_indices()
    assert count_dangling_gates(design) == 3
    assert remove_dangling_gates(design) == 3


# ----------------------------------------------------------------------
# Batch 34b-1 (U1-U3): multi-candidate ranking, cec fallback, stop on mismatch.
# `_run_abc_synthesis` is faked, so no real ABC runs here; `verify_equivalence`
# is faked too (the fake BLIFs are not functionally the chain).
# ----------------------------------------------------------------------


def _fake_chain_blif(depth: int, extra: int = 0, tag: int = 0, out: str = "y") -> "abc_synth_module._BlifNetlist":
    """A BLIF over a0..a7 -> y with exactly `depth` AND levels and `depth + extra` gates.
    Gate 0 = and(a0, a1); gate i = and(prev, b_i), b_i = a_{i+1}, except that the first `extra`
    of gates 1.. read b_i through a leaf buf1 (a gate that adds count, not depth). `tag`
    only changes which input the LAST gate reads (a{7-tag}), so ties can be told apart.
    `out` is the output token (the cone path expects `_CONE_OUT_TOKEN`)."""
    assert 1 <= depth <= 7 and 0 <= extra <= depth - 1
    gates = [_BlifGate("and2", {"a": "a0", "b": "a1", "O": "n0" if depth > 1 else out})]
    for i in range(1, depth):
        last = i == depth - 1
        b = f"a{7 - tag}" if last else f"a{i + 1}"
        if i <= extra:
            gates.append(_BlifGate("buf1", {"a": b, "O": f"e{i}"}))
            b = f"e{i}"
        gates.append(_BlifGate("and2", {"a": f"n{i - 1}", "b": b, "O": out if last else f"n{i}"}))
    return abc_synth_module._BlifNetlist([f"a{i}" for i in range(8)], [out], gates)


def _fake_candidates(monkeypatch, specs, out: str = "y"):
    """Candidate k (in `_DEPTH_SMALL_SCRIPTS` order) returns `_fake_chain_blif(*specs[k])`.
    Returns the list of scripts it was called with."""
    scripts: list[str] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        scripts.append(opt_script)
        return _fake_chain_blif(*specs[abc_synth_module._DEPTH_SMALL_SCRIPTS.index(opt_script)], out=out)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    return scripts


def _always_equivalent(monkeypatch):
    calls: list[Design] = []

    def _eq(original, candidate, timeout=None):
        calls.append(candidate)
        return EquivResult(True, "")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _eq)
    return calls


def _count_extracts(monkeypatch) -> list[int]:
    """Spy on `extract_combinational_view` (one call per prepare()); returns a list with one entry per call."""
    calls: list[int] = []
    real = abc_synth_module.extract_combinational_view

    def _spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(abc_synth_module, "extract_combinational_view", _spy)
    return calls


def _y_last_input(design: Design) -> str:
    (g,) = [g for g in design.gates if g.pins["O"] == _nb("y")]
    return g.pins["I1"].name


def test_u1_picks_the_lowest_depth(monkeypatch) -> None:
    """kills: ranking that ignores depth (e.g. first accepted wins)."""
    _fake_candidates(monkeypatch, [(5, 0), (3, 0), (4, 0), (6, 0)])
    _always_equivalent(monkeypatch)
    result = optimize_depth(_build_and_chain())
    assert result.changed and result.failure is None
    assert result.depth_after == 3
    assert NetlistGraph(result.design).max_design_depth() == 3
    assert result.candidates == 4 and result.incomplete == ()


def test_u1_same_depth_picks_the_fewest_gates(monkeypatch) -> None:
    """kills: dropping the gate count from the ranking key (then the first depth-4 candidate, 7 gates, wins)."""
    _fake_candidates(monkeypatch, [(4, 3), (4, 1), (4, 2), (5, 0)])
    _always_equivalent(monkeypatch)
    extracts = _count_extracts(monkeypatch)
    result = optimize_depth(_build_and_chain())
    assert result.depth_after == 4
    assert len(result.design.gates) == 5
    # Each candidate is extracted once; the winner is kept, never rebuilt. (Kills: the running-best
    # comparison losing its gate count, which would discard and rebuild the winner.)
    assert extracts == [1] * 4


def test_u1_full_tie_picks_the_first_candidate(monkeypatch) -> None:
    """kills: a tie broken by the LAST candidate (>= instead of > on the order)."""
    _fake_candidates(monkeypatch, [(4, 1, 0), (4, 1, 1), (4, 1, 2), (4, 1, 3)])
    _always_equivalent(monkeypatch)
    result = optimize_depth(_build_and_chain())
    assert result.depth_after == 4
    assert _y_last_input(result.design) == "a7"


def test_u1_cone_version_ranks_the_same_way(monkeypatch) -> None:
    """The cone path shares the ranking; here the gate-count tie-break picks candidate 2."""
    _fake_candidates(monkeypatch, [(4, 3), (4, 2), (4, 1), (5, 0)], out=abc_synth_module._CONE_OUT_TOKEN)
    _always_equivalent(monkeypatch)
    result = optimize_cone_depth(_build_and_chain(), _nb("y"))
    assert result.depth_after == 4
    assert len(result.design.gates) == 5


def test_u2_undecided_cec_falls_back_to_the_next_best(monkeypatch) -> None:
    """kills: giving up when the best candidate's cec is undecided; running cec on every candidate;
    re-running ABC to rebuild the runner-up."""
    scripts = _fake_candidates(monkeypatch, [(3, 0, 0), (4, 0, 1), (5, 0, 2), (6, 0, 3)])
    verified: list[Design] = []

    def _eq(original, candidate, timeout=None):
        verified.append(candidate)
        if len(verified) == 1:
            raise abc_bridge_module.ABCInconclusiveError("cec gave up")
        return EquivResult(True, "")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _eq)
    design = _build_and_chain()
    result = optimize_depth(design)
    assert result.changed and result.failure is None
    assert result.depth_after == 4
    assert _y_last_input(result.design) == "a6"
    assert len(verified) == 2  # the best (undecided) and the runner-up, not all four
    assert len(scripts) == 4  # the runner-up was rebuilt from its BLIF, no extra ABC run
    assert result.incomplete == ("equivalence check was inconclusive: cec gave up",)
    assert result.note.endswith(
        " 1 of 4 candidate scripts did not finish (equivalence check was inconclusive: cec gave up), "
        "so a lower depth may exist."
    )
    assert result.design is verified[1]


def test_u2_cec_runs_only_for_the_first_ranked_when_it_is_decided(monkeypatch) -> None:
    """kills: running cec on every accepted candidate."""
    _fake_candidates(monkeypatch, [(5, 0), (3, 0), (4, 0), (6, 0)])
    calls = _always_equivalent(monkeypatch)
    optimize_depth(_build_and_chain())
    assert len(calls) == 1


def test_u3_not_equivalent_stops_and_does_not_try_the_next(monkeypatch) -> None:
    """kills: treating a mismatch like an undecided cec (falling back to the runner-up)."""
    scripts = _fake_candidates(monkeypatch, [(3, 0), (4, 0), (5, 0), (6, 0)])
    calls: list[Design] = []

    def _eq(original, candidate, timeout=None):
        calls.append(candidate)
        return EquivResult(False, "counterexample: a0=1")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _eq)
    design = _build_and_chain()
    result = optimize_depth(design)
    assert len(calls) == 1
    assert len(scripts) == 4
    assert result.changed is False
    assert result.design is design
    assert "indicates a bug" in result.note
    assert result.failure is not None and "not equivalent" in result.failure
    assert result.depth_after == result.depth_before
    assert len(result.incomplete) <= result.candidates
    assert result.incomplete[-1] == result.failure


def test_u3_cone_not_equivalent_stops(monkeypatch) -> None:
    _fake_candidates(monkeypatch, [(3, 0), (4, 0), (5, 0), (6, 0)], out=abc_synth_module._CONE_OUT_TOKEN)
    calls: list[Design] = []

    def _eq(original, candidate, timeout=None):
        calls.append(candidate)
        return EquivResult(False, "counterexample")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _eq)
    design = _build_and_chain()
    result = optimize_cone_depth(design, _nb("y"))
    assert len(calls) == 1 and result.design is design and "indicates a bug" in result.note


def test_budget_reserve_decides_whether_candidates_after_the_first_run(monkeypatch) -> None:
    """D10: from the 2nd candidate on, avail = remaining - 0.35 * budget; avail < 10s => skipped
    (one `incomplete` entry each), else the candidate gets min(timeout, avail).
    kills: reserve 0 (would run with 100s); the 10s floor; no clamp to avail."""
    timeouts: list[float] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        timeouts.append(timeout)
        return _fake_chain_blif(3)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    _always_equivalent(monkeypatch)
    monkeypatch.setattr(abc_synth_module.deadline, "current_budget", lambda: 270.0)
    skipped = "skipped: not enough of this request's time budget was left for another candidate"

    # 270 * 0.35 = 94.5 reserved: 100 left => avail 5.5 < 10 => the 3 later candidates are skipped.
    monkeypatch.setattr(abc_synth_module.deadline, "remaining", lambda: 100.0)
    result = optimize_depth(_build_and_chain(), timeout=120.0)
    assert timeouts == [120.0]
    assert result.incomplete == (skipped,) * 3
    assert len(result.incomplete) <= result.candidates
    assert result.changed and "3 of 4 candidate scripts did not finish" in result.note

    # 110 left => avail 15.5: all run, candidates 2.. clamped to 15.5; the first keeps `timeout`.
    timeouts.clear()
    monkeypatch.setattr(abc_synth_module.deadline, "remaining", lambda: 110.0)
    result = optimize_depth(_build_and_chain(), timeout=120.0)
    assert timeouts == [120.0, 15.5, 15.5, 15.5]
    assert result.incomplete == ()


# ----------------------------------------------------------------------
# Batch 34b-1b (U4-U12): failures/budget/size groups/cone ranking/real ABC value/
# basis/dangling baseline. U8 (all fail) and U9 (completed but not improved, another
# failed) are the rewritten batch-30 tests above (`test_all_candidates_failing_keeps_every_reason`,
# `test_no_improvement_with_a_timed_out_candidate_is_not_ruled_out_not_a_failure`).
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [abc_synth_module._SynthTimeout("ABC synthesis timed out after 120s"), ABCBridgeError("ABC exploded")],
    ids=["timeout", "bridge-error"],
)
def test_u4_a_failed_candidate_is_recorded_and_the_others_still_run(monkeypatch, exc) -> None:
    """kills: not recording the failure in `incomplete`; `continue` turned into `break`
    (the later candidates would not run); a failure becoming `result.failure` although others won."""
    scripts: list[str] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        scripts.append(opt_script)
        k = abc_synth_module._DEPTH_SMALL_SCRIPTS.index(opt_script)
        if k == 1:
            raise exc
        return _fake_chain_blif(5 - k)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    _always_equivalent(monkeypatch)
    result = optimize_depth(_build_and_chain())
    assert scripts == list(abc_synth_module._DEPTH_SMALL_SCRIPTS)
    assert result.incomplete == (str(exc),)
    assert result.failure is None and result.changed
    assert result.depth_after == 2  # candidate 3 (k=3) gives depth 2
    assert result.candidates == 4
    assert result.note.endswith(f" 1 of 4 candidate scripts did not finish ({exc}), so a lower depth may exist.")


def test_u4_a_failed_first_candidate_does_not_stop_the_rest(monkeypatch) -> None:
    """kills: candidate 1 failing aborting the loop (the first candidate is `prepare()`d outside the loop body)."""

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        if opt_script == abc_synth_module._OPT_SCRIPT:
            raise abc_synth_module._SynthTimeout("ABC synthesis timed out after 120s")
        return _fake_chain_blif(4)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    _always_equivalent(monkeypatch)
    result = optimize_depth(_build_and_chain())
    assert result.changed and result.depth_after == 4
    assert result.incomplete == ("ABC synthesis timed out after 120s",)


_SKIPPED = "skipped: not enough of this request's time budget was left for another candidate"


def _timeouts_recorder(monkeypatch, depth: int = 3) -> list[float]:
    timeouts: list[float] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        timeouts.append(timeout)
        return _fake_chain_blif(depth)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    _always_equivalent(monkeypatch)
    return timeouts


def test_u5_budget_running_out_midway_skips_the_rest_one_entry_each(monkeypatch) -> None:
    """kills: not recording the skipped candidates; recording one entry for all; skipping only one
    candidate (no `continue` for later ones); the skip decision ignoring the shrinking remainder."""
    timeouts = _timeouts_recorder(monkeypatch)
    monkeypatch.setattr(abc_synth_module.deadline, "current_budget", lambda: 270.0)
    # t0 start, t0 end (t0 = 0), candidate 2 (before and after its prepare), then candidates 3, 4
    left = iter([200.0, 200.0, 200.0, 200.0, 100.0, 100.0])
    monkeypatch.setattr(abc_synth_module.deadline, "remaining", lambda: next(left))
    result = optimize_depth(_build_and_chain(), timeout=120.0)
    # candidate 2: avail = 200 - 94.5 = 105.5 -> runs, clamped; candidates 3, 4: avail 5.5 < 10 -> skipped.
    assert timeouts == [120.0, 105.5]
    assert result.incomplete == (_SKIPPED, _SKIPPED)
    assert len(result.incomplete) <= result.candidates
    assert "2 of 4 candidate scripts did not finish" in result.note


def test_u5_with_a_real_request_scope_a_tiny_budget_skips_candidates_after_the_first(monkeypatch) -> None:
    """kills: ignoring `deadline.remaining()` (all four would run)."""
    timeouts = _timeouts_recorder(monkeypatch)
    with abc_synth_module.deadline.request_scope(12.0):  # reserve 4.2s -> avail < 10
        result = optimize_depth(_build_and_chain(), timeout=120.0)
    assert timeouts == [120.0]
    assert result.incomplete == (_SKIPPED,) * 3


def test_u5_without_a_request_scope_every_candidate_runs_with_the_full_timeout(monkeypatch) -> None:
    """kills: treating "no scope" (remaining() is None) as "no time left"; clamping without a scope."""
    timeouts = _timeouts_recorder(monkeypatch)
    assert abc_synth_module.deadline.remaining() is None
    result = optimize_depth(_build_and_chain(), timeout=77.0)
    assert timeouts == [77.0] * 4
    assert result.incomplete == ()


def _record_scripts(monkeypatch, out: str = "y") -> list[str]:
    scripts: list[str] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        scripts.append(opt_script)
        return _fake_chain_blif(3, out=out)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    _always_equivalent(monkeypatch)
    return scripts


# The view ABC gets: 7 gates for the whole `_build_and_chain()`; 8 for y's restricted cone
# (`_restrict_to_fanin_cone` adds the output buffer).
_U6_VIEW_GATES = {False: 7, True: 8}


@pytest.mark.parametrize("cone", [False, True], ids=["whole", "cone"])
def test_u6_a_design_above_the_threshold_gets_the_large_group(monkeypatch, cone) -> None:
    """kills: always using the small list; the large list being used for the wrong comparison direction."""
    monkeypatch.setattr(abc_synth_module, "_DEPTH_LARGE_THRESHOLD", _U6_VIEW_GATES[cone] - 1)
    scripts = _record_scripts(monkeypatch, out=abc_synth_module._CONE_OUT_TOKEN if cone else "y")
    design = _build_and_chain()
    result = optimize_cone_depth(design, _nb("y")) if cone else optimize_depth(design)
    assert scripts == list(abc_synth_module._DEPTH_LARGE_SCRIPTS)
    assert result.candidates == len(abc_synth_module._DEPTH_LARGE_SCRIPTS)


@pytest.mark.parametrize("cone", [False, True], ids=["whole", "cone"])
def test_u6_a_design_exactly_at_the_threshold_gets_the_small_group(monkeypatch, cone) -> None:
    """kills: `<=` turned into `<` in the threshold comparison (view gates == threshold)."""
    monkeypatch.setattr(abc_synth_module, "_DEPTH_LARGE_THRESHOLD", _U6_VIEW_GATES[cone])
    scripts = _record_scripts(monkeypatch, out=abc_synth_module._CONE_OUT_TOKEN if cone else "y")
    design = _build_and_chain()
    result = optimize_cone_depth(design, _nb("y")) if cone else optimize_depth(design)
    assert scripts == list(abc_synth_module._DEPTH_SMALL_SCRIPTS)
    assert result.candidates == len(abc_synth_module._DEPTH_SMALL_SCRIPTS)


def test_u7_cone_version_ranks_by_the_cone_depth_not_the_whole_design_depth(monkeypatch) -> None:
    """y's cone is 7 deep, z's chain (untouched) is 5 deep, so the whole-design depth is 5 for every
    candidate (a tie) while the cone depths differ. Candidate 0: cone depth 4, 4 gates; candidate 1:
    cone depth 3, 5 gates. Ranking by whole-design depth would tie and pick candidate 0 on gate count.
    kills: the cone measure replaced by `max_design_depth`."""
    design = _build_two_cone_design(8)
    graph = NetlistGraph(design)
    assert graph.depth_to_sink(_nb("y")) == 7 and graph.depth_to_sink(_nb("z")) == 5
    _fake_candidates(monkeypatch, [(4, 0), (3, 2), (6, 0), (6, 0)], out=abc_synth_module._CONE_OUT_TOKEN)
    _always_equivalent(monkeypatch)
    result = optimize_cone_depth(design, _nb("y"))
    assert result.changed
    assert NetlistGraph(result.design).depth_to_sink(_nb("y")) == 3
    assert result.depth_after == 3
    assert NetlistGraph(result.design).max_design_depth() == 5


# U10: AND-OR alternating chain c[i+1] = (c[i] & a[i]) | b[i], 60 stages (depth 120).
# Measured locally (ABC vendored): `_OPT_SCRIPT` alone leaves depth 120 (unchanged);
# `optimize_depth` (all candidates) reaches 11; SOP x4/x8 alone 11, `&sopb` x2 alone 20,
# XOR script and `strash; balance` alone 120. The thresholds sit between those.
_U10_STAGES = 60


def _build_and_or_chain(stages: int = _U10_STAGES) -> Design:
    design = Design(module_name="top")
    names = ["c0"] + [f"a{i}" for i in range(stages)] + [f"b{i}" for i in range(stages)]
    for n in names:
        design.signals[n] = Signal(n, None, None, Direction.INPUT)
    design.signals["y"] = Signal("y", None, None, Direction.OUTPUT)
    design.ports = [Port(n, Direction.INPUT) for n in names] + [Port("y", Direction.OUTPUT)]
    prev = _nb("c0")
    for i in range(stages):
        t = design.fresh_net("t_")
        out = _nb("y") if i == stages - 1 else design.fresh_net("t_")
        design.add_gate(Gate(f"ga{i}", GateType.AND, {"O": t, "I0": prev, "I1": _nb(f"a{i}")}))
        design.add_gate(Gate(f"go{i}", GateType.OR, {"O": out, "I0": t, "I1": _nb(f"b{i}")}))
        prev = out
    design.build_indices()
    return design


def test_u10_real_abc_candidates_beat_the_single_opt_script_on_an_and_or_chain(monkeypatch) -> None:
    """kills: the candidate list collapsing to `_OPT_SCRIPT` only (depth stays 120 and `changed` False)."""
    design = _build_and_or_chain()
    assert NetlistGraph(design).max_design_depth() == 2 * _U10_STAGES
    original = copy.deepcopy(design)
    with monkeypatch.context() as m:
        m.setattr(abc_synth_module, "_depth_candidate_scripts", lambda gate_count: (abc_synth_module._OPT_SCRIPT,))
        single = optimize_depth(design)
    assert single.failure is None
    assert single.depth_after >= 100  # measured: 120 (no change)
    result = optimize_depth(design)
    assert result.failure is None and result.incomplete == ()
    assert result.changed and result.depth_after <= 30  # measured: 11
    assert result.depth_after < single.depth_after - 60
    assert NetlistGraph(result.design).max_design_depth() == result.depth_after
    assert verify_equivalence(original, result.design).equivalent


@pytest.mark.parametrize("script", [abc_synth_module._DEPTH_SMALL_SCRIPTS[2], abc_synth_module._DEPTH_SMALL_SCRIPTS[1], abc_synth_module._DEPTH_LARGE_SCRIPTS[1]], ids=["if-g", "if-x", "sopb"])
def test_u10_each_non_default_candidate_script_runs_in_the_local_abc(script) -> None:
    """`if -g`, `if -x` and `&sopb` all exist in the vendored ABC and produce a usable BLIF
    (a script ABC rejects would be silently recorded as an `incomplete` candidate in production)."""
    design = _build_and_or_chain(10)
    comb = abc_synth_module.extract_combinational_view(copy.deepcopy(design), "free_pi", {})
    blif = abc_synth_module._run_abc_synthesis(comb, None, 60.0, opt_script=script)
    assert blif.gates and blif.outputs == ["y"]


@pytest.mark.parametrize("basis", ["nand_not", "nor_not", "and_or_not"])
def test_u11_multi_candidate_result_stays_inside_the_basis_and_equivalent(basis) -> None:
    """kills: a candidate path (e.g. rebuilt BLIF / later script) not honoring the basis."""
    design = _build_and_chain(16)
    original = copy.deepcopy(design)
    result = optimize_depth(design, basis=basis)
    assert result.failure is None and result.changed and result.candidates == 4
    allowed = allowed_gate_types(basis)
    assert {g.gate_type for g in result.design.gates} <= allowed
    assert verify_equivalence(original, result.design).equivalent


def test_u11_every_candidate_receives_the_basis(monkeypatch) -> None:
    real_run = abc_synth_module._run_abc_synthesis
    bases: list = []

    def _spy(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        bases.append(basis)
        return real_run(view, basis, timeout, opt_script)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _spy)
    optimize_depth(_build_and_chain(16), basis="nand_not")
    assert bases == ["nand_not"] * 4


_ONE_AND_BLIF = """.model top
.inputs a b c
.outputs y
.gate and2 a=a b=b O=y
.end
"""


def _u12_fake(monkeypatch, text: str, cone: bool) -> list[str]:
    if cone:
        tok = abc_synth_module._CONE_OUT_TOKEN
        text = text.replace("O=y", f"O={tok}").replace(".outputs y", f".outputs {tok}")
    blif = parse_blif(text)
    scripts: list[str] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        scripts.append(opt_script)
        return blif

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    return scripts


@pytest.mark.parametrize("cone", [False, True], ids=["whole", "cone"])
def test_u12_multi_candidate_preexisting_dangling_gate_does_not_turn_optimal_into_changed(monkeypatch, cone) -> None:
    """Every candidate returns the same 1-gate result; the dead NOT(c) is swept from each, giving
    1 gate < len(design.gates)=2 but not < baseline=1. kills: baseline = `len(design.gates)`."""
    call = (lambda d: optimize_cone_depth(d, _nb("y"))) if cone else optimize_depth
    scripts = _u12_fake(monkeypatch, _ONE_AND_BLIF.replace(" c\n", "\n"), cone)
    control = call(_optimal_design_with_dead_gate(False))
    assert control.changed is False and len(scripts) == 4
    scripts.clear()
    design = _optimal_design_with_dead_gate(True)
    result = call(design)
    assert len(scripts) == 4
    assert result.changed is False and result.design is design and result.failure is None
    assert "already optimal" in result.note
    assert len(design.gates) == 2


@pytest.mark.parametrize("cone", [False, True], ids=["whole", "cone"])
def test_u12_multi_candidate_real_improvement_with_dangling_gate_is_still_changed(monkeypatch, cone) -> None:
    """Same depth, 2 gates vs baseline 3 (4 gates incl. the dead one). kills: baseline subtracting
    the dangling gates twice (2 < 2 would reject); baseline ignoring the dead gate is covered above."""
    call = (lambda d: optimize_cone_depth(d, _nb("y"))) if cone else optimize_depth
    scripts = _u12_fake(monkeypatch, _SAME_DEPTH_BLIF, cone)
    design = _redundant_design_with_dead_gate(True)
    result = call(design)
    assert len(scripts) == 4
    assert result.changed is True and len(result.design.gates) == 2
    assert "(4 to 2)" in result.note


# ----------------------------------------------------------------------
# Batch 34b-1 repair round 1: rebuild budget, boundary, rebuild failure, D6 pin
# ----------------------------------------------------------------------

_SKIPPED_REBUILD = (
    "skipped: not enough of this request's time budget was left to rebuild and verify another candidate"
)


def test_f1_no_time_left_after_the_first_cec_skips_rebuilding_the_runners_up(monkeypatch) -> None:
    """kills: rebuilding runner-ups without checking the request's remaining time."""
    _fake_candidates(monkeypatch, [(3, 0), (4, 0), (5, 0), (6, 0)])
    state = {"cec_done": False}

    def _eq(original, candidate, timeout=None):
        state["cec_done"] = True
        raise abc_bridge_module.ABCInconclusiveError("cec gave up")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _eq)
    monkeypatch.setattr(abc_synth_module.deadline, "current_budget", lambda: 270.0)
    monkeypatch.setattr(abc_synth_module.deadline, "remaining", lambda: 0.0 if state["cec_done"] else 1000.0)
    extracts = _count_extracts(monkeypatch)
    design = _build_and_chain()
    result = optimize_depth(design)
    assert len(extracts) == 4  # the four candidates only; no rebuild
    cec = "equivalence check was inconclusive: cec gave up"
    assert result.incomplete == (cec,) + (_SKIPPED_REBUILD,) * 3
    assert len(result.incomplete) <= result.candidates
    assert result.changed is False and result.design is design
    assert result.failure == cec


def _undecided_first_then_equivalent(monkeypatch, remaining_after_cec: float) -> None:
    """First cec is undecided, the rest are equivalent. `remaining()` reads 1000 until the first cec has run,
    then `remaining_after_cec`; `current_budget()` is 270 (a request scope is active)."""
    state = {"cec_done": False}
    verified: list[Design] = []

    def _eq(original, candidate, timeout=None):
        verified.append(candidate)
        if len(verified) == 1:
            state["cec_done"] = True
            raise abc_bridge_module.ABCInconclusiveError("cec gave up")
        return EquivResult(True, "")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _eq)
    monkeypatch.setattr(abc_synth_module.deadline, "current_budget", lambda: 270.0)
    monkeypatch.setattr(
        abc_synth_module.deadline, "remaining", lambda: remaining_after_cec if state["cec_done"] else 1000.0
    )


@pytest.mark.parametrize("remaining_after_cec", [1000.0, 50.0, 10.0])
def test_g1_enough_time_left_after_an_undecided_cec_still_rebuilds_the_runner_up(
    monkeypatch, remaining_after_cec
) -> None:
    """kills: a rebuild guard that fires too often -- threshold raised (e.g. 10 -> 100, which 50 catches),
    `left < 10` turned into `left <= 10` (10.0 catches), or a guard that skips whenever a request scope exists
    (1000 catches)."""
    _fake_candidates(monkeypatch, [(3, 0), (4, 0), (5, 0), (6, 0)])
    _undecided_first_then_equivalent(monkeypatch, remaining_after_cec)
    extracts = _count_extracts(monkeypatch)
    result = optimize_depth(_build_and_chain())
    assert len(extracts) == 5  # four candidates + one rebuild
    assert result.changed and result.depth_after == 4
    assert result.failure is None
    assert result.incomplete == ("equivalence check was inconclusive: cec gave up",)


def test_g1_just_under_ten_seconds_left_skips_the_rebuild(monkeypatch) -> None:
    """kills: `left < 10` turned into `left < 9` / a threshold lowered below 10 (9.99 must skip)."""
    _fake_candidates(monkeypatch, [(3, 0), (4, 0), (5, 0), (6, 0)])
    _undecided_first_then_equivalent(monkeypatch, 9.99)
    extracts = _count_extracts(monkeypatch)
    design = _build_and_chain()
    result = optimize_depth(design)
    assert len(extracts) == 4
    assert result.changed is False and result.design is design
    assert result.incomplete == ("equivalence check was inconclusive: cec gave up",) + (_SKIPPED_REBUILD,) * 3


def test_f1_without_a_request_scope_the_runner_up_is_still_rebuilt(monkeypatch) -> None:
    """kills: skipping the rebuild when `remaining()` is None (no request scope)."""
    _fake_candidates(monkeypatch, [(3, 0), (4, 0), (5, 0), (6, 0)])
    verified: list[Design] = []

    def _eq(original, candidate, timeout=None):
        verified.append(candidate)
        if len(verified) == 1:
            raise abc_bridge_module.ABCInconclusiveError("cec gave up")
        return EquivResult(True, "")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _eq)
    monkeypatch.setattr(abc_synth_module.deadline, "remaining", lambda: None)
    extracts = _count_extracts(monkeypatch)
    result = optimize_depth(_build_and_chain())
    assert result.changed and result.depth_after == 4
    assert len(extracts) == 5  # four candidates + one rebuild


def test_t_b_exactly_ten_seconds_of_avail_still_runs_the_candidate(monkeypatch) -> None:
    """kills: `avail <= 10` (boundary); reserve 0.35 * 270 = 94.5, so remaining 104.5 leaves exactly 10.0."""
    timeouts = _timeouts_recorder(monkeypatch)
    monkeypatch.setattr(abc_synth_module.deadline, "current_budget", lambda: 270.0)
    monkeypatch.setattr(abc_synth_module.deadline, "remaining", lambda: 104.5)
    result = optimize_depth(_build_and_chain(), timeout=120.0)
    assert timeouts == [120.0, 10.0, 10.0, 10.0]
    assert result.incomplete == ()


def test_t_c_a_failed_rebuild_is_recorded_and_the_next_candidate_is_tried(monkeypatch) -> None:
    """kills: the rebuild-failure `continue` turned into `break`; an unrecorded rebuild failure."""
    _fake_candidates(monkeypatch, [(3, 0, 0), (4, 0, 1), (5, 0, 2), (6, 0, 3)])
    verified: list[Design] = []

    def _eq(original, candidate, timeout=None):
        verified.append(candidate)
        if len(verified) == 1:
            raise abc_bridge_module.ABCInconclusiveError("cec gave up")
        return EquivResult(True, "")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _eq)
    real = abc_synth_module.extract_combinational_view
    n = {"calls": 0}

    def _extract(*a, **k):
        n["calls"] += 1
        if n["calls"] == 5:  # the first rebuild (the runner-up)
            raise ABCBridgeError("rebuild boom")
        return real(*a, **k)

    monkeypatch.setattr(abc_synth_module, "extract_combinational_view", _extract)
    result = optimize_depth(_build_and_chain())
    assert result.changed and result.depth_after == 5  # the third-ranked candidate
    assert result.incomplete == (
        "equivalence check was inconclusive: cec gave up",
        "could not rebuild a candidate result: rebuild boom",
    )
    assert len(result.incomplete) <= result.candidates


def test_g4_candidate_results_are_released_as_soon_as_they_are_no_longer_needed(monkeypatch) -> None:
    """F4. Candidates are (6,0),(3,0),(4,0),(5,0): #0 is the first best, #1 replaces it, #2 and #3 lose.
    #0 must beat the original (depth 7) to be accepted at all: with (7,0) it tied the 7-gate original,
    was never the best, and dropping `best.work = None` went unobserved (knife G4-c survived).
    kills: keeping `first` alive after candidate 0 (drop `first = None`); dropping the pre-`prepare()`
    `work = view = ... = None` (previous view/work alive inside prepare); dropping `best.work = None` when
    a better candidate replaces the best; dropping `cand.work = None` for a candidate that is not the best;
    dropping the `work = view = ... = None` after the loop (last view alive at the cec)."""
    import gc
    import weakref

    views: list = []  # weakrefs to the view given to ABC, per candidate
    works: list = []  # weakrefs to the work design, per candidate (taken at remove_dangling_gates)
    events: dict[str, bool] = {}

    def alive(refs, k):
        gc.collect()
        return refs[k]() is not None

    specs = [(6, 0), (3, 0), (4, 0), (5, 0)]

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        k = abc_synth_module._DEPTH_SMALL_SCRIPTS.index(opt_script)
        assert k == len(views)
        views.append(weakref.ref(view))
        if k >= 1:
            events[f"run{k}: view{k - 1} alive"] = alive(views, k - 1)
        if k == 2:
            events["run2: work0 alive"] = alive(works, 0)  # replaced as best by #1
            events["run2: work1 alive"] = alive(works, 1)  # the current best
        if k == 3:
            events["run3: work2 alive"] = alive(works, 2)  # lost to the best, not kept
            events["run3: work1 alive"] = alive(works, 1)
        return _fake_chain_blif(*specs[k])

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)

    real_rdg = abc_synth_module.remove_dangling_gates

    def _rdg(work):
        works.append(weakref.ref(work))
        return real_rdg(work)

    monkeypatch.setattr(abc_synth_module, "remove_dangling_gates", _rdg)

    real_extract = abc_synth_module.extract_combinational_view
    n_extract = {"n": 0}

    def _extract(*a, **k):
        i = n_extract["n"]  # prepare() number i builds candidate i
        n_extract["n"] += 1
        if i >= 1:
            events[f"prepare{i}: view{i - 1} alive"] = alive(views, i - 1)
            if i == 3:
                events["prepare3: work2 alive"] = alive(works, 2)
        return real_extract(*a, **k)

    monkeypatch.setattr(abc_synth_module, "extract_combinational_view", _extract)

    def _eq(original, candidate, timeout=None):
        gc.collect()
        for k in range(4):
            events[f"cec: view{k} alive"] = views[k]() is not None
        for k in (0, 2, 3):
            events[f"cec: work{k} alive"] = works[k]() is not None
        events["cec: work1 alive"] = works[1]() is not None
        events["cec: candidate is work1"] = works[1]() is candidate
        return EquivResult(True, "")

    monkeypatch.setattr(abc_synth_module, "verify_equivalence", _eq)
    result = optimize_depth(_build_and_chain())
    assert result.changed and result.depth_after == 3

    alive_expected = {"run2: work1 alive", "run3: work1 alive", "cec: work1 alive", "cec: candidate is work1"}
    assert alive_expected <= events.keys()
    for name, is_alive in events.items():
        assert is_alive == (name in alive_expected), name
    # every observation point actually fired
    assert {"run1: view0 alive", "run3: view2 alive", "prepare3: view2 alive", "run2: work0 alive",
            "prepare3: work2 alive", "cec: view3 alive", "cec: work3 alive"} <= events.keys()


def test_t_d_the_candidate_lists_match_spec_d6_verbatim() -> None:
    """Order is by cost (batch 35 D1: OPT, XOR, SOP x4, SOP x8).
    Hand-written from SPEC_34 D6 on purpose (not built with `_sop_script`): a changed list must fail here."""
    sop4 = "strash; " + "if -g -K 6; strash; " * 4 + "balance; dch"
    assert sop4 == (
        "strash; if -g -K 6; strash; if -g -K 6; strash; if -g -K 6; strash; if -g -K 6; strash; balance; dch"
    )
    assert abc_synth_module._DEPTH_SMALL_SCRIPTS == (
        "strash; balance; dch",
        "strash; balance; dch; if -x -K 6; strash; balance",
        "strash; if -g -K 6; strash; if -g -K 6; strash; if -g -K 6; strash; if -g -K 6; strash; balance; dch",
        "strash; if -g -K 6; strash; if -g -K 6; strash; if -g -K 6; strash; if -g -K 6; strash; "
        "if -g -K 6; strash; if -g -K 6; strash; if -g -K 6; strash; if -g -K 6; strash; balance; dch",
    )
    assert abc_synth_module._DEPTH_LARGE_SCRIPTS == (
        "strash; balance",
        "strash; &get -n; &st; &sopb; &sopb; &put; balance",
    )
    assert abc_synth_module._DEPTH_LARGE_THRESHOLD == 150_000


# ----------------------------------------------------------------------
# Batch 35 D2: the cec reserve follows machine speed (t0 = time of prepare + candidate 0)
# ----------------------------------------------------------------------


def _advancing_clock(monkeypatch, start: float, budget: float = 270.0):
    """A fake request clock: `remaining()` = start - elapsed; the test advances `elapsed`."""
    state = {"elapsed": 0.0}
    monkeypatch.setattr(abc_synth_module.deadline, "current_budget", lambda: budget)
    monkeypatch.setattr(abc_synth_module.deadline, "remaining", lambda: start - state["elapsed"])
    return state


def _clocked_candidates(monkeypatch, state, first_cost: float, first_exc=None):
    """Candidate 0 advances the clock by `first_cost` (and raises `first_exc` if given); later ones by 0."""
    timeouts: list[float] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        timeouts.append(timeout)
        if opt_script == abc_synth_module._OPT_SCRIPT:
            state["elapsed"] += first_cost
            if first_exc is not None:
                raise first_exc
        return _fake_chain_blif(3)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    _always_equivalent(monkeypatch)
    return timeouts


def test_d2_a_big_t0_raises_the_reserve_to_3_5_t0(monkeypatch) -> None:
    """t0 = 40 -> reserve = max(94.5, 140) = 140. remaining after #0 = 300 - 40 = 260 -> avail 120 -> candidates
    2.. get min(timeout=200, 120). With the old fixed 94.5 they would get 165.5.
    kills: dropping the t0 term; factor != 3.5."""
    state = _advancing_clock(monkeypatch, 300.0)
    timeouts = _clocked_candidates(monkeypatch, state, 40.0)
    result = optimize_depth(_build_and_chain(), timeout=200.0)
    assert timeouts == [200.0, 120.0, 120.0, 120.0]
    assert result.incomplete == ()


def test_d2_a_big_t0_makes_later_candidates_get_skipped(monkeypatch) -> None:
    """t0 = 50 -> reserve 175; remaining 150 - 50 = 100 -> avail < 0 -> 3 skipped (the fixed reserve 94.5 would
    leave avail 5.5, also skipped, so use remaining 170: 170 - 50 = 120 -> fixed avail 25.5 runs, t0 avail -55 skips).
    kills: the t0 term not applied to the skip decision."""
    state = _advancing_clock(monkeypatch, 170.0)
    timeouts = _clocked_candidates(monkeypatch, state, 50.0)
    result = optimize_depth(_build_and_chain(), timeout=200.0)
    assert timeouts == [200.0]
    assert result.incomplete == (_SKIPPED,) * 3


def test_d2_a_small_t0_keeps_the_fraction_reserve(monkeypatch) -> None:
    """t0 = 10 -> 3.5 * 10 = 35 < 94.5 -> reserve stays 0.35 * 270. remaining 200 - 10 = 190 -> avail 95.5.
    kills: reserve = t0-term only; max turned into min or sum."""
    state = _advancing_clock(monkeypatch, 200.0)
    timeouts = _clocked_candidates(monkeypatch, state, 10.0)
    optimize_depth(_build_and_chain(), timeout=200.0)
    assert timeouts == [200.0, 95.5, 95.5, 95.5]


def test_d2_a_failed_first_candidate_does_not_scale_the_reserve(monkeypatch) -> None:
    """Candidate 0 times out after burning 120s of the clock: that time says nothing about cec cost, so t0 = 0.
    remaining 400 - 120 = 280 -> avail 185.5 -> clamped to timeout 150 (with t0=120 the reserve 420 would skip all).
    kills: using t0 although candidate 0 failed."""
    state = _advancing_clock(monkeypatch, 400.0)
    timeouts = _clocked_candidates(
        monkeypatch, state, 120.0, first_exc=abc_synth_module._SynthTimeout("ABC synthesis timed out after 120s")
    )
    result = optimize_depth(_build_and_chain(), timeout=150.0)
    assert timeouts == [150.0, 150.0, 150.0, 150.0]
    assert result.incomplete == ("ABC synthesis timed out after 120s",)


def test_d2_t0_is_measured_across_prepare_not_only_the_abc_run(monkeypatch) -> None:
    """The clock advances inside `prepare()` (the deepcopy + extract), not in ABC: t0 must include it.
    prepare costs 40 -> reserve 140 -> avail = 260 - 140 = 120.
    kills: starting the t0 measurement after prepare()."""
    state = _advancing_clock(monkeypatch, 300.0)
    timeouts = _clocked_candidates(monkeypatch, state, 0.0)
    real_extract = abc_synth_module.extract_combinational_view
    calls = {"n": 0}

    def _slow_extract(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            state["elapsed"] += 40.0
        return real_extract(*a, **k)

    monkeypatch.setattr(abc_synth_module, "extract_combinational_view", _slow_extract)
    optimize_depth(_build_and_chain(), timeout=200.0)
    assert timeouts[:2] == [200.0, 120.0]


def _prepare_clock_hook(monkeypatch, state, on_call: int, cost: float) -> None:
    """The `on_call`-th `extract_combinational_view` (= prepare) advances the fake clock by `cost`."""
    real_extract = abc_synth_module.extract_combinational_view
    calls = {"n": 0}

    def _slow_extract(*a, **k):
        calls["n"] += 1
        if calls["n"] == on_call:
            state["elapsed"] += cost
        return real_extract(*a, **k)

    monkeypatch.setattr(abc_synth_module, "extract_combinational_view", _slow_extract)


def test_d2_prepare_of_a_later_candidate_eats_into_its_timeout(monkeypatch) -> None:
    """Candidate 1's prepare() advances the clock 50. t0 = 0 -> reserve 94.5. Before prepare: avail 205.5 (timeout 200
    would win); after: remaining 250 -> avail 155.5 -> candidate 1 gets 155.5 (later ones too: clock stays at 250).
    kills: computing cand_timeout before prepare() only (no re-read after it)."""
    state = _advancing_clock(monkeypatch, 300.0)
    timeouts = _clocked_candidates(monkeypatch, state, 0.0)
    _prepare_clock_hook(monkeypatch, state, 2, 50.0)
    optimize_depth(_build_and_chain(), timeout=200.0)
    assert timeouts == [200.0, 155.5, 155.5, 155.5]


def test_d2_prepare_that_leaves_under_10s_is_skipped(monkeypatch) -> None:
    """Candidate 1's prepare() advances the clock 210: remaining 90 - reserve 94.5 < 10 -> skipped, ABC not called
    for it, and the rest are skipped too (out_of_time).
    kills: no re-read after prepare() (ABC would be run with timeout 200).
    Not killed here: dropping `out_of_time = True` on this skip -- the fake clock never goes back up, so the later
    candidates fail the pre-prepare check on their own. In production `remaining()` cannot go back up inside one
    candidate loop either: `deadline.extend()` is only called before a handler/tool starts."""
    state = _advancing_clock(monkeypatch, 300.0)
    timeouts = _clocked_candidates(monkeypatch, state, 0.0)
    _prepare_clock_hook(monkeypatch, state, 2, 210.0)
    result = optimize_depth(_build_and_chain(), timeout=200.0)
    assert timeouts == [200.0]
    assert result.incomplete == (_SKIPPED,) * 3


def test_d2_t0_is_taken_from_candidate_0_only(monkeypatch) -> None:
    """Candidate 0 advances the clock 10 (t0 = 10 -> reserve max(94.5, 35) = 94.5); every later candidate advances 30.
    Start 400, timeout 300: #1 avail 400-10-94.5 = 295.5; #2 295.5-30 = 265.5; #3 265.5-30 = 235.5.
    Without the order == 0 guard t0 would be re-measured (40 after #1 -> reserve 140) and #3 would not be 235.5.
    kills: removing the `order == 0` guard (t0 updated for every candidate)."""
    state = _advancing_clock(monkeypatch, 400.0)
    timeouts: list[float] = []

    def _fake(view, basis, timeout, opt_script=abc_synth_module._OPT_SCRIPT):
        timeouts.append(timeout)
        state["elapsed"] += 10.0 if opt_script == abc_synth_module._OPT_SCRIPT else 30.0
        return _fake_chain_blif(3)

    monkeypatch.setattr(abc_synth_module, "_run_abc_synthesis", _fake)
    _always_equivalent(monkeypatch)
    optimize_depth(_build_and_chain(), timeout=300.0)
    assert timeouts == [300.0, 295.5, 265.5, 235.5]


def test_d2_t0_includes_the_splice_and_dangling_removal(monkeypatch) -> None:
    """The clock advances 40 inside the first `remove_dangling_gates` (after ABC and splice, before measure).
    t0 = 40 -> reserve 140 -> avail 260 - 140 = 120 (measuring t0 before it would give reserve 94.5 -> avail 165.5 -> min(200, 165.5) = 165.5).
    kills: measuring t0 before splice/remove_dangling_gates."""
    state = _advancing_clock(monkeypatch, 300.0)
    timeouts = _clocked_candidates(monkeypatch, state, 0.0)
    real_remove = abc_synth_module.remove_dangling_gates
    calls = {"n": 0}

    def _slow_remove(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            state["elapsed"] += 40.0
        return real_remove(*a, **k)

    monkeypatch.setattr(abc_synth_module, "remove_dangling_gates", _slow_remove)
    optimize_depth(_build_and_chain(), timeout=200.0)
    assert timeouts[:2] == [200.0, 120.0]


def test_d2_prepare_that_leaves_between_0_and_10s_is_skipped(monkeypatch) -> None:
    """Candidate 1's prepare() advances the clock 200: remaining 100 - reserve 94.5 = avail 5.5, positive but under
    _DEPTH_MIN_CANDIDATE_SECONDS -> skipped (before prepare it was 205.5, so only the post-prepare check can skip it).
    kills: the post-prepare threshold changed to `< 0` / `< 5` (or dropped in favour of `avail <= 0`)."""
    state = _advancing_clock(monkeypatch, 300.0)
    timeouts = _clocked_candidates(monkeypatch, state, 0.0)
    _prepare_clock_hook(monkeypatch, state, 2, 200.0)
    result = optimize_depth(_build_and_chain(), timeout=200.0)
    assert timeouts == [200.0]
    assert result.incomplete == (_SKIPPED,) * 3


def test_d2_t0_ends_after_measuring_the_depth_of_candidate_0(monkeypatch) -> None:
    """The clock advances 40 inside the depth measurement of candidate 0 (`max_design_depth` call 2; call 1 is
    depth_before). t0 must include it (SPEC D2: t0 ends after candidate 0's depth is measured):
    t0 = 40 -> reserve 140 -> avail 260 - 140 = 120. Ending t0 before measure would give avail 165.5.
    kills: taking the t0 reading before `depth_after = measure(work)`."""
    state = _advancing_clock(monkeypatch, 300.0)
    timeouts = _clocked_candidates(monkeypatch, state, 0.0)
    real_depth = abc_synth_module.NetlistGraph.max_design_depth
    calls = {"n": 0}

    def _slow_depth(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            state["elapsed"] += 40.0
        return real_depth(self, *a, **k)

    monkeypatch.setattr(abc_synth_module.NetlistGraph, "max_design_depth", _slow_depth)
    optimize_depth(_build_and_chain(), timeout=200.0)
    assert calls["n"] >= 2
    assert timeouts[:2] == [200.0, 120.0]
