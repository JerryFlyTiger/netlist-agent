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

from netlist_agent.abc_bridge import verify_equivalence
from netlist_agent.abc_synth import (
    BASIS_GATE_NAMES,
    DepthOptResult,
    _BlifGate,
    _SynthError,
    allowed_gate_types,
    optimize_cone_depth,
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
    must leave the DFF's Q bit exactly as-is (still driven by the DFF) and
    must not corrupt it via the floating-PO-bit artifact that
    extract_combinational_view's free_pi promotion creates for the split bit
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
        pytest.skip("Alpha_Testcase corpus not present -- put the released testcases under Alpha_Testcase/testcase/testNN/ to enable this test")
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
