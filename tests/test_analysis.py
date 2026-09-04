from __future__ import annotations

import os
import time

import pytest

from netlist_agent.analysis import (
    dffs_on_clock,
    direct_fanin,
    direct_fanout,
    fanin_cone_size,
    fanout_cone_size,
    fanout_count,
    gate_count_by_type,
    gate_lookup,
    gates_of_type,
    is_cut_signal,
    largest_fanin_cone,
    list_primary_inputs,
    list_primary_outputs,
    max_fanout_overall,
    max_fanout_pi,
    primary_input_bit_count,
    primary_input_port_count,
    primary_output_bit_count,
    primary_output_port_count,
)
from netlist_agent.graph import CombinationalCycleError, DffPin, NetlistGraph
from netlist_agent.ir import (
    Const,
    Design,
    Direction,
    Gate,
    GateType,
    NetBit,
    OUTPUT_PIN,
    Port,
    Signal,
)
from netlist_agent.parser import parse_verilog

from tests.helpers import corpus_netlist_paths, corpus_available

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATHS = corpus_netlist_paths()


# ----------------------------------------------------------------------
# Synthetic fixture
#
#   inputs:  a, b, clk, rn        outputs: y, z
#
#   Gdff: DFF(RN=rn, SN=1'b1, CK=clk, D=n8, Q=z)   -- Q wired straight to PO z
#   G1  : AND(n1  = a & b)
#   G2  : AND(n2  = n1 & z)                          -- reads DFF.Q as a source
#   G3  : OR (n3  = n1 | b)                          -- n1 and b both fan out
#   G4  : NOT(n4  = ~n3)
#   G5  : BUF(n5  =  n4)
#   G6  : AND(n6  = n2 & n5)                         -- diamond merge
#   G7  : NOT(y   = ~n6)                             -- feeds PO y
#   G8  : BUF(n8  =  n2)                             -- feeds DFF.D (reg-to-reg)
#   G9  : NOT(n9  = ~n1)                             -- dead-end, bumps n1's fanout to 3
#   G10 : AND(n10 = n4 & <unconnected>)              -- dead-end, tests unconnected fanin
#
# Hand-computed expectations (see PR/report for full derivation):
#   depth(a -> y)  = 6, longest path = [G1, G3, G4, G5, G6, G7]
#   path_count(a -> y) = 2   (via G2 branch, and via G3/G4/G5 branch)
#   max_design_depth = 6 (at G7/y); max_reg_to_reg_depth = 2 (G2,G8);
#   max_pi_to_dff_d_depth = 3 (G1,G2,G8)
#   fanin_cone_size(y) = 8 gates (7 non-DFF + boundary DFF Gdff, QA A94);
#   fanin_cone_size(n8, DFF.D) = 4 gates (likewise)
#   fanout_cone_size(a) = 10 gates (all non-dff gates)
#   fanout_count(n1) = 3 (unique max overall); fanout_count(b) = 2 (unique max among PIs)
#   cut_nets_between(a, y) = {n1, n6}; n2 is NOT a cut for any PI/PO pair
# ----------------------------------------------------------------------


def _build_synthetic_design() -> Design:
    design = Design(module_name="top")

    pi_names = ["a", "b", "clk", "rn"]
    po_names = ["y", "z"]
    internal_names = ["n1", "n2", "n3", "n4", "n5", "n6", "n8", "n9", "n10"]

    for name in pi_names:
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
        design.ports.append(Port(name=name, direction=Direction.INPUT))
    for name in po_names:
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.OUTPUT)
        design.ports.append(Port(name=name, direction=Direction.OUTPUT))
    for name in internal_names:
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)

    a, b, clk, rn = (NetBit(n) for n in pi_names)
    y, z = NetBit("y"), NetBit("z")
    n1, n2, n3, n4, n5, n6, n8, n9, n10 = (NetBit(n) for n in internal_names)

    design.gates = [
        Gate("Gdff", GateType.DFF, {"RN": rn, "SN": Const.ONE, "CK": clk, "D": n8, "Q": z}),
        Gate("G1", GateType.AND, {"O": n1, "I0": a, "I1": b}),
        Gate("G2", GateType.AND, {"O": n2, "I0": n1, "I1": z}),
        Gate("G3", GateType.OR, {"O": n3, "I0": n1, "I1": b}),
        Gate("G4", GateType.NOT, {"O": n4, "I0": n3}),
        Gate("G5", GateType.BUF, {"O": n5, "I0": n4}),
        Gate("G6", GateType.AND, {"O": n6, "I0": n2, "I1": n5}),
        Gate("G7", GateType.NOT, {"O": y, "I0": n6}),
        Gate("G8", GateType.BUF, {"O": n8, "I0": n2}),
        Gate("G9", GateType.NOT, {"O": n9, "I0": n1}),
        Gate("G10", GateType.AND, {"O": n10, "I0": n4, "I1": None}),
    ]
    design.build_indices()
    return design


@pytest.fixture
def design() -> Design:
    return _build_synthetic_design()


@pytest.fixture
def graph(design: Design) -> NetlistGraph:
    return NetlistGraph(design)


# ---- Counting / listing (capabilities 1-6) ----


def test_gate_count_by_type(design: Design) -> None:
    counts = gate_count_by_type(design)
    assert counts[GateType.AND] == 4
    assert counts[GateType.OR] == 1
    assert counts[GateType.NOT] == 3
    assert counts[GateType.BUF] == 2
    assert counts[GateType.NAND] == 0
    assert counts[GateType.DFF] == 1
    assert sum(counts.values()) == 11


def test_gates_of_type(design: Design) -> None:
    ands = {g.inst_name for g in gates_of_type(design, GateType.AND)}
    assert ands == {"G1", "G2", "G6", "G10"}


def test_list_primary_inputs_outputs(design: Design) -> None:
    pis = list_primary_inputs(design)
    pos = list_primary_outputs(design)
    assert {p.name for p in pis} == {"a", "b", "clk", "rn"}
    assert {p.name for p in pos} == {"y", "z"}
    assert all(p.width == 1 for p in pis + pos)


def test_pi_po_counts(design: Design) -> None:
    assert primary_input_bit_count(design) == 4
    assert primary_output_bit_count(design) == 2
    assert primary_input_port_count(design) == 4
    assert primary_output_port_count(design) == 2


def test_dffs_on_clock(design: Design) -> None:
    dffs = dffs_on_clock(design, "clk")
    assert [g.inst_name for g in dffs] == ["Gdff"]
    assert dffs_on_clock(design, "no_such_clock") == []


def test_gate_lookup(graph: NetlistGraph) -> None:
    gate = gate_lookup(graph, "G6")
    assert gate.gate_type == GateType.AND
    assert gate.pins["I0"] == NetBit("n2")
    assert gate.pins["I1"] == NetBit("n5")


# ---- Fanin / fanout (capabilities 7, 10, 11) ----


def test_direct_fanin_gate_and_dffq_sources(graph: NetlistGraph) -> None:
    entries = {e.pin: e for e in direct_fanin(graph, gate_lookup(graph, "G2"))}
    assert entries["I0"].source_kind == "gate"
    assert entries["I0"].source_gate.inst_name == "G1"
    assert entries["I1"].source_kind == "dff_q"
    assert entries["I1"].source_gate.inst_name == "Gdff"


def test_direct_fanin_pi_const_unconnected(graph: NetlistGraph) -> None:
    dff_entries = {e.pin: e for e in direct_fanin(graph, gate_lookup(graph, "Gdff"))}
    assert dff_entries["RN"].source_kind == "pi"
    assert dff_entries["SN"].source_kind == "const"
    assert dff_entries["SN"].value == Const.ONE
    assert dff_entries["CK"].source_kind == "pi"
    assert dff_entries["D"].source_kind == "gate"
    assert dff_entries["D"].source_gate.inst_name == "G8"

    g10_entries = {e.pin: e for e in direct_fanin(graph, gate_lookup(graph, "G10"))}
    assert g10_entries["I1"].source_kind == "unconnected"
    assert g10_entries["I1"].value is None


def test_direct_fanout_loads(graph: NetlistGraph) -> None:
    loads = direct_fanout(graph, NetBit("n1"))
    gate_pins = {(l.gate.inst_name, l.pin) for l in loads if l.kind == "gate"}
    assert gate_pins == {("G2", "I0"), ("G3", "I0"), ("G9", "I0")}
    assert not any(l.kind == "po" for l in loads)

    y_loads = direct_fanout(graph, NetBit("y"))
    assert len(y_loads) == 1
    assert y_loads[0].kind == "po"
    assert y_loads[0].port_name == "y"


def test_fanout_count_convention(graph: NetlistGraph) -> None:
    assert fanout_count(graph, NetBit("n1")) == 3
    assert fanout_count(graph, NetBit("b")) == 2
    # z is read by G2 (1 gate-pin load) AND wired to PO z (1 PO load) = 2.
    assert fanout_count(graph, NetBit("z")) == 2
    assert fanout_count(graph, NetBit("y")) == 1


def test_max_fanout_aggregates(graph: NetlistGraph) -> None:
    nb, count = max_fanout_overall(graph)
    assert (nb, count) == (NetBit("n1"), 3)

    nb, count = max_fanout_pi(graph)
    assert (nb, count) == (NetBit("b"), 2)


def test_cone_sizes(graph: NetlistGraph) -> None:
    # Per QA A94, a boundary DFF (one whose Q feeds a gate already in the
    # cone) counts as a gate in the fanin cone. G2 reads z (Gdff's Q), so
    # Gdff is a boundary DFF for both y's and n8's cones -- 7+1 and 3+1.
    assert fanin_cone_size(graph, NetBit("y")) == 8
    assert fanin_cone_size(graph, NetBit("n8")) == 4
    # A94's other clause: when the queried net IS itself a DFF's Q, the
    # cone is 1 (that DFF itself), not 0.
    assert fanin_cone_size(graph, NetBit("z")) == 1
    assert fanout_cone_size(graph, NetBit("a")) == 10


def test_largest_fanin_cone(graph: NetlistGraph) -> None:
    nb, size = largest_fanin_cone(graph)
    assert (nb, size) == (NetBit("y"), 8)


# ----------------------------------------------------------------------
# QA A94 (2026-08-27): a combinational signal's fanin cone must count the
# boundary DFF(s) its cone bottoms out at as gates in the cone -- built
# directly from the QA's own minimal example ("and g1 (X, q0, q1); where
# q0/q1 are the Q of DFF g2/g3 -> cone of X is 3 gates: AND:1, DFF:2"), not
# a number we derived ourselves.
# ----------------------------------------------------------------------


def _build_a94_example_design() -> Design:
    design = Design(module_name="top")
    for name in ("d0", "d1", "clk", "rn"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
        design.ports.append(Port(name=name, direction=Direction.INPUT))
    design.signals["x"] = Signal(name="x", msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports.append(Port(name="x", direction=Direction.OUTPUT))
    for name in ("q0", "q1"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)

    d0, d1, clk, rn = NetBit("d0"), NetBit("d1"), NetBit("clk"), NetBit("rn")
    q0, q1, x = NetBit("q0"), NetBit("q1"), NetBit("x")

    design.gates = [
        Gate("g2", GateType.DFF, {"RN": rn, "SN": Const.ONE, "CK": clk, "D": d0, "Q": q0}),
        Gate("g3", GateType.DFF, {"RN": rn, "SN": Const.ONE, "CK": clk, "D": d1, "Q": q1}),
        Gate("g1", GateType.AND, {"O": x, "I0": q0, "I1": q1}),
    ]
    design.build_indices()
    return design


def test_a94_fanin_cone_counts_boundary_dffs_qa_example() -> None:
    graph = NetlistGraph(_build_a94_example_design())
    assert fanin_cone_size(graph, NetBit("x")) == 3
    names = graph.backward_cone_with_boundary_dffs(NetBit("x"))
    assert names == {"g1", "g2", "g3"}


def test_a94_gate_type_breakdown_of_cone_qa_example() -> None:
    """QA A94's own worked example: "AND:1, DFF:2", total 3."""
    design = _build_a94_example_design()
    graph = NetlistGraph(design)
    names = graph.backward_cone_with_boundary_dffs(NetBit("x"))
    counts: dict[GateType, int] = {}
    for g in design.gates:
        if g.inst_name in names:
            counts[g.gate_type] = counts.get(g.gate_type, 0) + 1
    assert counts == {GateType.AND: 1, GateType.DFF: 2}


def test_a94_dff_q_itself_has_fanin_cone_size_one() -> None:
    """QA A94 point 3: "when X is itself a DFF's Q, the answer is 1 (that
    DFF itself), not 0"."""
    graph = NetlistGraph(_build_a94_example_design())
    assert fanin_cone_size(graph, NetBit("q0")) == 1
    assert fanin_cone_size(graph, NetBit("q1")) == 1


def _build_a94_shared_boundary_dff_design() -> Design:
    """One DFF (g2, Q=q0) feeds TWO different gates that are both inside the
    cone (g1 and g4), unlike `_build_a94_example_design` where every DFF
    feeds only a single gate. `backward_cone_with_boundary_dffs` collects
    boundary DFFs into a `set`, so g2 should still be counted once -- but no
    prior test exercises a DFF reached from more than one place in the cone,
    so a future change from `set` to an order-preserving `list` (e.g. to
    make cone contents deterministic) could start double-counting g2
    without any existing test going red."""
    design = Design(module_name="top")
    for name in ("d0", "d1", "d2", "clk", "rn"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
        design.ports.append(Port(name=name, direction=Direction.INPUT))
    design.signals["x"] = Signal(name="x", msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports.append(Port(name="x", direction=Direction.OUTPUT))
    for name in ("q0", "n1", "n4"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)

    d0, d1, d2, clk, rn = NetBit("d0"), NetBit("d1"), NetBit("d2"), NetBit("clk"), NetBit("rn")
    q0, n1, n4, x = NetBit("q0"), NetBit("n1"), NetBit("n4"), NetBit("x")

    design.gates = [
        Gate("g2", GateType.DFF, {"RN": rn, "SN": Const.ONE, "CK": clk, "D": d0, "Q": q0}),
        Gate("g1", GateType.AND, {"O": n1, "I0": q0, "I1": d1}),
        Gate("g4", GateType.OR, {"O": n4, "I0": q0, "I1": d2}),
        Gate("g5", GateType.AND, {"O": x, "I0": n1, "I1": n4}),
    ]
    design.build_indices()
    return design


def test_a94_shared_boundary_dff_counted_once_not_per_feed() -> None:
    """g2's Q (q0) feeds both g1 and g4 -- both already inside x's non-DFF
    cone -- so g2 is a boundary DFF reached twice, and must still appear
    exactly once in the cone (3 non-DFF gates + 1 DFF = 4), not twice (which
    a `list`-based, non-deduplicating cone would produce)."""
    graph = NetlistGraph(_build_a94_shared_boundary_dff_design())
    names = graph.backward_cone_with_boundary_dffs(NetBit("x"))
    assert names == {"g1", "g4", "g5", "g2"}
    assert len(names) == 4
    assert fanin_cone_size(graph, NetBit("x")) == 4


# ---- Depth (capabilities 12-16) ----


def test_max_design_depth(graph: NetlistGraph) -> None:
    assert graph.max_design_depth() == 6


def test_max_reg_to_reg_depth(graph: NetlistGraph) -> None:
    assert graph.max_reg_to_reg_depth() == 2


def test_max_pi_to_dff_d_depth(graph: NetlistGraph) -> None:
    assert graph.max_pi_to_dff_d_depth() == 3


def test_per_output_depths(graph: NetlistGraph) -> None:
    depths = graph.per_output_depths()
    assert depths[NetBit("y")] == 6
    assert depths[NetBit("z")] == 0
    assert depths[NetBit("n8")] == 3


def test_depth_through_gate_and_max_depth_path_judgment(graph: NetlistGraph) -> None:
    max_depth = graph.max_design_depth()
    assert max_depth == 6
    # G1 sits on the actual longest a->y path (G1,G3,G4,G5,G6,G7) -- see
    # test_depth_between_pi_and_po above.
    assert graph.depth_through_gate("G1") == max_depth
    # G2 reaches a sink (via G6/G7, or via G8/Gdff.D) but not along the
    # longest chain.
    assert graph.depth_through_gate("G2") == 4
    assert graph.depth_through_gate("G2") != max_depth
    # G9 is a dead end: it drives n9, which nothing else reads and which is
    # not a PO -- it reaches no true sink (PO/DFF.D) at all.
    assert graph.depth_through_gate("G9") is None


def test_depth_through_gate_sink_source_invariant(graph: NetlistGraph) -> None:
    """The milestone spec's own sanity invariant: for every gate that
    reaches a sink at all, dp_any + ext - 1 (the depth of the longest chain
    passing through it) never exceeds max_design_depth(), and at least one
    gate achieves equality."""
    dp = graph._dp_any_source()
    ext = graph._dp_to_sink_all()
    reaching = [n for n in dp if ext.get(n, 0) > 0]
    assert reaching
    assert max(dp[n] + ext[n] - 1 for n in reaching) == graph.max_design_depth()
    assert all(dp[n] + ext[n] - 1 <= graph.max_design_depth() for n in reaching)


def test_depth_between_pi_and_po(graph: NetlistGraph) -> None:
    result = graph.depth_between(NetBit("a"), NetBit("y"))
    assert result is not None
    depth, path = result
    assert depth == 6
    assert [g.inst_name for g in path] == ["G1", "G3", "G4", "G5", "G6", "G7"]


def test_depth_between_to_dff_pin(graph: NetlistGraph) -> None:
    result = graph.depth_between(NetBit("n2"), DffPin("Gdff", "D"))
    assert result is not None
    depth, path = result
    assert depth == 1
    assert [g.inst_name for g in path] == ["G8"]


def test_depth_between_direct_wire_is_zero(graph: NetlistGraph) -> None:
    result = graph.depth_between(NetBit("z"), NetBit("z"))
    assert result == (0, [])


def test_depth_between_unreachable_is_none(graph: NetlistGraph) -> None:
    assert graph.depth_between(NetBit("clk"), NetBit("y")) is None


# ---- Paths (capabilities 17-20) ----


def test_path_exists(graph: NetlistGraph) -> None:
    assert graph.path_exists(NetBit("a"), NetBit("y")) is True
    assert graph.path_exists(NetBit("a"), NetBit("y"), avoid=NetBit("n1")) is False
    assert graph.path_exists(NetBit("a"), NetBit("y"), avoid=NetBit("n2")) is True
    assert graph.path_exists(NetBit("a"), NetBit("y"), avoid=NetBit("n6")) is False


def test_path_exists_length_zero(graph: NetlistGraph) -> None:
    assert graph.path_exists(NetBit("z"), NetBit("z")) is True


def test_path_count_dp(graph: NetlistGraph) -> None:
    assert graph.path_count(NetBit("a"), NetBit("y")) == 2
    assert graph.path_count(NetBit("n2"), DffPin("Gdff", "D")) == 1
    assert graph.path_count(NetBit("z"), NetBit("z")) == 1


def test_path_enumeration_matches_hand_computed_set(graph: NetlistGraph) -> None:
    paths = list(graph.enumerate_paths(NetBit("a"), NetBit("y")))
    as_names = {tuple(g.inst_name for g in p) for p in paths}
    assert as_names == {
        ("G1", "G2", "G6", "G7"),
        ("G1", "G3", "G4", "G5", "G6", "G7"),
    }


def test_path_enumeration_length_zero(graph: NetlistGraph) -> None:
    paths = list(graph.enumerate_paths(NetBit("z"), NetBit("z")))
    assert paths == [[]]


# ---- Cut signals (capabilities 21, 22) ----


def test_cut_nets_between(graph: NetlistGraph) -> None:
    result = graph.cut_nets_between(NetBit("a"), NetBit("y"))
    assert result.path_exists is True
    assert set(result.cut_nets) == {NetBit("n1"), NetBit("n6")}
    # Per QA A87 ("articulation point" means a gate only, not a net): the
    # cone from a to y has a diamond (G2's branch and G3/G4/G5's branch
    # merging at G6), so most candidate gates in fwd&back (G2,G3,G4,G5,G8)
    # are NOT true cut points -- only G1 (drives n1) and G6 (drives n6) are.
    # This is the one assertion that would catch cut_gates being computed
    # as "every candidate", not "every candidate that IS actually a cut".
    assert set(result.cut_gates) == {"G1", "G6"}


def test_cut_nets_between_no_path(graph: NetlistGraph) -> None:
    result = graph.cut_nets_between(NetBit("clk"), NetBit("y"))
    assert result.path_exists is False
    assert result.cut_nets == []
    assert result.cut_gates == []


def test_cut_nets_between_source_equals_target(graph: NetlistGraph) -> None:
    """Degenerate case: source and target are the same net-bit -- a path
    trivially exists (length 0) and there is nothing to cut."""
    result = graph.cut_nets_between(NetBit("y"), NetBit("y"))
    assert result.path_exists is True
    assert result.cut_nets == []
    assert result.cut_gates == []


def test_is_cut_signal_for_some_pi_po_pair(graph: NetlistGraph) -> None:
    assert graph.is_cut_signal_for_some_pi_po_pair(NetBit("n6")) is True
    assert graph.is_cut_signal_for_some_pi_po_pair(NetBit("n1")) is True
    assert graph.is_cut_signal_for_some_pi_po_pair(NetBit("n2")) is False


# ---- Register-to-register combinational path counting (new capability) ----


def test_reg_to_reg_path_stats_synthetic(graph: NetlistGraph) -> None:
    # Exactly one reg-to-reg combinational path exists in the fixture:
    # Gdff.Q(z) -> G2 -> G8 -> Gdff.D(n8), a 2-gate chain -- matches the
    # existing test_path_count_dp assertion that path_count(n2, Gdff.D) == 1
    # (G8 is fed by n2, which G2 alone produces here).
    stats = graph.reg_to_reg_path_stats()
    assert stats.combinational_path_count == 1
    assert stats.direct_wire_count == 0
    assert stats.direct_wire_examples == []


def test_reg_to_reg_path_stats_direct_wire_injected() -> None:
    """Injected regression for the zero-gate direct-wire case: two DFFs
    whose Q is wired STRAIGHT into the other's D pin, with no combinational
    gate anywhere in the design at all. Must be counted in
    direct_wire_count/direct_wire_examples and NOT in
    combinational_path_count (the milestone spec's explicit convention)."""
    design = Design(module_name="top")
    for name in ("clk", "rn"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    for name in ("q1", "q2"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
    design.ports = [Port("clk", Direction.INPUT), Port("rn", Direction.INPUT)]
    design.gates = [
        Gate(
            "dffA",
            GateType.DFF,
            {"RN": NetBit("rn"), "SN": Const.ONE, "CK": NetBit("clk"), "D": NetBit("q2"), "Q": NetBit("q1")},
        ),
        Gate(
            "dffB",
            GateType.DFF,
            {"RN": NetBit("rn"), "SN": Const.ONE, "CK": NetBit("clk"), "D": NetBit("q1"), "Q": NetBit("q2")},
        ),
    ]
    design.build_indices()
    graph = NetlistGraph(design)
    stats = graph.reg_to_reg_path_stats()
    assert stats.combinational_path_count == 0
    assert stats.direct_wire_count == 2
    assert set(stats.direct_wire_examples) == {
        ("dffA", NetBit("q1"), "dffB"),
        ("dffB", NetBit("q2"), "dffA"),
    }


def test_reg_to_reg_seed_dedups_same_dff_q_on_two_pins() -> None:
    """A gate reading the SAME DFF's Q on two input pins seeds 1, not 2.

    A path is a distinct (source DFF, gate sequence, sink DFF) triple, so
    which pin the Q arrived on does not make a second path. Neither test32
    nor test37 contains a gate of this shape, so the two pinned corpus
    totals cannot observe this rule at all -- mutating the seed's set
    comprehension into a list survives the whole suite without this
    fixture.
    """
    design = Design(module_name="top")
    for name in ("clk", "rn"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    for name in ("q", "a"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
    design.ports = [Port("clk", Direction.INPUT), Port("rn", Direction.INPUT)]
    design.gates = [
        Gate(
            "dffA",
            GateType.DFF,
            {"RN": NetBit("rn"), "SN": Const.ONE, "CK": NetBit("clk"), "D": NetBit("a"), "Q": NetBit("q")},
        ),
        # Both AND inputs come from dffA.Q -- one source DFF, so one path.
        Gate("g1", GateType.AND, {"A": NetBit("q"), "B": NetBit("q"), "O": NetBit("a")}),
    ]
    design.build_indices()
    stats = NetlistGraph(design).reg_to_reg_path_stats()
    assert stats.combinational_path_count == 1
    assert stats.direct_wire_count == 0


def test_reg_to_reg_path_count_multiplies_by_dff_d_fanout() -> None:
    """F1's `dff_d_count` risk scenario: one gate's output net-bit is read by
    TWO different DFFs' D pins (`dff_d_count[out_nb]` must be 2, not 1 or a
    dedup-to-1). One source DFF (dffA) feeds g0, whose single output fans
    out to dffB.D and dffC.D -- exactly 2 distinct (source, chain, sink)
    reg-to-reg paths, not 1. Neither test32 nor test37 (the corpus's own
    max D-net reuse count is 1, per the milestone diagnosis) contains a gate
    of this shape, so this fixture is the only thing in the suite that would
    catch `dff_d_count[d] = dff_d_count.get(d, 0) + 1` regressing to
    `dff_d_count[d] = 1`."""
    design = Design(module_name="top")
    for name in ("clk", "rn"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    for name in ("qa", "db"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
    design.ports = [Port("clk", Direction.INPUT), Port("rn", Direction.INPUT)]
    design.gates = [
        Gate(
            "dffA",
            GateType.DFF,
            {"RN": NetBit("rn"), "SN": Const.ONE, "CK": NetBit("clk"), "D": Const.ZERO, "Q": NetBit("qa")},
        ),
        Gate("g0", GateType.BUF, {"O": NetBit("db"), "I0": NetBit("qa")}),
        # Both dffB and dffC read g0's SAME output net-bit on their D pin --
        # two distinct sink DFFs sharing one driver, the multiplier case.
        Gate(
            "dffB",
            GateType.DFF,
            {"RN": NetBit("rn"), "SN": Const.ONE, "CK": NetBit("clk"), "D": NetBit("db"), "Q": NetBit("qb")},
        ),
        Gate(
            "dffC",
            GateType.DFF,
            {"RN": NetBit("rn"), "SN": Const.ONE, "CK": NetBit("clk"), "D": NetBit("db"), "Q": NetBit("qc")},
        ),
    ]
    for name in ("qb", "qc"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
    design.build_indices()
    stats = NetlistGraph(design).reg_to_reg_path_stats()
    assert stats.combinational_path_count == 2
    assert stats.direct_wire_count == 0


def test_is_cut_signal_convenience(graph: NetlistGraph) -> None:
    assert is_cut_signal(graph, "n6") is True
    assert is_cut_signal(graph, "n2") is False


# ---- Cycle detection ----


def test_combinational_cycle_is_detected() -> None:
    design = Design(module_name="cyc")
    design.signals["a"] = Signal(name="a", msb=None, lsb=None, direction=Direction.INPUT)
    design.ports.append(Port(name="a", direction=Direction.INPUT))
    for n in ("n1", "n2"):
        design.signals[n] = Signal(name=n, msb=None, lsb=None, direction=Direction.INTERNAL)
    a = NetBit("a")
    n1, n2 = NetBit("n1"), NetBit("n2")
    # g1's output feeds g2, and g2's output feeds back into g1: a genuine
    # (unrealistic, but IR-legal) combinational cycle.
    design.gates = [
        Gate("g1", GateType.AND, {"O": n1, "I0": a, "I1": n2}),
        Gate("g2", GateType.BUF, {"O": n2, "I0": n1}),
    ]
    design.build_indices()
    graph = NetlistGraph(design)
    with pytest.raises(CombinationalCycleError):
        graph.max_design_depth()


def test_enumerate_paths_detects_combinational_cycle() -> None:
    """Regression test: `enumerate_paths` must raise CombinationalCycleError
    on a cyclic reachable subgraph the same way `path_count` already does,
    instead of backtracking-DFS-ing forever with no cycle guard.
    """
    design = Design(module_name="cyc")
    design.signals["a"] = Signal(name="a", msb=None, lsb=None, direction=Direction.INPUT)
    design.ports.append(Port(name="a", direction=Direction.INPUT))
    for n in ("n1", "n2"):
        design.signals[n] = Signal(name=n, msb=None, lsb=None, direction=Direction.INTERNAL)
    a = NetBit("a")
    n1, n2 = NetBit("n1"), NetBit("n2")
    # Same fixture as test_combinational_cycle_is_detected above: g1's output
    # feeds g2, and g2's output feeds back into g1.
    design.gates = [
        Gate("g1", GateType.AND, {"O": n1, "I0": a, "I1": n2}),
        Gate("g2", GateType.BUF, {"O": n2, "I0": n1}),
    ]
    design.build_indices()
    graph = NetlistGraph(design)
    with pytest.raises(CombinationalCycleError):
        graph.path_count(a, n2)
    with pytest.raises(CombinationalCycleError):
        list(graph.enumerate_paths(a, n2))


def test_enumerate_paths_prunes_dead_end_subgraph() -> None:
    """Regression test for a bug observed on the real 91k-gate test12.v:
    `enumerate_paths` pruned only forward-reachability from the source, so
    the backtracking DFS still walked every path inside subtrees that could
    never reach the target -- exponentially more work than the true result
    set (>10 CPU-minutes without yielding a single path, on a pair whose
    answer the DP counts instantly).

    Fixture: source `a` feeds one direct BUF to the target, plus a 32-stage
    diamond ladder (2 parallel BUFs + a merge AND per stage; ~2**32 distinct
    walks) that dead-ends without ever reaching the target. With backward
    pruning the ladder is excluded outright and enumeration is instant; if
    the pruning regresses, this test hangs rather than passes.
    """
    design = Design(module_name="deadend")
    design.signals["a"] = Signal(name="a", msb=None, lsb=None, direction=Direction.INPUT)
    design.ports.append(Port(name="a", direction=Direction.INPUT))

    def wire(name: str) -> NetBit:
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
        return NetBit(name)

    a = NetBit("a")
    s0 = wire("s0")
    t = wire("t")
    design.gates = [
        Gate("g_start", GateType.BUF, {"O": s0, "I0": a}),
        Gate("g_t", GateType.BUF, {"O": t, "I0": s0}),
    ]
    prev = s0
    for i in range(32):
        x0, x1, d = wire(f"x{i}_0"), wire(f"x{i}_1"), wire(f"d{i}")
        design.gates.append(Gate(f"b{i}_0", GateType.BUF, {"O": x0, "I0": prev}))
        design.gates.append(Gate(f"b{i}_1", GateType.BUF, {"O": x1, "I0": prev}))
        design.gates.append(Gate(f"m{i}", GateType.AND, {"O": d, "I0": x0, "I1": x1}))
        prev = d
    design.build_indices()
    graph = NetlistGraph(design)

    paths = list(graph.enumerate_paths(a, t))
    assert len(paths) == 1
    assert [g.inst_name for g in paths[0]] == ["g_start", "g_t"]
    assert graph.path_count(a, t) == 1


# ----------------------------------------------------------------------
# Smoke tests over all 40 real fixtures (capability requirement 2)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=[os.path.basename(p) for p in FIXTURE_PATHS])
def test_real_fixture_smoke(path: str) -> None:
    design = parse_verilog(path)
    graph = NetlistGraph(design)

    counts = gate_count_by_type(design)
    assert sum(counts.values()) == len(design.gates)

    max_depth = graph.max_design_depth()
    assert max_depth >= 0
    reg2reg = graph.max_reg_to_reg_depth()
    assert reg2reg >= 0
    pi2d = graph.max_pi_to_dff_d_depth()
    assert pi2d >= 0

    pos = list_primary_outputs(design)
    for po in pos[:2]:
        sig = design.signals[po.name]
        for bit in sig.bits()[:1]:
            size = fanin_cone_size(graph, bit)
            assert size >= 0


# ----------------------------------------------------------------------
# Performance check (capability requirement 3): test39/test12/test33
# ----------------------------------------------------------------------


def _endpoints_of_longest_chain(graph: NetlistGraph) -> tuple[NetBit, NetBit]:
    """Test-only diagnostic: reconstructs the actual source/target net-bits
    achieving `graph.max_design_depth()`, by walking the memoized `dp_any`
    table backward through `_pred`. Reaches into NetlistGraph's private
    precomputed tables (acceptable for a test helper) purely so the
    performance check below can run path_count on a pair that is *known* to
    be maximally connected, rather than guessing at a PI/PO pair that may
    turn out unrelated (heavily pipelined designs can have PI/PO pairs with
    zero connecting paths even when each side individually has a huge cone).
    """
    dp = graph._dp_any_source()
    sinks = graph._sink_gates_po | graph._sink_gates_dff
    best_gate_name = max(sinks, key=lambda n: dp[n])
    chain = [best_gate_name]
    cur = best_gate_name
    while True:
        pred = next((p for p in graph._pred.get(cur, ()) if dp.get(p) == dp[cur] - 1), None)
        if pred is None:
            break
        chain.append(pred)
        cur = pred
    chain.reverse()

    first_gate = graph.gate_by_name[chain[0]]
    out_pin = OUTPUT_PIN[first_gate.gate_type]
    source_nb = None
    for pin_name, value in first_gate.pins.items():
        if pin_name == out_pin or not isinstance(value, NetBit):
            continue
        driver = graph.design.net_driver.get(value)
        if driver is None or driver.gate_type == GateType.DFF:
            source_nb = value
            break
    last_gate = graph.gate_by_name[chain[-1]]
    target_nb = last_gate.pins[OUTPUT_PIN[last_gate.gate_type]]
    return source_nb, target_nb


@pytest.mark.parametrize("case", ["test39", "test12", "test33"])
@pytest.mark.skipif(not corpus_available(), reason="Alpha_Testcase corpus not present -- put the released testcases under Alpha_Testcase/testcase/testNN/ to enable this test")
def test_large_fixture_performance(case: str) -> None:
    path = os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase", case, f"{case}.v")
    design = parse_verilog(path)

    start = time.perf_counter()
    graph = NetlistGraph(design)
    max_depth = graph.max_design_depth()
    reg2reg = graph.max_reg_to_reg_depth()
    pi2d = graph.max_pi_to_dff_d_depth()
    depth_elapsed = time.perf_counter() - start

    source_nb, target_nb = _endpoints_of_longest_chain(graph)

    start = time.perf_counter()
    count = graph.path_count(source_nb, target_nb)
    path_count_elapsed = time.perf_counter() - start

    print(
        f"\n{case}.v: gates={len(design.gates)} max_depth={max_depth} "
        f"reg2reg={reg2reg} pi_to_dff_d={pi2d} "
        f"depth_computation={depth_elapsed:.3f}s "
        f"path_count({source_nb}->{target_nb})={count} path_count_dp={path_count_elapsed:.3f}s"
    )
    assert depth_elapsed < 60
    assert path_count_elapsed < 60


# ----------------------------------------------------------------------
# Corpus-verified exact numbers (new capabilities: max-depth-path membership
# and whole-design register-to-register path counting). Unlike the smoke/
# perf checks above (which only assert loose invariants across all 40
# fixtures), these pin down the EXACT figures independently confirmed
# against test32/test37's real corpus content -- a stub/placeholder
# implementation would not reproduce these.
# ----------------------------------------------------------------------


def _real_design(case: str) -> Design:
    _p = os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase")
    if not os.path.isdir(_p):
        pytest.skip("Alpha_Testcase corpus not present -- put the released testcases under Alpha_Testcase/testcase/testNN/ to enable this test")
    return parse_verilog(os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase", case, f"{case}.v"))


def test_gate_on_max_depth_path_test32_g0() -> None:
    graph = NetlistGraph(_real_design("test32"))
    assert graph.max_design_depth() == 58
    assert graph._dp_any_source()["g0"] == 15
    assert graph._dp_to_sink_all()["g0"] == 1
    assert graph.depth_through_gate("g0") == 15
    assert graph.depth_through_gate("g0") != graph.max_design_depth()


def test_reg_to_reg_path_stats_test32() -> None:
    graph = NetlistGraph(_real_design("test32"))
    stats = graph.reg_to_reg_path_stats()
    assert stats.combinational_path_count == 1176516
    assert stats.direct_wire_count == 7
    assert set(stats.direct_wire_examples) == {
        ("g908", NetBit("n1064"), "g925"),
        ("g991", NetBit("n1199"), "g992"),
        ("g993", NetBit("n1201"), "g994"),
        ("g995", NetBit("n39", 0), "g999"),
        ("g996", NetBit("n39", 1), "g1000"),
        ("g997", NetBit("n39", 2), "g1001"),
        ("g998", NetBit("n39", 3), "g1002"),
    }


def test_reg_to_reg_path_stats_test37() -> None:
    graph = NetlistGraph(_real_design("test37"))
    stats = graph.reg_to_reg_path_stats()
    assert stats.combinational_path_count == 16548172
    assert stats.direct_wire_count == 0
    assert stats.direct_wire_examples == []
