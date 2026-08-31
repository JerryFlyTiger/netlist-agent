"""Structural checks for the whole-netlist sweeps and the ir.py mutation
helpers, using small hand-built synthetic Designs where the expected
structural outcome is known exactly.
"""

from __future__ import annotations

from netlist_agent.analysis import fanout_count, iter_fanout_counts
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, Port, Signal
from netlist_agent.transform import (
    collapse_double_inverters,
    collapse_inverter_buffer_chains,
    deduplicate_gates,
    insert_buffer_per_load,
    limit_fanout,
    limit_fanout_net,
    remove_dangling_gates,
)


def _sig(design: Design, name: str, direction: Direction) -> None:
    design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=direction)
    if direction in (Direction.INPUT, Direction.OUTPUT):
        design.ports.append(Port(name=name, direction=direction))


def _nb(name: str) -> NetBit:
    return NetBit(name, None)


# ----------------------------------------------------------------------
# ir.py mutation helpers
# ----------------------------------------------------------------------


def test_fresh_net_and_gate_name_no_collision() -> None:
    design = Design(module_name="t")
    _sig(design, "n1", Direction.INPUT)
    design.signals["t_net_3"] = Signal(name="t_net_3", msb=None, lsb=None, direction=Direction.INTERNAL)
    design.gates.append(Gate(inst_name="t_gate_5", gate_type=GateType.BUF, pins={"O": _nb("n1"), "I0": _nb("n1")}))
    design.build_indices()

    n1 = design.fresh_net()
    n2 = design.fresh_net()
    assert n1.name not in ("t_net_3",)
    assert n1 != n2
    assert n1.name in design.signals and n2.name in design.signals

    g1 = design.fresh_gate_name()
    g2 = design.fresh_gate_name()
    assert g1 != "t_gate_5" and g2 != "t_gate_5"
    assert g1 != g2


def test_add_remove_gate_updates_indices() -> None:
    design = Design(module_name="t")
    _sig(design, "a", Direction.INPUT)
    _sig(design, "y", Direction.OUTPUT)
    design.build_indices()

    gate = Gate(inst_name="g0", gate_type=GateType.BUF, pins={"O": _nb("y"), "I0": _nb("a")})
    design.add_gate(gate)
    assert design.net_driver[_nb("y")] is gate
    assert gate in design.net_fanout[_nb("a")]

    design.remove_gate(gate)
    assert _nb("y") not in design.net_driver
    assert _nb("a") not in design.net_fanout or gate not in design.net_fanout.get(_nb("a"), [])
    assert gate not in design.gates


def test_rename_signal_rewrites_everywhere() -> None:
    design = Design(module_name="t")
    _sig(design, "n1214", Direction.INPUT)
    _sig(design, "y", Direction.OUTPUT)
    design.gates.append(Gate(inst_name="g0", gate_type=GateType.BUF, pins={"O": _nb("y"), "I0": _nb("n1214")}))
    design.build_indices()

    design.rename_signal("n1214", "renamed_wire")

    assert "n1214" not in design.signals
    assert design.signals["renamed_wire"].name == "renamed_wire"
    assert design.gates[0].pins["I0"] == _nb("renamed_wire")
    assert [p.name for p in design.ports] == ["renamed_wire", "y"]
    assert design.net_fanout[_nb("renamed_wire")] == [design.gates[0]]


# ----------------------------------------------------------------------
# 1. Back-to-back inverter collapsing
# ----------------------------------------------------------------------


def test_collapse_simple_chain() -> None:
    design = Design(module_name="t")
    _sig(design, "a", Direction.INPUT)
    _sig(design, "y", Direction.OUTPUT)
    _sig(design, "mid", Direction.INTERNAL)
    _sig(design, "z", Direction.INTERNAL)
    design.gates.append(Gate("g1", GateType.NOT, {"O": _nb("mid"), "I0": _nb("a")}))
    design.gates.append(Gate("g2", GateType.NOT, {"O": _nb("z"), "I0": _nb("mid")}))
    design.gates.append(Gate("g3", GateType.BUF, {"O": _nb("y"), "I0": _nb("z")}))
    design.build_indices()

    collapsed = collapse_double_inverters(design)
    assert collapsed == 1
    # g2 (whose output "z" is NOT a PO) is spliced out entirely; its
    # consumer (g3) is reconnected straight to what fed g1 (`a`).
    remaining = {g.inst_name: g for g in design.gates}
    assert "g2" not in remaining
    assert remaining["g3"].pins["I0"] == _nb("a")
    # g1 still exists (nothing removes a NOT with no remaining consumers --
    # that is dangling-removal's job, a separate transform).
    assert "g1" in remaining


def test_collapse_preserves_first_not_other_fanout() -> None:
    design = Design(module_name="t")
    _sig(design, "a", Direction.INPUT)
    _sig(design, "y1", Direction.OUTPUT)
    _sig(design, "y2", Direction.OUTPUT)
    _sig(design, "mid", Direction.INTERNAL)
    _sig(design, "z", Direction.INTERNAL)
    design.gates.append(Gate("g1", GateType.NOT, {"O": _nb("mid"), "I0": _nb("a")}))
    design.gates.append(Gate("g2", GateType.NOT, {"O": _nb("z"), "I0": _nb("mid")}))
    design.gates.append(Gate("g4", GateType.BUF, {"O": _nb("y1"), "I0": _nb("z")}))
    # g1's output also feeds a second, unrelated consumer -- must be untouched.
    design.gates.append(Gate("g3", GateType.BUF, {"O": _nb("y2"), "I0": _nb("mid")}))
    design.build_indices()

    collapsed = collapse_double_inverters(design)
    assert collapsed == 1
    remaining = {g.inst_name: g for g in design.gates}
    assert "g2" not in remaining
    assert remaining["g4"].pins["I0"] == _nb("a")  # spliced straight to what fed g1
    assert remaining["g3"].pins["I0"] == _nb("mid")  # g3 untouched


def test_collapse_output_is_primary_output_keeps_driver() -> None:
    design = Design(module_name="t")
    _sig(design, "a", Direction.INPUT)
    _sig(design, "y", Direction.OUTPUT)
    _sig(design, "mid", Direction.INTERNAL)
    design.gates.append(Gate("g1", GateType.NOT, {"O": _nb("mid"), "I0": _nb("a")}))
    design.gates.append(Gate("g2", GateType.NOT, {"O": _nb("y"), "I0": _nb("mid")}))
    design.build_indices()

    collapsed = collapse_double_inverters(design)
    assert collapsed == 1
    # y must still be driven (PO identity can't be dropped): g2 survives as a BUF.
    driver = design.net_driver[_nb("y")]
    assert driver.gate_type == GateType.BUF
    assert driver.pins["I0"] == _nb("a")


# ----------------------------------------------------------------------
# 2. Dangling gate removal
# ----------------------------------------------------------------------


def test_dangling_gate_removed_reachable_survives() -> None:
    design = Design(module_name="t")
    _sig(design, "a", Direction.INPUT)
    _sig(design, "y", Direction.OUTPUT)
    _sig(design, "unused", Direction.INTERNAL)
    design.gates.append(Gate("g_live", GateType.BUF, {"O": _nb("y"), "I0": _nb("a")}))
    design.gates.append(Gate("g_dead", GateType.NOT, {"O": _nb("unused"), "I0": _nb("a")}))
    design.build_indices()

    removed = remove_dangling_gates(design)
    assert removed == 1
    remaining = {g.inst_name for g in design.gates}
    assert remaining == {"g_live"}


def test_dangling_removal_respects_dff_boundary() -> None:
    design = Design(module_name="t")
    _sig(design, "a", Direction.INPUT)
    _sig(design, "clk", Direction.INPUT)
    _sig(design, "d", Direction.INTERNAL)
    design.gates.append(Gate("g_feed_d", GateType.BUF, {"O": _nb("d"), "I0": _nb("a")}))
    design.gates.append(
        Gate(
            "dff0",
            GateType.DFF,
            {"RN": Const.ONE, "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("d"), "Q": _nb("q_out")},
        )
    )
    _sig(design, "q_out", Direction.INTERNAL)
    design.build_indices()

    removed = remove_dangling_gates(design)
    assert removed == 0  # feeds a DFF's D pin -> not dangling; DFF itself never removed
    assert {g.inst_name for g in design.gates} == {"g_feed_d", "dff0"}


# ----------------------------------------------------------------------
# 3. Structural gate deduplication
# ----------------------------------------------------------------------


def test_dedup_merges_identical_gates_and_redirects_consumers() -> None:
    design = Design(module_name="t")
    _sig(design, "a", Direction.INPUT)
    _sig(design, "b", Direction.INPUT)
    _sig(design, "y1", Direction.OUTPUT)
    _sig(design, "y2", Direction.OUTPUT)
    _sig(design, "x1", Direction.INTERNAL)
    _sig(design, "x2", Direction.INTERNAL)
    design.gates.append(Gate("g1", GateType.AND, {"O": _nb("x1"), "I0": _nb("a"), "I1": _nb("b")}))
    # g2 is a duplicate of g1 (same type, same inputs, commutative order swapped).
    design.gates.append(Gate("g2", GateType.AND, {"O": _nb("x2"), "I0": _nb("b"), "I1": _nb("a")}))
    design.gates.append(Gate("c1", GateType.BUF, {"O": _nb("y1"), "I0": _nb("x1")}))
    design.gates.append(Gate("c2", GateType.BUF, {"O": _nb("y2"), "I0": _nb("x2")}))
    design.build_indices()

    merged = deduplicate_gates(design)
    assert merged == 1
    remaining = {g.inst_name for g in design.gates}
    assert len(remaining & {"g1", "g2"}) == 1  # exactly one of the two AND gates survives
    survivor_name = (remaining & {"g1", "g2"}).pop()
    survivor = next(g for g in design.gates if g.inst_name == survivor_name)
    survivor_out = survivor.pins["O"]
    # both consumers must now point at the surviving gate's output.
    c1 = next(g for g in design.gates if g.inst_name == "c1")
    c2 = next(g for g in design.gates if g.inst_name == "c2")
    assert c1.pins["I0"] == survivor_out
    assert c2.pins["I0"] == survivor_out


def test_dedup_keeps_both_when_each_drives_a_distinct_po() -> None:
    design = Design(module_name="t")
    _sig(design, "a", Direction.INPUT)
    _sig(design, "b", Direction.INPUT)
    _sig(design, "y1", Direction.OUTPUT)
    _sig(design, "y2", Direction.OUTPUT)
    design.gates.append(Gate("g1", GateType.AND, {"O": _nb("y1"), "I0": _nb("a"), "I1": _nb("b")}))
    design.gates.append(Gate("g2", GateType.AND, {"O": _nb("y2"), "I0": _nb("a"), "I1": _nb("b")}))
    design.build_indices()

    merged = deduplicate_gates(design)
    assert merged == 0
    assert {g.inst_name for g in design.gates} == {"g1", "g2"}


def test_dedup_non_po_duplicate_merges_into_po_driver() -> None:
    design = Design(module_name="t")
    _sig(design, "a", Direction.INPUT)
    _sig(design, "b", Direction.INPUT)
    _sig(design, "y", Direction.OUTPUT)
    _sig(design, "x", Direction.INTERNAL)
    design.gates.append(Gate("g_po", GateType.AND, {"O": _nb("y"), "I0": _nb("a"), "I1": _nb("b")}))
    design.gates.append(Gate("g_dup", GateType.AND, {"O": _nb("x"), "I0": _nb("a"), "I1": _nb("b")}))
    design.gates.append(Gate("c", GateType.BUF, {"O": _nb("consumed"), "I0": _nb("x")}))
    _sig(design, "consumed", Direction.INTERNAL)
    design.build_indices()

    merged = deduplicate_gates(design)
    assert merged == 1
    remaining = {g.inst_name for g in design.gates}
    assert "g_po" in remaining and "g_dup" not in remaining
    c = next(g for g in design.gates if g.inst_name == "c")
    assert c.pins["I0"] == _nb("y")


# ----------------------------------------------------------------------
# 6. Buffer insertion for fanout balancing
# ----------------------------------------------------------------------


def _fanout_design(n_loads: int) -> Design:
    design = Design(module_name="t")
    _sig(design, "src", Direction.INPUT)
    for i in range(n_loads):
        out = f"y{i}"
        _sig(design, out, Direction.OUTPUT)
        design.gates.append(Gate(f"c{i}", GateType.BUF, {"O": _nb(out), "I0": _nb("src")}))
    design.build_indices()
    return design


def test_limit_fanout_splits_into_tree_under_bound() -> None:
    design = _fanout_design(10)
    added = limit_fanout(design, max_fanout=4)
    assert added > 0
    graph = NetlistGraph(design)
    for nb, count in iter_fanout_counts(graph):
        assert count <= 4, f"{nb} still has fanout {count}"
    # functionality preserved: every original consumer must still trace back to `src`
    # (directly or through a chain of BUFs).
    for i in range(10):
        c = next(g for g in design.gates if g.inst_name == f"c{i}")
        cur = c.pins["I0"]
        seen = 0
        while cur != _nb("src"):
            driver = design.net_driver[cur]
            assert driver.gate_type == GateType.BUF
            cur = driver.pins["I0"]
            seen += 1
            assert seen < 10  # sanity bound against an infinite loop
        assert cur == _nb("src")


def test_limit_fanout_noop_when_within_bound() -> None:
    design = _fanout_design(3)
    added = limit_fanout(design, max_fanout=4)
    assert added == 0


def test_limit_fanout_net_scopes_to_named_net() -> None:
    design = _fanout_design(6)
    _sig(design, "other", Direction.INPUT)
    _sig(design, "z0", Direction.OUTPUT)
    design.gates.append(Gate("cz0", GateType.BUF, {"O": _nb("z0"), "I0": _nb("other")}))
    design.build_indices()

    added = limit_fanout_net(design, "src", max_fanout=2)
    assert added > 0
    graph = NetlistGraph(design)
    assert fanout_count(graph, _nb("src")) <= 2
    # "other" was untouched -- limit_fanout_net only touches the named net.
    assert fanout_count(graph, _nb("other")) == 1


def test_insert_buffer_per_load_one_to_one() -> None:
    design = _fanout_design(4)
    added = insert_buffer_per_load(design, "src")
    assert added == 4
    graph = NetlistGraph(design)
    assert fanout_count(graph, _nb("src")) == 4  # src now only drives the 4 new BUFs
    for i in range(4):
        c = next(g for g in design.gates if g.inst_name == f"c{i}")
        driver = design.net_driver[c.pins["I0"]]
        assert driver.gate_type == GateType.BUF
        assert driver.pins["I0"] == _nb("src")
    # each load gets its OWN dedicated buffer (not a shared one).
    drivers = {design.net_driver[next(g for g in design.gates if g.inst_name == f"c{i}").pins["I0"]].inst_name for i in range(4)}
    assert len(drivers) == 4


# ----------------------------------------------------------------------
# NOT -> BUF collapsing (spec 4.3: "Replace all inverters followed by
# buffers with a single inverter"). The interesting cases are all on the
# primary-output boundary, where the BUF cannot simply be deleted.
# ----------------------------------------------------------------------


def _not_buf_design(extra_load: bool = False) -> Design:
    """NOT drives one (or two) BUFs that each drive a primary output."""
    design = Design(module_name="t")
    _sig(design, "x", Direction.INPUT)
    _sig(design, "po1", Direction.OUTPUT)
    _sig(design, "po2", Direction.OUTPUT)
    _sig(design, "n", Direction.INTERNAL)
    design.gates.append(Gate("NOT1", GateType.NOT, {"O": _nb("n"), "I0": _nb("x")}))
    design.gates.append(Gate("BUF1", GateType.BUF, {"O": _nb("po1"), "I0": _nb("n")}))
    if extra_load:
        _sig(design, "m", Direction.INTERNAL)
        design.gates.append(Gate("AND1", GateType.AND, {"O": _nb("m"), "I0": _nb("n"), "I1": _nb("x")}))
        design.gates.append(Gate("BUF2", GateType.BUF, {"O": _nb("po2"), "I0": _nb("m")}))
    else:
        design.gates.append(Gate("BUF2", GateType.BUF, {"O": _nb("po2"), "I0": _nb("n")}))
    design.build_indices()
    return design


def test_collapse_inverter_buffer_sweeps_the_now_dead_inverter() -> None:
    """A BUF driving a primary output is retyped to NOT rather than deleted
    (the PO's net identity is pinned). That leaves the original NOT driving
    nothing -- it must be swept, or the request for "a single inverter" is
    answered with two, one of them dead weight in every later gate count."""
    design = _not_buf_design()
    assert len(design.gates) == 3

    collapsed = collapse_inverter_buffer_chains(design)

    assert collapsed == 2
    types = {g.inst_name: g.gate_type for g in design.gates}
    assert types == {"BUF1": GateType.NOT, "BUF2": GateType.NOT}  # NOT1 swept
    # Both now invert the original NOT's *input* directly.
    for g in design.gates:
        assert g.pins["I0"] == _nb("x")


def test_collapse_inverter_buffer_keeps_an_inverter_that_still_has_a_load() -> None:
    """The mirror image: the original NOT must NOT be swept while anything
    else still reads its output, or the rewrite silently breaks the design."""
    design = _not_buf_design(extra_load=True)

    collapsed = collapse_inverter_buffer_chains(design)

    assert collapsed == 1  # only BUF1 (BUF2's driver is the AND, not a NOT)
    types = {g.inst_name: g.gate_type for g in design.gates}
    assert types["NOT1"] == GateType.NOT  # kept: AND1 still reads net "n"
    assert types["BUF1"] == GateType.NOT
    assert types["BUF2"] == GateType.BUF
    assert design.net_fanout[_nb("n")] == [design.net_driver[_nb("m")]]


def test_collapse_inverter_buffer_multi_level_chain_ends_as_one_inverter() -> None:
    """NOT -> BUF -> BUF collapses all the way down, not just one level."""
    design = Design(module_name="t")
    _sig(design, "x", Direction.INPUT)
    _sig(design, "po1", Direction.OUTPUT)
    _sig(design, "n", Direction.INTERNAL)
    _sig(design, "m", Direction.INTERNAL)
    design.gates.append(Gate("NOT1", GateType.NOT, {"O": _nb("n"), "I0": _nb("x")}))
    design.gates.append(Gate("BUF1", GateType.BUF, {"O": _nb("m"), "I0": _nb("n")}))
    design.gates.append(Gate("BUF2", GateType.BUF, {"O": _nb("po1"), "I0": _nb("m")}))
    design.build_indices()

    collapsed = collapse_inverter_buffer_chains(design)

    assert collapsed == 2
    assert [(g.inst_name, g.gate_type) for g in design.gates] == [("BUF2", GateType.NOT)]
    assert design.gates[0].pins["I0"] == _nb("x")
