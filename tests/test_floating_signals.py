"""Unit tests for `netlist_agent.analysis.find_floating_signals` (new
capability: floating inputs / unconnected output ports).

The real corpus's own ground truth for this question (test37) is 0 across
every sub-check, so a "always answer 0" stub implementation would pass a
corpus-only check trivially -- every one of the five sub-checks below
therefore has its OWN injected-defect test that must catch a specific,
deliberately introduced structural problem. See analysis.py's
`FloatingSignalsResult` docstring for exactly what each sub-check means.
"""

from __future__ import annotations

from tests.helpers import corpus_available

import pytest

import os

from netlist_agent.analysis import find_floating_signals
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import Design, Direction, Gate, GateType, NetBit, Port, Signal
from netlist_agent.parser import parse_verilog

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pi(design: Design, name: str) -> None:
    design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)


def _po(design: Design, name: str) -> None:
    design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.OUTPUT)


def _wire(design: Design, name: str) -> None:
    design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)


def _build_design_with_every_defect() -> Design:
    """One design deliberately carrying exactly one instance of each of the
    five sub-check defects, plus a PI-directly-wired-to-PO passthrough (a
    NORMAL, non-defective construct -- see the "PI directly to PO" doc note
    below) and a dead internal wire (the sixth, non-headline observation).
    """
    design = Design(module_name="top")
    _pi(design, "used_in")  # consumed normally -- not a defect
    _pi(design, "unused_in")  # #2: declared PI, never consumed by anything
    _pi(design, "pass_in")  # PI directly wired through to a PO (see below)
    _po(design, "out_ok")  # driven normally -- not a defect
    _po(design, "out_dangling")  # #3: declared PO, nothing drives it
    _wire(design, "float_net")  # #1: read as a gate input, never driven, not a PI
    _wire(design, "dangle_net")  # #5: gate output, nothing consumes it, not a PO
    _wire(design, "dead_net")  # observation only: neither driven nor read at all

    design.ports = [
        Port("used_in", Direction.INPUT),
        Port("unused_in", Direction.INPUT),
        Port("pass_in", Direction.INPUT),
        # `pass_in` is ALSO declared as an output port -- the IR represents a
        # PI-directly-wired-to-a-PO net-bit (a zero-length path, see
        # `_h_zero_length_paths`/`graph.pi_bits & graph.po_bits`) this way:
        # two Port entries sharing one Signal/net-bit identity, one INPUT
        # and one OUTPUT. `declared_input_ports_completely_unused` (#2) must
        # exclude this net-bit (it IS used, by the PO connection) per the
        # milestone spec; `unconnected_output_ports_undriven` (#3) has no
        # such exclusion documented in the spec, so `pass_in` legitimately
        # also shows up there (nothing in `net_driver` drives it -- it's
        # driven by a PI, not a gate) -- asserted explicitly below so this
        # is a documented choice, not an accidental gap.
        Port("pass_in", Direction.OUTPUT),
        Port("out_ok", Direction.OUTPUT),
        Port("out_dangling", Direction.OUTPUT),
    ]

    design.gates = [
        Gate(
            "g1",
            GateType.AND,
            {"O": NetBit("out_ok"), "I0": NetBit("used_in"), "I1": NetBit("float_net")},
        ),
        Gate(
            "g2",
            GateType.AND,
            {"O": NetBit("dangle_net"), "I0": NetBit("used_in"), "I1": None},
        ),
    ]
    design.build_indices()
    return design


def test_check1_floating_input_net_referenced_but_undriven() -> None:
    graph = NetlistGraph(_build_design_with_every_defect())
    res = find_floating_signals(graph)
    assert res.floating_input_nets_referenced_but_undriven == [NetBit("float_net")]


def test_check2_declared_input_port_completely_unused() -> None:
    graph = NetlistGraph(_build_design_with_every_defect())
    res = find_floating_signals(graph)
    assert res.declared_input_ports_completely_unused == [NetBit("unused_in")]
    # `pass_in` is used (via its PO connection) -- must NOT appear here.
    assert NetBit("pass_in") not in res.declared_input_ports_completely_unused
    # `used_in` is consumed by g1/g2 -- must NOT appear here.
    assert NetBit("used_in") not in res.declared_input_ports_completely_unused


def test_check3_unconnected_output_port_undriven() -> None:
    graph = NetlistGraph(_build_design_with_every_defect())
    res = find_floating_signals(graph)
    assert set(res.unconnected_output_ports_undriven) == {NetBit("out_dangling"), NetBit("pass_in")}
    assert NetBit("out_ok") not in res.unconnected_output_ports_undriven


def test_check4_unconnected_gate_input_pin() -> None:
    graph = NetlistGraph(_build_design_with_every_defect())
    res = find_floating_signals(graph)
    hits = {(gate.inst_name, pin) for gate, pin in res.unconnected_gate_input_pins}
    assert hits == {("g2", "I1")}


def test_check5_dangling_gate_output_never_consumed() -> None:
    graph = NetlistGraph(_build_design_with_every_defect())
    res = find_floating_signals(graph)
    assert [g.inst_name for g in res.dangling_gate_outputs_never_consumed] == ["g2"]
    # g1's output drives a primary output -- must NOT be flagged dangling.
    assert "g1" not in [g.inst_name for g in res.dangling_gate_outputs_never_consumed]


def test_dead_internal_wire_bit_observation() -> None:
    graph = NetlistGraph(_build_design_with_every_defect())
    res = find_floating_signals(graph)
    assert res.dead_internal_wire_bits == [NetBit("dead_net")]
    # `float_net` is read (just undriven) and `dangle_net` is driven (just
    # unread) -- neither is "dead" (dead = BOTH undriven AND unread).
    assert NetBit("float_net") not in res.dead_internal_wire_bits
    assert NetBit("dangle_net") not in res.dead_internal_wire_bits


def test_headline_count_is_check1_plus_check3_only() -> None:
    graph = NetlistGraph(_build_design_with_every_defect())
    res = find_floating_signals(graph)
    # #1 (float_net) + #3 (out_dangling, pass_in) = 3; #2/#4/#5 and the dead-
    # wire observation deliberately do NOT inflate this count.
    assert res.headline_count == 3
    assert res.headline_count == (
        len(res.floating_input_nets_referenced_but_undriven) + len(res.unconnected_output_ports_undriven)
    )


def test_no_defects_gives_all_zero() -> None:
    """Baseline: a design with none of the five defects reports all-zero
    sub-checks and a zero headline count (the shape a stub implementation
    would ALWAYS report -- distinguished from a real one only by the
    injected-defect tests above)."""
    design = Design(module_name="top")
    _pi(design, "a")
    _po(design, "y")
    design.ports = [Port("a", Direction.INPUT), Port("y", Direction.OUTPUT)]
    design.gates = [Gate("g1", GateType.BUF, {"O": NetBit("y"), "I0": NetBit("a")})]
    design.build_indices()
    graph = NetlistGraph(design)
    res = find_floating_signals(graph)
    assert res.headline_count == 0
    assert res.floating_input_nets_referenced_but_undriven == []
    assert res.declared_input_ports_completely_unused == []
    assert res.unconnected_output_ports_undriven == []
    assert res.unconnected_gate_input_pins == []
    assert res.dangling_gate_outputs_never_consumed == []
    assert res.dead_internal_wire_bits == []


# ----------------------------------------------------------------------
# Corpus-verified exact numbers (test37)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# F6 numeric-vs-string ordering (D): inject same-name multi-bit signals
# whose defective bits are {1, 2, 10, 20} -- a plain string sort of the
# rendered tokens would put "n[10]" and "n[20]" before "n[2]"; the correct
# numeric order interleaves them as 1, 2, 10, 20. Each of these asserts an
# exact list (not a set), so it actually observes ordering, unlike the
# other tests above (single-element lists) or `set()`-based assertions
# elsewhere in this file/tests/test_llm_tools.py.
# ----------------------------------------------------------------------

_ORDER_BITS = (1, 2, 10, 20)


def test_ordering_floating_input_nets_referenced_but_undriven_numeric() -> None:
    """Four bits of the SAME undriven internal wire, each read by a
    different gate's I0, in an order (1, 2, 10, 20) that a lexicographic
    sort of "name[bit]" tokens would scramble."""
    design = Design(module_name="top")
    _wire(design, "float17")
    design.signals["float17"].msb = 20
    design.signals["float17"].lsb = 1
    design.ports = []
    design.gates = []
    for i, bit in enumerate(_ORDER_BITS):
        sink = f"sink{i}"
        _wire(design, sink)
        design.gates.append(Gate(f"g{i}", GateType.BUF, {"O": NetBit(sink), "I0": NetBit("float17", bit)}))
    design.build_indices()
    graph = NetlistGraph(design)
    res = find_floating_signals(graph)
    assert res.floating_input_nets_referenced_but_undriven == [NetBit("float17", b) for b in _ORDER_BITS]


def test_ordering_declared_input_ports_completely_unused_numeric() -> None:
    """A 20-bit PI port whose only unused bits are {1, 2, 10, 20}; every
    other bit is consumed by a dummy BUF so it does NOT show up here."""
    design = Design(module_name="top")
    design.signals["pi17"] = Signal(name="pi17", msb=20, lsb=1, direction=Direction.INPUT)
    design.ports = [Port("pi17", Direction.INPUT)]
    design.gates = []
    consumer = 0
    for bit in range(1, 21):
        if bit in _ORDER_BITS:
            continue
        sink = f"used{consumer}"
        _wire(design, sink)
        design.gates.append(Gate(f"consume{consumer}", GateType.BUF, {"O": NetBit(sink), "I0": NetBit("pi17", bit)}))
        consumer += 1
    design.build_indices()
    graph = NetlistGraph(design)
    res = find_floating_signals(graph)
    assert res.declared_input_ports_completely_unused == [NetBit("pi17", b) for b in _ORDER_BITS]


def test_ordering_unconnected_output_ports_undriven_numeric() -> None:
    """A 20-bit PO port whose only undriven bits are {1, 2, 10, 20}; every
    other bit IS driven by a dummy BUF so it does NOT show up here."""
    design = Design(module_name="top")
    design.signals["po17"] = Signal(name="po17", msb=20, lsb=1, direction=Direction.OUTPUT)
    design.ports = [Port("po17", Direction.OUTPUT)]
    design.gates = []
    for bit in range(1, 21):
        if bit in _ORDER_BITS:
            continue
        src = f"src{bit}"
        _pi(design, src)
        design.ports.append(Port(src, Direction.INPUT))
        design.gates.append(Gate(f"drive{bit}", GateType.BUF, {"O": NetBit("po17", bit), "I0": NetBit(src)}))
    design.build_indices()
    graph = NetlistGraph(design)
    res = find_floating_signals(graph)
    assert res.unconnected_output_ports_undriven == [NetBit("po17", b) for b in _ORDER_BITS]


@pytest.mark.skipif(not corpus_available(), reason="Alpha_Testcase corpus not present -- the 2026 CAD Contest at ICCAD publishes them with Problem A; put the released testcases under Alpha_Testcase/testcase/testNN/ to enable this test")
def test_floating_signals_test37_corpus() -> None:
    design = parse_verilog(os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase", "test37", "test37.v"))
    graph = NetlistGraph(design)
    res = find_floating_signals(graph)
    assert res.headline_count == 0
    assert res.floating_input_nets_referenced_but_undriven == []
    assert res.declared_input_ports_completely_unused == []
    assert res.unconnected_output_ports_undriven == []
    assert res.unconnected_gate_input_pins == []
    assert res.dangling_gate_outputs_never_consumed == []
    # 32 bits of n17 plus n57[0]/n58[0]/n59[0] = 35 dead internal wire bits,
    # per the ground truth's own dangling_signals list.
    dead_names = sorted(f"{nb.name}[{nb.bit}]" if nb.bit is not None else nb.name for nb in res.dead_internal_wire_bits)
    assert len(dead_names) == 35
    assert all(name.startswith("n17[") for name in dead_names if "n17" in name)
    assert {"n57[0]", "n58[0]", "n59[0]"} <= set(dead_names)
