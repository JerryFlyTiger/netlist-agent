"""Truth-table checks for the fixed-identity rewrites: constant-input
simplification rules and every basis-decomposition recipe (including the
verbatim-specified 4-NAND XOR and NOR-only XNOR recipes). Each check builds a
minimal single-gate synthetic Design, applies the rewrite, and asserts the
replacement's simulated truth table matches the original gate's, over every
input combination.
"""

from __future__ import annotations

import itertools

import pytest

from netlist_agent.ir import (
    Const,
    Design,
    Direction,
    Gate,
    GateType,
    NetBit,
    Port,
    Signal,
    TWO_INPUT_GATES,
)
from netlist_agent.transform import BASES, _CONST_RULES, remap_to_basis, simplify_constant_inputs
from tests.sim import _eval_gate, simulate

TWO_INPUT = sorted(TWO_INPUT_GATES, key=lambda g: g.value)


def _make_gate_design(gate_type: GateType) -> Design:
    d = Design(module_name="t")
    is_two = gate_type in TWO_INPUT_GATES
    for name in (["a", "b"] if is_two else ["a"]):
        d.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
        d.ports.append(Port(name=name, direction=Direction.INPUT))
    d.signals["y"] = Signal(name="y", msb=None, lsb=None, direction=Direction.OUTPUT)
    d.ports.append(Port(name="y", direction=Direction.OUTPUT))
    pins = {"O": NetBit("y"), "I0": NetBit("a")}
    if is_two:
        pins["I1"] = NetBit("b")
    d.gates.append(Gate(inst_name="g0", gate_type=gate_type, pins=pins))
    d.build_indices()
    return d


def _all_gate_types_in(design: Design) -> set[GateType]:
    return {g.gate_type for g in design.gates}


@pytest.mark.parametrize("basis_name", sorted(BASES))
@pytest.mark.parametrize("gate_type", TWO_INPUT)
def test_basis_decomposition_matches_truth_table(basis_name: str, gate_type: GateType) -> None:
    design = _make_gate_design(gate_type)
    expected = {}
    for a, b in itertools.product((0, 1), repeat=2):
        expected[(a, b)] = _eval_gate(gate_type, [a, b])

    remap_to_basis(design, basis_name)

    remaining_types = _all_gate_types_in(design)
    disallowed = remaining_types - (BASES[basis_name] | {GateType.BUF})
    assert not disallowed, f"basis remap left disallowed gate types: {disallowed}"

    for a, b in itertools.product((0, 1), repeat=2):
        values = simulate(design, {NetBit("a"): a, NetBit("b"): b})
        assert values[NetBit("y")] == expected[(a, b)], f"{gate_type}/{basis_name} mismatch at a={a},b={b}"


def test_nand_xor_uses_exactly_four_nands() -> None:
    """Spec calls out this exact recipe verbatim: "each 2-input XOR can be
    realized with 4 NAND gates"."""
    design = _make_gate_design(GateType.XOR)
    remap_to_basis(design, "nand_not")
    types = [g.gate_type for g in design.gates]
    assert types.count(GateType.NAND) == 4
    assert all(t == GateType.NAND for t in types)


def test_nor_xnor_is_nor_only() -> None:
    """Spec calls out this exact recipe verbatim: "convert every XNOR to an
    equivalent NOR-only implementation"."""
    design = _make_gate_design(GateType.XNOR)
    remap_to_basis(design, "nor_not")
    types = [g.gate_type for g in design.gates]
    assert all(t == GateType.NOR for t in types)
    assert len(types) == 4


@pytest.mark.parametrize("gate_type", [GateType.NOT, GateType.BUF])
def test_not_and_buf_pass_through_every_basis(gate_type: GateType) -> None:
    for basis_name in BASES:
        design = _make_gate_design(gate_type)
        replaced = remap_to_basis(design, basis_name)
        assert replaced == 0
        assert [g.gate_type for g in design.gates] == [gate_type]


def _const_design(gate_type: GateType, const_pin: str, const_value: Const) -> Design:
    design = _make_gate_design(gate_type)
    design.gates[0].pins[const_pin] = const_value
    design.build_indices()
    return design


@pytest.mark.parametrize("gate_type,const_value,rule", [(k[0], k[1], v) for k, v in _CONST_RULES.items()])
def test_constant_simplification_matches_truth_table(gate_type: GateType, const_value: Const, rule) -> None:
    for const_pin in ("I0", "I1"):
        design = _const_design(gate_type, const_pin, const_value)
        expected = {}
        for other in (0, 1):
            a = const_value.value if const_pin == "I0" else other
            b = other if const_pin == "I0" else const_value.value
            expected[other] = _eval_gate(gate_type, [a, b])

        eliminated = simplify_constant_inputs(design)
        assert eliminated == 1

        for other in (0, 1):
            # Both a/b are driven with the same value; only the pin that
            # wasn't const-tied actually affects the (now simplified) result.
            values = simulate(design, {NetBit("a"): other, NetBit("b"): other})
            assert values[NetBit("y")] == expected[other]


def test_constant_simplification_both_inputs_const() -> None:
    design = _make_gate_design(GateType.AND)
    design.gates[0].pins["I0"] = Const.ONE
    design.gates[0].pins["I1"] = Const.ONE
    design.build_indices()
    eliminated = simplify_constant_inputs(design)
    assert eliminated == 1
    values = simulate(design, {})
    assert values[NetBit("y")] == 1


def test_constant_simplification_gate_type_filter() -> None:
    design = _make_gate_design(GateType.OR)
    design.gates[0].pins["I1"] = Const.ONE
    design.build_indices()
    eliminated = simplify_constant_inputs(design, gate_types={GateType.NAND})
    assert eliminated == 0
    assert design.gates[0].gate_type == GateType.OR
