"""Tiny combinational simulator -- test utility only.

Topologically evaluates every non-DFF gate in a Design given fixed primary-
input values and fixed DFF.Q values, returning the resolved value (0/1) of
every net-bit that has one. Deliberately simple (fixed-point iteration, not a
real topo sort) -- intended only for the small synthetic fixtures used by
transform.py's truth-table-style tests, not for real 100k-gate designs.
"""

from __future__ import annotations

from typing import Optional

from netlist_agent.ir import Const, Design, GateType, NetBit, OUTPUT_PIN, Pin


def _eval_gate(gate_type: GateType, ins: list[int]) -> int:
    if gate_type == GateType.NOT:
        return 1 - ins[0]
    if gate_type == GateType.BUF:
        return ins[0]
    a, b = ins
    if gate_type == GateType.AND:
        return a & b
    if gate_type == GateType.OR:
        return a | b
    if gate_type == GateType.NAND:
        return 1 - (a & b)
    if gate_type == GateType.NOR:
        return 1 - (a | b)
    if gate_type == GateType.XOR:
        return a ^ b
    if gate_type == GateType.XNOR:
        return 1 - (a ^ b)
    raise ValueError(f"cannot evaluate gate type {gate_type!r}")


def simulate(
    design: Design,
    inputs: dict[NetBit, int],
    dff_q: Optional[dict[NetBit, int]] = None,
) -> dict[NetBit, int]:
    """Evaluate every combinational net-bit in `design` given fixed PI values
    (`inputs`) and fixed DFF.Q values (`dff_q`). Returns every resolved
    net-bit's value (0/1), including PO/DFF.D nets. Raises ValueError if a
    gate's inputs never all resolve (unconnected pin or a genuine cycle).
    """
    values: dict[NetBit, int] = dict(inputs)
    values.update(dff_q or {})

    def resolve(pin: Pin) -> Optional[int]:
        if pin is None:
            return None
        if isinstance(pin, Const):
            return pin.value
        return values.get(pin)

    remaining = [g for g in design.gates if g.gate_type != GateType.DFF]
    progress = True
    while remaining and progress:
        progress = False
        still = []
        for g in remaining:
            out_key = OUTPUT_PIN[g.gate_type]
            in_pins = [p for k, p in g.pins.items() if k != out_key]
            resolved = [resolve(p) for p in in_pins]
            if any(r is None for r in resolved):
                still.append(g)
                continue
            out_nb = g.pins.get(out_key)
            if isinstance(out_nb, NetBit):
                values[out_nb] = _eval_gate(g.gate_type, resolved)
            progress = True
        remaining = still
    if remaining:
        raise ValueError(
            f"simulate(): {len(remaining)} gate(s) never resolved (cycle or unconnected input)"
        )
    return values
