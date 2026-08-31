from __future__ import annotations

from itertools import groupby

from netlist_agent.ir import (
    Const,
    DFF_PIN_ORDER,
    Design,
    Direction,
    Gate,
    GateType,
    NetBit,
    Pin,
    POSITIONAL_PIN_ORDER,
    Signal,
)

DECL_KEYWORD = {
    Direction.INPUT: "input",
    Direction.OUTPUT: "output",
    Direction.INTERNAL: "wire",
}


def _render_pin(value: Pin) -> str:
    if value is None:
        return ""
    if isinstance(value, Const):
        return "1'b0" if value == Const.ZERO else "1'b1"
    if value.bit is None:
        return value.name
    return f"{value.name}[{value.bit}]"


def _render_decls(signals: dict[str, Signal]) -> list[str]:
    lines: list[str] = []
    for (direction, msb, lsb), group in groupby(signals.values(), key=lambda s: (s.direction, s.msb, s.lsb)):
        names = [s.name for s in group]
        width = f"[{msb}:{lsb}] " if msb is not None else ""
        lines.append(f"  {DECL_KEYWORD[direction]} {width}{', '.join(names)};\n")
    return lines


def _render_gate(gate: Gate) -> str:
    if gate.gate_type == GateType.DFF:
        conns = ", ".join(f".{pin}({_render_pin(gate.pins.get(pin))})" for pin in DFF_PIN_ORDER)
    else:
        pin_order = POSITIONAL_PIN_ORDER[gate.gate_type]
        conns = ", ".join(_render_pin(gate.pins.get(pin)) for pin in pin_order)
    return f"  {gate.gate_type.value} {gate.inst_name}({conns});\n"


def write_verilog(design: Design, path: str) -> None:
    with open(path, "w") as f:
        f.write(f"module {design.module_name}({', '.join(p.name for p in design.ports)});\n")
        f.writelines(_render_decls(design.signals))
        f.writelines(_render_gate(gate) for gate in design.gates)
        f.write("endmodule\n")
