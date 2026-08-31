from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


class Direction(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INTERNAL = "wire"


class Const(Enum):
    ZERO = 0
    ONE = 1


class GateType(Enum):
    AND = "and"
    OR = "or"
    NAND = "nand"
    NOR = "nor"
    XOR = "xor"
    XNOR = "xnor"
    NOT = "not"
    BUF = "buf"
    DFF = "dff"


TWO_INPUT_GATES = {
    GateType.AND,
    GateType.OR,
    GateType.NAND,
    GateType.NOR,
    GateType.XOR,
    GateType.XNOR,
}
ONE_INPUT_GATES = {GateType.NOT, GateType.BUF}

# Fixed positional pin order for non-dff primitives, output first.
POSITIONAL_PIN_ORDER: dict[GateType, list[str]] = {}
for _gt in TWO_INPUT_GATES:
    POSITIONAL_PIN_ORDER[_gt] = ["O", "I0", "I1"]
for _gt in ONE_INPUT_GATES:
    POSITIONAL_PIN_ORDER[_gt] = ["O", "I0"]

DFF_PIN_ORDER = ["RN", "SN", "CK", "D", "Q"]

# The pin that is the driven output for each gate type (rest are inputs).
OUTPUT_PIN: dict[GateType, str] = {gt: "O" for gt in POSITIONAL_PIN_ORDER}
OUTPUT_PIN[GateType.DFF] = "Q"


@dataclass(frozen=True)
class NetBit:
    name: str
    bit: Optional[int] = None


Pin = Union[NetBit, Const, None]


@dataclass
class Signal:
    name: str
    msb: Optional[int]
    lsb: Optional[int]
    direction: Direction

    @property
    def width(self) -> int:
        # abs(): an ASCENDING declaration (`wire [0:7] x`, msb < lsb) is legal
        # Verilog. `msb - lsb + 1` returns -6 for it, and that negative width
        # used to leak verbatim into user-facing text ("x is a -6-bit signal").
        # `msb is None` iff `lsb is None` at every construction site; testing
        # both is what lets a type checker narrow them, and costs nothing.
        if self.msb is None or self.lsb is None:
            return 1
        return abs(self.msb - self.lsb) + 1

    def bits(self) -> list[NetBit]:
        # Walk from msb towards lsb whichever way they are declared. The old
        # hard-coded `range(msb, lsb - 1, -1)` silently returned [] for an
        # ascending declaration -- and an empty bit list is indistinguishable
        # from "this signal has no bits", so every aggregate over it (fanout,
        # cut-signal, gates-connected) answered a confident 0/none. Descending
        # declarations (every net in the corpus) keep the exact same order.
        if self.msb is None or self.lsb is None:
            return [NetBit(self.name, None)]
        step = -1 if self.msb >= self.lsb else 1
        return [NetBit(self.name, i) for i in range(self.msb, self.lsb + step, step)]


@dataclass
class Port:
    name: str
    direction: Direction


@dataclass
class Gate:
    inst_name: str
    gate_type: GateType
    pins: dict[str, Pin] = field(default_factory=dict)


@dataclass
class Design:
    module_name: str
    ports: list[Port] = field(default_factory=list)
    signals: dict[str, Signal] = field(default_factory=dict)
    gates: list[Gate] = field(default_factory=list)
    # net -> driving gate / fanout gates, keyed by flattened per-bit net identity.
    net_driver: dict[NetBit, Gate] = field(default_factory=dict)
    net_fanout: dict[NetBit, list[Gate]] = field(default_factory=dict)
    # Lazy bookkeeping caches for the mutation helpers below (transform.py).
    # Not part of the design's logical content: excluded from repr/equality so
    # existing structural-equality-based tests are unaffected.
    _gate_index: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _fresh_counters: dict[tuple[str, str], int] = field(default_factory=dict, repr=False, compare=False)

    def build_indices(self) -> None:
        self.net_driver = {}
        self.net_fanout = {}
        for gate in self.gates:
            out_key = OUTPUT_PIN[gate.gate_type]
            for pin_name, value in gate.pins.items():
                if not isinstance(value, NetBit):
                    continue
                if pin_name == out_key:
                    self.net_driver[value] = gate
                else:
                    self.net_fanout.setdefault(value, []).append(gate)

    # ------------------------------------------------------------------
    # Mutation helpers (used by transform.py). These keep net_driver/
    # net_fanout continuously consistent -- no separate build_indices()
    # call is needed after using them, unlike direct list/dict surgery.
    # ------------------------------------------------------------------

    def _ensure_gate_index(self) -> dict[str, int]:
        # Length mismatch is our only staleness signal: valid as long as
        # design.gates is only ever mutated via add_gate/remove_gate once a
        # Design is in play for transforms (parser.py's direct .append calls
        # happen before any of these helpers are ever invoked).
        if len(self._gate_index) != len(self.gates):
            self._gate_index = {g.inst_name: i for i, g in enumerate(self.gates)}
        return self._gate_index

    def add_gate(self, gate: Gate) -> None:
        """Register a new gate: append to design.gates and index its pins."""
        idx = self._ensure_gate_index()
        idx[gate.inst_name] = len(self.gates)
        self.gates.append(gate)
        out_key = OUTPUT_PIN[gate.gate_type]
        for pin_name, value in gate.pins.items():
            if not isinstance(value, NetBit):
                continue
            if pin_name == out_key:
                self.net_driver[value] = gate
            else:
                self.net_fanout.setdefault(value, []).append(gate)

    def remove_gate(self, gate: Gate) -> None:
        """Remove a gate: drop it from design.gates and un-index its pins.

        O(1) amortized: removal is a swap-with-last-and-pop, so design.gates
        does not preserve declaration order after removals.
        """
        idx = self._ensure_gate_index()
        pos = idx.pop(gate.inst_name)
        last_pos = len(self.gates) - 1
        if pos != last_pos:
            moved = self.gates[last_pos]
            self.gates[pos] = moved
            idx[moved.inst_name] = pos
        self.gates.pop()
        out_key = OUTPUT_PIN[gate.gate_type]
        for pin_name, value in gate.pins.items():
            if not isinstance(value, NetBit):
                continue
            if pin_name == out_key:
                if self.net_driver.get(value) is gate:
                    del self.net_driver[value]
            else:
                lst = self.net_fanout.get(value)
                if lst is not None:
                    try:
                        lst.remove(gate)
                    except ValueError:
                        pass
                    if not lst:
                        del self.net_fanout[value]

    def rewire_pin(self, gate: Gate, pin_name: str, new_value: Pin) -> None:
        """Repoint one pin of an already-registered gate to a new value,
        keeping net_driver/net_fanout in sync with the change."""
        old_value = gate.pins.get(pin_name)
        out_key = OUTPUT_PIN[gate.gate_type]
        if isinstance(old_value, NetBit):
            if pin_name == out_key:
                if self.net_driver.get(old_value) is gate:
                    del self.net_driver[old_value]
            else:
                lst = self.net_fanout.get(old_value)
                if lst is not None:
                    try:
                        lst.remove(gate)
                    except ValueError:
                        pass
                    if not lst:
                        del self.net_fanout[old_value]
        gate.pins[pin_name] = new_value
        if isinstance(new_value, NetBit):
            if pin_name == out_key:
                self.net_driver[new_value] = gate
            else:
                self.net_fanout.setdefault(new_value, []).append(gate)

    def fresh_gate_name(self, prefix: str = "t_gate_") -> str:
        """Allocate an instance name guaranteed not to collide with any
        existing gate (including ones allocated earlier by this call)."""
        idx = self._ensure_gate_index()
        counters = self._fresh_counters
        key = ("gate", prefix)
        if key not in counters:
            start = 0
            for name in idx:
                if name.startswith(prefix) and name[len(prefix):].isdigit():
                    start = max(start, int(name[len(prefix):]) + 1)
            counters[key] = start
        while f"{prefix}{counters[key]}" in idx:
            counters[key] += 1
        name = f"{prefix}{counters[key]}"
        counters[key] += 1
        return name

    def fresh_net(self, prefix: str = "t_net_") -> NetBit:
        """Allocate a fresh single-bit internal wire (name guaranteed not to
        collide with any existing signal) and register it in design.signals.
        """
        counters = self._fresh_counters
        key = ("net", prefix)
        if key not in counters:
            start = 0
            for name in self.signals:
                if name.startswith(prefix) and name[len(prefix):].isdigit():
                    start = max(start, int(name[len(prefix):]) + 1)
            counters[key] = start
        while f"{prefix}{counters[key]}" in self.signals:
            counters[key] += 1
        name = f"{prefix}{counters[key]}"
        counters[key] += 1
        self.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
        return NetBit(name, None)

    def rename_signal(self, old_name: str, new_name: str) -> None:
        """Whole-netlist rename of a signal: updates design.signals, every
        gate pin referencing it, and design.ports if it is a port. Rebuilds
        net_driver/net_fanout at the end since their keys embed the name.
        """
        if old_name not in self.signals:
            raise KeyError(f"no such signal: {old_name!r}")
        if new_name != old_name and new_name in self.signals:
            raise ValueError(f"signal name already in use: {new_name!r}")
        sig = self.signals.pop(old_name)
        sig.name = new_name
        self.signals[new_name] = sig
        for gate in self.gates:
            for pin_name, value in gate.pins.items():
                if isinstance(value, NetBit) and value.name == old_name:
                    gate.pins[pin_name] = NetBit(new_name, value.bit)
        for port in self.ports:
            if port.name == old_name:
                port.name = new_name
        self.build_indices()


def design_fingerprint(design: Optional[Design]) -> Optional[str]:
    """Structure-sensitive digest: every gate's name, type and pin-to-net
    wiring.

    The count cannot answer "did anything happen" -- a remap swaps types
    without changing the total. Neither can the type histogram, which was the
    first attempt here: a transform that rewires same-type gates, or renames
    them, leaves the histogram identical and would have been recorded as
    "design unchanged". Since the whole point of this function is to catch a
    model editing a netlist it was not asked to edit, under-reporting a change
    is the one error that must not be cheap."""
    if design is None:
        return None
    h = hashlib.sha256()
    # repr() throughout, and a length-prefixed record per line: an identifier
    # is allowed to contain the separators (escaped Verilog identifiers), and
    # an unescaped digest can be fooled at a field boundary.
    def rec(*fields) -> None:
        payload = json.dumps([repr(f) for f in fields])
        h.update(f"{len(payload)}:{payload}\n".encode())

    rec("module", design.module_name)
    for g in sorted(design.gates, key=lambda x: x.inst_name):
        rec("gate", g.inst_name, getattr(g.gate_type, "name", g.gate_type),
            sorted((k, repr(v)) for k, v in g.pins.items()))
    # Ports and signals are in here because a rename can touch a net that no
    # gate references -- a floating primary input is exactly the kind this
    # repo has tools to hunt for -- and a digest over gates alone reports that
    # edit as "design unchanged". Under-reporting a change is the one error
    # this function must not make.
    for prt in sorted(design.ports, key=lambda x: x.name):
        rec("port", prt.name, getattr(prt.direction, "name", prt.direction))
    for name in sorted(design.signals):
        sig = design.signals[name]
        rec("signal", name, sig.msb, sig.lsb,
            getattr(sig.direction, "name", sig.direction))
    return h.hexdigest()
