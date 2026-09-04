"""Higher-level "answer this kind of contest question" functions, built on
top of the structural primitives in graph.py.

Counting/listing functions (capabilities 1-6) operate directly on a
:class:`~netlist_agent.ir.Design` -- they need no precomputed adjacency.
Fanin/fanout and cone-size functions (capabilities 7, 10, 11) take a
:class:`~netlist_agent.graph.NetlistGraph` since they rely on its precomputed
PI/PO bit sets and/or reachability traversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from netlist_agent.graph import Load, NetlistGraph
from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, OUTPUT_PIN, Pin
from netlist_agent.netref import netbit_sort_key

# ----------------------------------------------------------------------
# Counting / listing (capabilities 1-6)
# ----------------------------------------------------------------------


def gate_count_by_type(design: Design) -> dict[GateType, int]:
    """Capability 1: gate count broken down by GateType (including DFF)."""
    counts: dict[GateType, int] = {gt: 0 for gt in GateType}
    for gate in design.gates:
        counts[gate.gate_type] += 1
    return counts


def gates_of_type(design: Design, gate_type: GateType) -> list[Gate]:
    """Capability 2: list gates of a given type (with their pin connections,
    already present on each Gate)."""
    return [g for g in design.gates if g.gate_type == gate_type]


def gates_by_name_substring(design: Design, substring: str, gate_type: Optional[GateType] = None) -> list[Gate]:
    """"Find all the buffers which name include '_gc__'" style query: gates
    whose instance name contains `substring` as a literal substring (not a
    regex/glob -- every phrasing this backs says "name include(s) X").
    `gate_type`, if given, additionally restricts to that gate type so a
    substring that happens to also appear in, e.g., a DFF's or AND's instance
    name doesn't get swept in unintentionally."""
    return [g for g in design.gates if substring in g.inst_name and (gate_type is None or g.gate_type == gate_type)]


@dataclass(frozen=True)
class PortInfo:
    name: str
    width: int
    msb: Optional[int]
    lsb: Optional[int]


def list_primary_inputs(design: Design) -> list[PortInfo]:
    """Capability 3: list primary inputs with bit widths."""
    return [
        PortInfo(p.name, design.signals[p.name].width, design.signals[p.name].msb, design.signals[p.name].lsb)
        for p in design.ports
        if p.direction == Direction.INPUT
    ]


def list_primary_outputs(design: Design) -> list[PortInfo]:
    """Capability 3: list primary outputs with bit widths."""
    return [
        PortInfo(p.name, design.signals[p.name].width, design.signals[p.name].msb, design.signals[p.name].lsb)
        for p in design.ports
        if p.direction == Direction.OUTPUT
    ]


def primary_input_bit_count(design: Design) -> int:
    """Capability 4: total PI bit count (the operationally useful count given buses)."""
    return sum(pi.width for pi in list_primary_inputs(design))


def primary_output_bit_count(design: Design) -> int:
    """Capability 4: total PO bit count."""
    return sum(po.width for po in list_primary_outputs(design))


def primary_input_port_count(design: Design) -> int:
    """Capability 4: PI port count (as opposed to bit count)."""
    return len(list_primary_inputs(design))


def primary_output_port_count(design: Design) -> int:
    """Capability 4: PO port count (as opposed to bit count)."""
    return len(list_primary_outputs(design))


def dffs_on_clock(design: Design, clock_net_name: str) -> list[Gate]:
    """Capability 5: all DFFs whose CK pin is driven by the net named
    `clock_net_name` (matched by net name, ignoring bit index)."""
    result = []
    for gate in design.gates:
        if gate.gate_type != GateType.DFF:
            continue
        ck = gate.pins.get("CK")
        if isinstance(ck, NetBit) and ck.name == clock_net_name:
            result.append(gate)
    return result


def gate_lookup(graph: NetlistGraph, inst_name: str) -> Gate:
    """Capability 6: gate lookup by instance name -> type + pin connections
    (the returned Gate carries both)."""
    return graph.gate_by_name[inst_name]


# ----------------------------------------------------------------------
# Fanin / fanout (capabilities 7, 10, 11)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FaninEntry:
    """One resolved input pin of a gate: the raw pin value, plus what kind of
    source it traces to."""

    pin: str
    value: Pin
    # "const": tied to 1'b0/1'b1.
    # "unconnected": pin absent / None.
    # "pi": net-bit with no driver at all -- a primary input.
    # "dff_q": driven by a DFF's Q (a pseudo-source per the boundary rule).
    # "gate": driven by another (non-DFF) gate.
    source_kind: str
    source_gate: Optional[Gate] = None


def direct_fanin(graph: NetlistGraph, gate: Gate) -> list[FaninEntry]:
    """Capability 7: direct fanin of a gate -- what drives each input pin.

    Only reports pins actually present as keys in `gate.pins`; it does not
    reconcile against a fixed positional/DFF pin-order union, so a pin
    parser.py ever omits entirely from `gate.pins` (rather than including it
    with value None) will silently be absent here rather than reported as
    unconnected.
    """
    out_pin = OUTPUT_PIN[gate.gate_type]
    entries: list[FaninEntry] = []
    for pin_name, value in gate.pins.items():
        if pin_name == out_pin:
            continue
        if value is None:
            entries.append(FaninEntry(pin_name, value, "unconnected"))
        elif isinstance(value, Const):
            entries.append(FaninEntry(pin_name, value, "const"))
        else:
            driver = graph.design.net_driver.get(value)
            if driver is None:
                entries.append(FaninEntry(pin_name, value, "pi"))
            elif driver.gate_type == GateType.DFF:
                entries.append(FaninEntry(pin_name, value, "dff_q", driver))
            else:
                entries.append(FaninEntry(pin_name, value, "gate", driver))
    return entries


def direct_fanout(graph: NetlistGraph, nb: NetBit) -> list[Load]:
    """Capability 7: direct fanout of a net-bit -- every gate-pin load plus,
    per the fanout-counting convention, an extra explicit `Load("po", ...)`
    entry if `nb` is wired to a primary-output port."""
    loads: list[Load] = []
    seen_gates: set[str] = set()
    for gate in graph.design.net_fanout.get(nb, ()):
        if gate.inst_name in seen_gates:
            continue
        seen_gates.add(gate.inst_name)
        out_pin = OUTPUT_PIN[gate.gate_type]
        for pin_name, value in gate.pins.items():
            if pin_name != out_pin and value == nb:
                loads.append(Load("gate", gate=gate, pin=pin_name))
    port_name = graph.po_port_of.get(nb)
    if port_name is not None:
        loads.append(Load("po", port_name=port_name))
    return loads


def fanout_count(graph: NetlistGraph, nb: NetBit) -> int:
    """Capability 10: fanout count of a net-bit, per the pin-counting
    convention (every consuming gate pin, plus one if `nb` also drives a PO).
    This is deliberately the raw `net_fanout` list length (not deduplicated)
    -- unlike the graph-traversal adjacency in graph.py, which dedups at
    gate granularity for path/depth purposes, fanout counting is explicitly
    per-pin per the spec.

    Assumes `nb` is wired to at most one output Port: the `+1` is a
    membership check against the `po_bits` set, so if two different output
    ports both happened to include the same net-bit in their `Signal.bits()`
    (unusual, but not IR-forbidden), that PO load would only ever be counted
    once here, not twice.
    """
    return len(graph.design.net_fanout.get(nb, ())) + (1 if nb in graph.po_bits else 0)


def iter_fanout_counts(
    graph: NetlistGraph, netbits: Optional[Iterable[NetBit]] = None
) -> Iterator[tuple[NetBit, int]]:
    """One pass yielding (net-bit, fanout_count) for every net-bit that has
    any fanout (or, if `netbits` given, restricted to that set). Used by the
    max-fanout aggregates below; kept as a single generator so a hard
    constraint check ("no signal drives more than N loads") can be done in
    one linear pass without materializing every count.
    """
    if netbits is None:
        netbits = sorted(set(graph.design.net_fanout.keys()) | graph.po_bits, key=netbit_sort_key)
    for nb in netbits:
        yield nb, fanout_count(graph, nb)


def max_fanout_among(graph: NetlistGraph, netbits: Iterable[NetBit]) -> tuple[Optional[NetBit], int]:
    best_nb: Optional[NetBit] = None
    best_count = 0
    for nb in sorted(netbits, key=netbit_sort_key):
        c = fanout_count(graph, nb)
        if c > best_count:
            best_nb, best_count = nb, c
    return best_nb, best_count


def max_fanout_overall(graph: NetlistGraph) -> tuple[Optional[NetBit], int]:
    """Capability 10: which signal (net-bit) has the highest fanout, and what
    is the current maximum fanout, across the whole design. One linear pass
    over all net-bits that have any fanout."""
    best_nb: Optional[NetBit] = None
    best_count = 0
    for nb, c in iter_fanout_counts(graph):
        if c > best_count:
            best_nb, best_count = nb, c
    return best_nb, best_count


def max_fanout_pi(graph: NetlistGraph) -> tuple[Optional[NetBit], int]:
    """Capability 10: which PI has the highest fanout in the design."""
    return max_fanout_among(graph, graph.pi_bits)


def fanin_cone_size(graph: NetlistGraph, nb: NetBit) -> int:
    """Capability 11: gate count of a signal's fanin cone.

    Per QA A94, boundary DFFs (a DFF whose Q feeds a gate in the cone, or
    the DFF driving `nb` itself when `nb` IS a DFF's Q) count as gates in
    the cone -- see `NetlistGraph.backward_cone_with_boundary_dffs`.
    """
    return len(graph.backward_cone_with_boundary_dffs(nb))


def fanout_cone_size(graph: NetlistGraph, nb: NetBit) -> int:
    """Companion to capability 11: gate count of a signal's fanout cone."""
    return len(graph.forward_reachable_gates(nb))


def po_cone_sizes(graph: NetlistGraph) -> dict[NetBit, int]:
    """Companion to `fanin_cone_size`/`per_output_depths` (graph.py): gate
    count of each primary output's fanin cone, keyed by the PO's net-bit.
    Unlike `per_output_depths`, this is restricted to `graph.po_bits` only
    (no DFF.D pins) -- callers asking specifically about "primary outputs"
    (as opposed to "outputs" in the looser depth-query sense) get exactly
    the PO ports, nothing else."""
    return {nb: fanin_cone_size(graph, nb) for nb in graph.po_bits}


def largest_fanin_cone(graph: NetlistGraph) -> tuple[Optional[NetBit], int]:
    """Capability 11: which output (PO or DFF.D) has the largest fanin cone."""
    best_nb: Optional[NetBit] = None
    best_size = -1
    candidates: set[NetBit] = set(graph.po_bits)
    for gate in graph.dff_gates:
        d = gate.pins.get("D")
        if isinstance(d, NetBit):
            candidates.add(d)
    for nb in sorted(candidates, key=netbit_sort_key):
        size = fanin_cone_size(graph, nb)
        if size > best_size:
            best_nb, best_size = nb, size
    return best_nb, best_size


def deepest_fanin_cone(graph: NetlistGraph) -> tuple[Optional[NetBit], int]:
    """Which output (PO or DFF.D) has the DEEPEST fanin cone -- distinct
    from `largest_fanin_cone` (gate COUNT). Two different questions the
    router used to conflate: "deepest"/"biggest" fanin cone all reached
    `largest_fanin_cone`, but depth and gate count can disagree (a long
    thin chain can be deeper yet have fewer gates than a short, wide one).
    Same candidate set and tie-break convention as `largest_fanin_cone`
    (first-in-sort-order among ties, via strict `>`), driven by
    `graph.per_output_depths()` instead of `fanin_cone_size`.
    """
    depths = graph.per_output_depths()
    best_nb: Optional[NetBit] = None
    best_depth = -1
    for nb in sorted(depths, key=netbit_sort_key):
        depth = depths[nb]
        if depth > best_depth:
            best_nb, best_depth = nb, depth
    return best_nb, best_depth


# ----------------------------------------------------------------------
# Cut-signal convenience (capability 21, signal-name granularity)
# ----------------------------------------------------------------------


def is_cut_signal(graph: NetlistGraph, signal_name: str) -> bool:
    """Capability 21, signal-name convenience: true if ANY bit of the named
    signal is a cut signal for at least one PI/PO pair."""
    sig = graph.design.signals[signal_name]
    return any(graph.is_cut_signal_for_some_pi_po_pair(nb) for nb in sig.bits())


def is_cut_signal_bits(graph: NetlistGraph, netbits: Iterable[NetBit]) -> bool:
    """Capability 21 over an already-resolved net-bit set: true if ANY of
    `netbits` is a cut signal for at least one PI/PO pair. Callers that must
    validate a user-given net reference against the design's declared
    widths first (`netref.resolve_bits`) use this instead of `is_cut_signal`
    -- a bare vector signal name resolves to ALL of its bits (matching
    `is_cut_signal`'s own "any bit" semantics), while a bit-selected token
    resolves to just that one bit."""
    return any(graph.is_cut_signal_for_some_pi_po_pair(nb) for nb in netbits)


# ----------------------------------------------------------------------
# Floating input / unconnected output port sweep (new capability)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FloatingSignalsResult:
    """Capability N: structural "floating"/dangling-declaration sweep of a
    design, per five DISTINCT sub-checks (deliberately not merged into one
    bucket -- "floating input" and "declared-but-unused input port" are
    different defects, and callers/tests need to tell them apart):

      1. `floating_input_nets_referenced_but_undriven` -- a net-bit that
         some gate's input pin actually reads, with no driver in
         `design.net_driver`, and that is NOT a declared primary input (a
         real PI with no driver is normal and NOT floating -- see the
         explicit `graph.pi_bits` exclusion in `find_floating_signals`
         below). This is the strict "floating input" defect.
      2. `declared_input_ports_completely_unused` -- a bit of a declared
         primary input port with zero consumers in `design.net_fanout`,
         EXCLUDING PI-straight-through-to-PO bits (also present in
         `graph.po_bits`) -- those bits are used (by the PO connection),
         just not by any gate.
      3. `unconnected_output_ports_undriven` -- a bit of a declared primary
         output port with no driver in `design.net_driver`.
      4. `unconnected_gate_input_pins` -- a specific (gate, pin) whose pin
         value is literally `None` (present in `gate.pins` as an
         unconnected slot, distinct from #1's "referenced but driverless
         net" -- this is "no net referenced at all").
      5. `dangling_gate_outputs_never_consumed` -- a gate whose output net
         is neither a primary output nor has any entry in
         `design.net_fanout` (drives nothing observable at all).

    `headline_count` (floating inputs + unconnected output ports, per the
    strict-definition convention this project adopted for the corpus's own
    "Check if there are any floating inputs or unconnected output ports"
    question) is #1 + #3 ONLY -- #2, #4, #5 are real structural findings
    worth reporting, but are not what "floating input" or "unconnected
    output port" mean in the strict sense, so they do not inflate the
    headline count.

    `dead_internal_wire_bits` is a SEPARATE observation (not one of the five
    sub-checks, and not part of `headline_count`): a declared INTERNAL wire
    bit that is both undriven and unread -- a dead declaration that cannot
    possibly affect any logic, but the corpus's own ground truth explicitly
    keeps this out of the strict floating-input/unconnected-output-port
    count while still wanting it surfaced.

    Sub-checks #1/#4/#5 (`find_floating_signals` below) iterate
    `design.gates` directly, so they scan DFFs too -- this is DELIBERATE and
    NOT the same "combinational gates only" boundary `graph.py`'s
    `_all_gate_names` convention uses elsewhere in this codebase. Floating
    checks care about physical connection completeness of every declared
    pin; a DFF's D/CK/RN/SN/Q pin can be just as unconnected or dangling as
    a combinational gate's, and excluding DFFs would silently miss those
    defects. Not applying the combinational-only boundary here is
    consistent with what this check is actually for, not an inconsistency.
    """

    floating_input_nets_referenced_but_undriven: list[NetBit]
    declared_input_ports_completely_unused: list[NetBit]
    unconnected_output_ports_undriven: list[NetBit]
    unconnected_gate_input_pins: list[tuple[Gate, str]]
    dangling_gate_outputs_never_consumed: list[Gate]
    dead_internal_wire_bits: list[NetBit]

    @property
    def headline_count(self) -> int:
        return len(self.floating_input_nets_referenced_but_undriven) + len(
            self.unconnected_output_ports_undriven
        )


def find_floating_signals(graph: NetlistGraph) -> FloatingSignalsResult:
    """Run all five sub-checks (see `FloatingSignalsResult`) plus the dead-
    internal-wire observation, over `graph.design`. Deliberately does NOT
    reuse `FaninEntry.source_kind == "pi"` (see `direct_fanin` above) --
    that classification only checks "no driver at all", without checking
    whether the net-bit is actually a DECLARED primary input, so an
    undriven internal wire referenced as a gate input would be misclassified
    as "pi" by it. Sub-check #1 here does the real `graph.pi_bits` check
    that distinction requires.
    """
    design = graph.design

    floating_inputs_seen: set[NetBit] = set()
    unconnected_pins: list[tuple[Gate, str]] = []
    for gate in design.gates:
        out_pin = OUTPUT_PIN[gate.gate_type]
        for pin_name, value in gate.pins.items():
            if pin_name == out_pin:
                continue
            if value is None:
                unconnected_pins.append((gate, pin_name))
            elif isinstance(value, NetBit):
                if value not in graph.pi_bits and design.net_driver.get(value) is None:
                    floating_inputs_seen.add(value)
    # Deduplicated (the same undriven net-bit can be read by several gate
    # input pins -- it is one floating NET, not one entry per pin that
    # reads it), sorted for a deterministic result.
    floating_inputs = sorted(floating_inputs_seen, key=lambda nb: (nb.name, nb.bit if nb.bit is not None else -1))

    unused_input_ports: list[NetBit] = []
    for nb in graph.pi_bits:
        # PI-straight-through-to-PO exclusion: NOT mirrored below for
        # `unconnected_outputs` (deliberately left asymmetric -- see that
        # block's comment). `design.signals` keys each Signal by name and
        # gives it exactly one `Direction`, so `graph.pi_bits` and
        # `graph.po_bits` are built from disjoint sets of signals -- a
        # NetBit can never be a member of both, regardless of whether this
        # Verilog subset supports `assign` (it doesn't, which is a
        # secondary reason: even with `assign` support, a distinct
        # PI-direction Signal and PO-direction Signal wired together are
        # still two different NetBits, not one shared one). This branch is
        # therefore dead code under every input this parser can ever
        # produce. Kept anyway (rather than removed) as defensive
        # documentation of intent, not because it's reachable.
        if nb in graph.po_bits:
            continue  # PI-straight-through-to-PO: used, just not by a gate
        if not design.net_fanout.get(nb):
            unused_input_ports.append(nb)
    # F6: `graph.pi_bits` is a set, so iterating it directly is ordered by
    # PYTHONHASHSEED -- sort for a deterministic result, same as
    # `floating_inputs` above (this file already has that comment two
    # sections up; this list and `unconnected_outputs` below were
    # inconsistently left unsorted).
    unused_input_ports.sort(key=lambda nb: (nb.name, nb.bit if nb.bit is not None else -1))

    # Deliberately does NOT exclude PI-straight-through-to-PO bits the way
    # `unused_input_ports` above does (asymmetric on purpose, not an
    # oversight): as explained above, `graph.pi_bits` and `graph.po_bits`
    # can never share a NetBit (each Signal has exactly one Direction), so
    # a PI-straight-through-to-PO bit can never actually occur here either,
    # which makes the exclusion above dead code too -- adding a matching
    # exclusion here would just make the two blocks symmetrically
    # unreachable instead of asymmetrically unreachable. Left as-is rather
    # than "fixed" for that reason.
    unconnected_outputs: list[NetBit] = sorted(
        (nb for nb in graph.po_bits if design.net_driver.get(nb) is None),
        key=lambda nb: (nb.name, nb.bit if nb.bit is not None else -1),
    )

    dangling_outputs: list[Gate] = []
    for gate in design.gates:
        out_pin = OUTPUT_PIN[gate.gate_type]
        out_nb = gate.pins.get(out_pin)
        if not isinstance(out_nb, NetBit):
            continue
        if out_nb in graph.po_bits:
            continue
        if not design.net_fanout.get(out_nb):
            dangling_outputs.append(gate)

    dead_internal: list[NetBit] = []
    for sig in design.signals.values():
        if sig.direction != Direction.INTERNAL:
            continue
        for nb in sig.bits():
            if design.net_driver.get(nb) is None and not design.net_fanout.get(nb):
                dead_internal.append(nb)

    return FloatingSignalsResult(
        floating_input_nets_referenced_but_undriven=floating_inputs,
        declared_input_ports_completely_unused=unused_input_ports,
        unconnected_output_ports_undriven=unconnected_outputs,
        unconnected_gate_input_pins=unconnected_pins,
        dangling_gate_outputs_never_consumed=dangling_outputs,
        dead_internal_wire_bits=dead_internal,
    )
