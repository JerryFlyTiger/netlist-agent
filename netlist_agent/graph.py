"""Structural/connectivity graph layer over a parsed :class:`~netlist_agent.ir.Design`.

This module builds a purely structural view of a netlist's combinational
connectivity and exposes traversal primitives (reachability, depth, path
counting/enumeration, cut-signal queries) on top of it. No Boolean-function
reasoning is performed anywhere in this module.

Sequential-boundary convention (enforced everywhere in this module, even
though ``Design.net_driver``/``Design.net_fanout`` are purely structural and
know nothing about it):
  * Backward traversal stops the moment it reaches a net-bit driven by a DFF
    (its Q output) -- that DFF is treated as a pseudo primary-input/source,
    never recursing into the DFF's D/CK/RN/SN pins.
  * Forward traversal never continues past a DFF gate reached via one of its
    input pins -- the DFF is treated as a pseudo primary-output/sink.
  * A "path" is measured in number of *combinational gates* traversed. A
    direct wire (source net-bit literally identical to the target net-bit,
    e.g. a DFF.Q pin wired straight into a PO) has depth 0 / is a length-0
    path, since zero gates sit between the two.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Union

from netlist_agent.ir import (
    Design,
    Direction,
    Gate,
    GateType,
    NetBit,
    OUTPUT_PIN,
    Pin,
)


class CombinationalCycleError(Exception):
    """Raised when a topological pass detects a cycle in the combinational graph.

    The combinational portion of a netlist (after excluding DFF D->Q
    traversal) is required to be a DAG; if it isn't, callers must find out
    explicitly rather than have the traversal loop forever or "handle" it.
    """


@dataclass(frozen=True)
class DffPin:
    """Identifies one specific DFF instance's D/CK/RN/SN pin.

    Needed (as opposed to just a NetBit) because several DFF instances' pins
    can be driven by the *same* net-bit, yet a caller may want to address one
    specific instance's pin (e.g. "depth from PI x to gate g4's D pin").
    Only ever meaningful as a sink/target endpoint, never as a traversal
    source (a DFF pin does not drive anything forward -- see module docstring).
    """

    inst_name: str
    pin: str  # one of "D", "CK", "RN", "SN"


# A traversal endpoint: either a plain net-bit (usable as a source -- PI, PO,
# or a DFF's Q -- or as a target), or a specific DFF instance's D/CK/RN/SN pin
# (target/sink only).
Endpoint = Union[NetBit, DffPin]


@dataclass(frozen=True)
class Load:
    """One unit of fanout: either a gate consuming a net-bit on one input pin,
    or a primary-output port connection. A PO connection is not a Gate pin in
    the IR (it is simply "this net-bit is wired to an output port"), so it is
    represented here as its own variant rather than folded into the gate case.
    This is the one PO-fanout-load representation used consistently by every
    fanout-reporting function in analysis.py.
    """

    kind: str  # "gate" or "po"
    gate: Optional[Gate] = None
    pin: Optional[str] = None
    port_name: Optional[str] = None


def _gate_output_pin(gate: Gate) -> Pin:
    return gate.pins.get(OUTPUT_PIN[gate.gate_type])


def _topo_order(
    nodes: Iterable[str], succs_within: Callable[[str], Iterable[str]]
) -> list[str]:
    """Kahn topological sort restricted to `nodes`, using only edges that stay
    within `nodes` (as reported by `succs_within`). Raises
    CombinationalCycleError if not all nodes can be ordered.
    """
    nodes = list(nodes)
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for n in nodes:
        for s in succs_within(n):
            indeg[s] += 1
    queue: deque[str] = deque(n for n in nodes if indeg[n] == 0)
    order: list[str] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for s in succs_within(n):
            indeg[s] -= 1
            if indeg[s] == 0:
                queue.append(s)
    if len(order) != len(nodes):
        raise CombinationalCycleError(
            f"combinational cycle detected: {len(nodes) - len(order)} gate(s) "
            "could not be topologically ordered"
        )
    return order


class NetlistGraph:
    """Precomputed structural graph over a Design, plus traversal primitives.

    Construct once per Design; the precomputed adjacency/PI/PO bit sets are
    then reused by every query method (and by analysis.py).
    """

    def __init__(self, design: Design) -> None:
        self.design = design
        self.gate_by_name: dict[str, Gate] = {g.inst_name: g for g in design.gates}

        self.pi_bits: set[NetBit] = set()
        self.po_bits: set[NetBit] = set()
        # Which output port each PO net-bit belongs to (a PO connection is not
        # a Gate pin in the IR, so this is precomputed once for O(1) lookup
        # rather than re-scanning ports/Signal.bits() on every query).
        self.po_port_of: dict[NetBit, str] = {}
        for port in design.ports:
            sig = design.signals[port.name]
            if port.direction == Direction.INPUT:
                self.pi_bits.update(sig.bits())
            elif port.direction == Direction.OUTPUT:
                self.po_bits.update(sig.bits())
                for nb in sig.bits():
                    self.po_port_of[nb] = port.name

        self.dff_gates: list[Gate] = [g for g in design.gates if g.gate_type == GateType.DFF]
        self.dff_q_bits: set[NetBit] = set()
        for g in self.dff_gates:
            q = g.pins.get("Q")
            if isinstance(q, NetBit):
                self.dff_q_bits.add(q)

        # Deduplicated gate-level adjacency over non-DFF gates only: an edge
        # g1 -> g2 exists iff g1's output feeds one of g2's input pins and
        # g2 is not a DFF (a DFF never propagates forward past itself, and
        # never appears as a source of an outgoing edge since its Q output
        # is a boundary/pseudo-source, not a combinational hop).
        self._succ: dict[str, set[str]] = {}
        self._pred: dict[str, set[str]] = {}
        # Gates whose output directly drives a DFF pin (any of D/CK/RN/SN).
        self._feeds_dff: set[str] = set()
        for g in design.gates:
            if g.gate_type == GateType.DFF:
                continue
            out_nb = _gate_output_pin(g)
            if not isinstance(out_nb, NetBit):
                continue
            for consumer in design.net_fanout.get(out_nb, ()):
                if consumer.gate_type == GateType.DFF:
                    self._feeds_dff.add(g.inst_name)
                    continue
                self._succ.setdefault(g.inst_name, set()).add(consumer.inst_name)
                self._pred.setdefault(consumer.inst_name, set()).add(g.inst_name)

        self._sink_gates_po: set[str] = set()
        self._sink_gates_dff: set[str] = set(self._feeds_dff)
        for g in design.gates:
            if g.gate_type == GateType.DFF:
                continue
            out_nb = _gate_output_pin(g)
            if isinstance(out_nb, NetBit) and out_nb in self.po_bits:
                self._sink_gates_po.add(g.inst_name)

        self._all_gate_names: set[str] = {
            g.inst_name for g in design.gates if g.gate_type != GateType.DFF
        }
        self._global_topo_order: Optional[list[str]] = None
        self._dp_any: Optional[dict[str, int]] = None
        self._dp_from_pi: Optional[dict[str, Optional[int]]] = None
        self._dp_from_dffq: Optional[dict[str, Optional[int]]] = None
        self._dp_to_sink: Optional[dict[str, int]] = None
        self._reg_to_reg_stats: Optional["RegToRegPathStats"] = None

    # ------------------------------------------------------------------
    # Endpoint resolution helpers
    # ------------------------------------------------------------------

    def resolve_endpoint(self, ep: Endpoint) -> Optional[NetBit]:
        """Resolve an Endpoint to the underlying NetBit it refers to, or None
        if it refers to a Const/unconnected pin (no connectivity to trace).
        """
        if isinstance(ep, NetBit):
            return ep
        gate = self.gate_by_name[ep.inst_name]
        value = gate.pins.get(ep.pin)
        return value if isinstance(value, NetBit) else None

    def _fed_by_gate(self, nb: NetBit) -> Optional[Gate]:
        """Non-DFF driver gate of nb, or None if driven by a DFF/PI/nothing."""
        driver = self.design.net_driver.get(nb)
        if driver is not None and driver.gate_type != GateType.DFF:
            return driver
        return None

    # ------------------------------------------------------------------
    # Fanin/fanout cones (capabilities 8, 9)
    # ------------------------------------------------------------------

    def forward_reachable_gates(self, start: NetBit) -> set[str]:
        """Instance names of every non-DFF gate forward-reachable from `start`
        (the transitive fanout cone), respecting the DFF forward boundary.
        """
        visited: set[str] = set()
        frontier: deque[str] = deque()
        for consumer in self.design.net_fanout.get(start, ()):
            if consumer.gate_type != GateType.DFF and consumer.inst_name not in visited:
                visited.add(consumer.inst_name)
                frontier.append(consumer.inst_name)
        while frontier:
            name = frontier.popleft()
            for succ in self._succ.get(name, ()):
                if succ not in visited:
                    visited.add(succ)
                    frontier.append(succ)
        return visited

    def backward_reachable_gates(self, start: NetBit) -> set[str]:
        """Instance names of every non-DFF gate backward-reachable from
        `start` (the transitive fanin cone), respecting the DFF backward
        boundary.
        """
        visited: set[str] = set()
        frontier: deque[str] = deque()
        driver = self._fed_by_gate(start)
        if driver is not None:
            visited.add(driver.inst_name)
            frontier.append(driver.inst_name)
        while frontier:
            name = frontier.popleft()
            for pred in self._pred.get(name, ()):
                if pred not in visited:
                    visited.add(pred)
                    frontier.append(pred)
        return visited

    def backward_cone_with_boundary_dffs(self, start: NetBit) -> set[str]:
        """Instance names of `start`'s fanin cone per QA A94: the non-DFF
        cone (`backward_reachable_gates`) PLUS every boundary DFF -- a DFF
        whose Q output feeds (directly) a gate already in that non-DFF cone
        -- PLUS, when `start` itself is a DFF's Q, that DFF (A94's "when X
        is itself a DFF's Q, the cone is 1: that DFF itself" rule, which the
        first clause alone would miss since such a DFF has no downstream
        gate inside the cone to make it a boundary).

        This is a distinct traversal from `backward_reachable_gates`, not a
        replacement for it: `backward_reachable_gates` still stops AT the
        DFF boundary (never includes the DFF), which is exactly what the
        cone-remapping/decomposition transforms in router.py need (they
        rewrite non-DFF gates only, and A72/A80 say DFFs are cone
        boundaries -- reached but not traversed past). Only analysis-style
        "how big/what's in this cone" callers want the DFF counted.
        """
        non_dff = self.backward_reachable_gates(start)
        result: set[str] = set(non_dff)
        for name in non_dff:
            gate = self.gate_by_name[name]
            out_pin = OUTPUT_PIN[gate.gate_type]
            for k, v in gate.pins.items():
                if k == out_pin or not isinstance(v, NetBit):
                    continue
                driver = self.design.net_driver.get(v)
                if driver is not None and driver.gate_type == GateType.DFF:
                    result.add(driver.inst_name)
        driver = self.design.net_driver.get(start)
        if driver is not None and driver.gate_type == GateType.DFF:
            result.add(driver.inst_name)
        return result

    # ------------------------------------------------------------------
    # Global depth aggregates (capabilities 13, 14, 15, 16)
    # ------------------------------------------------------------------

    def _ensure_global_topo(self) -> list[str]:
        if self._global_topo_order is None:
            self._global_topo_order = _topo_order(
                self._all_gate_names, lambda n: self._succ.get(n, ())
            )
        return self._global_topo_order

    def _dp_any_source(self) -> dict[str, int]:
        """dp[g] = length (#gates, inclusive of g) of the longest chain ending
        at g, from ANY true source (PI, DFF.Q, Const, or unconnected pin).
        A gate with no gate-predecessors is, by construction, fed only by
        such true sources, so its base depth is 1.
        """
        if self._dp_any is not None:
            return self._dp_any
        order = self._ensure_global_topo()
        dp: dict[str, int] = {}
        for g in order:
            preds = self._pred.get(g, ())
            dp[g] = 1 + max((dp[p] for p in preds), default=0)
        self._dp_any = dp
        return dp

    def _dp_to_sink_all(self) -> dict[str, int]:
        """ext[g] = length (#gates, inclusive of g) of the longest chain
        STARTING at g and ending at any true sink (PO or DFF.D); 0 if g
        cannot reach any sink at all (dangling logic that never influences
        an observable output). Exactly symmetric to `_dp_any_source` above,
        but walked backwards: seeded at the same `_sink_gates_po |
        _sink_gates_dff` sets `max_design_depth` uses, over `_succ` instead
        of `_pred`, in REVERSE topological order.
        """
        if self._dp_to_sink is not None:
            return self._dp_to_sink
        order = self._ensure_global_topo()
        sinks = self._sink_gates_po | self._sink_gates_dff
        ext: dict[str, int] = {}
        for g in reversed(order):
            candidates: list[int] = []
            if g in sinks:
                candidates.append(1)
            for s in self._succ.get(g, ()):
                sv = ext.get(s, 0)
                if sv > 0:
                    candidates.append(1 + sv)
            ext[g] = max(candidates) if candidates else 0
        self._dp_to_sink = ext
        return ext

    def depth_through_gate(self, name: str) -> Optional[int]:
        """Depth (#gates) of the longest source-to-sink combinational chain
        that passes through gate `name` (inclusive of `name` itself); None
        if `name` cannot reach any true sink (PO or DFF.D) at all, in which
        case "does it lie on a maximum-depth path" is vacuously no. Combines
        `_dp_any_source` (source -> name) with `_dp_to_sink_all` (name ->
        sink); the `-1` avoids double-counting `name`, which both DPs
        include. `name` must be a non-DFF gate instance name (DFFs are a
        sequential boundary, not part of this module's combinational depth
        DPs at all -- callers should special-case them before calling this).
        """
        ext = self._dp_to_sink_all()
        e = ext.get(name, 0)
        if e == 0:
            return None
        return self._dp_any_source()[name] + e - 1

    def _dp_rooted_at(self, fed_by: Callable[[Gate], bool]) -> dict[str, Optional[int]]:
        """dp[g] = longest chain ending at g (inclusive) among chains whose
        first gate is directly fed (on any pin) per `fed_by`; None if no such
        chain reaches g at all. One global topological pass, reused for both
        the DFF.Q-rooted (#14) and PI-rooted (#15) aggregates -- this avoids
        a separate BFS/DP per individual DFF or PI, which would not scale.
        """
        order = self._ensure_global_topo()
        dp: dict[str, Optional[int]] = {}
        for name in order:
            gate = self.gate_by_name[name]
            candidates: list[int] = []
            if fed_by(gate):
                candidates.append(1)
            for p in self._pred.get(name, ()):
                pv = dp.get(p)
                if pv is not None:
                    candidates.append(1 + pv)
            dp[name] = max(candidates) if candidates else None
        return dp

    def _dp_from_pi_source(self) -> dict[str, Optional[int]]:
        if self._dp_from_pi is None:
            self._dp_from_pi = self._dp_rooted_at(
                lambda gate: any(
                    isinstance(v, NetBit) and v in self.pi_bits
                    for k, v in gate.pins.items()
                    if k != OUTPUT_PIN[gate.gate_type]
                )
            )
        return self._dp_from_pi

    def _dp_from_dffq_source(self) -> dict[str, Optional[int]]:
        if self._dp_from_dffq is None:
            self._dp_from_dffq = self._dp_rooted_at(
                lambda gate: any(
                    isinstance(v, NetBit) and v in self.dff_q_bits
                    for k, v in gate.pins.items()
                    if k != OUTPUT_PIN[gate.gate_type]
                )
            )
        return self._dp_from_dffq

    def max_design_depth(self) -> int:
        """Overall maximum combinational depth, over all PI/DFF.Q -> PO/DFF.D
        pairs in the whole design (capability 13). 0 if the design has no
        combinational gates feeding any sink (all connections are direct wires).
        """
        dp = self._dp_any_source()
        sinks = self._sink_gates_po | self._sink_gates_dff
        return max((dp[g] for g in sinks), default=0)

    def max_reg_to_reg_depth(self) -> int:
        """Maximum combinational depth restricted to DFF.Q -> DFF.D paths only
        (capability 14). 0 if no combinational-gate-mediated reg-to-reg path
        exists (including the degenerate case of no DFFs at all).
        """
        dp = self._dp_from_dffq_source()
        return max((dp[g] for g in self._sink_gates_dff if dp.get(g) is not None), default=0)

    def max_pi_to_dff_d_depth(self) -> int:
        """Maximum combinational depth from any PI to any DFF D/CK/RN/SN pin
        (capability 15)."""
        dp = self._dp_from_pi_source()
        return max((dp[g] for g in self._sink_gates_dff if dp.get(g) is not None), default=0)

    def depth_to_sink(self, nb: NetBit) -> int:
        """Depth (#gates) of the longest chain (from any true source) driving
        `nb`. 0 if `nb` is undriven or driven directly by a DFF (Q wired
        straight through, no combinational gates).
        """
        driver = self.design.net_driver.get(nb)
        if driver is None or driver.gate_type == GateType.DFF:
            return 0
        return self._dp_any_source()[driver.inst_name]

    def per_output_depths(self) -> dict[NetBit, int]:
        """Depth of each PO's and each DFF.D pin's fanin cone (capability 16),
        keyed by the PO's net-bit / the D-pin's driving net-bit. Note distinct
        DFF instances whose D pins share one net-bit collapse to one entry
        here (same depth value anyway, since depth depends only on the net).
        """
        result: dict[NetBit, int] = {}
        for nb in self.po_bits:
            result[nb] = self.depth_to_sink(nb)
        for gate in self.dff_gates:
            d = gate.pins.get("D")
            if isinstance(d, NetBit):
                result[d] = self.depth_to_sink(d)
        return result

    # ------------------------------------------------------------------
    # Point-to-point depth (capability 12)
    # ------------------------------------------------------------------

    def _forward_reachable_with_restricted_dp(
        self, source: NetBit
    ) -> tuple[set[str], dict[str, int], dict[str, Optional[str]]]:
        """Reachable non-DFF gate set S from `source`, plus a longest-path DP
        restricted to S: dp[g] = #gates from `source` to g inclusive, seeded
        by whether g is directly fed by `source` itself. parent[g] records the
        predecessor achieving the max (or None if g is a base case).
        """
        reachable = self.forward_reachable_gates(source)
        order = _topo_order(
            reachable, lambda n: (s for s in self._succ.get(n, ()) if s in reachable)
        )
        dp: dict[str, int] = {}
        parent: dict[str, Optional[str]] = {}
        for name in order:
            gate = self.gate_by_name[name]
            direct = any(
                v == source for k, v in gate.pins.items() if k != OUTPUT_PIN[gate.gate_type]
            )
            best = 1 if direct else None
            best_parent: Optional[str] = None
            # `_pred` values are `set[str]`, whose iteration order depends on
            # PYTHONHASHSEED -- sorted() here makes the tie-break among
            # multiple predecessors achieving the same max DP value
            # deterministic, so `depth_between`'s returned path is
            # reproducible across processes/seeds, not just within one
            # (the depth VALUE was already deterministic via `max()`; only
            # the specific `best_parent` chosen on a tie was not).
            for p in sorted(self._pred.get(name, ())):
                if p in reachable and p in dp:
                    cand = 1 + dp[p]
                    if best is None or cand > best:
                        best = cand
                        best_parent = p
            # `direct` alone (best==1, best_parent None) may still be beaten
            # by a longer predecessor chain computed above.
            if best is None:
                # Unreachable via this restricted subgraph's edges (shouldn't
                # happen for nodes in `reachable`, but guard defensively).
                continue
            dp[name] = best
            parent[name] = best_parent
        return reachable, dp, parent

    def depth_between(self, source: NetBit, target: Endpoint) -> Optional[tuple[int, list[Gate]]]:
        """Depth and longest path between two specific points (capability 12).

        Returns (depth, path_gates) or None if `target` is unreachable from
        `source`. `path_gates` is the ordered list of combinational gates
        traversed (empty for a depth-0 direct-wire connection).
        """
        target_nb = self.resolve_endpoint(target)
        if target_nb is None:
            return None
        if target_nb == source:
            return 0, []
        driver = self._fed_by_gate(target_nb)
        if driver is None:
            return None
        reachable, dp, parent = self._forward_reachable_with_restricted_dp(source)
        if driver.inst_name not in dp:
            return None
        depth = dp[driver.inst_name]
        path: list[str] = []
        cur: Optional[str] = driver.inst_name
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return depth, [self.gate_by_name[n] for n in path]

    # ------------------------------------------------------------------
    # Path existence / counting / enumeration (capabilities 17-20)
    # ------------------------------------------------------------------

    def _succ_avoiding(self, name: str, avoid: Optional[NetBit]) -> Iterable[str]:
        if avoid is not None:
            gate = self.gate_by_name[name]
            out_nb = _gate_output_pin(gate)
            if out_nb == avoid:
                return ()
        # `_succ` values are `set[str]`, whose iteration order depends on
        # PYTHONHASHSEED -- sorted() here is what makes `enumerate_paths`'s
        # DFS (and thus `_iter_reg_to_reg_paths`'s sample order in
        # router.py) reproducible across processes/seeds, not just within
        # one. Cheap: successor sets are small (fanout of one gate).
        return sorted(self._succ.get(name, ()))

    def _forward_reachable_avoiding(self, source: NetBit, avoid: Optional[NetBit]) -> set[str]:
        if avoid is not None and source == avoid:
            return set()
        visited: set[str] = set()
        frontier: deque[str] = deque()
        if avoid is None or source != avoid:
            for consumer in self.design.net_fanout.get(source, ()):
                if consumer.gate_type != GateType.DFF and consumer.inst_name not in visited:
                    visited.add(consumer.inst_name)
                    frontier.append(consumer.inst_name)
        while frontier:
            name = frontier.popleft()
            for succ in self._succ_avoiding(name, avoid):
                if succ not in visited:
                    visited.add(succ)
                    frontier.append(succ)
        return visited

    def path_exists(
        self, source: NetBit, target: Endpoint, avoid: Optional[NetBit] = None
    ) -> bool:
        """Whether a (possibly length-0) path exists from `source` to `target`,
        optionally with `avoid`'s net-bit node removed from the graph
        (capability 17).
        """
        target_nb = self.resolve_endpoint(target)
        if target_nb is None:
            return False
        if target_nb == source:
            return avoid is None or avoid != source
        if avoid is not None and target_nb == avoid:
            return False
        driver = self._fed_by_gate(target_nb)
        if driver is None:
            return False
        reachable = self._forward_reachable_avoiding(source, avoid)
        return driver.inst_name in reachable

    def path_count(
        self, source: NetBit, target: Endpoint, avoid: Optional[NetBit] = None
    ) -> int:
        """Number of distinct paths (as gate sequences) from `source` to
        `target`, via DP over the DAG in topological order (capability 18).
        Uses Python's arbitrary-precision ints, so no overflow risk.
        """
        target_nb = self.resolve_endpoint(target)
        if target_nb is None:
            return 0
        if target_nb == source:
            return 1 if (avoid is None or avoid != source) else 0
        if avoid is not None and target_nb == avoid:
            return 0
        driver = self._fed_by_gate(target_nb)
        if driver is None:
            return 0
        reachable = self._forward_reachable_avoiding(source, avoid)
        if driver.inst_name not in reachable:
            return 0
        order = _topo_order(
            reachable,
            lambda n: (s for s in self._succ_avoiding(n, avoid) if s in reachable),
        )
        count: dict[str, int] = {}
        for name in order:
            gate = self.gate_by_name[name]
            direct = 1 if any(
                v == source for k, v in gate.pins.items() if k != OUTPUT_PIN[gate.gate_type]
            ) else 0
            total = direct + sum(
                count[p] for p in self._pred.get(name, ()) if p in reachable
            )
            count[name] = total
        return count[driver.inst_name]

    def enumerate_paths(
        self, source: NetBit, target: Endpoint, avoid: Optional[NetBit] = None
    ) -> Iterator[list[Gate]]:
        """Generator yielding each path from `source` to `target` as an
        ordered list of Gates, via backtracking DFS -- memory-safe for huge
        result sets since it does not memoize full path lists (capability 19).
        A length-0 path (direct wire) is yielded as an empty list.
        """
        target_nb = self.resolve_endpoint(target)
        if target_nb is None:
            return
        if target_nb == source:
            if avoid is None or avoid != source:
                yield []
            return
        if avoid is not None and target_nb == avoid:
            return
        driver = self._fed_by_gate(target_nb)
        if driver is None:
            return
        reachable = self._forward_reachable_avoiding(source, avoid)
        if driver.inst_name not in reachable:
            return
        target_name = driver.inst_name

        # Bidirectional pruning: `reachable` alone (forward from source) is
        # not enough -- the DFS below would still walk every path inside
        # dead-end subtrees that can never reach the target, which on large
        # cones is exponentially more work than the actual result set
        # (observed for real: a 91k-gate design where enumeration between
        # one PI/PO pair burned CPU for >10 minutes without yielding a
        # single path). Restricting to `useful` = (forward-reachable from
        # source) AND (backward-reachable to target) makes every DFS
        # descent end at the target, so total work is proportional to the
        # emitted paths' total length -- output-bound, not cone-bound. The
        # backward sweep mirrors `_succ_avoiding`'s edge semantics: an edge
        # out of a gate whose output net is `avoid` is cut.
        useful: set[str] = {target_name}
        frontier: deque[str] = deque([target_name])
        while frontier:
            name = frontier.popleft()
            for pred in self._pred.get(name, ()):
                if pred not in reachable or pred in useful:
                    continue
                if avoid is not None and _gate_output_pin(self.gate_by_name[pred]) == avoid:
                    continue
                useful.add(pred)
                frontier.append(pred)

        # Same cycle guard `path_count`/`depth_between` get for free from
        # their own topo-sort DP: the backtracking DFS below has no cycle
        # detection of its own, so without this upfront check a cyclic
        # subgraph would make it recurse forever instead of raising
        # CombinationalCycleError like every other path/depth query in this
        # module. The resulting order is discarded -- only the cycle check
        # (a `CombinationalCycleError` raise on failure) matters here.
        _topo_order(
            useful,
            lambda n: (s for s in self._succ_avoiding(n, avoid) if s in useful),
        )

        # Explicit-stack DFS: each frame is (gate_name, iterator over
        # successors-within-useful still to try). `path` mirrors the
        # stack's gate_name sequence for O(1) yield.
        start_names = [
            c.inst_name
            for c in self.design.net_fanout.get(source, ())
            if c.gate_type != GateType.DFF and c.inst_name in useful
        ]
        for start_name in start_names:
            path = [start_name]
            stack: list[Iterator[str]] = [
                iter(s for s in self._succ_avoiding(start_name, avoid) if s in useful)
            ]
            if start_name == target_name:
                yield [self.gate_by_name[n] for n in path]
            while stack:
                try:
                    nxt = next(stack[-1])
                except StopIteration:
                    stack.pop()
                    path.pop()
                    continue
                path.append(nxt)
                if nxt == target_name:
                    yield [self.gate_by_name[n] for n in path]
                stack.append(
                    iter(s for s in self._succ_avoiding(nxt, avoid) if s in useful)
                )

    # ------------------------------------------------------------------
    # Cut signals (capabilities 21, 22)
    # ------------------------------------------------------------------

    def is_cut_signal_for_some_pi_po_pair(self, nb: NetBit) -> bool:
        """Whether removing `nb` disconnects at least one PI->PO pair
        (capability 21). Exhaustive-with-early-exit over the PIs that reach
        `nb` and the POs `nb` reaches; correct for reasonable design sizes,
        but O(#reaching-PIs * #reached-POs) path_exists calls in the worst
        case -- not intended for use on huge whole-design sweeps.
        """
        back = self.backward_reachable_gates(nb)
        # A PI reaches `nb` if it IS `nb`, or if `nb`'s backward cone (`back`,
        # gate instance names) contains a gate directly fed by that PI.
        reaching_pis: set[NetBit] = set()
        for pi in self.pi_bits:
            if pi == nb:
                reaching_pis.add(pi)
                continue
            if any(
                any(
                    v == pi
                    for k, v in self.gate_by_name[gname].pins.items()
                    if k != OUTPUT_PIN[self.gate_by_name[gname].gate_type]
                )
                for gname in back
            ):
                reaching_pis.add(pi)

        fwd = self.forward_reachable_gates(nb)
        reached_pos = set()
        for po in self.po_bits:
            if po == nb:
                reached_pos.add(po)
                continue
            driver = self._fed_by_gate(po)
            if driver is not None and driver.inst_name in fwd:
                reached_pos.add(po)

        for pi in reaching_pis:
            for po in reached_pos:
                if self.path_exists(pi, po) and not self.path_exists(pi, po, avoid=nb):
                    return True
        return False

    def cut_nets_between(
        self, source: NetBit, target: Endpoint
    ) -> "CutResult":
        """All net-bits (excluding `source`/`target` themselves) whose removal
        disconnects `source` from `target` (capability 22), alongside the
        GATE instance driving each such net-bit (`cut_gates` -- per QA A87,
        "articulation point" means a gate only, not a net; `cut_nets` is kept
        for the pre-A87 net-level callers -- `analysis`/`llm/tools_schema`'s
        "cut nets between" tool -- which ask a related but distinct question
        and were not part of A87's ruling).

        O(#candidates * (V+E)) -- one `path_exists` (full BFS) call per
        candidate gate in `fwd & back`; correct for reasonable design sizes,
        but intended for a specific bounded (source, target) query, not a
        whole-design sweep (mirroring the same caveat on
        `is_cut_signal_for_some_pi_po_pair` above).
        """
        target_nb = self.resolve_endpoint(target)
        if target_nb is None or not self.path_exists(source, target):
            return CutResult(path_exists=False, cut_nets=[], cut_gates=[])
        if target_nb == source:
            return CutResult(path_exists=True, cut_nets=[], cut_gates=[])

        fwd = self.forward_reachable_gates(source)
        back = self.backward_reachable_gates(target_nb)
        candidates = fwd & back

        cuts: list[NetBit] = []
        cut_gates: list[str] = []
        for gname in candidates:
            gate = self.gate_by_name[gname]
            out_nb = _gate_output_pin(gate)
            if not isinstance(out_nb, NetBit) or out_nb in (source, target_nb):
                continue
            if not self.path_exists(source, target, avoid=out_nb):
                cuts.append(out_nb)
                cut_gates.append(gname)
        return CutResult(path_exists=True, cut_nets=cuts, cut_gates=cut_gates)


    # ------------------------------------------------------------------
    # Register-to-register combinational path counting (whole-design)
    # ------------------------------------------------------------------

    def reg_to_reg_path_stats(self) -> "RegToRegPathStats":
        """Count every register-to-register combinational path in the whole
        design: a distinct (source DFF, gate sequence, sink DFF) triple where
        `source`'s Q feeds the first gate and the last gate's output feeds
        `sink`'s D pin (CK/RN/SN never count as a sink).

        One topological-order DP over the whole design, NOT one `path_count`
        call per (source DFF, sink DFF) pair -- with O(#DFF) DFFs on each
        side that would be O(#DFF^2) `path_count` calls, each itself an
        O(V+E) pass; on a real ~2000-DFF design that is many orders of
        magnitude too slow. Instead:
          seed[g]  = number of DISTINCT source DFFs whose Q net feeds g on
                     any input pin (deduplicated -- the same DFF's Q wired to
                     two of g's pins still counts once)
          count[g] = seed[g] + sum(count[p] for p in _pred[g])
        `count[g]` is then exactly the number of (source DFF, gate-chain
        ending at g) pairs, and `count[g] * (#DFF D pins g's output drives)`
        summed over every gate is exactly the number of >=1-gate reg-to-reg
        paths -- the same "sum of predecessor DP values" shape as
        `path_count` above, generalized to many sources at once instead of
        one, so it stays a single linear DP pass.

        Zero-gate direct connections (a DFF's Q wired straight into another
        --or the same-- DFF's D, no combinational gate in between) are
        counted and reported SEPARATELY (`direct_wire_count`, with a few
        example triples in `direct_wire_examples`) -- NOT folded into
        `combinational_path_count`, matching the corpus's own convention.
        """
        if self._reg_to_reg_stats is not None:
            return self._reg_to_reg_stats
        order = self._ensure_global_topo()

        seed: dict[str, int] = {}
        for g in order:
            gate = self.gate_by_name[g]
            out_pin = OUTPUT_PIN[gate.gate_type]
            sources: set[NetBit] = {
                v
                for k, v in gate.pins.items()
                if k != out_pin and isinstance(v, NetBit) and v in self.dff_q_bits
            }
            seed[g] = len(sources)

        count: dict[str, int] = {}
        for g in order:
            count[g] = seed[g] + sum(count[p] for p in self._pred.get(g, ()))

        # Precompute, once, how many DFFs have their D pin on each net-bit --
        # O(#DFF) -- instead of scanning all DFFs for every gate in the total
        # loop below (that was O(#gates * #DFF), ~10^8 Python-level
        # iterations on a ~45k-gate/2.3k-DFF design).
        dff_d_count: dict[NetBit, int] = {}
        for dff in self.dff_gates:
            d = dff.pins.get("D")
            if isinstance(d, NetBit):
                dff_d_count[d] = dff_d_count.get(d, 0) + 1

        total = 0
        for g in order:
            out_nb = _gate_output_pin(self.gate_by_name[g])
            if not isinstance(out_nb, NetBit):
                continue
            n_dff_d = dff_d_count.get(out_nb, 0)
            if n_dff_d:
                total += count[g] * n_dff_d

        direct_wire_count = 0
        direct_wire_examples: list[tuple[str, NetBit, str]] = []
        for sink in sorted(self.dff_gates, key=lambda dff: dff.inst_name):
            d = sink.pins.get("D")
            if not isinstance(d, NetBit) or d not in self.dff_q_bits:
                continue
            for source in sorted(self.dff_gates, key=lambda dff: dff.inst_name):
                if source.pins.get("Q") == d:
                    direct_wire_count += 1
                    if len(direct_wire_examples) < 20:
                        direct_wire_examples.append((source.inst_name, d, sink.inst_name))

        self._reg_to_reg_stats = RegToRegPathStats(
            combinational_path_count=total,
            direct_wire_count=direct_wire_count,
            direct_wire_examples=direct_wire_examples,
        )
        return self._reg_to_reg_stats


@dataclass(frozen=True)
class RegToRegPathStats:
    """Result of `NetlistGraph.reg_to_reg_path_stats`. `direct_wire_examples`
    is capped at 20 triples (not exhaustive) purely to bound memory on
    designs with many direct connections; `direct_wire_count` is always the
    true total."""

    combinational_path_count: int
    direct_wire_count: int
    direct_wire_examples: list[tuple[str, NetBit, str]]


@dataclass(frozen=True)
class CutResult:
    """Result of `NetlistGraph.cut_nets_between`. `path_exists=False`
    distinguishes "no path at all" from "path exists but no cut nets" (an
    empty `cut_nets` list alone would be ambiguous between those two cases).

    `cut_gates` is the same set of cuts, reported as gate instance names
    instead of net-bits -- per QA A87, "articulation point" means a gate
    only, not a net, so `_h_articulation_points` (router.py) reads
    `cut_gates`; `cut_nets` remains for the net-level "cut nets between"
    query, a related but distinct question A87 did not rule on.
    """

    path_exists: bool
    cut_nets: list[NetBit]
    cut_gates: list[str]
