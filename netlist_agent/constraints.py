"""Structural hard constraints that must keep holding across every
subsequent request in a testcase, per QA A63 ("if request 1 enforces max
fanout 4 and request 2 remaps the entire design, must the final design still
satisfy max fanout 4?" -> Yes) and `Problem_Description/A_20260212.pdf` sec 5
("If any hard requirement is violated, the testcase gets no credit.").

A `StructuralConstraint` is recorded once, at the request that establishes a
numeric bound (see router.py's `handle_request` docstring for the single
enforcement point), and is re-checked -- and, where possible, re-enforced --
after every later request for the rest of the testcase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional

from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import OUTPUT_PIN, GateType, NetBit
from netlist_agent.netref import NetRefError, resolve_bit, resolve_bits
from netlist_agent.transform import limit_fanout, limit_fanout_nets

if TYPE_CHECKING:
    from netlist_agent.ir import Design

from netlist_agent.analysis import direct_fanout, max_fanout_among, max_fanout_overall


def _fanout_tree_closure(graph: NetlistGraph, netbits: Iterable[NetBit]) -> list[NetBit]:
    """Every net-bit a `max_fanout` bound on `netbits` actually constrains:
    `netbits` themselves, plus the transitive closure of BUF gates that
    `transform._insert_fanout_tree` would have inserted (or already has) to
    relieve them.

    `_insert_fanout_tree`'s docstring promises the bound covers not just the
    scoped net-bit(s) but "no intermediate BUF feeds more than `arity`
    nodes" -- `_balance_one_net` passes `arity=max_fanout`, i.e. the SAME
    bound this constraint records. So a `max_fanout` constraint's scope is
    the whole relief tree, not just the net-bit(s) it was written against;
    checking/restoring only the root would miss every BUF the tree grew (or
    already had) to relieve it, including ones a later `deduplicate_gates`
    collapsed back down (see the module docstring's dedup regression).

    Deliberately walks EVERY BUF forwarding the constrained signal's value,
    not just ones `limit_fanout_net` itself inserted: the `_LIMIT_FANOUT_NET_RE`
    that establishes this constraint accepts two wordings, routed to the same
    handler and recording the same constraint -- "to reduce its fanout to at
    most N" and "so that no gate has fanout greater than N". Under the second
    wording, a BUF that forwards this signal to N-or-more loads is squarely in
    scope no matter who inserted it (e.g. one left behind when some other
    operation retyped a gate to BUF in place, such as
    `simplify_constant_inputs` folding `AND(x, 1)`). Narrowing the scope to
    "only the tree `limit_fanout_net` itself built" would make the same
    constraint's meaning depend on which of the two equivalent wordings the
    user happened to use. It would also need a "is this BUF part of the
    tree" test, and the only one available -- `fresh_net()`'s `t_net_`
    name prefix -- is a naming convention, not a guarantee: it stops meaning
    anything after a rename or after `deduplicate_gates` merges the tree's
    BUFs with pre-existing ones. Erring wide is functionally safe (inserting
    an extra buffer preserves equivalence), and QA A63 + spec sec 5 make a
    missed violation zero-credit for the whole testcase, so erring wide is
    the right side to be wrong on. The cost: this can insert a buffer on a
    net the user never named.
    """
    seen: set[NetBit] = set()
    frontier: list[NetBit] = list(netbits)
    closure: list[NetBit] = []
    while frontier:
        nb = frontier.pop()
        if nb in seen:
            continue
        seen.add(nb)
        closure.append(nb)
        for load in direct_fanout(graph, nb):
            if load.kind != "gate" or load.gate.gate_type != GateType.BUF:
                continue
            out_pin = load.gate.pins.get(OUTPUT_PIN[GateType.BUF])
            if out_pin is None:
                continue
            frontier.append(out_pin)
    return closure


@dataclass(frozen=True)
class CheckResult:
    """Outcome of re-checking one `StructuralConstraint` against the current
    design.

    `unresolvable` (the token this constraint scopes to no longer resolves
    on the current design -- e.g. a later request renamed it) takes
    precedence over `holds`/`measured`, both of which are meaningless in
    that case (`measured` is `None`).

    `culprit` is the net-bit `measured` was actually taken from -- for
    `max_fanout`, whichever node in the checked scope (the whole relief
    tree, per `_fanout_tree_closure`, or the whole design) turned out to
    have the highest fanout; it need NOT be `scope_token`/`cone_of` itself
    (see the constraints.py module for why the tree can extend past the
    node named in the request). Without this, a caller has no way to say
    which node `measured` is actually about, and ends up implying it is the
    constrained token's own fanout even when it is not. `max_depth` has no
    such "which node" concept (`measured` there is a single depth number,
    not an aggregate over a set of nodes), so it is always `None`."""

    unresolvable: bool
    holds: bool
    measured: Optional[int]
    culprit: Optional[NetBit] = None


@dataclass(frozen=True)
class StructuralConstraint:
    """One hard structural bound established by an earlier request.

    `scope_token` and `cone_of` are stored as the user-given TOKEN (e.g.
    "n1"), not a resolved `NetBit` -- a later request may rename the signal
    (see test38's own L13), and QA A61 already establishes that a later
    reference resolves against the CURRENT design, not the one in force
    when it was written down. Storing a `NetBit` here would silently turn
    into a reference to a since-renamed ghost; re-resolving the token fresh
    on every `check()` call is what QA A61 requires anyway."""

    kind: str  # "max_fanout" | "max_depth"
    bound: int
    scope_token: Optional[str]  # None = whole design; else the net token this bounds
    cone_of: Optional[str]  # None = whole-design depth; else the cone root token
    request_id: int  # response id of the request that established this constraint

    @property
    def bound_desc(self) -> str:
        """The bound itself ("max fanout <= 4"), with no location or
        request-id suffix -- shared by `describe()` and by callers (see
        `router._enforce_structural_constraints`) that need to splice the
        location in themselves (e.g. "recorded on n1 (request 4)")."""
        return f"max fanout <= {self.bound}" if self.kind == "max_fanout" else f"max depth <= {self.bound}"

    def describe(self) -> str:
        base = self.bound_desc
        if self.kind == "max_fanout" and self.scope_token is not None:
            base += f" on {self.scope_token}"
        elif self.kind == "max_depth" and self.cone_of is not None:
            base += f" on the cone of {self.cone_of}"
        return f"{base} (from request {self.request_id})"

    @property
    def restorable(self) -> bool:
        # max_fanout has a general "insert buffers until it holds" operation
        # (transform.limit_fanout/limit_fanout_net). max_depth does not: there
        # is no general "reduce depth to exactly N" primitive -- the ABC-backed
        # depth optimizers *minimize* depth on a best-effort basis and are not
        # guaranteed to reach any particular target, so pretending this is
        # restorable (and silently swallowing a failure to do so) would be
        # exactly the "no try/except pretending it worked" anti-pattern this
        # module is meant to avoid.
        return self.kind == "max_fanout"

    def check(self, design: "Design") -> CheckResult:
        graph = NetlistGraph(design)
        culprit: Optional[NetBit] = None
        if self.kind == "max_fanout":
            if self.scope_token is None:
                culprit, measured = max_fanout_overall(graph)
            else:
                try:
                    netbits = resolve_bits(design, self.scope_token)
                except NetRefError:
                    return CheckResult(unresolvable=True, holds=False, measured=None)
                tree = _fanout_tree_closure(graph, netbits)
                culprit, measured = max_fanout_among(graph, tree)
        else:  # max_depth
            if self.cone_of is None:
                measured = graph.max_design_depth()
            else:
                try:
                    nb = resolve_bit(design, self.cone_of)
                except NetRefError:
                    return CheckResult(unresolvable=True, holds=False, measured=None)
                measured = graph.depth_to_sink(nb)
        return CheckResult(unresolvable=False, holds=measured <= self.bound, measured=measured, culprit=culprit)

    def restore(self, design: "Design") -> int:
        """Re-enforce this constraint in place, returning the number of BUF
        gates inserted to do so. Only valid when `restorable` is True --
        callers must check that first (see its docstring); raises otherwise
        rather than silently doing nothing.

        Cost, measured rather than guessed: this used to call `limit_fanout_net`
        once per snapshotted node, and that function builds its own fresh
        `NetlistGraph` per call, so restoring a relief tree of N nodes was
        O(N x design) -- on test33 (64k gates), capping its worst net (fanout
        4211) grows a 1408-node tree, and after a `remap_to_basis` to
        nand_not one restore pass measured **384.7s**, over the problem
        statement's 300s budget (`Problem_Description/A_20260212.pdf` p.4)
        for a non-basic request -- and that was to insert 0 buffers, since
        every node in the snapshot was already back in bound by the time it
        was reached. Below `resolve_bits`/`_fanout_tree_closure`, this now
        calls `transform.limit_fanout_nets`, which hoists ONE `NetlistGraph`
        out of the loop and drives `transform._balance_one_net` directly for
        every node in the tree, the same shape `transform.limit_fanout` (the
        `scope_token is None` branch just below) already uses. This is sound
        for the same reason that one already is: distinct nets' load sets are
        disjoint (a gate pin has exactly one driver), so relieving one node
        cannot invalidate another node's load list. It is also sound because
        `analysis.direct_fanout`/`fanout_count` -- the only reads
        `_balance_one_net` does through `graph` on this path -- read
        `graph.design.net_fanout`/`net_driver` directly, i.e. live, mutable
        state on the `Design`, not anything `NetlistGraph.__init__` cached up
        front. The only caches `NetlistGraph.__init__` does build (`po_bits`/
        `po_port_of`, both derived from `design.ports`) are never touched by
        inserting a buffer, so this one graph object cannot go stale for
        anything this path reads from it. Measured equivalent
        (same buffers-inserted count, same final gate count, same final
        worst fanout) across 15 before/after cases spanning 4 designs x 4
        constraint-breaking transforms; the test33/nand_not case above drops
        from 384.7s to 0.32s (~1202x)."""
        if not self.restorable:
            raise ValueError(f"{self.kind} constraints cannot be restored (no general operation exists)")
        if self.scope_token is None:
            return limit_fanout(design, self.bound)
        netbits = resolve_bits(design, self.scope_token)
        # Snapshot the whole relief tree's node list up front, then process
        # it as a single `limit_fanout_nets` call -- NOT interleaved with the
        # walk -- because inserting a buffer changes `net_fanout`/
        # `net_driver`, which would invalidate an in-progress traversal of
        # the very tree being fixed. `limit_fanout_nets` re-measures each
        # net-bit's current fanout itself off its own single `NetlistGraph`,
        # so passing every snapshotted node (not just the ones measured
        # over-bound before any of them were touched) is safe: it is a no-op
        # for any node that isn't (or is no longer) over `self.bound`.
        graph = NetlistGraph(design)
        tree = _fanout_tree_closure(graph, netbits)
        return limit_fanout_nets(design, tree, self.bound)
