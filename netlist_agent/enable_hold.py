"""Flip-flop D-input enable/hold structure detection.

Definition adopted (matches the corpus's own Q12 ground truth, see
`ground_truth.json`): a flop `f` has a hold structure in its D-input logic
iff there exists a net `c` in D's fanin cone and a value `v` such that
forcing `c = v` makes D identically equal to `f`'s own Q, for every
assignment of every primary input and every OTHER flop's Q (all of those are
treated as free variables -- the same "free_pi" convention `abc_bridge.py`
and `property_check.py` already use elsewhere in this codebase). This is a
FREE-VARIABLE model: a conclusion here can legitimately describe a state
that is unreachable from reset, and says nothing about reset-reachability.

Four-stage pipeline, cost increasing stage over stage so the expensive
stages only ever run on what the cheaper ones couldn't settle:

  Stage 0 (`_fanin_cone`)      -- structural: is the flop's own Q even in
                                   its own D fanin cone? If not, hold is
                                   structurally impossible (`no_self_reference`).
  Stage 1 (`_candidates`,      -- bit-parallel random simulation, used ONLY
           `_degenerate_candidates`) to shrink Stage 2/3's search space
                                   (a necessary-condition filter for the
                                   ordinary case, an exhaustive near-to-far
                                   scan of the whole cone for the `x == 0`
                                   case -- see `_degenerate_candidates`);
                                   never a verdict by itself. See each
                                   function's own docstring for its
                                   soundness argument.
  Stage 2 (`_constprop`)       -- symbolic constant propagation over the
                                   forced cone: an exact structural PROOF
                                   when it succeeds, not a sample.
  Stage 3 (`_truth_table_proof`,  -- for whatever Stage 2 could not settle:
           `_prove_via_abc`)         a truth table when the forced cone's
                                   support is small enough to be cheap (its
                                   cost tracks the cone's gate count, not
                                   ABC's), else a formal `abc_bridge`
                                   equivalence check -- both routes share
                                   ONE combined per-flop attempt budget
                                   (`MAX_ABC_CANDIDATES_PER_FLOP`; see its
                                   own comment for why a separate, larger
                                   truth-table-only budget was tried and
                                   reverted -- 3x the per-query cost for a
                                   labeling-precision-only gain). Whatever is
                                   STILL unsettled after that is reported as
                                   "simulated support only" -- never silently
                                   folded into either verdict.

One thing this module deliberately does NOT do is treat `x == 0` in Stage 1
(D == Q on every simulated sample with nothing forced at all -- see
`EnableHoldResult.degenerate_sim`) as its own separate proof path. An
earlier version of this module did, and that was a real bug, caught by an
independent cross-check: at test40's own sample count the "unconditional
identity" hypothesis those flops' `x == 0` signal suggests is FALSE for
essentially all of them -- but they are still perfectly ordinary hold
structures, just ones whose control net is asserted so rarely that plain
random sampling never once saw the "load" branch fire (D = rare_condition ?
load : Q). Concretely: `x == 0` here is a sample-count artifact, not a
distinct circuit category -- see `_degenerate_candidates`'s docstring for
the fix, and the module docstring's own note on why `degenerate_sim` is
reported purely as a diagnostic, not a verdict bucket.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Union, cast

from netlist_agent.abc_bridge import ABCBridgeError, are_equivalent
from netlist_agent.ir import (
    Const,
    Design,
    Direction,
    Gate,
    GateType,
    NetBit,
    OUTPUT_PIN,
    Pin,
    Port,
    Signal,
)
from netlist_agent.netref import netbit_sort_key

# ---------------------------------------------------------------- tunables
NUM_SAMPLES = 2048
SIM_SEED = 20260812

# Stage 1 evidence threshold. MUST stay at 1: raising it drops real hold
# structures (measured on test40: threshold 64 loses 295 genuine flops),
# which would make the "no false negatives" guarantee this module leans on
# in Stage 3 (see `_candidates`'s docstring) simply false.
MIN_EVIDENCE = 1

# Stage 3 cost controls. Neither of these is tuned to reproduce any
# particular headline count (see module docstring) -- they are a bounded
# amount of extra formal effort spent on whatever Stage 2's constant
# propagation could not settle structurally.
TRUTH_TABLE_SUPPORT_LIMIT = 20
# Measured on test40: raising this from 1 to 2, 3, or 6 proves not a single
# additional flop (every candidate past the top-evidence-ranked one that
# Stage 3 ever succeeds on was already caught by Stage 2's constant
# propagation) -- it only adds cost (roughly +20s per extra attempt across
# the corpus's ~330 Stage-3-eligible flops). Kept at 1 for that reason, not
# to hit any particular headline count. A design where a real hold's ONLY
# provable control is not its highest-evidence Stage 1 candidate would need
# this raised; nothing in this corpus exercises that case. Re-measured with
# cap=2 after the `_degenerate_candidates` leaf-inclusion fix and the `q`
# exclusion (both changed WHICH candidates exist, so this claim needed
# re-checking, not just trusting): `proven_hold` still 1834 either way --
# the extra attempt only moves flops between `forced_no_candidate` and
# `forced_simulated_only` (91/288 at cap=1 -> 123/256 at cap=2), for +~17s.
#
# This ONE cap bounds BOTH the cheap truth-table route and the expensive
# ABC route, combined, not separately. A separate, larger budget for just
# the truth table was tried and measured: even though it is cheap PER
# CANDIDATE (O(cone size), not row count -- see `_truth_table_proof`), a
# flop can have hundreds of Stage-1 survivors (measured: up to several
# hundred on test40's degenerate-branch exhaustive scan), and enough
# individually-cheap attempts still adds up -- test40's per-query time
# roughly TRIPLED (40s -> 122s) for a benefit that turned out to be purely a
# `forced_simulated_only` -> `forced_no_candidate` labeling-precision shift
# (`proven_hold` itself never changed -- no NEW hold was ever found past the
# first candidate tried, only additional DISPROOFS of ones already correctly
# excluded from `proven_hold`). Not worth 3x the cost, so reverted to one
# shared budget; see `detect_enable_hold`'s own Stage 3 comment for how the
# labeling-precision fix (a candidate proven definitively FALSE, not merely
# timed out, must not be reported as "unresolved") was kept anyway, at zero
# extra cost, since it only changes how the SAME already-budgeted attempt(s)
# are interpreted afterward.
MAX_ABC_CANDIDATES_PER_FLOP = 1
ABC_CANDIDATE_TIMEOUT = 5.0

# Symbolic constant-propagation values: either a constant, or a named net
# with a polarity (its own gate's output net names itself when the
# simplification rules below can't reduce it any further -- see `_constprop`).
SymVal = tuple[str, Union[int, "NetBit"], int]
_C0: SymVal = ("c", 0, 0)
_C1: SymVal = ("c", 1, 0)


@dataclass(frozen=True)
class HoldFinding:
    """One flop this module PROVED has a hold structure (Stage 2 or Stage 3
    -- never Stage 1 simulation alone)."""

    flop: str
    q: NetBit
    d: NetBit
    control_net: Optional[NetBit]  # None when D == Q was proven with NOTHING forced (an unconditional identity)
    control_value: Optional[int]
    proof: str  # "constprop" | "truth_table" | "abc"
    and_gated: Optional[bool]  # True if the disabled branch structurally reduces to constant 0


@dataclass(frozen=True)
class EnableHoldResult:
    """See the module docstring for the full pipeline. Every count below is
    disjoint from every other -- summing the leaf categories reproduces
    `total_flops`.

    Control-net non-uniqueness (measured on test40: a flop that has ANY
    valid control net has a median of ~25 that all individually satisfy the
    Stage 2/3 proof): this module reports the FIRST one its search order
    happens to prove, sorted by Stage 1 evidence strength (see
    `_candidates`) -- not a canonical or unique choice, and callers must not
    treat `HoldFinding.control_net` as "the" control signal.
    """

    total_flops: int
    no_self_reference: int  # D's fanin cone never reaches this flop's own Q: hold is structurally impossible

    # Flops where D's fanin cone DOES reach its own Q (`total_flops -
    # no_self_reference` of them), split into three disjoint groups. Every
    # self-referencing flop -- including the `x == 0` ones (see
    # `degenerate_sim` below) -- goes through the SAME proof pipeline and
    # lands in exactly one of these three; there is no separate "degenerate"
    # verdict bucket (see the module docstring for why an earlier version
    # having one was a bug).
    forced_proven: int  # a control net (c, v), or no forcing at all, formally proven to make D == Q
    # Proven no REPORTABLE hold: either no Stage-1 candidate survived at all, or every survivor was
    # individually, definitively disproven by Stage 3 (never a timeout/budget skip -- see
    # `detect_enable_hold`'s `exhausted` tracking). "No candidate" is scoped to the candidate POOL
    # `_candidates`/`_degenerate_candidates` actually search -- which deliberately excludes the
    # flop's own D and Q (see `_candidates`'s docstring for why forcing Q makes the whole D==Q
    # comparison vacuous, not just uninteresting) -- NOT a claim that no net anywhere, d/q included,
    # satisfies this module's literal top-of-file definition. A flop whose ONLY working control is
    # its own Q lands here too, correctly, since this module never reports that as a control anyway.
    forced_no_candidate: int
    forced_simulated_only: int  # candidate(s) survived Stage 1 but none could be settled within budget

    # PURELY DIAGNOSTIC, not a verdict, not a partition of `total_flops`, and
    # sensitive to `NUM_SAMPLES` (measured: 2048 samples found 96 of these on
    # test40, 16384 samples with a different seed found 94 -- the count
    # itself is a sampling artifact, not a stable structural classification;
    # see the module docstring). Counts how many self-referencing flops had
    # D == Q on EVERY simulated sample with NOTHING forced, and, of those,
    # how many ended up in `forced_proven` / `forced_simulated_only` above
    # (already counted there -- these two are a breakdown, not additional
    # flops).
    degenerate_sim: int
    degenerate_proven: int
    degenerate_unresolved: int

    findings: list[HoldFinding] = field(default_factory=list)

    @property
    def self_referencing(self) -> int:
        return self.total_flops - self.no_self_reference

    @property
    def proven_hold(self) -> int:
        return self.forced_proven

    @property
    def simulated_only(self) -> int:
        return self.forced_simulated_only

    @property
    def and_gated_findings(self) -> list[HoldFinding]:
        return [f for f in self.findings if f.and_gated]


# ----------------------------------------------------------------------
# Stage 0: structural cone / self-reference
# ----------------------------------------------------------------------


def _fanin_cone(d: NetBit, driver: dict[NetBit, Gate], comb_ins: dict[str, list[Pin]]) -> set[NetBit]:
    """All net-bits in `d`'s fanin, walking only combinational gates (a
    DFF's Q is a source, never traversed past)."""
    seen: set[NetBit] = set()
    stack = [d]
    while stack:
        nb = stack.pop()
        if nb in seen:
            continue
        seen.add(nb)
        gate = driver.get(nb)
        if gate is None:
            continue
        for pin in comb_ins[gate.inst_name]:
            if isinstance(pin, NetBit) and pin not in seen:
                stack.append(pin)
    return seen


# ----------------------------------------------------------------------
# Stage 1: bit-parallel simulation + necessary-condition candidate filter
# ----------------------------------------------------------------------


def _topo_order(comb_gates: list[Gate], driver: dict[NetBit, Gate], comb_ins: dict[str, list[Pin]]) -> list[Gate]:
    indeg: dict[str, int] = {g.inst_name: 0 for g in comb_gates}
    succs: dict[str, list[str]] = {g.inst_name: [] for g in comb_gates}
    for g in comb_gates:
        for pin in comb_ins[g.inst_name]:
            if isinstance(pin, NetBit):
                dg = driver.get(pin)
                if dg is not None:
                    succs[dg.inst_name].append(g.inst_name)
                    indeg[g.inst_name] += 1
    order_names = [n for n, d in indeg.items() if d == 0]
    qi = 0
    while qi < len(order_names):
        for s in succs[order_names[qi]]:
            indeg[s] -= 1
            if indeg[s] == 0:
                order_names.append(s)
        qi += 1
    if len(order_names) != len(comb_gates):
        raise ValueError("combinational cycle detected while building the enable/hold simulation order")
    by_name = {g.inst_name: g for g in comb_gates}
    return [by_name[n] for n in order_names]


def _eval(gate_type: GateType, ins: list[int], mask: int) -> int:
    if gate_type == GateType.NOT:
        return (~ins[0]) & mask
    if gate_type == GateType.BUF:
        return ins[0]
    a, b = ins
    if gate_type == GateType.AND:
        return a & b
    if gate_type == GateType.OR:
        return a | b
    if gate_type == GateType.NAND:
        return (~(a & b)) & mask
    if gate_type == GateType.NOR:
        return (~(a | b)) & mask
    if gate_type == GateType.XOR:
        return a ^ b
    if gate_type == GateType.XNOR:
        return (~(a ^ b)) & mask
    raise ValueError(f"unexpected gate type in enable/hold simulation: {gate_type!r}")


def _simulate(
    design: Design,
    order: list[Gate],
    comb_ins: dict[str, list[Pin]],
    out_of: dict[str, Pin],
    pi_bits: list[NetBit],
    q_bits: list[NetBit],
    mask: int,
    seed: int,
) -> dict[NetBit, int]:
    """One `NUM_SAMPLES`-wide bit-parallel simulation: every PI bit and
    every flop's Q gets an independent random pattern (Q is a free variable
    -- see module docstring), propagated through one topological pass."""
    rng = random.Random(seed)
    sig: dict[NetBit, int] = {}
    for nb in pi_bits:
        sig[nb] = rng.getrandbits(NUM_SAMPLES)
    for nb in q_bits:
        sig[nb] = rng.getrandbits(NUM_SAMPLES)

    def resolve(p: Pin) -> int:
        if isinstance(p, Const):
            return mask if p == Const.ONE else 0
        if isinstance(p, NetBit):
            return sig.get(p, 0)
        return 0

    for g in order:
        o = out_of[g.inst_name]
        if isinstance(o, NetBit):
            sig[o] = _eval(g.gate_type, [resolve(p) for p in comb_ins[g.inst_name]], mask)
    return sig


def _candidates(
    x: int, cone: set[NetBit], d: NetBit, q: NetBit, sig: dict[NetBit, int], mask: int
) -> list[tuple[NetBit, int]]:
    """Stage 1's necessary-condition filter: NEVER a verdict, only a way to
    shrink Stage 2/3's search space, so it must never drop a real hold
    structure (no false negatives -- false positives are fine and expected,
    Stage 2/3 exist to weed them out).

    Soundness argument: forcing `c = v` at a sampled input pattern where `c`
    ALREADY evaluates to `v` under that pattern changes nothing anywhere in
    the circuit (the forced and unforced circuits agree bit-for-bit on that
    pattern by construction). So if forcing `c = v` really makes D == Q on
    EVERY pattern, then restricted to just the patterns where c == v
    naturally, D and Q must ALREADY agree in the unforced simulation. Taking
    the contrapositive: a pattern where `x = D ^ Q` is nonzero while c == v
    is direct evidence that (c, v) is NOT a valid hold control -- so
    "candidate" is exactly "no such counterexample pattern was observed",
    i.e. `x & (c==v ? sig[c] : ~sig[c]) == 0`.

    `MIN_EVIDENCE = 1` (not any higher -- see module constant) is the
    weakest possible admission bar: at least one sampled pattern actually
    has c == v, otherwise "no counterexample was observed" is vacuous (c
    never took that value at all) rather than real supporting evidence.

    `d` and `q` are BOTH DELIBERATELY EXCLUDED from the candidate pool --
    this is a POLICY choice about what counts as a reportable control, NOT a
    mathematical consequence of anything above, and it is NOT claimed to be
    exhaustive over the full definition in the module docstring (see
    `forced_no_candidate`'s field comment on `EnableHoldResult` and
    `detect_enable_hold`'s own comment for how that gap is kept honest in
    what gets reported). `d` itself: "forcing D to a constant" is not a
    control signal reachable by any real gate wired ahead of D. `q` (the
    flop's OWN output): forcing `q` makes the "D == Q" comparison this whole
    module is built around VACUOUS, not merely uninteresting -- Stage 2/3
    would be comparing D-forced-with-q-substituted against Q, but Q's own
    value under that forcing IS the same forced constant, so the check
    degenerates into "does D reduce to exactly the value I just forced Q
    to", which can genuinely succeed even when D is NOT constant for other
    values of Q (verified with a real counterexample: `D = XOR(AND(Q, x),
    AND(Q, y))` with `x, y` free -- forcing q=0 makes D reduce to 0 = the
    forced Q, satisfying this module's literal definition, even though D is
    XOR(x, y) -- NOT a constant -- whenever Q=1). So excluding `q` is not
    "this case is handled elsewhere and found some other way" (an earlier
    version of this docstring claimed exactly that, and it was false, checked
    by building the counterexample above and running it through
    `detect_enable_hold`: it lands in `forced_no_candidate`, not
    `forced_proven`, via NEITHER route) -- it is "this module chooses not to
    report a flop's own output as its own hold control", full stop, and
    whatever a flop like the counterexample above gets bucketed as must be
    honest about that choice, not silent about it.
    """
    hits: list[tuple[NetBit, int]] = []
    for c in sorted(cone, key=netbit_sort_key):
        if c == d or c == q:
            continue
        sc = sig.get(c)
        if sc is None:
            continue
        if x & sc == 0 and sc.bit_count() >= MIN_EVIDENCE:
            hits.append((c, 1))
        nsc = (~sc) & mask
        if x & nsc == 0 and nsc.bit_count() >= MIN_EVIDENCE:
            hits.append((c, 0))
    # Try the most-often-asserted control first: a real enable/hold control
    # is active on a large fraction of samples, a coincidental survivor
    # usually is not, so this ordering makes Stage 2/3's early exit pay off
    # (and gives a deterministic, evidence-ranked `HoldFinding.control_net`
    # when a flop admits more than one valid control -- see
    # `EnableHoldResult`'s docstring on why that choice is not unique).
    # `list.sort` is stable, so ties in evidence count fall back to `hits`'s
    # own order -- which is only deterministic because `cone` is walked in
    # `netbit_sort_key` order above, NOT because of anything in this sort
    # call itself.
    hits.sort(key=lambda cv: -((sig[cv[0]] if cv[1] else (~sig[cv[0]]) & mask).bit_count()))
    return hits


def _degenerate_candidates(
    cone: set[NetBit], driver: dict[NetBit, Gate], rank: dict[str, int], d: NetBit, q: NetBit
) -> list[tuple[NetBit, int]]:
    """Candidate controls for a flop where `x == 0` (D == Q on EVERY
    simulated sample with nothing forced -- see the module docstring for why
    this is NOT its own proof path). `_candidates`'s necessary-condition
    filter is VACUOUS in this case: `x == 0` trivially satisfies
    `x & sig[c] == 0` (and its complement) for every net `c` in the cone, so
    every net looks like maximal evidence -- that is a loss of Stage 1's
    FILTERING power, not a loss of Stage 2's PROVING power, and the earlier
    version of this module conflated the two.

    So this scans the WHOLE cone -- every net in it, LEAVES (raw PI bits,
    OTHER flops' free Q) included, excluding only `d` and this flop's OWN
    `q` for the same two reasons `_candidates` does -- ordered CLOSEST TO D
    FIRST for the
    internally-driven nets (descending topological rank of each net's
    driving gate), with leaves tried last in a deterministic order. Leaf
    inclusion is NOT optional: an earlier version of this function excluded
    every net without a driving gate (`if nb in driver`), which quietly
    narrowed "the whole cone" to "the whole cone minus its own leaves" --
    the exact same "I didn't look there" -> "proven absent" failure mode
    this module's own docstring warns about, just one level down, caught by
    an independent reviewer's hand-derived counterexample: `D =
    OR(AND(en, load), AND(NOT(en), Q))` with `en` a bare, undriven primary
    input referenced directly by two different gates. Forcing any
    INTERNALLY-DRIVEN net downstream of `en` (e.g. the NOT gate's own
    output) only fixes ONE of `en`'s two uses, so no derived wire can
    substitute for forcing `en` itself -- if leaves are excluded from the
    search, this flop's real, provable hold control is never found, and an
    empty (leaves-excluded) result gets reported as a formal proof of no
    hold, which is false. See `test_degenerate_candidates_includes_leaf_nets`
    (unit-level) and the `bug_repro`-named fixture test in
    tests/test_enable_hold.py (end-to-end through `detect_enable_hold`) for
    both the counterexample and the fix.

    The near-to-far ordering (like leaf-inclusion, unlike the exhaustiveness
    claim above) IS just a performance heuristic, not a correctness
    requirement -- a real hold's control tends to sit close to D
    structurally (this module's own NAND-NAND-mux-collapse observation), so
    trying near-to-far, driven-nets-first, finds it fast when it works, but
    nothing here bounds the search to some prefix: it runs over the ENTIRE
    cone including leaves, so an empty result is a genuine proof that no
    single net anywhere in the cone -- leaf or derived -- can hold D at Q,
    not just "none found in the first K tried". Cost is bounded by Stage 2
    being cheap per candidate (O(cone size) per constant-propagation pass,
    not by ABC), which is why this can afford to be exhaustive where
    `_candidates` is only a filter.
    """
    driven = sorted(
        (nb for nb in cone if nb != d and nb != q and nb in driver),
        key=lambda nb: -rank[driver[nb].inst_name],
    )
    leaves = sorted(
        (nb for nb in cone if nb != d and nb != q and nb not in driver),
        key=lambda nb: (nb.name, nb.bit if nb.bit is not None else -1),
    )
    return [(nb, v) for nb in driven + leaves for v in (1, 0)]


# ----------------------------------------------------------------------
# Stage 2: symbolic constant propagation (exact proof, not sampling)
# ----------------------------------------------------------------------


def _inv(v: SymVal) -> SymVal:
    if v[0] == "c":
        # v[1] is statically int | NetBit (SymVal's second slot is shared
        # between the "c" and "n" cases); this branch's own invariant (see
        # _C0/_C1) is that it's always the int 0/1 here -- the cast documents
        # that invariant for the type checker rather than narrowing anything
        # at runtime.
        return ("c", 1 - cast(int, v[1]), 0)
    return ("n", v[1], 1 - v[2])


def _s_and(a: SymVal, b: SymVal) -> Optional[SymVal]:
    if a == _C0 or b == _C0:
        return _C0
    if a == _C1:
        return b
    if b == _C1:
        return a
    if a == b:
        return a
    if a == _inv(b):
        return _C0
    return None


def _s_or(a: SymVal, b: SymVal) -> Optional[SymVal]:
    if a == _C1 or b == _C1:
        return _C1
    if a == _C0:
        return b
    if b == _C0:
        return a
    if a == b:
        return a
    if a == _inv(b):
        return _C1
    return None


def _s_xor(a: SymVal, b: SymVal) -> Optional[SymVal]:
    if a == _C0:
        return b
    if b == _C0:
        return a
    if a == _C1:
        return _inv(b)
    if b == _C1:
        return _inv(a)
    if a == b:
        return _C0
    if a == _inv(b):
        return _C1
    return None


def _constprop(
    cone_gates: list[Gate],
    comb_ins: dict[str, list[Pin]],
    out_of: dict[str, Pin],
    d: NetBit,
    forced: dict[NetBit, SymVal],
) -> SymVal:
    """Symbolic constant propagation of D over `cone_gates` (already a
    topologically-sorted subset of the design's combinational gates) with
    `forced` net(s) pinned to a constant symbolic value. Returns D's
    resulting symbolic value: a constant, or `("n", net, polarity)` naming
    the net whose output this reduced to -- crucially, a gate the
    simplification rules below could NOT reduce still gets named after its
    OWN output net (rather than left opaque), so a later gate in the same
    cone can still recognize `x AND (NOT x)`-shaped cancellations built on
    top of it (this is exactly what collapses a NAND-NAND mux once its
    enable is pinned: D = NAND(NAND(e, L), NAND(NOT e, Q)); force e = 0 ->
    NAND(C1, NAND(C1, Q)) -> NAND(C1, NOT Q) -> Q).
    """
    val: dict[NetBit, SymVal] = dict(forced)

    def resolve(p: Pin) -> SymVal:
        if isinstance(p, Const):
            return _C1 if p == Const.ONE else _C0
        if isinstance(p, NetBit):
            v = val.get(p)
            return v if v is not None else ("n", p, 0)
        return _C0

    binop = {
        GateType.AND: _s_and,
        GateType.OR: _s_or,
        GateType.XOR: _s_xor,
    }
    for g in cone_gates:
        o = out_of[g.inst_name]
        if not isinstance(o, NetBit) or o in forced:
            continue
        gt = g.gate_type
        ins = [resolve(p) for p in comb_ins[g.inst_name]]
        if gt == GateType.NOT:
            out: Optional[SymVal] = _inv(ins[0])
        elif gt == GateType.BUF:
            out = ins[0]
        elif gt in (GateType.NAND, GateType.NOR, GateType.XNOR):
            base = {GateType.NAND: _s_and, GateType.NOR: _s_or, GateType.XNOR: _s_xor}[gt](ins[0], ins[1])
            out = None if base is None else _inv(base)
        else:
            out = binop[gt](ins[0], ins[1])
        val[o] = out if out is not None else ("n", o, 0)
    x = val.get(d)
    return x if x is not None else ("n", d, 0)


def _support(cone_gates: list[Gate], comb_ins: dict[str, list[Pin]], out_of: dict[str, Pin], d: NetBit) -> set[NetBit]:
    """The set of leaf net-bits (PIs / flop Qs -- anything with no driving
    gate INSIDE `cone_gates`) that D actually, structurally depends on."""
    driven = {out_of[g.inst_name]: g for g in cone_gates if isinstance(out_of[g.inst_name], NetBit)}
    leaves: set[NetBit] = set()
    seen: set[NetBit] = set()
    stack = [d]
    while stack:
        nb = stack.pop()
        if nb in seen:
            continue
        seen.add(nb)
        gate = driven.get(nb)
        if gate is None:
            leaves.add(nb)
            continue
        for p in comb_ins[gate.inst_name]:
            if isinstance(p, NetBit):
                stack.append(p)
    return leaves


# ----------------------------------------------------------------------
# Stage 3: exhaustive truth table (small support) / bounded ABC calls
# ----------------------------------------------------------------------


def _truth_table_proof(
    cone_gates: list[Gate],
    comb_ins: dict[str, list[Pin]],
    out_of: dict[str, Pin],
    d: NetBit,
    q: NetBit,
    forced: dict[NetBit, int],  # net -> 0/1
    leaves: list[NetBit],
) -> bool:
    """Exhaustive proof for a small support: bit-parallel over EVERY
    assignment of `leaves` at once (2**len(leaves) samples packed into one
    big-int signature per net) -- cost tracks the cone's gate count, not the
    row count, so this is cheap even at the `TRUTH_TABLE_SUPPORT_LIMIT`."""
    n = len(leaves)
    mask = (1 << (1 << n)) - 1
    sig: dict[NetBit, int] = {}
    for i, leaf in enumerate(leaves):
        period = 1 << i
        pattern = 0
        block = (1 << period) - 1
        pos = 0
        total = 1 << n
        while pos < total:
            pattern |= block << (pos + period)
            pos += 2 * period
        sig[leaf] = pattern & mask
    for nb, v in forced.items():
        sig[nb] = mask if v else 0

    def resolve(p: Pin) -> int:
        if isinstance(p, Const):
            return mask if p == Const.ONE else 0
        if isinstance(p, NetBit):
            return sig.get(p, 0)
        return 0

    for g in cone_gates:
        o = out_of[g.inst_name]
        if isinstance(o, NetBit) and o not in forced:
            sig[o] = _eval(g.gate_type, [resolve(p) for p in comb_ins[g.inst_name]], mask)
    return sig.get(d, 0) == sig.get(q, 0)


def _build_forced_cone_design(
    design: Design,
    cone_gates: list[Gate],
    comb_ins: dict[str, list[Pin]],
    out_of: dict[str, Pin],
    d: NetBit,
    q: NetBit,
    forced: tuple[NetBit, int],
) -> Design:
    """A fresh, purely-combinational `Design` containing exactly `d`'s
    forced cone -- everything upstream of `forced[0]` is dropped and
    replaced by a single tie gate driving `forced[0]` to the constant
    `forced[1]`, mirroring `abc_bridge.extract_combinational_view`'s
    "const_zero" DFF-tie convention but for one net, not every flop. Every
    other leaf (PI bit, or another flop's free Q, INCLUDING this flop's own
    Q) becomes an ordinary primary input. Built directly from `cone_gates`
    rather than via `abc_bridge.extract_combinational_view` +
    `_restrict_to_fanin_cone` on the WHOLE design, which would re-walk (and
    re-copy) the entire netlist on every single candidate -- measured to
    matter at this design's scale (26k gates, hundreds of candidates).

    `forced` is required, not optional: the only caller (`_prove_via_abc`)
    only ever runs on genuine forced candidates -- ABC is never asked to
    prove an unconditional identity with nothing forced at all, since Step
    1 of `detect_enable_hold`'s per-flop loop (cheap symbolic constant
    propagation with an empty `forced` dict) already covers that case for
    every self-referencing flop before Stage 3 ever runs."""
    tie_net = forced[0]
    gate_names = {g.inst_name for g in cone_gates if out_of[g.inst_name] != tie_net}

    work = Design(module_name="__enable_hold_cone__")
    c, v = forced
    work.signals[c.name] = Signal(
        name=c.name, msb=design.signals[c.name].msb, lsb=design.signals[c.name].lsb, direction=Direction.INTERNAL
    )
    work.add_gate(
        Gate(inst_name="__force__", gate_type=GateType.BUF, pins={"O": c, "I0": Const.ONE if v else Const.ZERO})
    )

    for g in cone_gates:
        if g.inst_name not in gate_names:
            continue
        work.add_gate(Gate(inst_name=g.inst_name, gate_type=g.gate_type, pins=dict(g.pins)))
        o = out_of[g.inst_name]
        if isinstance(o, NetBit) and o.name not in work.signals:
            orig = design.signals[o.name]
            work.signals[o.name] = Signal(name=o.name, msb=orig.msb, lsb=orig.lsb, direction=Direction.INTERNAL)

    leaf_names: set[str] = set()
    for name in gate_names:
        for p in comb_ins[name]:
            if isinstance(p, NetBit) and p.name not in work.signals:
                leaf_names.add(p.name)
    for nb in (d, q):
        if isinstance(nb, NetBit) and nb.name not in work.signals:
            leaf_names.add(nb.name)
    for name in leaf_names:
        orig = design.signals[name]
        work.signals[name] = Signal(name=name, msb=orig.msb, lsb=orig.lsb, direction=Direction.INPUT)
        work.ports.append(Port(name=name, direction=Direction.INPUT))
    return work


def _leaves_for_candidate(
    cone_gates: list[Gate], comb_ins: dict[str, list[Pin]], out_of: dict[str, Pin], d: NetBit, forced: NetBit
) -> list[NetBit]:
    """D's structural support with `forced` itself removed (it's pinned, not
    a free leaf anymore) -- shared by both `_truth_table_proof` (needs the
    leaf list to enumerate) and `detect_enable_hold` (needs just the COUNT,
    to decide which Stage 3 route -- truth table or ABC -- a candidate
    takes; see `MAX_ABC_CANDIDATES_PER_FLOP`'s comment for why that decision
    does NOT get its own separate budget)."""
    return sorted(_support(cone_gates, comb_ins, out_of, d) - {forced}, key=lambda nb: (nb.name, nb.bit or 0))


def _prove_via_abc(
    design: Design,
    cone_gates: list[Gate],
    comb_ins: dict[str, list[Pin]],
    out_of: dict[str, Pin],
    d: NetBit,
    q: NetBit,
    forced: tuple[NetBit, int],
    timeout: float = ABC_CANDIDATE_TIMEOUT,
) -> Optional[bool]:
    """Stage 3's expensive route, for a candidate whose forced support is
    too large for `_truth_table_proof` to be cheap: a single bounded
    `abc_bridge.are_equivalent` formal check. Returns True (proven hold),
    False (proven NOT a hold for this candidate), or None (ABC timed out /
    errored -- genuinely unresolved, NOT the same as False)."""
    work = _build_forced_cone_design(design, cone_gates, comb_ins, out_of, d, q, forced)
    try:
        return are_equivalent(work, d, q, timeout=timeout)
    except ABCBridgeError:
        return None


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def detect_enable_hold(design: Design) -> EnableHoldResult:
    dffs = [g for g in design.gates if g.gate_type == GateType.DFF]
    comb_gates = [g for g in design.gates if g.gate_type != GateType.DFF]

    comb_ins: dict[str, list[Pin]] = {}
    out_of: dict[str, Pin] = {}
    driver: dict[NetBit, Gate] = {}
    for g in comb_gates:
        ok = OUTPUT_PIN[g.gate_type]
        out_of[g.inst_name] = g.pins.get(ok)
        comb_ins[g.inst_name] = [v for k, v in g.pins.items() if k != ok]
        o = out_of[g.inst_name]
        if isinstance(o, NetBit):
            driver[o] = g

    order = _topo_order(comb_gates, driver, comb_ins)
    rank = {g.inst_name: i for i, g in enumerate(order)}

    q_of = {g.inst_name: g.pins.get("Q") for g in dffs}
    d_of = {g.inst_name: g.pins.get("D") for g in dffs}
    pi_bits = [nb for p in design.ports if p.direction == Direction.INPUT for nb in design.signals[p.name].bits()]
    q_bits = [q for q in q_of.values() if isinstance(q, NetBit)]

    # ---- Stage 0: structural cones + self-reference -----------------
    # `d_ref`/`q_ref` (as opposed to `d_of`/`q_of` above, typed `dict[str,
    # Pin]` since a DFF's D/Q pin can be unconnected or tied to a Const) hold
    # only the NetBit-typed D/Q pair for flops that make it into `self_ref`
    # -- narrowing here once, rather than re-checking `isinstance` on every
    # access below, is what lets every downstream function in this module
    # declare its `d`/`q` parameters as plain `NetBit` instead of `Pin`.
    cones: dict[str, set[NetBit]] = {}
    self_ref: list[str] = []
    d_ref: dict[str, NetBit] = {}
    q_ref: dict[str, NetBit] = {}
    no_self_ref = 0
    for g in dffs:
        d, q = d_of[g.inst_name], q_of[g.inst_name]
        if not isinstance(d, NetBit) or not isinstance(q, NetBit):
            no_self_ref += 1
            continue
        cone = _fanin_cone(d, driver, comb_ins)
        cones[g.inst_name] = cone
        if q in cone:
            self_ref.append(g.inst_name)
            d_ref[g.inst_name] = d
            q_ref[g.inst_name] = q
        else:
            no_self_ref += 1

    mask = (1 << NUM_SAMPLES) - 1
    sig = _simulate(design, order, comb_ins, out_of, pi_bits, q_bits, mask, SIM_SEED)

    cone_gates: dict[str, list[Gate]] = {}
    by_name = {g.inst_name: g for g in comb_gates}
    for name in self_ref:
        gs = [driver[nb].inst_name for nb in cones[name] if nb in driver]
        gs.sort(key=rank.__getitem__)
        cone_gates[name] = [by_name[n] for n in gs]

    findings: list[HoldFinding] = []
    degenerate_sim = 0
    degenerate_proven = 0
    degenerate_unresolved = 0
    forced_proven = 0
    forced_no_candidate = 0
    forced_simulated_only = 0

    for name in self_ref:
        d, q = d_ref[name], q_ref[name]
        gates = cone_gates[name]
        x = (sig[d] ^ sig[q]) & mask
        # `x == 0` (D == Q on every simulated sample with nothing forced) is
        # tracked PURELY as a diagnostic -- see `EnableHoldResult`'s and this
        # module's docstrings for why it must not steer which proof path
        # runs below. Every self-referencing flop, `is_degenerate_sim` or
        # not, goes through the exact same three steps.
        is_degenerate_sim = x == 0
        if is_degenerate_sim:
            degenerate_sim += 1

        # Step 1 (always attempted, cheap): is D == Q with NOTHING forced --
        # a genuine unconditional identity? Harmless to always try: if D and
        # Q ever disagreed on a sampled pattern (x != 0), no unforced
        # structural reduction could make them equal for every assignment
        # including that one, so this can only ever succeed when x == 0.
        sym = _constprop(gates, comb_ins, out_of, d, {})
        if sym == ("n", q, 0):
            forced_proven += 1
            if is_degenerate_sim:
                degenerate_proven += 1
            findings.append(HoldFinding(name, q, d, None, None, "constprop", None))
            continue

        # Step 2: forced-candidate search. `_candidates`'s necessary-
        # condition filter is vacuous when `x == 0` (every net trivially
        # "survives" it), so that case uses `_degenerate_candidates`'s
        # exhaustive near-to-far scan of the whole cone instead -- a
        # different SEARCH ORDER, not a different PROOF (Stage 2/3 below are
        # identical either way).
        cands = (
            _degenerate_candidates(cones[name], driver, rank, d, q)
            if is_degenerate_sim
            else _candidates(x, cones[name], d, q, sig, mask)
        )
        if not cands:
            # No candidate survived an exhaustive (degenerate case) or
            # no-false-negatives (ordinary case -- see `_candidates`'s
            # docstring) search over the searched pool (d and q excluded on
            # purpose -- see `_candidates`'s docstring): this PROVES no (c,
            # v) drawn from that pool can make D == Q, not just "none
            # found" -- see `forced_no_candidate`'s own field comment for
            # why this is scoped, not a claim about q itself.
            forced_no_candidate += 1
            continue

        proven_here = False
        for c, v in cands:
            sym = _constprop(gates, comb_ins, out_of, d, {c: (_C1 if v else _C0)})
            if sym == ("n", q, 0):
                forced_proven += 1
                if is_degenerate_sim:
                    degenerate_proven += 1
                and_gated = _classify_and_gated(gates, comb_ins, out_of, d, c, v)
                findings.append(HoldFinding(name, q, d, c, v, "constprop", and_gated))
                proven_here = True
                break
        if proven_here:
            continue

        # Stage 3, ONE combined attempt budget (see MAX_ABC_CANDIDATES_PER_FLOP's
        # own comment for why this is NOT split into a separate, larger
        # truth-table budget -- measured going that route: test40's
        # per-query time roughly tripled, 40s -> 122s, for zero change in
        # `proven_hold` -- purely a `forced_simulated_only` ->
        # `forced_no_candidate` labeling-precision shift that isn't worth
        # 3x the cost). `exhausted` tracks whether the candidate(s) actually
        # attempted (within budget) all got a DEFINITIVE verdict (the truth
        # table always gives one; ABC gives one unless it timed out/errored)
        # -- if so, and none of them proved True, that is a real proof of no
        # hold via those candidates, not merely "ran out of budget", EVEN
        # THOUGH the budget cap means untested candidates may remain (that
        # is the same limitation `forced_no_candidate`'s docstring already
        # describes for the Stage-1-vacuity route).
        attempts = 0
        exhausted = True
        for c, v in cands:
            if attempts >= MAX_ABC_CANDIDATES_PER_FLOP:
                exhausted = False  # real, untried candidates remain -- genuinely budget-limited
                break
            attempts += 1
            leaves = _leaves_for_candidate(gates, comb_ins, out_of, d, c)
            if len(leaves) <= TRUTH_TABLE_SUPPORT_LIMIT:
                proof: Optional[bool] = _truth_table_proof(gates, comb_ins, out_of, d, q, {c: v}, leaves)
                proof_kind = "truth_table"
            else:
                proof = _prove_via_abc(design, gates, comb_ins, out_of, d, q, (c, v))
                proof_kind = "abc"
                if proof is None:
                    exhausted = False  # ABC itself couldn't settle this one (timeout/error)
            if proof is True:
                forced_proven += 1
                if is_degenerate_sim:
                    degenerate_proven += 1
                and_gated = _classify_and_gated(gates, comb_ins, out_of, d, c, v)
                findings.append(HoldFinding(name, q, d, c, v, proof_kind, and_gated))
                proven_here = True
                break
        if not proven_here:
            if exhausted:
                # Every Stage-1 survivor was individually, DEFINITIVELY
                # disproven (never just timed out, never skipped for
                # budget) -- a complete proof of no hold via any candidate
                # this search identified, same verdict category as an
                # empty candidate list (`forced_no_candidate` above), just
                # reached by exhaustive disproof instead of Stage 1
                # vacuity.
                forced_no_candidate += 1
            else:
                forced_simulated_only += 1
                if is_degenerate_sim:
                    degenerate_unresolved += 1

    return EnableHoldResult(
        total_flops=len(dffs),
        no_self_reference=no_self_ref,
        forced_proven=forced_proven,
        forced_no_candidate=forced_no_candidate,
        forced_simulated_only=forced_simulated_only,
        degenerate_sim=degenerate_sim,
        degenerate_proven=degenerate_proven,
        degenerate_unresolved=degenerate_unresolved,
        findings=findings,
    )


def _classify_and_gated(
    cone_gates: list[Gate], comb_ins: dict[str, list[Pin]], out_of: dict[str, Pin], d: NetBit, c: NetBit, v: int
) -> Optional[bool]:
    """Structural check (not sampling): with the flop's control disabled
    (`c` forced to the OPPOSITE of the hold value `v`, i.e. the branch that
    actually loads new data), does D reduce, by constant propagation alone,
    to the constant 0? That is a real, structural pure-AND-gating pattern
    (the "AND gates" the corpus question asks about). Returns None (not
    False) when constant propagation can't fully reduce that branch --
    "not structurally confirmed either way", not "confirmed not AND-gated".
    """
    disabled = _constprop(cone_gates, comb_ins, out_of, d, {c: (_C0 if v else _C1)})
    if disabled == _C0:
        return True
    if disabled[0] == "c":
        return False
    return None
