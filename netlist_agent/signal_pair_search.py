"""Existential signal-pair search: "is there a pair of existing signals (a, b)
in the design whose OP(a, b) is functionally equivalent to a
named target net z?" for OP in {AND, NAND, OR, NOR, XOR, XNOR}.

Real designs in the corpus have on the order of 35k nets, so the naive O(n^2)
pairwise check (~1.2 billion pairs for test35) is not viable. This module
instead runs a three-stage pipeline:

  1. Bit-parallel random simulation (`_simulate_signatures`): every net-bit
     of the design's combinational view gets an N-sample signature (a Python
     big integer, one bit per sample), computed in one topological pass.
  2. Linear signature filtering (`find_pair_for_op`): a cheap necessary
     condition specific to each operator narrows the ~35k nets down to a
     small survivor set / a small number of signature-matching pairs (see
     the docstring on `find_pair_for_op` for the per-operator algebra).
  3. Formal verification (`_verify_pair`): each signature-matching candidate
     pair is checked for real via `abc_bridge.are_equivalent` (which is
     exact, not probabilistic) on a throwaway copy of the design with one
     extra OP gate spliced in; the first one that formally holds is
     returned. The search has three possible conclusions: "Yes." (a pair
     formally holds), a clean "No." (every candidate was formally refuted, the
     candidate set was not truncated), or "Undetermined." (a candidate was
     undecided by ABC, left unverified by a limit, the set was truncated, or no
     signature could be computed for the target -- a matching pair may still
     exist).

DFF boundary handling is delegated entirely to
`abc_bridge.extract_combinational_view(..., "free_pi", ...)` (each DFF's Q
net keeps its original name in the view, re-driven from a synthetic
`__dff_Q__<inst>` primary input) -- this module never re-derives that
convention itself. The synthetic `__dff_Q__` / `__dff_D__` nets are not
signals of the design and are never pair candidates.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from netlist_agent.abc_bridge import (
    DEFAULT_ABC_TIMEOUT,
    ABCInconclusiveError,
    are_equivalent,
    extract_combinational_view,
)
from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, OUTPUT_PIN
from netlist_agent.netref import netbit_token

SUPPORTED_OPS = ("AND", "NAND", "OR", "NOR", "XOR", "XNOR")

# Synthetic boundary nets that `extract_combinational_view` adds; neither is a
# signal that exists in the original netlist, so neither is a pair candidate.
# (If tap_control_pins is ever turned on here, add the __dff_RN__/__dff_SN__/__dff_CK__ prefixes.)
_SYNTHETIC_PREFIXES = ("__dff_D__", "__dff_Q__")

DEFAULT_NUM_SAMPLES = 2048
# Cap on the size of the signature-surviving candidate set (AND/NAND/OR/NOR)
# that gets pairwise-enumerated (O(k^2)); a design where this cap actually
# bites is reported as such (`stats["truncated"]`), never silently.
DEFAULT_MAX_CANDIDATES_FOR_PAIRING = 4000
# Cap on how many signature-matching pairs are collected for reporting/
# verification, independent of how the candidate set was built.
DEFAULT_PAIR_COLLECT_CAP = 200
# Cap on how many candidate pairs are actually run through ABC (the
# expensive step) before giving up. Hitting this cap with candidates left
# over is NOT a "no": the result is "Undetermined." (see `find_pair_for_op`).
DEFAULT_MAX_VERIFY = 20
# Cap on how many pairs ABC may fail to decide (timeout / undecided) before the
# search stops sending further pairs. Each undecided pair can cost a full ABC
# timeout, so without this cap the worst case is max_verify * timeout. The
# pairs not yet sent are counted as unverified (never as refuted).
DEFAULT_MAX_UNDECIDED = 2


@dataclass(frozen=True)
class PairSearchResult:
    found: bool
    op: str
    target: str
    pair: Optional[tuple[str, str]]
    explanation: str
    stats: dict[str, int] = field(default_factory=dict)
    # True when the answer is definite: a verified pair was found, or the
    # search was exhaustive with every candidate formally refuted. False means
    # "Undetermined." -- a matching pair may still exist (some candidate pair
    # was undecided / not verified, the candidate set was truncated, or no
    # signature could be computed for the target so nothing was checked).
    conclusive: bool = True


# ----------------------------------------------------------------------
# Stage 1: bit-parallel random simulation
# ----------------------------------------------------------------------


def _topo_sort_gates(design: Design) -> list[Gate]:
    """Kahn topological sort of every gate in `design`, which must already be
    purely combinational (no DFF instances -- see module docstring). Built
    directly off `design.net_driver`/`design.gates` rather than reusing
    graph.py's `NetlistGraph._ensure_global_topo` (private, and built for a
    different purpose -- depth/path DP over gate-instance-name adjacency)."""
    indeg: dict[str, int] = {g.inst_name: 0 for g in design.gates}
    succs: dict[str, list[str]] = {g.inst_name: [] for g in design.gates}
    for g in design.gates:
        out_key = OUTPUT_PIN[g.gate_type]
        for pin, val in g.pins.items():
            if pin == out_key or not isinstance(val, NetBit):
                continue
            driver = design.net_driver.get(val)
            if driver is not None:
                succs[driver.inst_name].append(g.inst_name)
                indeg[g.inst_name] += 1

    queue: deque[str] = deque(name for name, d in indeg.items() if d == 0)
    order_names: list[str] = []
    while queue:
        name = queue.popleft()
        order_names.append(name)
        for s in succs[name]:
            indeg[s] -= 1
            if indeg[s] == 0:
                queue.append(s)
    if len(order_names) != len(design.gates):
        raise ValueError(
            "combinational cycle detected while topologically sorting for signature simulation"
        )
    by_name = {g.inst_name: g for g in design.gates}
    return [by_name[n] for n in order_names]


def _eval_gate_signature(gate_type: GateType, ins: list[int], mask: int) -> int:
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
    raise ValueError(f"cannot compute a bit-parallel signature for gate type {gate_type!r}")


def _simulate_signatures(design: Design, num_samples: int, seed: Optional[int]) -> dict[NetBit, int]:
    """Assign every PI net-bit of `design` (a purely combinational design --
    see `_topo_sort_gates`) an independent `num_samples`-bit random pattern,
    then propagate every gate's output signature through one topological
    pass. Returns every net-bit that got a signature (every PI plus every
    gate output)."""
    rng = random.Random(seed)
    mask = (1 << num_samples) - 1
    sig: dict[NetBit, int] = {}
    for port in design.ports:
        if port.direction != Direction.INPUT:
            continue
        for nb in design.signals[port.name].bits():
            sig[nb] = rng.getrandbits(num_samples)

    def _resolve(pin) -> int:
        if pin is None:
            return 0
        if isinstance(pin, Const):
            return mask if pin == Const.ONE else 0
        return sig.get(pin, 0)

    for gate in _topo_sort_gates(design):
        out_key = OUTPUT_PIN[gate.gate_type]
        in_pins = [v for k, v in gate.pins.items() if k != out_key]
        out_val = _eval_gate_signature(gate.gate_type, [_resolve(p) for p in in_pins], mask)
        out_nb = gate.pins.get(out_key)
        if isinstance(out_nb, NetBit):
            sig[out_nb] = out_val
    return sig


# ----------------------------------------------------------------------
# Stage 3: formal verification
# ----------------------------------------------------------------------


def _resolve_to_original(nb: NetBit) -> Optional[NetBit]:
    """Map a combinational-view net-bit back to the net-bit that names it in
    the ORIGINAL (pre-extraction) design. Original nets keep their names in
    the view (a DFF's Q net is re-driven by a BUF from its own synthetic
    `__dff_Q__<inst>` PI), so this is the identity for every net that exists
    in the design. Returns None for a synthetic `__dff_D__...` / `__dff_Q__...`
    boundary net, which is not a signal that exists in the original netlist
    at all."""
    if nb.name.startswith(_SYNTHETIC_PREFIXES):
        return None
    return nb


def _verify_pair(design: Design, a: NetBit, b: NetBit, op: str, target: NetBit, timeout: float) -> bool:
    """Splice a fresh OP(a, b) gate into a throwaway deep copy of `design`
    and formally check (via `abc_bridge.are_equivalent`, exact, not
    probabilistic) whether its output is equivalent to `target`."""
    work = copy.deepcopy(design)
    new_net = work.fresh_net("t_pairsearch_")
    gate = Gate(
        inst_name=work.fresh_gate_name("t_pairsearch_gate_"),
        gate_type=GateType(op.lower()),
        pins={"O": new_net, "I0": a, "I1": b},
    )
    work.add_gate(gate)
    return are_equivalent(work, new_net, target, timeout=timeout)


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def find_pair_for_op(
    design: Design,
    target: NetBit,
    op: str,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    max_candidates_for_pairing: int = DEFAULT_MAX_CANDIDATES_FOR_PAIRING,
    pair_collect_cap: int = DEFAULT_PAIR_COLLECT_CAP,
    max_verify: int = DEFAULT_MAX_VERIFY,
    max_undecided: int = DEFAULT_MAX_UNDECIDED,
    seed: Optional[int] = 0,
    timeout: float = DEFAULT_ABC_TIMEOUT,
) -> PairSearchResult:
    """Search for a pair of signals (a, b) -- already present in `design`,
    neither syntactically equal to `target` itself -- such that OP(a, b) is
    functionally equivalent to `target`. `a == b` is explicitly allowed.

    Per-operator necessary-condition algebra used for the stage-2 linear
    filter (Z = target's N-sample signature, M = the all-ones N-bit mask):
      AND(a,b)==z  <=> a&b==Z   => both a,b superset of Z   (s&Z==Z)
      NAND(a,b)==z <=> a&b==~Z  => both a,b superset of ~Z
      OR(a,b)==z   <=> a|b==Z   => both a,b subset of Z     (s&~Z==0)
      NOR(a,b)==z  <=> a|b==~Z  => both a,b subset of ~Z
      XOR(a,b)==z  <=> b==a^Z   => hash lookup, O(n)
      XNOR(a,b)==z <=> b==a^~Z  => hash lookup, O(n)
    Signature agreement is only a NECESSARY condition (2^-N-ish false-positive
    rate per random sample set, negligible at N=2048 but not zero) -- every
    surviving candidate pair is re-checked by exact formal verification
    (`_verify_pair`) before being reported as a real answer.

    Conclusions: "Yes. ..." (`found`, conclusive), a clean "No. ..." (not
    `found`, conclusive: nothing truncated, nothing undecided, nothing left
    unverified), or "Undetermined. ..." (not `found`, `conclusive` False).
    ABC timeouts / undecided verdicts on a pair are counted, not raised;
    other `ABCBridgeError`s still propagate. The search stops sending pairs
    once `max_verify` pairs were sent or `max_undecided` of them were
    undecided; the remaining pairs count as `pairs_unverified`.

    Note: `stats["pairs_verified"]` is the number of pairs sent to ABC
    (including undecided ones), while the "formally verified" figure in
    messages excludes undecided pairs.
    """
    op = op.upper()
    if op not in SUPPORTED_OPS:
        raise ValueError(f"unsupported operator {op!r}; choose one of {SUPPORTED_OPS}")
    if max_undecided < 1:
        raise ValueError(f"max_undecided must be >= 1, got {max_undecided}")

    comb = extract_combinational_view(design, "free_pi")
    sig = _simulate_signatures(comb, num_samples, seed)

    if target not in sig:
        return PairSearchResult(
            False,
            op,
            netbit_token(target),
            None,
            f"Undetermined. Could not compute a signature for {netbit_token(target)} in the combinational "
            "view, so no candidate pair was checked; a matching pair may still exist.",
            {"nets_scanned": 0, "signature_survivors": 0, "candidate_pairs_considered": 0, "pairs_verified": 0, "truncated": 0,
             "pairs_undecided": 0, "pairs_unverified": 0},
            False,
        )

    mask = (1 << num_samples) - 1
    Z = sig[target]

    # Every signature-tracked net-bit except the target itself and the
    # synthetic DFF-D boundary taps (not "signals already in the netlist").
    candidates: dict[NetBit, int] = {
        nb: s for nb, s in sig.items() if nb != target and not nb.name.startswith(_SYNTHETIC_PREFIXES)
    }
    nets_scanned = len(candidates)

    pairs: list[tuple[NetBit, NetBit]] = []
    signature_survivors = 0
    truncated = False

    if op in ("XOR", "XNOR"):
        delta = Z if op == "XOR" else (~Z) & mask
        by_sig: dict[int, list[NetBit]] = {}
        for nb, s in candidates.items():
            by_sig.setdefault(s, []).append(nb)
        seen: set[frozenset] = set()
        for a_nb, a_s in candidates.items():
            b_s = a_s ^ delta
            for b_nb in by_sig.get(b_s, ()):
                key = frozenset((netbit_token(a_nb), netbit_token(b_nb)))
                if key in seen:
                    continue
                seen.add(key)
                signature_survivors += 1
                if len(pairs) < pair_collect_cap:
                    pairs.append((a_nb, b_nb))
        truncated = signature_survivors > len(pairs)
    else:
        if op == "AND":
            zz, is_superset = Z, True
        elif op == "NAND":
            zz, is_superset = (~Z) & mask, True
        elif op == "OR":
            zz, is_superset = Z, False
        else:  # NOR
            zz, is_superset = (~Z) & mask, False

        if is_superset:
            survivors = [nb for nb, s in candidates.items() if (s & zz) == zz]
        else:
            survivors = [nb for nb, s in candidates.items() if (s & ~zz & mask) == 0]
        signature_survivors = len(survivors)

        s_capped = survivors
        if len(survivors) > max_candidates_for_pairing:
            s_capped = survivors[:max_candidates_for_pairing]
            truncated = True

        n = len(s_capped)
        combine = (lambda a, b: a & b) if op in ("AND", "NAND") else (lambda a, b: a | b)
        for i in range(n):
            a_nb = s_capped[i]
            a_s = candidates[a_nb]
            for j in range(i, n):
                b_nb = s_capped[j]
                if combine(a_s, candidates[b_nb]) == zz:
                    if len(pairs) < pair_collect_cap:
                        pairs.append((a_nb, b_nb))
                    else:
                        truncated = True

    # stats["pairs_verified"] is verified_count, so it includes undecided pairs;
    # the "formally verified" figure in messages excludes them.
    verified_count = 0  # pairs actually sent to ABC (decided or not); counts against max_verify
    undecided_count = 0  # of those, pairs where ABC timed out / reported undecided
    unverified_count = 0  # pairs never sent to ABC (max_verify or max_undecided reached, or unresolvable)
    found_pair: Optional[tuple[str, str]] = None
    stopped_for_verify = False  # the loop broke because verified_count reached max_verify
    stopped_for_undecided = False  # the loop broke because undecided_count reached max_undecided
    for idx, (a_nb, b_nb) in enumerate(pairs):
        # Both limits can be reached at the same break point; record each one.
        if verified_count >= max_verify or undecided_count >= max_undecided:
            unverified_count += len(pairs) - idx
            stopped_for_verify = verified_count >= max_verify
            stopped_for_undecided = undecided_count >= max_undecided
            break
        orig_a = _resolve_to_original(a_nb)
        orig_b = _resolve_to_original(b_nb)
        if orig_a is None or orig_b is None:
            # Not reachable today (synthetic nets are never candidates), but a
            # pair that was skipped was not refuted, so it must not count toward a clean "No".
            unverified_count += 1
            continue
        verified_count += 1
        try:
            holds = _verify_pair(design, orig_a, orig_b, op, target, timeout)
        except ABCInconclusiveError:
            undecided_count += 1
            continue
        if holds:
            found_pair = (netbit_token(orig_a), netbit_token(orig_b))
            break

    stats = {
        "nets_scanned": nets_scanned,
        "signature_survivors": signature_survivors,
        "candidate_pairs_considered": len(pairs),
        "pairs_verified": verified_count,
        "truncated": int(truncated),
        "pairs_undecided": undecided_count,
        # Pairs left unchecked when the search stopped without a Yes (verification
        # limit, undecided limit, or unresolvable). Candidates after a found pair
        # are never counted here: a Yes is definite, so they do not matter.
        "pairs_unverified": unverified_count,
    }

    if found_pair is not None:
        a_tok, b_tok = found_pair
        explanation = (
            f"Yes. {op}({a_tok}, {b_tok}) is formally verified equivalent to {netbit_token(target)} "
            f"(scanned {nets_scanned} net(s), {signature_survivors} passed the {num_samples}-sample "
            f"signature filter, {verified_count - undecided_count} pair(s) formally verified)."
        )
        if undecided_count:
            explanation += (
                f" {undecided_count} earlier candidate pair(s) could not be decided by ABC "
                f"(timeout/undecided); that does not affect this result."
            )
        return PairSearchResult(True, op, netbit_token(target), found_pair, explanation, stats, True)

    if not truncated and unverified_count == 0 and undecided_count == 0:
        explanation = (
            f"No. Scanned {nets_scanned} net(s); {signature_survivors} signal(s)/pair(s) passed the "
            f"{num_samples}-sample signature filter, {verified_count} candidate pair(s) were formally "
            f"verified and none held."
        )
        return PairSearchResult(False, op, netbit_token(target), None, explanation, stats, True)

    refuted = verified_count - undecided_count
    reasons = []
    if stopped_for_undecided:
        reasons.append(
            f"The search stopped once {undecided_count} pair(s) could not be decided "
            f"(max_undecided={max_undecided}), to bound the time spent in ABC."
        )
    if stopped_for_verify:
        reasons.append(f"The verification limit was reached (max_verify={max_verify}).")
    reason = " ".join(reasons) + (" " if reasons else "")
    explanation = (
        f"Undetermined. Scanned {nets_scanned} net(s); {signature_survivors} signal(s)/pair(s) passed the "
        f"{num_samples}-sample signature filter; {refuted} candidate pair(s) were formally verified and "
        f"none held; {undecided_count} pair(s) could not be decided (ABC timeout or undecided); "
        f"{unverified_count} pair(s) were not checked; the candidate set was "
        f"{'truncated' if truncated else 'not truncated'}. "
        f"{reason}So a matching pair may still exist."
    )
    return PairSearchResult(False, op, netbit_token(target), None, explanation, stats, False)
