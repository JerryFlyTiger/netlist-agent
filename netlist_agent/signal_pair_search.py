"""Existential signal-pair search: "does there exist a pair of signals (a, b)
already in the netlist such that OP(a, b) is functionally equivalent to a
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
     returned.

DFF boundary handling is delegated entirely to
`abc_bridge.extract_combinational_view(..., "free_pi", ...)` (DFF Q treated
as a free pseudo primary input) -- this module never re-derives that
convention itself.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from netlist_agent.abc_bridge import DEFAULT_ABC_TIMEOUT, are_equivalent, extract_combinational_view
from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, OUTPUT_PIN
from netlist_agent.netref import netbit_token

SUPPORTED_OPS = ("AND", "NAND", "OR", "NOR", "XOR", "XNOR")

DEFAULT_NUM_SAMPLES = 2048
# Cap on the size of the signature-surviving candidate set (AND/NAND/OR/NOR)
# that gets pairwise-enumerated (O(k^2)); a design where this cap actually
# bites is reported as such (`stats["truncated"]`), never silently.
DEFAULT_MAX_CANDIDATES_FOR_PAIRING = 4000
# Cap on how many signature-matching pairs are collected for reporting/
# verification, independent of how the candidate set was built.
DEFAULT_PAIR_COLLECT_CAP = 200
# Cap on how many candidate pairs are actually run through ABC (the
# expensive step) before giving up and answering "no".
DEFAULT_MAX_VERIFY = 20


@dataclass(frozen=True)
class PairSearchResult:
    found: bool
    op: str
    target: str
    pair: Optional[tuple[str, str]]
    explanation: str
    stats: dict[str, int] = field(default_factory=dict)


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


def _resolve_to_original(nb: NetBit, promoted_q_source: dict[str, NetBit]) -> Optional[NetBit]:
    """Map a combinational-view net-bit back to the net-bit that names it in
    the ORIGINAL (pre-extraction) design -- trivial identity for an ordinary
    internal/PI net, but a real remap for a promoted-DFF-Q pseudo-PI (whether
    a plain same-name promotion or a `_split_bit_to_fresh_input` split, both
    recorded uniformly in `promoted_q_source`). Returns None for a synthetic
    `__dff_D__...` boundary tap, which is not a signal that exists in the
    original netlist at all."""
    if nb.name.startswith("__dff_D__"):
        return None
    return promoted_q_source.get(nb.name, nb)


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
    """
    op = op.upper()
    if op not in SUPPORTED_OPS:
        raise ValueError(f"unsupported operator {op!r}; choose one of {SUPPORTED_OPS}")

    promoted_q_source: dict[str, NetBit] = {}
    comb = extract_combinational_view(design, "free_pi", promoted_q_source)
    sig = _simulate_signatures(comb, num_samples, seed)

    if target not in sig:
        return PairSearchResult(
            False,
            op,
            netbit_token(target),
            None,
            f"No: could not compute a signature for {netbit_token(target)} in the combinational view "
            "(check that the net name/bit exists in the design).",
            {"nets_scanned": 0, "signature_survivors": 0, "candidate_pairs_considered": 0, "pairs_verified": 0, "truncated": 0},
        )

    mask = (1 << num_samples) - 1
    Z = sig[target]

    # Every signature-tracked net-bit except the target itself and the
    # synthetic DFF-D boundary taps (not "signals already in the netlist").
    candidates: dict[NetBit, int] = {
        nb: s for nb, s in sig.items() if nb != target and not nb.name.startswith("__dff_D__")
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

    verified_count = 0
    found_pair: Optional[tuple[str, str]] = None
    for a_nb, b_nb in pairs:
        if verified_count >= max_verify:
            break
        orig_a = _resolve_to_original(a_nb, promoted_q_source)
        orig_b = _resolve_to_original(b_nb, promoted_q_source)
        if orig_a is None or orig_b is None:
            continue
        verified_count += 1
        if _verify_pair(design, orig_a, orig_b, op, target, timeout):
            found_pair = (netbit_token(orig_a), netbit_token(orig_b))
            break

    stats = {
        "nets_scanned": nets_scanned,
        "signature_survivors": signature_survivors,
        "candidate_pairs_considered": len(pairs),
        "pairs_verified": verified_count,
        "truncated": int(truncated),
    }

    if found_pair is not None:
        a_tok, b_tok = found_pair
        explanation = (
            f"Yes. {op}({a_tok}, {b_tok}) is formally verified equivalent to {netbit_token(target)} "
            f"(scanned {nets_scanned} net(s), {signature_survivors} passed the {num_samples}-sample "
            f"signature filter, {verified_count} pair(s) formally verified)."
        )
        return PairSearchResult(True, op, netbit_token(target), found_pair, explanation, stats)

    explanation = (
        f"No. Scanned {nets_scanned} net(s); {signature_survivors} signal(s)/pair(s) passed the "
        f"{num_samples}-sample signature filter, {verified_count} candidate pair(s) were formally "
        f"verified and none held."
    )
    if truncated:
        explanation += " The candidate set exceeded the search cap and was truncated."
    return PairSearchResult(False, op, netbit_token(target), None, explanation, stats)
