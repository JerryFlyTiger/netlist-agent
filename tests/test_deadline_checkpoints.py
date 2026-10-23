"""Deadline checkpoints in the read-only analysis loops (batch 21d, SPEC_21d T1-T3).

Public-safe: imports only graph / analysis / signal_pair_search / deadline / parser / ir (no router,
no llm). The router-side checkpoints are in `test_deadline_router_checkpoints.py` (private).

Technique: `_Probe` replaces `deadline.check` with a counter that can raise
`DeadlineExceeded` on the k-th call. Every checkpoint site calls `deadline.check()` through the
module attribute, so the replacement sees them all. A site is pinned three ways: the exact call
count (deleting a checkpoint lowers it), raising at every k (each call is reachable and the
exception escapes), and a real `check()` with a fake clock (the wiring to the scope)."""

from __future__ import annotations

import pytest

from netlist_agent import deadline
from netlist_agent.analysis import fanin_cone_size, largest_fanin_cone, po_cone_sizes
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import NetBit
from netlist_agent.parser import parse_verilog
from netlist_agent import signal_pair_search
from netlist_agent.signal_pair_search import find_pair_for_op


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(deadline, "_monotonic", lambda: now["t"])
    return now


class _Probe:
    """Stands in for `deadline.check`: counts calls, raises on the `raise_at`-th."""

    def __init__(self, monkeypatch, raise_at=None) -> None:
        self.n = 0
        self.raise_at = raise_at
        monkeypatch.setattr(deadline, "check", self)

    def __call__(self) -> None:
        self.n += 1
        if self.n == self.raise_at:
            raise deadline.DeadlineExceeded(45.0)


def _design(tmp_path, text: str):
    path = tmp_path / "d.v"
    path.write_text(text)
    return parse_verilog(str(path))


# Three diamonds in series: 2 * 2 * 2 = 8 distinct gate sequences from a to y, 9 gates.
LADDER = """
module top(a, y);
  input a;
  output y;
  wire p0, q0, w0, p1, q1, w1, p2, q2;
  buf s0p(p0, a);
  buf s0q(q0, a);
  and m0(w0, p0, q0);
  buf s1p(p1, w0);
  buf s1q(q1, w0);
  and m1(w1, p1, q1);
  buf s2p(p2, w1);
  buf s2q(q2, w1);
  and m2(y, p2, q2);
endmodule
"""

# p is NOT a cut (a reaches y through q too), so is_cut_signal_for_some_pi_po_pair walks every loop.
NOT_A_CUT = """
module top(a, b, y, z);
  input a, b;
  output y, z;
  wire p, q;
  buf gp(p, a);
  buf gq(q, a);
  or gy(y, p, q);
  buf gz(z, b);
endmodule
"""

# dff0.Q feeds one gate on both pins, that gate feeds dff1.D. Outputs: o. Candidates for the
# largest cone: PO o, and the D nets d0 and w.
TWO_DFF = """
module top(clk, rn, d0, o);
  input clk, rn, d0;
  output o;
  wire q0, w, q1;
  dff dff0(.RN(rn), .SN(1'b1), .CK(clk), .D(d0), .Q(q0));
  and g1(w, q0, q0);
  dff dff1(.RN(rn), .SN(1'b1), .CK(clk), .D(w), .Q(q1));
  buf g2(o, q1);
endmodule
"""

# z = a & b; w, v are other nets. Candidates (all but z): a b c w v = 5.
PAIR = """
module top(a, b, c, z, w, v);
  input a, b, c;
  output z, w, v;
  and gz(z, a, b);
  buf g1(w, z);
  or g2(v, a, c);
endmodule
"""


def _many_outputs(n: int = 12) -> str:
    """n primary outputs of different cone sizes (o_i is a chain of i+1 bufs), so the order of
    graph.po_bits (a set) has a negligible chance of equalling the sorted order."""
    lines = ["module top(a, " + ", ".join(f"o{i}" for i in range(n)) + ");", "  input a;"]
    lines.append("  output " + ", ".join(f"o{i}" for i in range(n)) + ";")
    for i in range(n):
        prev = "a"
        for j in range(i):
            lines.append(f"  buf c{i}_{j}(t{i}_{j}, {prev});")
            prev = f"t{i}_{j}"
        lines.append(f"  buf e{i}(o{i}, {prev});")
    lines.append("endmodule")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# T1: one case per checkpoint (C3-C10). Each builder returns (thunk, exact check count).
# ----------------------------------------------------------------------


def _case_c3(tmp_path, monkeypatch):
    graph = NetlistGraph(_design(tmp_path, LADDER))
    return (lambda: graph.cut_nets_between(NetBit("a"), NetBit("y"))), 9


def _case_c4_c5(tmp_path, monkeypatch):
    """C4 runs once per PI (2), C5 once per (reaching PI, reached PO) pair (1)."""
    graph = NetlistGraph(_design(tmp_path, NOT_A_CUT))
    return (lambda: graph.is_cut_signal_for_some_pi_po_pair(NetBit("p"))), 3


def _case_c6(tmp_path, monkeypatch):
    design = _design(tmp_path, TWO_DFF)
    return (lambda: NetlistGraph(design).reg_to_reg_path_stats()), 2


def _case_c7(tmp_path, monkeypatch):
    graph = NetlistGraph(_design(tmp_path, _many_outputs()))
    return (lambda: po_cone_sizes(graph)), 12


def _case_c8(tmp_path, monkeypatch):
    graph = NetlistGraph(_design(tmp_path, TWO_DFF))
    return (lambda: largest_fanin_cone(graph)), 3


def _case_pair(op: str, expected: int):
    def build(tmp_path, monkeypatch):
        design = _design(tmp_path, PAIR)
        # The checkpoints under test are in the pairing loops; keep ABC out of it.
        monkeypatch.setattr(signal_pair_search, "_verify_pair", lambda *a, **k: False)
        return (lambda: find_pair_for_op(design, NetBit("z"), op)), expected

    return build


_CASES = {
    "C3_cut_nets_between": _case_c3,
    "C4_C5_is_cut_signal": _case_c4_c5,
    "C6_reg_to_reg_path_stats": _case_c6,
    "C7_po_cone_sizes": _case_c7,
    "C8_largest_fanin_cone": _case_c8,
    "C9_pair_xor": _case_pair("XOR", 5),
    "C10_pair_and": _case_pair("AND", 4),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_t1_exact_check_count(case, tmp_path, monkeypatch) -> None:
    """kills: deleting any one of C3-C10 (the count drops); an extra or stride-gated check.
    Does NOT distinguish a check at the top of the loop body from one at its tail for C4, C5,
    C7, C8, C9 and C10: the count is the same (near-equivalent, it only delays the check by
    one iteration). For C3 and C6 a move to the tail is caught, but only because the loop body
    has a `continue` that skips the tail check, so the count drops. C2 is not a T1 case: T9 and
    R1-3 (private repo) pin that it exists and sits in the inner `for source` loop, but a move to
    the tail of that body survives, as for C4/C5."""
    thunk, expected = _CASES[case](tmp_path, monkeypatch)
    probe = _Probe(monkeypatch)
    thunk()
    assert probe.n == expected


@pytest.mark.parametrize("case", sorted(_CASES))
def test_t1_exception_escapes_from_every_check(case, tmp_path, monkeypatch) -> None:
    """kills: a checkpoint inside a try/except that swallows DeadlineExceeded; an unreachable
    check. A check moved to the tail of its loop body is NOT killed for C4, C5, C7, C8, C9, C10
    (same count, same reachability; the cost is one extra iteration of work before the deadline
    is seen). C3 and C6 moved to the tail change the count (see test_t1_exact_check_count)."""
    thunk, expected = _CASES[case](tmp_path, monkeypatch)
    for k in range(1, expected + 1):
        _Probe(monkeypatch, raise_at=k)
        with pytest.raises(deadline.DeadlineExceeded):
            thunk()


@pytest.mark.parametrize("case", sorted(_CASES))
def test_t1_real_check_with_spent_budget_raises(case, tmp_path, monkeypatch, clock) -> None:
    """kills: a checkpoint that is not wired to the request scope (e.g. a local clock, or
    check() replaced by a no-op)."""
    thunk, _ = _CASES[case](tmp_path, monkeypatch)
    with deadline.request_scope(deadline.ANALYSIS_BUDGET):
        clock["t"] = 46.0
        with pytest.raises(deadline.DeadlineExceeded) as info:
            thunk()
    assert info.value.budget == 45.0


@pytest.mark.parametrize("case", sorted(_CASES))
def test_t1_enough_budget_result_equals_no_scope(case, tmp_path, monkeypatch, clock) -> None:
    """kills: a checkpoint that changes a result when time is plentiful (reordering, early
    return, truncation)."""
    thunk, _ = _CASES[case](tmp_path, monkeypatch)
    plain = thunk()
    with deadline.request_scope(deadline.ANALYSIS_BUDGET):
        clock["t"] = 10.0
        scoped = thunk()
    assert scoped == plain


# ----------------------------------------------------------------------
# T2: enumerate_paths de-duplicates a start gate fed on two pins (D3)
# ----------------------------------------------------------------------


def test_t2_double_pin_start_gate_is_one_path(tmp_path) -> None:
    """kills: removing the `dict.fromkeys` dedupe of start_names (2 identical paths vs path_count 1)."""
    design = _design(
        tmp_path,
        """
        module top(a, y);
          input a;
          output y;
          wire w;
          and g1(w, a, a);
          buf g2(y, w);
        endmodule
        """,
    )
    graph = NetlistGraph(design)
    paths = [[g.inst_name for g in p] for p in graph.enumerate_paths(NetBit("a"), NetBit("y"))]
    assert paths == [["g1", "g2"]]
    assert graph.path_count(NetBit("a"), NetBit("y")) == 1


def test_t2_dedupe_does_not_merge_distinct_paths(tmp_path) -> None:
    """kills: a dedupe that is too aggressive (e.g. dropping every repeated first gate, or
    de-duplicating whole paths by endpoints): the ladder has 8 distinct gate sequences."""
    graph = NetlistGraph(_design(tmp_path, LADDER))
    paths = [tuple(g.inst_name for g in p) for p in graph.enumerate_paths(NetBit("a"), NetBit("y"))]
    assert len(paths) == len(set(paths)) == graph.path_count(NetBit("a"), NetBit("y")) == 8


def test_t2_dff_double_pin_counts_once_in_stats(tmp_path) -> None:
    """kills: reg_to_reg_path_stats drifting from the one-per-gate-sequence convention (the
    router-side half, `_iter_reg_to_reg_paths` agreeing, is in the private checkpoints test)."""
    graph = NetlistGraph(_design(tmp_path, TWO_DFF))
    assert graph.reg_to_reg_path_stats().combinational_path_count == 1


def test_r1_2_multi_start_path_order_is_fanout_order(tmp_path) -> None:
    """kills: D3's `dict.fromkeys` replaced by `set(...)` (start gates would come out in hash
    order, varying with PYTHONHASHSEED), `sorted(...)` or `reversed(...)`. Eight start gates,
    declared in the non-lexicographic order 5,2,7,0,3,6,1,4; the paths must follow that order."""
    n = 8
    decl = [5, 2, 7, 0, 3, 6, 1, 4]
    # OR chain m1..m7 (2-input gates): m1 = w0|w1, m_k = m_{k-1}|w_k, m7 drives y.
    outs = [f"t{k}" for k in range(1, n - 1)] + ["y"]
    lines = ["module top(a, y);", "  input a;", "  output y;"]
    lines.append("  wire " + ", ".join([f"w{i}" for i in range(n)] + outs[:-1]) + ";")
    lines += [f"  buf s{i}(w{i}, a);" for i in decl]
    lines.append(f"  or m1({outs[0]}, w0, w1);")
    lines += [f"  or m{k}({outs[k - 1]}, {outs[k - 2]}, w{k});" for k in range(2, n)]
    lines.append("endmodule")
    graph = NetlistGraph(_design(tmp_path, "\n".join(lines)))
    got = [[g.inst_name for g in p] for p in graph.enumerate_paths(NetBit("a"), NetBit("y"))]
    tail = lambda k: [f"m{j}" for j in range(k, n)]
    expected = [[f"s{i}"] + tail(max(i, 1)) for i in decl]
    assert got == expected


# ----------------------------------------------------------------------
# T3: po_cone_sizes keeps graph.po_bits iteration order (C7 rewrite)
# ----------------------------------------------------------------------


def test_t3_po_cone_sizes_order_matches_old_comprehension(tmp_path) -> None:
    """kills: rewriting C7's loop in a different order (e.g. `sorted(...)`, reversed), dropping
    an entry, or changing a value. Oracle = the pre-21d dict comprehension."""
    graph = NetlistGraph(_design(tmp_path, _many_outputs()))
    oracle = {nb: fanin_cone_size(graph, nb) for nb in graph.po_bits}
    got = po_cone_sizes(graph)
    assert list(got.items()) == list(oracle.items())
    assert len(got) == 12
    assert sorted(got.values()) == list(range(1, 13))
