"""Regression anchors pinned to `Beta_Testcase/testcase/test58/test58.v` (the
0813 release corpus), recorded during the A94/A87 audit
(`experiments/a94_a87_deepest_2026-09-03/RESULTS.md`) so the numbers live in
a real test instead of only in that prose file.

The corpus itself is not part of the public export (see `_FORBIDDEN_TOP_DIRS`
in `scripts/export_public.py`), so this whole module is skipped there --
`scripts/export_public.py` inserts a module-level
`pytestmark = pytest.mark.skipif(not corpus_available_beta(), ...)` at export
time via `_skip_whole_module_without_corpus`. In this (private) repo the
corpus is always present, so no guard is needed here, mirroring
`tests/test_abc_bridge_real_files.py`'s unconditional direct load of
`Alpha_Testcase/`.

A1 and A2 are regression anchors for this implementation's own fix (QA A94:
a fanin cone counts boundary DFFs as members, and a DFF's own Q has a fanin
cone size of 1, not 0) -- their expected values come from re-deriving A94's
rule against this specific testcase, not from anything the organizers
published about test58. A3 is different in kind: its expected value (107) is
not derived from this implementation at all -- the organizers directly named
test58 in QA/A_QA_20260827.pdf (Q73/Q81) and ruled that the correct
register-to-register combinational path count for it is 107 (distinct
combinational paths, self-loops included; 36 is the reading they rejected).
"""

from __future__ import annotations

import pytest

import os

from netlist_agent.analysis import fanin_cone_size
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import NetBit
from netlist_agent.parser import parse_verilog


from tests.helpers import corpus_available_beta

pytestmark = pytest.mark.skipif(
    not corpus_available_beta(),
    reason="Beta_Testcase corpus not present -- "
    "the 2026 CAD Contest at ICCAD publishes them with Problem A; "
    "put the released testcases under Beta_Testcase/testcase/testNN/ to enable this module",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST58_PATH = os.path.join(REPO_ROOT, "Beta_Testcase", "testcase", "test58", "test58.v")


def _test58_graph() -> NetlistGraph:
    return NetlistGraph(parse_verilog(_TEST58_PATH))


def test_a94_dff_q_fanin_cone_size_is_one_on_test58() -> None:
    """DFF `g15` in test58.v: `dff g15(.RN(n1), .SN(1'b1), .CK(n0), .D(n93),
    .Q(n6[1]))` -- its Q is bit 1 of the 8-bit output net `n6`. QA A94 point
    3 says that when X is itself a DFF's Q, the fanin cone size is 1 (that
    DFF itself), not 0. Before the A94 fix this returned 0."""
    graph = _test58_graph()
    assert fanin_cone_size(graph, NetBit("n6", 1)) == 1


def test_a94_combinational_gate_fanin_cone_counts_boundary_dffs_on_test58() -> None:
    """Gate `g2` in test58.v: `nand g2(n112, n6[0], n110)` -- its output net
    is `n112`. `n6` is the 8-bit vector `output [7:0] n6`, six of whose bits
    (n6[0], n6[1], n6[2], n6[3], n6[4], n6[5]/n6[6]/n6[7] as reached deeper
    in the cone) are each driven by an independent DFF instance (g15-g21,
    g25). QA A94 counts each such boundary DFF as a member of the fanin
    cone. Before the fix this returned 6 (the non-DFF gates only); the A94
    reading is 12 (6 non-DFF gates + 6 boundary DFF instances)."""
    graph = _test58_graph()
    assert fanin_cone_size(graph, NetBit("n112")) == 12


def test_reg_to_reg_path_count_test58_matches_the_organizers_ruling() -> None:
    """Not a regression anchor for this implementation -- the expected value
    here comes from the organizers, not from re-deriving anything ourselves.
    QA/A_QA_20260827.pdf Q73/Q81 names test58 by number: "For released 0813
    testcase test58 the two readings give 107 and 36 respectively", and
    A73/A81 rule that the correct reading (distinct combinational paths,
    self-loops included) is 107, not 36."""
    graph = _test58_graph()
    stats = graph.reg_to_reg_path_stats()
    assert stats.combinational_path_count == 107
