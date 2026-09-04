"""Integration tests for the ABC bridge against the real Alpha_Testcase
corpus. Expensive (each test shells out to ABC), so only a handful of
small-to-medium files are used, plus one/some of the LARGEST files purely for
a timing measurement -- mirrors tests/test_transform_real_files.py's
glob/REPO_ROOT pattern.
"""

from __future__ import annotations

import os
import time

import pytest

from netlist_agent.abc_bridge import verify_equivalence
from netlist_agent.parser import parse_verilog
from netlist_agent.transform import remap_to_basis, remove_dangling_gates


from tests.helpers import corpus_available

pytestmark = pytest.mark.skipif(
    not corpus_available(),
    reason="Alpha_Testcase corpus not present -- "
    "the 2026 CAD Contest at ICCAD publishes them with Problem A; "
    "put the released testcases under Alpha_Testcase/testcase/testNN/ to enable this module",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LARGEST = ["test39", "test12", "test33"]
# test39 used to be a documented exception here: it has at least one DFF Q
# pin that is a bit-select of a wider bus whose OTHER bits are independently
# driven by combinational gates, which extract_combinational_view's free_pi
# mode (Signal-granularity Direction promotion) couldn't handle without a
# multiply-driven net. Fixed by _split_bit_to_fresh_input in abc_bridge.py,
# which splits just the conflicting bit off into its own fresh single-bit
# input instead of promoting (or refusing to promote) the whole bus -- see
# test_extract_combinational_view_dff_q_shares_bus_with_combinational_bit in
# tests/test_abc_bridge.py for the targeted synthetic regression test.

# Small-to-medium files without any dff instance (per a direct grep of the
# corpus: `grep -c '^\s*dff ' Alpha_Testcase/testcase/test*/test*.v`).
SMALL_NO_DFF = ["test04", "test03", "test01"]

# Small-to-medium files confirmed (by the same grep) to contain dff
# instances, so verify_equivalence's extraction actually exercises the
# DFF-boundary path end-to-end here, not just the trivial no-op path.
SMALL_WITH_DFF = ["test21", "test23"]


def _path(name: str) -> str:
    return os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase", name, f"{name}.v")


@pytest.mark.parametrize("name", SMALL_NO_DFF + SMALL_WITH_DFF)
def test_remap_to_basis_preserves_equivalence(name: str) -> None:
    path = _path(name)
    original = parse_verilog(path)
    transformed = parse_verilog(path)
    remap_to_basis(transformed, "nand_not")

    result = verify_equivalence(original, transformed)
    assert result.equivalent, f"{name}: remap_to_basis changed behavior -- {result.detail}"


@pytest.mark.parametrize("name", SMALL_NO_DFF + SMALL_WITH_DFF)
def test_remove_dangling_gates_preserves_equivalence(name: str) -> None:
    path = _path(name)
    original = parse_verilog(path)
    transformed = parse_verilog(path)
    remove_dangling_gates(transformed)

    result = verify_equivalence(original, transformed)
    assert result.equivalent, f"{name}: remove_dangling_gates changed behavior -- {result.detail}"


@pytest.mark.parametrize("name", LARGEST)
def test_verify_equivalence_on_largest_files_timing(name: str) -> None:
    path = _path(name)
    original = parse_verilog(path)
    transformed = parse_verilog(path)
    # remap_to_basis actually rewrites a large fraction of gates (unlike
    # remove_dangling_gates, which is a no-op on some of these files), giving
    # a more representative equivalence-checking workload/timing measurement.
    remap_to_basis(transformed, "nand_not")

    start = time.monotonic()
    result = verify_equivalence(original, transformed, timeout=600)
    elapsed = time.monotonic() - start

    print(f"\n{name}.v verify_equivalence (remap_to_basis nand_not) wall-clock: {elapsed:.3f}s")
    assert result.equivalent, f"{name}: remap_to_basis changed behavior -- {result.detail}"
    assert elapsed < 600
