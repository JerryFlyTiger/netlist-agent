"""Cross-validate our parser's gate counts against iverilog's independent
Verilog front end, on a small representative subset of the corpus.

See scripts/crosscheck_gate_counts.py for the full method and the two
pitfalls (BUFT/BUFZ exclusion, `-s top`) this relies on. This test is the
fast, always-run version of that script: it only checks a handful of
designs, chosen to cover both a design with source `buf` gates and no DFFs
(test01) and a design with DFFs and no `buf` gates (test58), plus one more
from each corpus for a little extra spread.

Skips (not fails) when either the corpus or `iverilog` itself is missing --
this test is exported to the public repo, where neither is guaranteed to be
present (see the module docstring in scripts/crosscheck_gate_counts.py for
why the check exists at all).
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA_ROOT = os.path.join(REPO_ROOT, "Alpha_Testcase")
BETA_ROOT = os.path.join(REPO_ROOT, "Beta_Testcase")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from netlist_agent.analysis import gate_count_by_type
from netlist_agent.ir import GateType
from netlist_agent.parser import parse_verilog

from crosscheck_gate_counts import iverilog_counts  # noqa: E402

IVERILOG_MISSING = shutil.which("iverilog") is None

# (corpus root, testcase name) -- test01 has BUF but no DFF, test58 has DFF
# but no BUF; test12 and test83 are larger designs from each corpus thrown
# in for extra spread.
CASES = [
    (ALPHA_ROOT, "test01"),
    (ALPHA_ROOT, "test12"),
    (BETA_ROOT, "test58"),
    (BETA_ROOT, "test83"),
]


def _corpus_missing(root: str) -> bool:
    return not os.path.isdir(root)


@pytest.mark.skipif(IVERILOG_MISSING, reason="iverilog not installed")
@pytest.mark.parametrize("root,name", CASES, ids=[f"{os.path.basename(r)}/{n}" for r, n in CASES])
def test_gate_counts_match_iverilog(root, name, tmp_path):
    if _corpus_missing(root):
        pytest.skip(f"{root} corpus not present")

    design_path = os.path.join(root, "testcase", name, f"{name}.v")
    ours = gate_count_by_type(parse_verilog(design_path))
    theirs = iverilog_counts(design_path, str(tmp_path))

    for gate_type in GateType:
        assert ours[gate_type] == theirs[gate_type], (
            f"{name} {gate_type.value}: ours={ours[gate_type]} iverilog={theirs[gate_type]}"
        )


@pytest.mark.skipif(IVERILOG_MISSING, reason="iverilog not installed")
def test_test01_has_bufs_and_no_dffs(tmp_path):
    """Sanity check on the fixture choice itself: test01 must actually
    exercise the BUF side of the cross-check (it is otherwise possible for
    this whole file to pass vacuously if test01 stopped having any buf
    gates)."""
    if _corpus_missing(ALPHA_ROOT):
        pytest.skip("Alpha_Testcase corpus not present")
    design_path = os.path.join(ALPHA_ROOT, "testcase", "test01", "test01.v")
    ours = gate_count_by_type(parse_verilog(design_path))
    assert ours[GateType.BUF] > 0
    assert ours[GateType.DFF] == 0


@pytest.mark.skipif(IVERILOG_MISSING, reason="iverilog not installed")
def test_test58_has_dffs_and_no_bufs(tmp_path):
    """Sanity check on the fixture choice itself: test58 must actually
    exercise the DFF side of the cross-check."""
    if _corpus_missing(BETA_ROOT):
        pytest.skip("Beta_Testcase corpus not present")
    design_path = os.path.join(BETA_ROOT, "testcase", "test58", "test58.v")
    ours = gate_count_by_type(parse_verilog(design_path))
    assert ours[GateType.DFF] > 0
    assert ours[GateType.BUF] == 0
