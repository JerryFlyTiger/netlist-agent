from __future__ import annotations

import os
import time

import pytest

from netlist_agent.parser import parse_verilog
from netlist_agent.writer import write_verilog
from tests.helpers import assert_structurally_equal, corpus_netlist_paths, corpus_available

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATHS = corpus_netlist_paths()


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=[os.path.basename(p) for p in FIXTURE_PATHS])
def test_roundtrip(path: str, tmp_path) -> None:
    design = parse_verilog(path)
    out_path = tmp_path / "roundtrip.v"
    write_verilog(design, str(out_path))
    design2 = parse_verilog(str(out_path))
    assert_structurally_equal(design, design2)


@pytest.mark.skipif(not corpus_available(), reason="Alpha_Testcase corpus not present -- put the released testcases under Alpha_Testcase/testcase/testNN/ to enable this test")
def test_test39_performance(tmp_path) -> None:
    path = os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase", "test39", "test39.v")
    out_path = tmp_path / "test39_out.v"

    start = time.perf_counter()
    design = parse_verilog(path)
    write_verilog(design, str(out_path))
    elapsed = time.perf_counter() - start

    design2 = parse_verilog(str(out_path))
    assert_structurally_equal(design, design2)

    print(f"\ntest39.v parse+write round trip: {elapsed:.3f}s")
    assert elapsed < 60
