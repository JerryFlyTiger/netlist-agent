"""Smoke tests against all 40 real testcase files: every transform must not
crash, must leave a valid round-trippable Design, and (where cheap) must
satisfy a structural invariant. Full semantic equivalence checking on these
100k-gate files is out of scope here (that needs the future ABC-based
checker) -- structural invariants + "doesn't crash, stays fast" is the bar.
"""

from __future__ import annotations

import os
import time

import pytest

from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import GateType
from netlist_agent.parser import parse_verilog
from netlist_agent.transform import (
    BASES,
    collapse_double_inverters,
    deduplicate_gates,
    limit_fanout,
    remove_dangling_gates,
    remap_to_basis,
    simplify_constant_inputs,
)
from netlist_agent.writer import write_verilog
from tests.helpers import assert_structurally_equal, corpus_netlist_paths, corpus_available

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATHS = corpus_netlist_paths()
LARGEST = ["test39", "test12", "test33"]


def _roundtrip(design, tmp_path, name):
    out_path = tmp_path / name
    write_verilog(design, str(out_path))
    reparsed = parse_verilog(str(out_path))
    assert_structurally_equal(design, reparsed)
    return reparsed


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=[os.path.basename(p) for p in FIXTURE_PATHS])
def test_collapse_double_inverters_smoke(path: str, tmp_path) -> None:
    design = parse_verilog(path)
    collapse_double_inverters(design)
    _roundtrip(design, tmp_path, "out.v")


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=[os.path.basename(p) for p in FIXTURE_PATHS])
def test_dangling_removal_smoke_and_invariant(path: str, tmp_path) -> None:
    design = parse_verilog(path)
    remove_dangling_gates(design)
    _roundtrip(design, tmp_path, "out.v")
    graph = NetlistGraph(design)
    live: set[str] = set()
    for po in graph.po_bits:
        live |= graph.backward_reachable_gates(po)
    for dff in graph.dff_gates:
        for pin in ("D", "CK", "RN", "SN"):
            v = dff.pins.get(pin)
            if v is not None and hasattr(v, "name"):
                live |= graph.backward_reachable_gates(v)
    for gate in design.gates:
        if gate.gate_type != GateType.DFF:
            assert gate.inst_name in live, f"{gate.inst_name} survived removal but is unreachable"


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=[os.path.basename(p) for p in FIXTURE_PATHS])
def test_dedup_smoke(path: str, tmp_path) -> None:
    design = parse_verilog(path)
    deduplicate_gates(design)
    _roundtrip(design, tmp_path, "out.v")


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=[os.path.basename(p) for p in FIXTURE_PATHS])
def test_constant_simplification_smoke(path: str, tmp_path) -> None:
    design = parse_verilog(path)
    simplify_constant_inputs(design)
    _roundtrip(design, tmp_path, "out.v")


@pytest.mark.parametrize("basis_name", sorted(BASES))
@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=[os.path.basename(p) for p in FIXTURE_PATHS])
def test_basis_remap_smoke_and_invariant(path: str, basis_name: str, tmp_path) -> None:
    design = parse_verilog(path)
    remap_to_basis(design, basis_name)
    _roundtrip(design, tmp_path, "out.v")
    allowed = BASES[basis_name] | {GateType.BUF, GateType.DFF}
    disallowed = {g.gate_type for g in design.gates} - allowed
    assert not disallowed, f"{path} still has disallowed types after {basis_name} remap: {disallowed}"


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=[os.path.basename(p) for p in FIXTURE_PATHS])
def test_limit_fanout_smoke_and_invariant(path: str, tmp_path) -> None:
    design = parse_verilog(path)
    limit_fanout(design, max_fanout=8)
    _roundtrip(design, tmp_path, "out.v")
    from netlist_agent.analysis import iter_fanout_counts

    graph = NetlistGraph(design)
    for nb, count in iter_fanout_counts(graph):
        assert count <= 8, f"{path}: {nb} still has fanout {count} after limit_fanout"


@pytest.mark.parametrize("name", LARGEST)
@pytest.mark.skipif(not corpus_available(), reason="Alpha_Testcase corpus not present -- put the released testcases under Alpha_Testcase/testcase/testNN/ to enable this test")
def test_transform_sweep_performance_on_largest_files(name: str, tmp_path) -> None:
    path = os.path.join(REPO_ROOT, "Alpha_Testcase", "testcase", name, f"{name}.v")
    design = parse_verilog(path)

    start = time.perf_counter()
    collapse_double_inverters(design)
    remove_dangling_gates(design)
    deduplicate_gates(design)
    simplify_constant_inputs(design)
    remap_to_basis(design, "nand_not")
    elapsed = time.perf_counter() - start

    print(f"\n{name}.v full transform sweep: {elapsed:.3f}s")
    assert elapsed < 60
