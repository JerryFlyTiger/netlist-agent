from __future__ import annotations

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from netlist_agent.ir import Design


def canonical_ports(design: Design) -> list[tuple[str, str]]:
    return [(p.name, p.direction.value) for p in design.ports]


def canonical_signals(design: Design) -> dict[str, tuple[int | None, int | None, str]]:
    return {name: (s.msb, s.lsb, s.direction.value) for name, s in design.signals.items()}


def canonical_gates(design: Design) -> set[tuple[str, str, tuple]]:
    result = set()
    for gate in design.gates:
        pins = tuple(sorted(gate.pins.items(), key=lambda kv: kv[0]))
        result.add((gate.inst_name, gate.gate_type.value, pins))
    return result


def assert_structurally_equal(d1: Design, d2: Design) -> None:
    assert canonical_ports(d1) == canonical_ports(d2)
    assert canonical_signals(d1) == canonical_signals(d2)
    assert canonical_gates(d1) == canonical_gates(d2)


def corpus_netlist_paths(root: str | None = None) -> list[str]:
    """The 40 corpus INPUT netlists, sorted.

    Not just `test*/test*.v`: that glob also matches the generated
    `testNN_out.v` a corpus run writes next to each input. Because those
    artifacts are gitignored, `git status` stayed clean whether or not they
    were present, while every module that parametrised over this glob
    silently changed size -- 1904 tests collected on a clean tree, 2304 once
    someone had run the corpus. The "pytest N passed" figure this project
    records as each batch's acceptance number was therefore not comparable
    between runs.

    Three test modules had their own copy of the glob (test_analysis,
    test_roundtrip, test_transform_real_files), so patching one left the
    other two inflating the count. One definition, three consumers.

    `root` exists so the `_out.v` exclusion is testable: with no `_out.v`
    lying around, a test that only checks the real corpus cannot tell a
    working filter from a deleted one. A test can point this at a synthetic
    tree containing both and watch the filter actually do something.
    """
    import glob
    import os

    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return sorted(
        p
        for p in glob.glob(os.path.join(root, "Alpha_Testcase", "testcase", "test*", "test*.v"))
        if not os.path.basename(p).endswith("_out.v")
    )


def corpus_available() -> bool:
    """True if the Alpha_Testcase/ corpus (not part of this public repo) has
    been placed alongside it. Most tests that exercise real testcase files
    are skipped (not failed/errored) when this is False -- see the modules
    that check it."""
    return os.path.isdir(os.path.join(_REPO_ROOT, "Alpha_Testcase", "testcase"))
