"""Tests for scripts/mutation_check.py's stale-.pyc defence.

CPython decides whether a cached .pyc is still valid by comparing
(int(source mtime in seconds), source size) against the numbers baked into
the .pyc header -- not by hashing the source. If a mutation's replacement
text is the same length as the original AND the write lands within the
same wall-clock second as the .pyc already on disk, the interpreter will
import the OLD bytecode while the file on disk holds the NEW text. The
tests below exercise that hazard directly (T1), by forging a stale .pyc
byte-for-byte so no clock race is involved (F3, F4), and end-to-end
against the real tool running under a real second-boundary race (T2).
"""

from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import shutil
import stat
import struct
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import mutation_check  # noqa: E402


def test_stale_pyc_hazard_is_real_and_invalidate_bytecode_fixes_it(tmp_path):
    """T1: mechanism + fix, fully deterministic (no timing races).

    Builds mod.py containing VALUE = "GOOD", compiles it to a .pyc, then
    overwrites the source with an EQUAL-LENGTH replacement, VALUE = "BAD!",
    and forces the file's mtime back to exactly the second the .pyc was
    compiled in -- reproducing the (mtime, size) collision without racing
    a real clock.

    (a) asserts that, in that state, `import mod` reads "GOOD": the STALE
        bytecode, not the file on disk. This assertion records the hazard
        itself, not our code -- it exists so that if a future CPython
        switches .pyc validation to a content hash (making this hazard
        vanish), this line goes red and tells us to come back and decide
        whether `_invalidate_bytecode` is still needed.
    (b) asserts that after calling `_invalidate_bytecode(mod_path)`,
        `import mod` reads "BAD!" -- the mutation is no longer hidden.

    Mutation: replace `_invalidate_bytecode`'s body with `pass`. Then (b)
    must go red, because nothing removed the stale .pyc that (a) already
    proved is being served.
    """
    mod_path = tmp_path / "mod.py"
    mod_path.write_text('VALUE = "GOOD"\n')

    py_compile.compile(str(mod_path), doraise=True)

    pre_stat = mod_path.stat()

    mod_path.write_text('VALUE = "BAD!"\n')  # same length as "GOOD"
    os.utime(mod_path, (pre_stat.st_atime, pre_stat.st_mtime))

    def _read_value() -> str:
        proc = subprocess.run(
            [sys.executable, "-c", "import mod; print(mod.VALUE)"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    # (a) the hazard itself: stale bytecode wins over the file on disk.
    assert _read_value() == "GOOD"

    # (b) the fix: after invalidation, the real (mutated) source is read.
    mutation_check._invalidate_bytecode(str(mod_path))
    assert _read_value() == "BAD!"


def test_invalidate_bytecode_glob_fallback_removes_mismatched_magic_tag_pyc(tmp_path):
    """F4: the glob fallback in `_invalidate_bytecode` is the ONLY thing
    that can clear a .pyc compiled under a DIFFERENT interpreter's magic
    tag than the one running this test/tool -- which is exactly what
    happens in real use, since mutation_check.py normally runs under
    whatever Python launched it while the tests it drives run under
    `.venv/bin/python`. Neither T1 nor T2 exercises this path: both only
    ever produce .pyc files whose magic tag matches
    importlib.util.cache_from_source()'s own prediction (same interpreter
    throughout), so removing the glob block entirely leaves both green.

    Mutation: delete the glob-fallback loop in `_invalidate_bytecode`,
    keeping only the `cache_from_source()` branch. This test must go red.
    """
    mod_path = tmp_path / "mod.py"
    mod_path.write_text("VALUE = 1\n")

    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    fake_pyc = pycache / "mod.cpython-999.pyc"
    fake_pyc.write_bytes(b"not a real pyc; contents are irrelevant to this test")

    real_cache_path = importlib.util.cache_from_source(str(mod_path))
    # Sanity check that the fake file really is a different name than what
    # cache_from_source() itself would look for -- otherwise this test
    # would not actually be exercising the glob branch at all.
    assert os.path.basename(real_cache_path) != fake_pyc.name

    mutation_check._invalidate_bytecode(str(mod_path))

    assert not fake_pyc.exists()


def test_failed_line_regex_also_matches_collection_errors(tmp_path):
    """F2: a collection/setup error's short-summary line is
    "ERROR path" (not "FAILED path::test"), and its FAILURES header is
    "___ ERROR collecting path ___" -- multiple tokens between the
    underscores, which `_FAILURES_HEADER_RE`'s `^_+ (\\S+) _+$` cannot
    match at all (it requires exactly one non-space token). So
    `_FAILED_LINE_RE` must match ERROR lines directly; the header can't be
    a fallback for this case the way it is for a normal FAILED line.

    This is a narrow, cheap check of the regex against real pytest output,
    not a full run through the tool: producing a collection error through
    the tool's own two-knife A/B pipeline would require a knife that
    breaks pkg/m.py's importability rather than just its behaviour, which
    is a materially different (and more expensive) synthetic-project setup
    for the same regex-level coverage this gets directly.

    Mutation: narrow `_FAILED_LINE_RE` back to `^FAILED (\\S+)` only (drop
    the `ERROR` alternative). This test must go red.
    """
    (tmp_path / "broken.py").write_text("def broken(:\n")  # syntax error
    (tmp_path / "test_broken.py").write_text("from broken import broken\n")

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_broken.py", "-q", "-x", "--no-header", "-rfE"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    match = mutation_check._FAILED_LINE_RE.search(output)
    assert match is not None, output
    assert match.group(1) == "test_broken.py", output


def test_failed_line_regex_finds_all_failures_not_just_first():
    """`_FAILED_LINE_RE` is read with `.findall()` in `_run()`, not
    `.search()` -- this reports every red test a knife causes, not just
    whichever pytest happens to print first. This is the regex-level unit
    backing that change.

    Mutation: revert `_run()`'s `.findall()` back to `.search()` (wrapping
    the single result in a list). This test does not touch `_run()`
    directly, so it stays green under that mutation on its own -- it is
    `test_mutation_check_reports_all_failed_tests_not_just_first` below
    that catches it end to end. This test instead pins the regex itself:
    it must go red if `_FAILED_LINE_RE`'s pattern is narrowed so it can no
    longer be found more than once per line boundary (e.g. anchoring
    beyond MULTILINE).
    """
    output = (
        "FAILED tests/test_a.py::test_one - AssertionError\n"
        "FAILED tests/test_b.py::test_two - AssertionError\n"
        "ERROR tests/test_c.py::test_three - ImportError\n"
    )
    matches = mutation_check._FAILED_LINE_RE.findall(output)
    assert matches == [
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two",
        "tests/test_c.py::test_three",
    ]


def test_failures_header_regex_finds_all_headers():
    """Same as above, for the bare-header fallback pattern."""
    output = "____ test_one ____\nsome text\n____ test_two ____\nmore text\n"
    matches = mutation_check._FAILURES_HEADER_RE.findall(output)
    assert matches == ["test_one", "test_two"]


def _build_synthetic_project(tmp_path, name="proj"):
    """Build a minimal project used by several tests below: pkg/m.py (a
    one-branch `sign` function), tests/test_m.py, a copy of the real
    mutation_check.py, a shim .venv/bin/python, a pyproject.toml
    mirroring this repo's own pytest addopts, and knives.json with two
    equal-length knives:
      A: "nonpos" -> "nonneg"   (changes behaviour -- must be KILLED)
      B: "x > 0"  -> "x > 1"    (does not change behaviour for x=0 -- must
                                  be SURVIVED)

    The shim .venv/bin/python MUST be a shell wrapper script, not a
    symlink: a symlink resolves to the real interpreter's own path
    resolution and was observed to pick up the system Python instead
    (which lacks pytest), turning the baseline red.

    COST, on macOS: writing a fresh unsigned executable at a path never
    seen before makes Gatekeeper run a synchronous `GK performScan` on the
    first exec, and that scan includes a network lookup against Apple's
    notarization service. Every test that calls this helper gets a new
    tmp_path, so every test pays for a new scan. With the service
    reachable that is ~0.12s; with it unreachable the lookup runs to its
    own timeout and the exec stalls for 40-60 seconds instead.

    That was measured here, not guessed: a run of this file took 220-350s
    with four tests landing at 38-61s, against 6s for the same file an
    hour later. The wrapper mtimes across that run were exactly ~60s
    apart, the kernel log named this exact path
    ("ASP: Security policy would not allow process: .../proj/.venv/bin/python"),
    and syspolicyd's own network flows in the same window were
    disconnecting with "Operation timed out" at 60s intervals over a link
    marked `expensive, constrained`.

    Nothing in this repo is wrong when that happens, and it cannot happen
    on CI, which is Linux. It matters here only because it looks exactly
    like a test regression: the failure surfaces as a bare
    `subprocess.TimeoutExpired` on whichever call site has the tightest
    budget, which reads as "the tool got slower" rather than "this machine
    could not reach Apple". If a future run of this file is inexplicably
    slow on a Mac, check the network before reading the diff.

    Giving every synthetic project a symlink to one already-scanned shim
    would avoid the repeat scans (measured: 0.021s and no scan, against
    0.14s and a scan per copy). That is a symlink to the WRAPPER, which
    the paragraph above does not forbid -- its hazard is symlinking to the
    interpreter. It is left undone deliberately: the cost is invisible
    with a working network, and the shared shim would have to outlive
    tmp_path to pay off.

    Returns (proj_path, wrapper_path, original_m_source).
    """
    real_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proj = tmp_path / name
    (proj / "pkg").mkdir(parents=True)
    (proj / "tests").mkdir()
    (proj / "scripts").mkdir()
    (proj / ".venv" / "bin").mkdir(parents=True)

    original_m = "def sign(x):\n    if x > 0:\n        return \"pos\"\n    return \"nonpos\"\n"

    (proj / "pkg" / "__init__.py").write_text("")
    (proj / "pkg" / "m.py").write_text(original_m)
    (proj / "tests" / "__init__.py").write_text("")
    (proj / "tests" / "test_m.py").write_text(
        "from pkg.m import sign\n\n\ndef test_zero_is_nonpos():\n    assert sign(0) == \"nonpos\"\n"
    )
    # F1: mirrors this repo's own pyproject.toml, which sets
    # addopts = ["-rs"] -- notably NOT including f/E. Without this file,
    # the synthetic project would exercise a pytest reporting mode (plain
    # defaults) the real project never runs under, and would not have
    # caught the dead fully-qualified-name branch that -rfE now fixes:
    # measured directly against this repo's real settings, `-q
    # --no-header` alone prints only the bare "____ test_name ____"
    # FAILURES header and no "FAILED path::test_name" line at all.
    (proj / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = ["-rs"]\n'
    )

    shutil.copy2(
        os.path.join(real_root, "scripts", "mutation_check.py"),
        str(proj / "scripts" / "mutation_check.py"),
    )

    wrapper_path = proj / ".venv" / "bin" / "python"
    wrapper_path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    knives = [
        {"name": "A: nonpos -> nonneg (behaviour change)", "file": "pkg/m.py", "old": "nonpos", "new": "nonneg"},
        {"name": "B: x > 0 -> x > 1 (invisible at x=0)", "file": "pkg/m.py", "old": "x > 0", "new": "x > 1"},
    ]
    (proj / "knives.json").write_text(json.dumps(knives))

    return proj, wrapper_path, original_m


def test_pre_baseline_invalidation_prevents_a_forged_stale_baseline(tmp_path):
    """F3: fully deterministic (no timing race) coverage for the
    pre-baseline `_invalidate_bytecode` loop in `main()`.

    Neither T1 nor T2 actually exercises this loop: T2's `_reset_project`
    deletes the whole __pycache__ before every tool invocation, so by the
    time the tool's own pre-baseline loop runs there is nothing left for
    it to delete -- removing that loop entirely still leaves T1 and T2
    both green.

    This test forges a stale .pyc directly, with no clock race involved:
    it compiles a .pyc from the MUTATED ("nonneg") source, then overwrites
    the mtime and size fields in its 16-byte header (magic 4 + flags 4 +
    mtime 4 + size 4; struct.pack("<I", ...) at byte offsets 8 and 12) so
    they read as the ORIGINAL ("nonpos") source's real (mtime, size). The
    file on disk is then restored to the clean original source with its
    original mtime, so the forged .pyc's header matches the file on disk
    exactly, byte for byte, on every field CPython checks -- while the
    code inside the .pyc is the mutated version. If the pre-baseline
    invalidation loop does not run, the tool's own baseline pytest run
    imports the forged bytecode, sees sign(0) == "nonneg" where the test
    expects "nonpos", and the BASELINE itself goes red.

    Mutation: delete the `for path in backups: _invalidate_bytecode(path)`
    loop that runs before the baseline in `main()`. This test must go red
    -- and reliably so (10/10), since nothing about it depends on
    wall-clock timing.
    """
    proj, wrapper_path, original_m = _build_synthetic_project(tmp_path)
    pkg_m = proj / "pkg" / "m.py"

    pkg_m.write_text(original_m)
    py_compile.compile(str(pkg_m), doraise=True)
    pyc_path = importlib.util.cache_from_source(str(pkg_m))
    pre_stat = pkg_m.stat()

    mutated_m = original_m.replace("nonpos", "nonneg")
    pkg_m.write_text(mutated_m)
    py_compile.compile(str(pkg_m), doraise=True)

    header = bytearray(open(pyc_path, "rb").read())
    header[8:12] = struct.pack("<I", int(pre_stat.st_mtime) & 0xFFFFFFFF)
    header[12:16] = struct.pack("<I", pre_stat.st_size & 0xFFFFFFFF)
    with open(pyc_path, "wb") as handle:
        handle.write(header)

    # Restore the clean original source with its original mtime, so the
    # forged .pyc's header now matches the file on disk exactly.
    pkg_m.write_text(original_m)
    os.utime(pkg_m, (pre_stat.st_atime, pre_stat.st_mtime))

    proc = subprocess.run(
        [str(wrapper_path), "scripts/mutation_check.py", "knives.json", "--tests", "tests/test_m.py"],
        cwd=str(proj),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BASELINE IS RED" not in proc.stdout, proc.stdout + proc.stderr


def _pyc_header_mtime_size(pyc_path: str) -> tuple[int, int]:
    """Read the (mtime, size) fields CPython's timestamp-based .pyc
    invalidation checks, straight out of the 16-byte header (magic 4 +
    flags 4 + mtime 4 + size 4). Used only to directly probe whether the
    stale-.pyc collision window is reachable on the current machine --
    see the detection-power probe at the end of T2, below.
    """
    with open(pyc_path, "rb") as handle:
        header = handle.read(16)
    mtime = struct.unpack("<I", header[8:12])[0]
    size = struct.unpack("<I", header[12:16])[0]
    return mtime, size


def test_mutation_check_end_to_end_survives_repeated_stale_pyc_exposure(tmp_path):
    """T2: end-to-end, running the real tool twice against a synthetic project.

    Runs `.venv/bin/python scripts/mutation_check.py knives.json --tests
    tests/test_m.py` inside the synthetic project (see
    `_build_synthetic_project`) TWICE and asserts every run reports A as
    KILLED (naming the specific test that failed) and B as SURVIVED.

    RESET BEFORE EVERY CALL, IN THIS ORDER (clock alignment, then reset,
    then invoke -- with nothing else in between): sleep until just past a
    wall-clock second boundary, THEN rewrite pkg/m.py back to its original
    contents and remove every __pycache__ directory in the synthetic
    project, THEN immediately invoke the tool. Both the reset and the
    ordering are load-bearing, not decoration:

      * without the reset, the previous invocation's compiled .pyc and its
        mtime carry over into the next iteration, breaking the causal
        chain the hazard depends on. An earlier revision of this test
        reused one synthetic project directory across 4 iterations with no
        reset at all; measured against the pre-fix tool, that version only
        caught the M2 mutation (below) in 5 of 10 outer pytest runs,
        because leftover state made the iterations correlated instead of
        independent trials.
      * without the clock alignment coming BEFORE the reset, the hazard
        cannot fire at all, regardless of how many iterations are run. The
        pyc the baseline compiles records pkg/m.py's mtime at reset time;
        the race only fires if the first knife's write lands in that SAME
        integer second. A revision of this test that reset first and then
        slept to align the clock was measured to MISS the M2 mutation
        10 of 10 times: the sleep burns up to a full second after the
        reset but before the tool even starts, so the baseline run and the
        first knife-write always land a full second later than the
        recorded mtime and can never collide.

    Why two runs and not one: with the reset and ordering both correct, a
    single call against the pre-fix tool was measured to catch the M2
    mutation 10 of 10 times, so one run already demonstrates the hazard
    reliably -- this is NOT "N independent coin flips, multiply the miss
    probabilities"; that reasoning was tried in an earlier revision of
    this test and was measured to be wrong (see above; a 4-iteration
    version with no reset caught M2 only 5/10 times, not the ~99.6% that
    reasoning predicted). The second run here exists only as slack for
    machine-to-machine jitter, not as a probability lever.

    ASSERTION ORDER, AND WHY: the per-iteration correctness assertions
    (KILLED/SURVIVED verdicts, and A naming the fully-qualified test) run
    UNCONDITIONALLY, before any skip decision. An earlier revision of this
    test measured a single machine-speed threshold (0.8s) up front and
    skipped the ENTIRE test -- including those timing-independent
    assertions -- if a probe pytest invocation exceeded it. That was
    measured to be wrong in two ways: (1) it is a proxy, not the actual
    condition -- "how long did one pytest invocation take" is not the same
    fact as "did the mtime-second collision actually happen", and a fixed
    threshold picked without measuring the true condition is exactly the
    kind of unmeasured number this project's own history warns against;
    (2) because the skip covered the whole test, a real regression in the
    -rfE flag (F1, unrelated to timing at all) was measured to be missed
    1 time in 5 by that version, purely because the machine-speed probe
    happened to trip the skip on that run. The fix is to run the tool and
    check its output for correctness first, every time, and only ask "did
    the timing-sensitive part of this test have any power to detect
    anything" as the very last step -- see the detection-power probe
    below, which is placed after the loop.

    Mutation A2 (main hazard): remove the `_invalidate_bytecode(path)`
    call that runs right after a mutation is applied. This test must go
    red -- A's verdict flips to SURVIVED (or its failing-test name goes
    missing) because the mutated source is masked by the still-valid
    pre-mutation .pyc.

    Mutation A3 (failing-test name): make `_run()` always return "" for
    the failing-test component. The assertion that A's KILLED line names
    tests/test_m.py::test_zero_is_nonpos exists specifically to catch
    this -- "the suite went red" is not the same claim as "it went red
    for the reason the knife intended", and A3's whole purpose is telling
    those apart.

    Mutation F1 (the -rfE flag): remove "-rfE" from `_run()`'s pytest
    invocation. With this synthetic project's pyproject.toml setting
    addopts = ["-rs"] (mirroring the real repo), pytest then prints only
    the bare "____ test_name ____" FAILURES header with no file path, so
    the "tests/test_m.py::test_zero_is_nonpos" assertion below must go
    red -- and, per the ordering above, it must go red EVERY time this
    test runs, never skip, since that assertion carries no timing
    dependency at all.
    """
    proj, wrapper_path, original_m = _build_synthetic_project(tmp_path)

    def _reset_project() -> None:
        (proj / "pkg" / "m.py").write_text(original_m)
        for pycache in proj.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)

    for _ in range(2):
        # Align to just past a second boundary BEFORE the reset, not after:
        # the pyc that the baseline run compiles records pkg/m.py's mtime
        # at reset time, and the race only fires if the first knife's write
        # lands in that SAME integer second. Sleeping after the reset (an
        # earlier revision of this test did that) burns the rest of that
        # second before the tool even starts, so the baseline+knife-write
        # window always lands a full second later than the recorded mtime
        # and the two can never collide -- this was measured directly: with
        # the sleep in the wrong place, M2 (see below) was missed 10/10.
        # With it in the right place, one call already lands the race
        # reliably; the alignment removes any dependency on how fast this
        # particular machine's baseline pytest happens to start up.
        time.sleep(max(0.0, 1.0 - time.time() % 1.0) + 0.02)
        _reset_project()
        proc = subprocess.run(
            [
                str(wrapper_path),
                "scripts/mutation_check.py",
                "knives.json",
                "--tests",
                "tests/test_m.py",
            ],
            cwd=str(proj),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        out = proc.stdout
        assert "BASELINE IS RED" not in out, out
        out_lines = out.splitlines()
        a_index, a_line = next(
            (i, line) for i, line in enumerate(out_lines) if line.startswith(("KILLED", "SURVIVED", "BROKEN")) and "A:" in line
        )
        b_line = next(line for line in out_lines if line.startswith(("KILLED", "SURVIVED", "BROKEN")) and "B:" in line)
        # These three assertions are timing-independent: they hold
        # regardless of whether the stale-pyc race window ever opened on
        # this machine, so they run unconditionally, before any skip.
        assert a_line.startswith("KILLED"), out
        # The failing-test name must be attributed to A specifically --
        # not merely present somewhere in the output -- so this checks the
        # line immediately following A's verdict line, which is where the
        # tool prints each knife's own "red:" lines.
        assert out_lines[a_index + 1].strip() == "red: tests/test_m.py::test_zero_is_nonpos", out
        assert b_line.startswith("SURVIVED"), out

    # F5, detection-power probe: run LAST, after every timing-independent
    # assertion above has already executed and passed. Rather than compare
    # a measured duration against a guessed constant threshold (an earlier
    # revision used 0.8s; it was never derived from anything, it was
    # picked, and picking numbers without measuring them is exactly the
    # kind of mistake this project's own history warns about), this probes
    # the ACTUAL condition the hazard depends on: does an equal-length
    # overwrite, delayed by one real pytest-startup gap, still collide
    # with a .pyc compiled before that gap? If it does not collide here,
    # it will not collide in the tool's own baseline-to-first-knife gap
    # either, and this test has no power to catch the M2 mutation this run.
    # Align to just past a second boundary before measuring, exactly as
    # each loop iteration above does. Without this the probe is a
    # different experiment from the one it is estimating: unaligned, it
    # is systematically more pessimistic than the aligned loop, so it
    # would skip runs that did in fact have detection power.
    time.sleep(max(0.0, 1.0 - time.time() % 1.0) + 0.02)
    gap_start = time.perf_counter()
    gap_probe = subprocess.run(
        [str(wrapper_path), "-m", "pytest", "tests/test_m.py", "-q", "--no-header"],
        cwd=str(proj),
        capture_output=True,
        text=True,
        timeout=60,
    )
    gap = time.perf_counter() - gap_start
    assert gap_probe.returncode == 0, gap_probe.stdout + gap_probe.stderr

    probe_mod = tmp_path / "probe_mod.py"
    probe_mod.write_text("VALUE = 1\n")
    py_compile.compile(str(probe_mod), doraise=True)
    pyc_path = importlib.util.cache_from_source(str(probe_mod))
    header_mtime, header_size = _pyc_header_mtime_size(pyc_path)

    time.sleep(gap)

    probe_mod.write_text("VALUE = 2\n")  # same length as "VALUE = 1\n"
    current_stat = probe_mod.stat()
    window_open = int(current_stat.st_mtime) == header_mtime and current_stat.st_size == header_size

    if not window_open:
        pytest.skip(
            "Every timing-independent assertion above already ran and "
            "passed. Skipping only the timing-dependent part: after "
            f"sleeping {gap:.3f}s (one measured pytest startup in this "
            "synthetic project, the same gap the tool itself has between "
            "its baseline run and its first knife's write), a source "
            "file's (mtime, size) no longer matched a .pyc compiled just "
            "before the sleep -- so the stale-pyc collision window this "
            "test relies on to catch the M2 mutation did not open on this "
            "machine/run. This is a directly measured fact about this "
            "run, not a guessed threshold."
        )


def test_invalidate_bytecode_never_propagates_a_failed_delete(tmp_path, monkeypatch) -> None:
    """`_invalidate_bytecode` must swallow a failing delete, not raise.

    This is not hypothetical tidiness. One of its call sites is `main()`'s
    `finally` restore loop, which walks every mutated file and copies its
    backup back. An exception escaping from here aborts that walk partway,
    leaving every file it had not reached yet sitting on disk in its
    MUTATED state -- the precise outcome the restore loop exists to
    prevent, produced by the cleanup step meant to help it.

    The glob branch has always guarded its `os.unlink`; the
    `cache_from_source` branch did not, so a delete that lost a TOCTOU race
    (the file vanishing between `os.path.exists` and `os.unlink`) or hit a
    permission error propagated.

    Mutation this kills: drop the `try`/`except OSError` around the
    `cache_from_source` branch's `os.unlink(cached)`.
    """
    source = tmp_path / "mod.py"
    source.write_text("VALUE = 1\n")
    py_compile.compile(str(source), doraise=True)
    cached = importlib.util.cache_from_source(str(source))
    assert os.path.exists(cached), "precondition: the .pyc must exist to be deleted"

    real_unlink = os.unlink

    def failing_unlink(target, *args, **kwargs):
        if os.path.abspath(target) == os.path.abspath(cached):
            raise OSError(13, "Permission denied")
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", failing_unlink)

    # Must not raise.
    mutation_check._invalidate_bytecode(str(source))


def test_mutation_check_accepts_multiple_test_paths(tmp_path):
    """`--tests` takes `nargs="+"`: passing two paths must run BOTH
    against the same knife, and a second file the knife never touches
    (test_other.py, which does not even import pkg.m) must not change the
    verdict.

    Mutation: revert `--tests` to a bare string argument (drop
    `nargs="+"`/`action="extend"`). This call passes TWO values after
    --tests ("tests/test_m.py" "tests/test_other.py"), so with a plain
    single-value `--tests` argparse rejects the extra token at the
    parsing stage and exits with status 2 before `subprocess.run` is ever
    reached -- `proc.returncode == 0` goes red. (Character-by-character
    unpacking of a single un-split string, the failure mode the earlier
    version of this docstring described, only arises when exactly one
    path is passed after --tests; that is not what this call does.)
    """
    proj, wrapper_path, _ = _build_synthetic_project(tmp_path)
    (proj / "tests" / "test_other.py").write_text("def test_always_passes():\n    assert True\n")

    proc = subprocess.run(
        [
            str(wrapper_path),
            "scripts/mutation_check.py",
            "knives.json",
            "--tests",
            "tests/test_m.py",
            "tests/test_other.py",
        ],
        cwd=str(proj),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "BASELINE IS RED" not in out, out
    # Baseline collects both files: test_zero_is_nonpos (from test_m.py)
    # and test_always_passes (from test_other.py) -- "2 passed" in the
    # baseline line is the observable proof both paths were actually
    # handed to pytest, not just the first one.
    baseline_line = next(line for line in out.splitlines() if line.startswith("[baseline]"))
    assert "2 passed" in baseline_line, baseline_line
    a_line = next(line for line in out.splitlines() if line.startswith(("KILLED", "SURVIVED", "BROKEN")) and "A:" in line)
    b_line = next(line for line in out.splitlines() if line.startswith(("KILLED", "SURVIVED", "BROKEN")) and "B:" in line)
    assert a_line.startswith("KILLED"), out
    assert "red: tests/test_m.py::test_zero_is_nonpos" in out, out
    assert b_line.startswith("SURVIVED"), out


def _build_multi_failure_project(tmp_path, name="multiproj"):
    """Synthetic project whose single knife reddens TWO tests in the same
    pytest run. Mirrors `_build_synthetic_project`'s pyproject.toml
    (`addopts = ["-rs"]`) for the same reason documented there (F1): a
    synthetic project missing that file would not exercise this repo's
    real pytest reporting settings.
    """
    real_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proj = tmp_path / name
    (proj / "pkg").mkdir(parents=True)
    (proj / "tests").mkdir()
    (proj / "scripts").mkdir()
    (proj / ".venv" / "bin").mkdir(parents=True)

    original_m = 'def sign(x):\n    if x > 0:\n        return "pos"\n    return "nonpos"\n'

    (proj / "pkg" / "__init__.py").write_text("")
    (proj / "pkg" / "m.py").write_text(original_m)
    (proj / "tests" / "__init__.py").write_text("")
    (proj / "tests" / "test_m.py").write_text(
        "from pkg.m import sign\n\n\n"
        'def test_zero_is_nonpos():\n    assert sign(0) == "nonpos"\n\n\n'
        'def test_negative_is_nonpos():\n    assert sign(-5) == "nonpos"\n'
    )
    (proj / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = ["-rs"]\n')

    shutil.copy2(
        os.path.join(real_root, "scripts", "mutation_check.py"),
        str(proj / "scripts" / "mutation_check.py"),
    )

    wrapper_path = proj / ".venv" / "bin" / "python"
    wrapper_path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    knives = [
        {"name": "break nonpos for both callers", "file": "pkg/m.py", "old": "nonpos", "new": "nonneg"},
    ]
    (proj / "knives.json").write_text(json.dumps(knives))

    return proj, wrapper_path


def test_mutation_check_reports_all_failed_tests_not_just_first(tmp_path):
    """Core of this batch: a single knife can redden more than one test in
    the same run. With `-x` dropped, the tool must run the whole target
    suite to completion and report EVERY test that went red, not just
    whichever one pytest happened to hit first.

    Mutation: this is exactly what re-adding `-x` to `_run()`'s pytest
    invocation, or reverting `_FAILED_LINE_RE`/`_FAILURES_HEADER_RE` back
    to `.search()`, would break -- either would leave only one of the two
    `red:` lines (or none) in the output. Without this test, that
    regression would not be caught: the existing A/B knives in
    `_build_synthetic_project` are each written to affect at most one
    test, so nothing else in this file exercises the "one knife, several
    red tests" case.
    """
    proj, wrapper_path = _build_multi_failure_project(tmp_path)
    proc = subprocess.run(
        [str(wrapper_path), "scripts/mutation_check.py", "knives.json", "--tests", "tests/test_m.py"],
        cwd=str(proj),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "BASELINE IS RED" not in out, out
    assert "KILLED" in out, out
    assert "red: tests/test_m.py::test_zero_is_nonpos" in out, out
    assert "red: tests/test_m.py::test_negative_is_nonpos" in out, out


def test_mutation_check_repeated_tests_flag_accumulates_rather_than_overwrites(tmp_path):
    """`--tests` uses `action="extend"`: passing the flag twice
    (`--tests a.py --tests b.py`) must ACCUMULATE both paths, not let the
    second occurrence silently discard the first. This is the exact
    failure mode measured directly against an earlier draft of this
    flag: plain `action="store"` (argparse's default) with `--tests`
    given twice kept only the LAST occurrence, with no warning and exit
    0 -- the tool would print a complete-looking "N/M killed" report
    while having actually tested only one of the two intended files.

    Mutation: drop `action="extend"` (revert to plain `store`). The
    baseline line's "2 passed" count would then reflect only ONE file's
    tests, not both, and this test's assertions catch that.
    """
    proj, wrapper_path, _ = _build_synthetic_project(tmp_path)
    (proj / "tests" / "test_other.py").write_text("def test_always_passes():\n    assert True\n")

    proc = subprocess.run(
        [
            str(wrapper_path),
            "scripts/mutation_check.py",
            "knives.json",
            "--tests",
            "tests/test_m.py",
            "--tests",
            "tests/test_other.py",
        ],
        cwd=str(proj),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "BASELINE IS RED" not in out, out
    baseline_line = next(line for line in out.splitlines() if line.startswith("[baseline]"))
    assert "2 passed" in baseline_line, baseline_line
    assert "red: tests/test_m.py::test_zero_is_nonpos" in out, out


def _build_collection_error_project(tmp_path, name="collproj"):
    """Synthetic project where the single knife breaks pkg/shared.py's
    syntax, which:
      * breaks tests/test_a.py's collection outright (it imports
        pkg.shared at module scope) -- pytest reports this as
        "ERROR tests/test_a.py", not a per-test FAILURES entry.
      * does NOT break tests/test_b.py's own collection (it never
        imports pkg.shared; instead it reads and ast.parses the file's
        source text at runtime), but DOES make its one test's assertion
        fail -- "FAILED tests/test_b.py::test_shared_module_parses".
    This is deliberately arranged so ONE knife produces one collection
    ERROR and one ordinary test FAILURE in two different files at once --
    exactly the shape --continue-on-collection-errors exists to keep
    visible. Mirrors the real repo's pyproject.toml (addopts = ["-rs"])
    for the same reason as _build_synthetic_project's F1.
    """
    real_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proj = tmp_path / name
    (proj / "pkg").mkdir(parents=True)
    (proj / "tests").mkdir()
    (proj / "scripts").mkdir()
    (proj / ".venv" / "bin").mkdir(parents=True)

    (proj / "pkg" / "__init__.py").write_text("")
    (proj / "pkg" / "shared.py").write_text("VALUE = 1\n")
    (proj / "tests" / "__init__.py").write_text("")
    (proj / "tests" / "test_a.py").write_text("from pkg.shared import VALUE\n\n\ndef test_a():\n    assert VALUE == 1\n")
    (proj / "tests" / "test_b.py").write_text(
        "import ast\n"
        "import os\n\n\n"
        "def test_shared_module_parses():\n"
        '    path = os.path.join(os.path.dirname(__file__), "..", "pkg", "shared.py")\n'
        "    with open(path) as f:\n"
        "        source = f.read()\n"
        "    try:\n"
        "        ast.parse(source)\n"
        "        ok = True\n"
        "    except SyntaxError:\n"
        "        ok = False\n"
        "    assert ok\n"
    )
    (proj / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = ["-rs"]\n')

    shutil.copy2(
        os.path.join(real_root, "scripts", "mutation_check.py"),
        str(proj / "scripts" / "mutation_check.py"),
    )

    wrapper_path = proj / ".venv" / "bin" / "python"
    wrapper_path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    knives = [
        {"name": "break pkg/shared.py's syntax", "file": "pkg/shared.py", "old": "VALUE = 1", "new": "VALUE = 1)"},
    ]
    (proj / "knives.json").write_text(json.dumps(knives))

    return proj, wrapper_path


def test_mutation_check_collection_error_in_one_file_does_not_hide_another_files_red_test(tmp_path):
    """Core of item 2: without --continue-on-collection-errors, a
    collection ERROR in one target file makes pytest abort the ENTIRE run
    before any other file's tests execute at all -- measured directly:
    "Interrupted: 1 error during collection", and test_b.py's failure
    never even runs, so its FAILED line can never appear no matter how
    the output is parsed. The verdict would still print KILLED (the
    ERROR alone is enough) while silently discarding whether the rest of
    the multi-file coverage this batch exists for ran at all.

    Mutation: drop --continue-on-collection-errors from _run()'s pytest
    invocation. This test must go red because
    "red: tests/test_b.py::test_shared_module_parses" would be missing
    from the output -- test_b.py never runs.
    """
    proj, wrapper_path = _build_collection_error_project(tmp_path)
    proc = subprocess.run(
        [
            str(wrapper_path),
            "scripts/mutation_check.py",
            "knives.json",
            "--tests",
            "tests/test_a.py",
            "tests/test_b.py",
        ],
        cwd=str(proj),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "BASELINE IS RED" not in out, out
    assert "KILLED" in out, out
    assert "red: tests/test_a.py" in out, out
    assert "red: tests/test_b.py::test_shared_module_parses" in out, out


def test_mutation_check_default_tests_path_is_used_when_tests_is_omitted(tmp_path):
    """`--tests` is optional -- the module docstring's usage line puts it
    in brackets -- and omitting it falls back to tests/test_router.py.

    That fallback moved out of argparse's own `default=` and into a
    hand-written `args.tests or [...]` when `--tests` became
    `action="extend"`, because an extend action with a *list* default
    appends the command line onto it instead of replacing it. Nothing
    exercised the fallback afterwards: every other test in this file
    passes `--tests` explicitly, so reverting the line to
    `tests = args.tests` left all twelve of them green, while an omitted
    `--tests` would die with `TypeError: Value after * must be an
    iterable, not NoneType`.

    The synthetic project gets a tests/test_router.py that does catch
    knife A, so a fallback pointing somewhere unintended shows up as A
    surviving rather than only as a crash.

    Mutation: `tests = args.tests or ["tests/test_router.py"]` ->
    `tests = args.tests`. This test must go red.
    """
    proj, wrapper_path, _ = _build_synthetic_project(tmp_path)
    (proj / "tests" / "test_router.py").write_text(
        'from pkg.m import sign\n\n\ndef test_zero_is_nonpos_via_default_path():\n    assert sign(0) == "nonpos"\n'
    )

    proc = subprocess.run(
        [str(wrapper_path), "scripts/mutation_check.py", "knives.json"],
        cwd=str(proj),
        capture_output=True,
        text=True,
        # The five other call sites in this file all pass timeout=60; this
        # one was the only one without it, which meant a wedged child hung
        # the whole suite forever instead of failing one test. That is not
        # hypothetical: see `_build_synthetic_project`'s note on Gatekeeper
        # for a measured condition under which this exact subprocess takes
        # ~60s, and every other call site here turns that into a red test
        # while this one would have waited indefinitely.
        timeout=60,
    )
    out = proc.stdout + proc.stderr

    assert proc.returncode == 0, out
    assert "[baseline] 1 passed" in out, out
    assert "red: tests/test_router.py::test_zero_is_nonpos_via_default_path" in out, out
    assert "1/2 killed" in out, out


def _write_misfire_project(tmp_path):
    """A two-test module where one test asserts a guard REJECTS something and
    the other asserts it ACCEPTS something -- the minimum needed to tell a
    knife that hit its target apart from one that broke the code outright.
    Both tests pass against the unmutated source."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "m.py").write_text(
        "import re\n"
        'GUARDED = re.compile(r"count (?:the )?gates in the (?:design|netlist)")\n'
        "\n"
        "def claims(text):\n"
        "    return GUARDED.search(text) is not None\n"
    )
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_m.py").write_text(
        "import sys, os\n"
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        "from pkg.m import claims\n"
        "\n"
        "def test_guard_accepts_the_real_thing():\n"
        '    assert claims("count the gates in the design")\n'
        "\n"
        "def test_guard_rejects_a_different_object():\n"
        '    assert not claims("count the gates in the module")\n'
    )


def _run_tool(tmp_path, knives):
    """The tool resolves `knife["file"]` against the directory ABOVE its own
    location, so it has to be copied into the synthetic project rather than
    invoked from the real repo -- same arrangement the .pyc tests above use."""
    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copy2(mutation_check.__file__, tmp_path / "scripts" / "mutation_check.py")
    # The tool shells out to `.venv/bin/python` by name, so the synthetic
    # project needs one -- same shim the .pyc tests above build.
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    wrapper = venv_bin / "python"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    (tmp_path / "knives.json").write_text(json.dumps(knives))
    proc = subprocess.run(
        [sys.executable, "scripts/mutation_check.py", "knives.json", "--tests", "tests/test_m.py"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.stdout + proc.stderr


def test_a_knife_that_reddens_the_wrong_tests_is_MISFIRED_not_KILLED(tmp_path):
    """The regression this feature exists for, reproduced end to end.

    The knife below aims at the guard's object binding: it should make
    `claims("...in the module")` true, reddening the "rejects" test. Instead
    the replacement leaves a regex that matches nothing -- the shape a
    mis-escaped `\\\\w` produces when a knife is hand-typed into JSON. The
    suite DOES go red, but on the "accepts" test, the opposite direction.

    Without `expect_red` the tool prints KILLED and the run reads as proof
    the guard is covered, which is exactly what happened on 2026-09-01 and
    cost two review rounds to unwind.
    """
    _write_misfire_project(tmp_path)
    out = _run_tool(
        tmp_path,
        [
            {
                "name": "mis-escaped wildcard",
                "file": "pkg/m.py",
                "old": "(?:design|netlist)",
                "new": "(?:\\\\w+)",
                "expect_red": ["test_guard_rejects_a_different_object"],
            }
        ],
    )
    assert "MISFIRED" in out, out
    assert "0/1 killed" in out, out
    # The unmet expectation is named, not just counted -- the reader has to be
    # able to see WHICH aim was missed.
    assert "expected red, but nothing matched: test_guard_rejects_a_different_object" in out, out
    # And the test that did go red is still printed, so the mis-escape is
    # diagnosable from this output alone.
    assert "test_guard_accepts_the_real_thing" in out, out


def test_a_knife_that_reddens_its_intended_test_is_still_KILLED(tmp_path):
    """The control. Same guard, same expectation, a knife that actually lands:
    without it, `expect_red` could be satisfied by reporting MISFIRED for
    everything and this file would not notice."""
    _write_misfire_project(tmp_path)
    out = _run_tool(
        tmp_path,
        [
            {
                "name": "object binding -> wildcard",
                "file": "pkg/m.py",
                "old": "(?:design|netlist)",
                "new": "\\w+",
                "expect_red": ["test_guard_rejects_a_different_object"],
            }
        ],
    )
    assert "KILLED" in out, out
    assert "MISFIRED" not in out, out
    assert "1/1 killed" in out, out


def test_a_knife_without_expect_red_keeps_the_old_behaviour(tmp_path):
    """`expect_red` is optional: every knives.json written before this feature
    existed must keep working unchanged, or adding it silently invalidates the
    project's existing mutation records."""
    _write_misfire_project(tmp_path)
    out = _run_tool(
        tmp_path,
        [{"name": "no expectation", "file": "pkg/m.py", "old": "(?:design|netlist)", "new": "(?:\\\\w+)"}],
    )
    assert "KILLED" in out, out
    assert "MISFIRED" not in out, out
