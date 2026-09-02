#!/usr/bin/env python3
"""Apply one deliberate defect at a time and report which ones the tests catch.

    python scripts/mutation_check.py knives.json [--tests tests/test_a.py tests/test_b.py]

`--tests` accepts one or more paths and MUST come last: it uses argparse's
`nargs="+"`, which is greedy and would otherwise swallow the `knives.json`
positional argument too (e.g. `--tests a.py b.py knives.json` puts
knives.json into --tests and then fails with a confusing "missing knives"
error). Put `knives.json` before `--tests`, as in the usage line above.

A "knife" is a single textual substitution that should break behaviour:

    [
      {"name": "drop the 'zero' spelling from the length",
       "file": "netlist_agent/router.py",
       "old": "(?:0|zero))",
       "new": "(?:0))",
       "expect_red": ["test_length_accepts_the_word_zero"]}
    ]

`expect_red` is optional but strongly recommended: a list of substrings, each
of which must appear in the name of at least one test that went red. If the
suite goes red but an expectation matches nothing, the verdict is MISFIRED
rather than KILLED.

Why it is worth the extra line: KILLED only says "something broke". It does
not say the guard under test caught anything. On 2026-09-01 a knife aimed at
an overreach guard was written with the wrong escaping; the substitution left
a regex that matched nothing, three tests went red, and the run printed
KILLED. The three that reddened were the ones asserting the pattern still
matches -- the opposite direction from the guard the knife was aimed at, which
was never exercised at all. Nothing in the output distinguished that from a
real kill. `expect_red` makes the comparison the tool's job instead of the
reader's.

Write `expect_red` from the intent, before running: name the test that SHOULD
notice this defect. If you cannot name one, that is the finding -- the defect
has no observer -- and it is worth knowing before the run rather than after.

Each knife is applied alone, the tests are run, and the file is restored --
including on exit, so an interrupted run does not leave a mutated tree behind.

Why this exists rather than "do we have tests for it"
-----------------------------------------------------
Test count is not evidence. This project has repeatedly shipped code where the
tests ran, passed, and were aimed one layer above the thing that could be
wrong: assertions on a regex match object while the bug lived in the handler
reading it, or on an internal value while the user saw a separately-rendered
string. A knife that SURVIVES is the finding -- it names a change that the
whole suite cannot see.

Three rules this tool enforces, all learned the expensive way:

  * the anchor must appear EXACTLY once, or the knife is reported BROKEN. A
    knife that never lands is indistinguishable from one that was killed, and
    reads as reassurance. This is not hypothetical: a guard being edited here
    once appeared four times in the file rather than twice, because the
    comments quote the regex verbatim.
  * the baseline must be green before any knife is cut. Against a red suite
    every knife "fails" and the whole run reads as perfect coverage.
  * a stale .pyc can make a knife's verdict silently wrong. CPython decides
    whether a cached .pyc is still valid by comparing (int(source mtime in
    seconds), source size) against the numbers baked into the .pyc header --
    not by hashing the source. If a mutation's replacement text is the same
    length as the original AND the write lands within the same wall-clock
    second as the .pyc that is already on disk, the interpreter will happily
    import the OLD bytecode while the file on disk holds the NEW text. This
    was reproduced directly and deterministically (see
    tests/test_mutation_check.py, the mechanism test that forces the mtime
    collision by hand rather than racing a real clock) and end-to-end
    against this tool itself (the same file's test that runs the tool
    repeatedly against a synthetic project). Two windows open this up in
    practice: (1) cutting the first knife within the same second as
    whatever run last compiled the target module (e.g. running this tool
    right after editing the file by hand), and (2) two knives of equal
    length cut back-to-back, where the gap is only as long as one pytest
    startup -- which is precisely what happens when the target test file
    is fast, i.e. the normal iteration workflow this tool is meant to
    support. Restoring a file (via shutil.copy2) is NOT part of this
    hazard: copy2 preserves the backup's original mtime, which will not
    match the freshly-compiled .pyc, forcing a recompile. The hazard is
    specifically in the *write that applies a mutation*
    (open(path, "w").write(...)), whose mtime is "now".

A knife that survives for a *structural* reason (the mutated behaviour is
unreachable, or the operation is commutative so the change is invisible) is
not the same as one that survives because nothing tests it. Both print
SURVIVED; work out which you have, and write down the reason.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _invalidate_bytecode(path: str) -> None:
    """Delete any cached .pyc for `path` so a stale one cannot be imported.

    Uses importlib.util.cache_from_source() for the normal case, then also
    globs for `__pycache__/<stem>.*.pyc` next to the source file as a
    fallback. The glob is not redundant: cache_from_source() reports where
    *this* interpreter (the one running mutation_check.py) would look, but
    the tests are actually run under `.venv/bin/python` -- a different
    interpreter that may use a different magic tag (so a different pyc
    filename) or a different sys.pycache_prefix. The glob makes cache
    invalidation immune to that mismatch. Deleting a .pyc that turns out
    not to be stale is always harmless -- the interpreter just recompiles
    it -- so there is no need to be more surgical than "delete every
    <stem>.*.pyc next to this file".

    Missing files are ignored, and so is a delete that fails: this is
    best-effort hygiene, not a step whose failure should abort a mutation
    run. That matters more than it looks, because one of the call sites is
    the `finally` restore loop -- an exception escaping from here would
    abort that loop partway and leave every not-yet-restored file sitting
    on disk in its mutated state, which is exactly what the restore loop
    exists to prevent.
    """
    try:
        cached = importlib.util.cache_from_source(path)
    except (ValueError, OSError):
        cached = None
    if cached and os.path.exists(cached):
        try:
            os.unlink(cached)
        except OSError:
            pass

    # `os.path.dirname(os.path.abspath(path))` and `os.path.split(path)[0]`
    # pick out the same directory for every shape of path, bare filenames
    # included (both resolve relative to the cwd there, which is the only
    # thing a bare filename can mean). The abspath form is used purely
    # because it says so explicitly.
    directory = os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    for stale in glob.glob(os.path.join(directory, "__pycache__", f"{stem}.*.pyc")):
        try:
            os.unlink(stale)
        except OSError:
            pass


_FAILED_LINE_RE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)
_FAILURES_HEADER_RE = re.compile(r"^_+ (\S+) _+$", re.MULTILINE)


def _run(tests: list[str]) -> tuple[int, str, list[str]]:
    proc = subprocess.run(
        # -rfE forces the short summary line (FAILED/ERROR path::test) to
        # be printed regardless of the target project's own pytest config.
        # This project's own pyproject.toml sets `addopts = ["-rs"]`, which
        # does NOT include f/E -- measured directly (deliberately reddening
        # a test in tests/test_session.py): without -rfE, `-q --no-header`
        # prints only the bare "____ test_name ____" FAILURES header and no
        # "FAILED path::test_name" line at all, so the fully-qualified-name
        # branch below was a dead path under this project's real settings,
        # not a fallback -- adding -rfE here makes the tool's own diagnostics
        # independent of whatever addopts the target project happens to set.
        #
        # No -x: this project's verdicts are read off of WHICH tests turned
        # red, not just whether any did (see the module docstring's "a red
        # test is not proof the knife cut" lesson) -- stopping at the first
        # failure would silently discard every other red test a knife
        # caused. The cost is that a KILLED knife now runs the whole target
        # suite to completion instead of stopping at the first failure, so
        # each cut is somewhat slower; nobody has asked for a flag to trade
        # that back, so none is added here.
        # --continue-on-collection-errors: a knife is a blind textual
        # substitution, so it can wreck one target file's import/syntax
        # without touching another's. Without this flag, pytest aborts
        # collection entirely on the first such error and NOTHING in any
        # of the other --tests paths runs at all -- the verdict would
        # still print KILLED (from that one collection ERROR) while
        # silently hiding that the rest of the multi-file coverage this
        # batch exists for never executed.
        [
            ".venv/bin/python",
            "-m",
            "pytest",
            *tests,
            "-q",
            "--no-header",
            "-rfE",
            "--continue-on-collection-errors",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    lines = output.strip().splitlines()
    tail = lines[-1] if lines else ""
    # A red suite is not, by itself, evidence the knife hit the intended
    # target -- a parametrized case can fail while its siblings stay green,
    # or an unrelated pre-existing failure could be the first one pytest
    # happens to print. Recording *which* tests failed lets a human check
    # that, and recording ALL of them (not just the first) matters because
    # a knife can redden more than one test at once, and which ones is
    # itself diagnostic of whether the knife hit its intended target.
    #
    # pytest -q prints two candidates: a bare "____ test_name ____" header
    # in the FAILURES section (appears first, no file path or parametrize
    # id), and a "FAILED path::test_name" / "ERROR path - ..." line in the
    # short summary at the end (appears later, fully qualified -- and the
    # only form collection/setup errors produce, since those never reach a
    # per-test FAILURES header at all). Prefer the fully-qualified
    # short-summary lines when present -- they disambiguate which file and
    # which parametrized case actually failed.
    failures = _FAILED_LINE_RE.findall(output)
    if not failures:
        failures = _FAILURES_HEADER_RE.findall(output)
    return proc.returncode, tail, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("knives", help="JSON list of {name, file, old, new}")
    parser.add_argument(
        "--tests",
        nargs="+",
        action="extend",
        default=None,
        help=(
            "one or more test paths to run per knife. Repeatable -- "
            "'--tests a.py --tests b.py' accumulates both, it does not "
            "overwrite. Must come LAST on the command line: this flag is "
            "greedy (nargs='+') and will swallow the knives.json positional "
            "argument if it is given after --tests, e.g. '--tests a.py "
            "b.py knives.json' puts knives.json into --tests instead of "
            "being read as the knives file."
        ),
    )
    args = parser.parse_args()
    # `action="extend"` with a *list* default appends the CLI value onto
    # that default instead of replacing it (e.g. default=["d.py"] plus
    # "--tests a.py" yields ["d.py", "a.py"], not ["a.py"]) -- measured
    # directly. default=None sidesteps that: when --tests is omitted,
    # args.tests stays None and the real default is applied here instead.
    tests = args.tests or ["tests/test_router.py"]

    with open(args.knives) as handle:
        knives = json.load(handle)

    backups: dict[str, str] = {}
    for knife in knives:
        path = os.path.join(_ROOT, knife["file"])
        if path not in backups:
            descriptor, backup = tempfile.mkstemp(suffix=".bak")
            os.close(descriptor)
            shutil.copy2(path, backup)
            backups[path] = backup

    # A pre-existing stale .pyc for any of these files would make the
    # baseline itself run the wrong code, which poisons every verdict
    # that follows -- so clear the cache before the baseline runs, too.
    for path in backups:
        _invalidate_bytecode(path)

    code, tail, _ = _run(tests)
    print(f"[baseline] {tail}")
    if code != 0:
        print("BASELINE IS RED -- fix that first; against a red suite every knife reads as KILLED")
        for path, backup in backups.items():
            os.unlink(backup)
        return 1

    results: list[tuple[str, str]] = []
    try:
        for knife in knives:
            path = os.path.join(_ROOT, knife["file"])
            source = open(path).read()
            hits = source.count(knife["old"])
            if hits != 1:
                print(f"BROKEN    {knife['name']}  (anchor appears {hits}x, expected exactly 1)")
                results.append((knife["name"], "BROKEN"))
                continue
            with open(path, "w") as handle:
                handle.write(source.replace(knife["old"], knife["new"]))
            _invalidate_bytecode(path)
            code, tail, failures = _run(tests)
            shutil.copy2(backups[path], path)
            _invalidate_bytecode(path)
            verdict = "KILLED" if code != 0 else "SURVIVED"
            # `expect_red` turns "which tests went red" from something a human
            # is trusted to eyeball into something this tool checks. Measured
            # cost of not having it (2026-09-01): a knife aimed at an
            # overreach guard was written with the wrong escaping, so the
            # mutated regex matched nothing at all. Three tests went red and
            # the run printed KILLED -- but they were the three asserting the
            # pattern still MATCHES, not the three asserting it does not
            # over-claim. The guard under test was never exercised, and the
            # report read as reassurance. A knife that breaks the code
            # outright is indistinguishable from a knife its target caught,
            # unless someone compares the red list against the intent, and
            # the whole reason this file exists is that "someone will
            # remember to compare" is not a control.
            unmet: list[str] = []
            if verdict == "KILLED" and knife.get("expect_red"):
                unmet = [
                    want for want in knife["expect_red"]
                    if not any(want in failed for failed in failures)
                ]
                if unmet:
                    verdict = "MISFIRED"
            print(f"{verdict:9s} {knife['name']}   [{tail}]")
            # Every red test is printed, not just the first -- this is the
            # whole point of dropping -x above. Deliberately not truncated,
            # even when there are many: which tests went red is exactly the
            # information a human needs to tell "the knife hit its intended
            # target" apart from "the knife broke something else entirely".
            for failed in failures:
                print(f"    red: {failed}")
            for want in unmet:
                print(f"    ⚠ expected red, but nothing matched: {want}")
            if unmet:
                print("      -- tests DID go red, but not the ones this knife was aimed at.")
                print("      -- treat this as the knife missing, not as coverage.")
            results.append((knife["name"], verdict))
    finally:
        for path, backup in backups.items():
            shutil.copy2(backup, path)
            os.unlink(backup)
            # Covers the normal case, which is already safe on its own
            # (copy2 preserves the backup's original mtime, so a stale
            # freshly-mutated .pyc will not match it and gets recompiled).
            # This call matters for the case that ISN'T safe on its own:
            # an interrupted run. If this process is killed mid-_run(),
            # pytest may already have compiled the mutated source into a
            # .pyc before the interruption, and this `finally` block only
            # restores the file's content/mtime, not the cache -- leaving
            # a mutated .pyc keyed to the restored (original) mtime for
            # the NEXT knife or the next invocation to collide with. Known
            # gap: this path is not covered by a test, because reliably
            # interrupting a subprocess mid-compile is itself a race.
            _invalidate_bytecode(path)

    killed = sum(1 for _, verdict in results if verdict == "KILLED")
    print(f"\n{killed}/{len(results)} killed")
    survivors = [(name, verdict) for name, verdict in results if verdict != "KILLED"]
    for name, verdict in survivors:
        print(f"  {verdict}: {name}")
    misfired = [name for name, verdict in results if verdict == "MISFIRED"]
    if misfired:
        print("\nMISFIRED is not a weaker KILLED. The mutation broke something, but not the")
        print("thing the knife was aimed at -- most often the edit made the code invalid")
        print("rather than merely wrong. Fix the knife and re-cut; nothing has been")
        print("learned about the guard it was meant to test.")
    plain_survivors = [(n, v) for n, v in survivors if v == "SURVIVED"]
    if plain_survivors:
        print("\nEach survivor is either a coverage gap or a structurally unreachable change.")
        print("Decide which, and record the reason -- they look identical from here.")
        print("Before writing \"unreachable\": write down one input that WOULD reach it, and")
        print("run that input. If you cannot produce one, say what shapes you tried.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
