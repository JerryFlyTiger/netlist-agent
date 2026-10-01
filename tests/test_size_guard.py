"""No git-tracked file may exceed a small size ceiling.

Why: a corpus replay once got pointed at a git-tracked directory
(`experiments/`) and `git add`ed whole, staging 348 files / 530 MB and
hitting GitHub's 100 MB single-file cap. The .gitignore patch that fixed
that push only blocks specific filename shapes; a future artifact with a
name nobody thought of would sail straight through. This test is the
size-based backstop: it doesn't care what the file is called, only how
big it is.

The real assertion (10 MB, must find nothing) can never go red on its own
merit while the tree happens to be clean -- any mutation to the comparison
logic would pass just as silently as the correct code. The synthetic
observer test below pins the same logic against a threshold (1 KB) that
the tracked tree is guaranteed to exceed somewhere, so a broken comparison
(e.g. `>` flipped to `<`, or the limit ignored) shows up as a failure here
even when nothing oversized is actually tracked.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEN_MB = 10 * 1024 * 1024

# Narrow exemption (2026-09-28): the official contest testcase corpora are
# DELIBERATELY tracked in full, and three final_release_100 designs
# (test002, test016, test051) are 11,337,078 bytes each -- over the 10 MB
# ceiling on their own merit, not by accident. That is the opposite failure
# mode from the one this guard exists to catch (an accidental run_corpus
# artifact getting `git add`ed), so a blanket "anything under these
# directories is fine" would defeat the guard's own purpose: a stray 500 MB
# enumeration dump dropped into final_release_100/ by a future corpus replay
# would sail through unnoticed. The exemption is therefore shaped, not
# directory-wide -- only a path matching `<corpus root>/testNNN/testNNN.v`
# (the corpus's own design-file naming convention, confirmed against what
# `git ls-files` actually tracks under each root) is let through; anything
# else under these same directories (an `_out.v`, an `enum_paths.v`, a
# stray `.txt`) is still held to the 10 MB ceiling like every other file.
_CORPUS_DESIGN_FILE_RE = re.compile(
    r"^(?:Alpha_Testcase/testcase|Beta_Testcase/testcase|final_release_100)/(test\d+)/\1\.v$"
)


def _is_exempt_corpus_design_file(rel_path: str) -> bool:
    return bool(_CORPUS_DESIGN_FILE_RE.match(rel_path))


def _in_git_work_tree(root: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", root, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def oversized_tracked_files(limit_bytes: int, root: str = REPO_ROOT) -> list[tuple[str, int]]:
    """Return (path, size_bytes) for every git-tracked file under `root`
    whose size exceeds `limit_bytes`, sorted largest first.

    Untracked and gitignored files are not considered -- this checks what
    would actually be pushed, not what happens to sit in the work tree.
    """
    proc = subprocess.run(
        ["git", "-C", root, "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [p for p in proc.stdout.split("\0") if p]
    oversized = []
    for rel_path in paths:
        full_path = os.path.join(root, rel_path)
        try:
            size = os.path.getsize(full_path)
        except OSError:
            continue  # tracked-but-deleted in the working tree; nothing to weigh
        if size > limit_bytes and not _is_exempt_corpus_design_file(rel_path):
            oversized.append((rel_path, size))
    oversized.sort(key=lambda entry: entry[1], reverse=True)
    return oversized


def test_no_tracked_file_exceeds_10mb():
    if not _in_git_work_tree(REPO_ROOT):
        pytest.skip("not running inside a git work tree")

    oversized = oversized_tracked_files(TEN_MB)
    assert oversized == [], (
        "git-tracked file(s) over the 10 MB ceiling (a run_corpus enumeration "
        "artifact, or something like it, likely got committed by accident):\n"
        + "\n".join(f"  {path}: {size:,} bytes" for path, size in oversized)
    )


def test_ten_mb_constant_is_ten_megabytes():
    """Pins the actual ceiling value: a change that quietly widens (or
    narrows) it changes the real assertion's meaning without changing
    whether it passes on this tree."""
    assert TEN_MB == 10 * 1024 * 1024


def test_oversized_tracked_files_finds_something_at_a_tiny_threshold(tmp_path):
    """Synthetic observer, run against a throwaway git repo rather than this
    one: a 2 KB tracked file, a 10-byte tracked file, and a 4 KB file that
    is never `git add`ed. At a 1 KB threshold the result must be exactly the
    2 KB file.

    Why a throwaway repo: an earlier version only asserted "non-empty",
    which survives flipping `>` to `<` (almost every tracked file is on one
    side of 1 KB or the other). The next version named two real files of
    this repo -- but `scripts/export_public.py` copies this test into the
    public repo, which structurally has no `experiments/`, so the named
    "known large" file could never exist there. Building the repo here pins
    the direction (`<` would return only the 10-byte file), the
    tracked-only rule (the untracked 4 KB file must not appear), and the
    reported size, with nothing that can differ between the two repos."""
    try:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable: {exc}")
    (tmp_path / "big.bin").write_bytes(b"x" * 2048)
    (tmp_path / "small.txt").write_bytes(b"y" * 10)
    (tmp_path / "untracked.bin").write_bytes(b"z" * 4096)
    subprocess.run(["git", "-C", str(tmp_path), "add", "big.bin", "small.txt"], check=True, capture_output=True)

    assert oversized_tracked_files(1024, root=str(tmp_path)) == [("big.bin", 2048)], (
        "oversized_tracked_files() looks broken: expected only the 2 KB tracked file"
    )


# ----------------------------------------------------------------------
# Corpus design file exemption (2026-09-28)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "Alpha_Testcase/testcase/test16/test16.v",
        "Beta_Testcase/testcase/test39/test39.v",
        "final_release_100/test002/test002.v",
        "final_release_100/test100/test100.v",
    ],
)
def test_corpus_design_file_shape_is_exempt(rel_path: str) -> None:
    """The exact `<corpus root>/testNNN/testNNN.v` shape, for each of the
    three tracked corpus roots, is let through -- this is the positive side
    of the exemption."""
    assert _is_exempt_corpus_design_file(rel_path)


@pytest.mark.parametrize(
    "rel_path",
    [
        # Same directory, wrong filename -- not the design file itself.
        "final_release_100/test002/enum_paths.v",
        "final_release_100/test002/foo.txt",
        # Same directory, the OTHER tracked .v file in Alpha_Testcase (the
        # synthesized/optimized output, not the original design).
        "Alpha_Testcase/testcase/test16/test16_out.v",
        # Right filename shape, wrong root -- outside any corpus directory.
        "experiments/some_run/test002/test002.v",
        # Right filename shape, root name typo'd -- must not fuzzy-match.
        "Final_Release_100/test002/test002.v",
        # Directory number and filename number disagree.
        "final_release_100/test002/test003.v",
    ],
)
def test_non_design_shapes_are_not_exempt(rel_path: str) -> None:
    """Everything else under a corpus root -- a differently-named file, a
    mismatched directory/filename pair, or a lookalike root -- is still
    held to the size ceiling like any other tracked file."""
    assert not _is_exempt_corpus_design_file(rel_path)


def test_real_oversized_final_release_designs_are_exempt():
    """The three actual final_release_100 designs that motivated this
    exemption (11,337,078 bytes each, over the 10 MB ceiling) exist on disk
    in this tree and match the shape -- pins the exemption against the
    real files, not just a synthetic string."""
    for name in ("test002", "test016", "test051"):
        rel_path = f"final_release_100/{name}/{name}.v"
        full_path = os.path.join(REPO_ROOT, rel_path)
        if not os.path.exists(full_path):
            pytest.skip(f"{rel_path} not present in this checkout")
        assert os.path.getsize(full_path) > TEN_MB
        assert _is_exempt_corpus_design_file(rel_path)


def _init_repo_with_tracked_file(root: str, rel_path: str, size_bytes: int) -> None:
    full_path = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as f:
        f.write(b"\0" * size_bytes)
    subprocess.run(["git", "-C", root, "init", "-q"], check=True)
    subprocess.run(["git", "-C", root, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", root, "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", root, "add", rel_path], check=True)
    subprocess.run(["git", "-C", root, "commit", "-q", "-m", "init"], check=True)


def test_oversized_tracked_files_exempts_design_shape_in_a_synthetic_repo():
    """A large file matching the exact corpus design shape is exempt --
    exercised against a real tmp git repo, not just the string matcher, so
    a mismatch between `_is_exempt_corpus_design_file` and how
    `oversized_tracked_files` actually calls it would show up here."""
    with tempfile.TemporaryDirectory() as tmp:
        _init_repo_with_tracked_file(tmp, "final_release_100/test099/test099.v", TEN_MB + 1)
        oversized = oversized_tracked_files(TEN_MB, root=tmp)
    assert oversized == []


def test_oversized_tracked_files_still_flags_lookalike_file_in_corpus_dir():
    """A large file that merely SITS in a corpus directory, but does not
    match the `testNNN/testNNN.v` shape, is still flagged -- the exemption
    is shaped, not directory-wide."""
    with tempfile.TemporaryDirectory() as tmp:
        _init_repo_with_tracked_file(tmp, "final_release_100/test099/enum_paths.v", TEN_MB + 1)
        oversized = oversized_tracked_files(TEN_MB, root=tmp)
    assert oversized == [("final_release_100/test099/enum_paths.v", TEN_MB + 1)]


def test_oversized_tracked_files_still_flags_large_v_file_outside_corpus():
    """A large `.v` file outside any of the three corpus roots is still
    flagged, even though its filename shape happens to match
    `testNNN/testNNN.v` -- the exemption is scoped to the corpus roots, not
    to any directory with that naming convention."""
    with tempfile.TemporaryDirectory() as tmp:
        _init_repo_with_tracked_file(tmp, "scratch/test099/test099.v", TEN_MB + 1)
        oversized = oversized_tracked_files(TEN_MB, root=tmp)
    assert oversized == [("scratch/test099/test099.v", TEN_MB + 1)]
