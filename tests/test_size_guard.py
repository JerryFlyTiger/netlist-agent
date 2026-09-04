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
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEN_MB = 10 * 1024 * 1024


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
        if size > limit_bytes:
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


def test_oversized_tracked_files_finds_something_at_a_tiny_threshold():
    """Synthetic observer: with a 1 KB threshold, some git-tracked file in
    this repo is guaranteed to exceed it. If this ever comes back empty,
    the comparison logic itself is broken (wrong operator, ignored limit,
    empty file list, ...) -- not that the repo got smaller than 1 KB."""
    if not _in_git_work_tree(REPO_ROOT):
        pytest.skip("not running inside a git work tree")

    oversized = oversized_tracked_files(1024)
    assert oversized, (
        "expected at least one git-tracked file over 1 KB -- "
        "oversized_tracked_files() looks broken, not the repo"
    )
