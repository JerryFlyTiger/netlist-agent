"""Guard against sweeping corpus-replay artifacts into a git-tracked,
non-ignored location.

Why this exists: a corpus replay writes a full path-enumeration side file
per case (one hit 104 MB / 289,368 lines -- see QA A16, "list all paths" is
correct behaviour, the file just happens to be huge). The driver sweeps
those artifacts out of the (read-only-intent) testcase dir into
`--results-dir` afterwards. `--results-dir` is a free-form path, so nothing
stopped it from being pointed at a git-tracked directory that isn't
`.gitignore`d -- which is exactly what happened once (`experiments/`
itself), landing 348 files / 530 MB in `git add` and blocking the push
past GitHub's 100 MB file cap. This module is the fix: refuse to write
artifacts into such a location instead of writing them and letting the
next `git add` discover the problem.

Shared by scripts/run_corpus.py and
experiments/self_score_2026-09-04/self_score.py.
"""

from __future__ import annotations

import os
import subprocess


def git_root(start_dir: str) -> str | None:
    """Return the git work tree root containing `start_dir`, or None if
    git isn't installed or `start_dir` isn't inside a git repo."""
    try:
        proc = subprocess.run(
            ["git", "-C", start_dir, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def ignored_paths(root: str, paths: list[str]) -> set[str]:
    """Return the subset of `paths` (absolute) that git ignores, checked in
    one batched `git check-ignore --stdin` call under `root`."""
    if not paths:
        return set()
    proc = subprocess.run(
        ["git", "-C", root, "check-ignore", "--stdin"],
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
    )
    return {line for line in proc.stdout.splitlines() if line}


def check_case_artifacts(root: str | None, results_dir: str, artifact_names: list[str]) -> None:
    """Raise SystemExit if writing `artifact_names` under `results_dir`
    would land git-tracked, non-ignored files in the work tree.

    No-op when `root` is None (not a git repo / git unavailable, resolved
    once up front via `git_root()`), or when `results_dir` sits outside
    that work tree entirely (the destination can't be tracked by that repo
    at all).
    """
    if root is None:
        return
    results_dir_abs = os.path.abspath(results_dir)
    root_abs = os.path.abspath(root)
    if os.path.commonpath([results_dir_abs, root_abs]) != root_abs:
        return  # results_dir is outside the repo work tree -- safe
    dest_paths = [os.path.join(results_dir_abs, name) for name in artifact_names]
    if not dest_paths:
        return
    ignored = ignored_paths(root_abs, dest_paths)
    unignored = [p for p in dest_paths if p not in ignored]
    if unignored:
        offenders = "\n".join(f"  {os.path.relpath(p, root_abs)}" for p in unignored)
        rel_dir = os.path.relpath(results_dir_abs, root_abs)
        raise SystemExit(
            "refusing to sweep corpus-replay artifacts into a git-tracked, "
            "non-ignored location:\n"
            f"{offenders}\n"
            "add a line like this to .gitignore (or point --results-dir "
            "somewhere already ignored, e.g. under /tmp):\n"
            f"  {rel_dir}/*\n"
        )
