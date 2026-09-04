"""scripts/results_dir_guard.py: refuse to sweep corpus-replay artifacts
into a git-tracked, non-ignored location.

Exercised against disposable `git init` repos under tmp_path rather than
this repo itself, so the guard's own decision logic (not the state of any
particular directory) is what gets pinned down.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA_ROOT = os.path.join(REPO_ROOT, "Alpha_Testcase")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from results_dir_guard import check_case_artifacts, git_root, ignored_paths  # noqa: E402


def _init_repo(path: str) -> None:
    subprocess.run(["git", "init", "-q", path], check=True)


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not installed",
)
class TestGitRoot:
    def test_none_outside_any_repo(self, tmp_path):
        outside = tmp_path / "not_a_repo"
        outside.mkdir()
        assert git_root(str(outside)) is None

    def test_finds_the_repo_root_from_a_subdirectory(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(str(repo))
        sub = repo / "a" / "b"
        sub.mkdir(parents=True)
        found = git_root(str(sub))
        assert found is not None
        assert os.path.realpath(found) == os.path.realpath(str(repo))


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not installed",
)
class TestIgnoredPaths:
    def test_batches_ignored_and_tracked_paths_in_one_call(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(str(repo))
        (repo / ".gitignore").write_text("ignored_dir/*\n")
        (repo / "ignored_dir").mkdir()
        (repo / "tracked_dir").mkdir()

        ignored_path = str(repo / "ignored_dir" / "a.log")
        tracked_path = str(repo / "tracked_dir" / "b.log")

        calls = []
        real_run = subprocess.run

        def counting_run(*args, **kwargs):
            calls.append((args, kwargs))
            return real_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", counting_run)

        result = ignored_paths(str(repo), [ignored_path, tracked_path])

        assert result == {ignored_path}
        assert len(calls) == 1, "expected a single batched git check-ignore call, not one per path"

    def test_empty_input_makes_no_subprocess_call(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(str(repo))

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not shell out for an empty list")),
        )
        assert ignored_paths(str(repo), []) == set()


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not installed",
)
class TestCheckCaseArtifacts:
    def test_noop_when_root_is_none(self, tmp_path):
        # No git repo involved at all -- must not raise regardless of what's
        # in results_dir or artifact_names.
        check_case_artifacts(None, str(tmp_path), ["anything.log"])

    def test_noop_when_results_dir_is_outside_the_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(str(repo))
        outside = tmp_path / "elsewhere"
        outside.mkdir()

        check_case_artifacts(str(repo), str(outside), ["test01.log"])

    def test_passes_when_the_destination_is_gitignored(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(str(repo))
        (repo / ".gitignore").write_text("results/*\n")
        results_dir = repo / "results"
        results_dir.mkdir()

        check_case_artifacts(str(repo), str(results_dir), ["test01.log", "test01_out.v"])

    def test_raises_when_the_destination_is_tracked_and_not_ignored(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(str(repo))
        results_dir = repo / "results"
        results_dir.mkdir()

        with pytest.raises(SystemExit) as excinfo:
            check_case_artifacts(str(repo), str(results_dir), ["test01.log"])
        assert "test01.log" in str(excinfo.value)

    def test_empty_artifact_list_never_raises(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_repo(str(repo))
        results_dir = repo / "results"
        results_dir.mkdir()

        check_case_artifacts(str(repo), str(results_dir), [])


@pytest.mark.skipif(not os.path.isdir(ALPHA_ROOT), reason="Alpha_Testcase corpus not present")
@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not installed",
)
def test_run_corpus_refuses_a_tracked_unignored_results_dir():
    """End-to-end: scripts/run_corpus.py wired the guard into its sweep
    loop, not just imported it. Runs one real case against a scratch
    directory inside this actual repo that is tracked and NOT gitignored,
    and checks the driver exits non-zero and leaves nothing behind there."""
    probe_dir = os.path.join(REPO_ROOT, "pytest_run_corpus_guard_probe")
    assert git_root(REPO_ROOT) is not None
    assert not os.path.exists(probe_dir), "leftover probe dir from a previous run"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "scripts", "run_corpus.py"),
                "test01",
                "--results-dir",
                probe_dir,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode != 0, (
            f"expected a non-zero exit when --results-dir is tracked and unignored; "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        assert "non-ignored location" in proc.stderr
        # the artifacts must NOT have been swept into the tracked probe dir
        assert glob.glob(os.path.join(probe_dir, "test01*")) == []
    finally:
        if os.path.isdir(probe_dir):
            shutil.rmtree(probe_dir)
        # the run also writes test01's artifacts back where the case lives
        # if the guard tripped before the sweep -- clean those up too.
        for stray in glob.glob(os.path.join(ALPHA_ROOT, "test01.log")):
            os.remove(stray)
        for stray in glob.glob(os.path.join(ALPHA_ROOT, "testcase", "test01", "test01_out.v")):
            os.remove(stray)
