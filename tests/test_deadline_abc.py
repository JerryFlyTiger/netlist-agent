"""Tests for the ABC subprocess leaves clamped to the request deadline (batch 21b).

Must not import `netlist_agent.router`: this file is exported to the public
repo, where the router is a stub. All time is a fake clock and every ABC
subprocess is a fake."""

from __future__ import annotations

import subprocess

import pytest

from netlist_agent import abc_bridge as abc_bridge_module
from netlist_agent import abc_synth as abc_synth_module
from netlist_agent import deadline
from netlist_agent.abc_bridge import (
    ABCBridgeError,
    ABCInconclusiveError,
    ABCTimeoutError,
    _resolve_abc,
    _run_abc,
    extract_combinational_view,
    is_constant,
)
from netlist_agent.abc_synth import _run_abc_synthesis, _SynthError, optimize_depth
from netlist_agent.ir import NetBit
from netlist_agent.parser import parse_verilog

_BUDGET_MSG = "(the time left in this request's budget)"


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(deadline, "_monotonic", lambda: now["t"])
    return now


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Spawns:
    """Fake `subprocess.run`: records every call's timeout; `action(timeout)` decides the outcome."""

    def __init__(self, monkeypatch, action=None):
        self.timeouts: list = []
        self.action = action or (lambda t: _Result())
        monkeypatch.setattr(subprocess, "run", self)

    def __call__(self, *args, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        return self.action(kwargs.get("timeout"))


def _raise_timeout(t):
    raise subprocess.TimeoutExpired(cmd="abc", timeout=t)


_CHAIN_V = """
module top(a, b, c, d, y);
  input a, b, c, d;
  output y;
  wire n1, n2;
  and g0(n1, a, b);
  and g1(n2, n1, c);
  and g2(y, n2, d);
endmodule
"""

_CONST_V = """
module top(a, y);
  input a;
  output y;
  wire n1;
  and g0(n1, a, 1'b0);
  buf g1(y, n1);
endmodule
"""


def _design(tmp_path, src):
    p = tmp_path / "d.v"
    p.write_text(src)
    return parse_verilog(str(p))


# ----------------------------------------------------------------------
# T1 spawn_timeout
# ----------------------------------------------------------------------


def test_spawn_timeout_no_scope_returns_same_object(clock) -> None:
    """kills: spawn_timeout converting/copying the timeout when no scope is active."""
    seven = 7
    assert deadline.spawn_timeout(seven) is seven
    with deadline.disabled():
        assert deadline.spawn_timeout(seven) is seven


def test_spawn_timeout_clamps_and_boundary(clock) -> None:
    """kills: min() dropped; `<` turned into `<=`; the 0.5s check removed; None branch returning timeout."""
    with deadline.request_scope(10.0):
        assert deadline.spawn_timeout(120.0) == 10.0
    limit = 120.0
    with deadline.request_scope(200.0):
        assert deadline.spawn_timeout(limit) is limit
    with deadline.request_scope(0.5):
        assert deadline.spawn_timeout(120.0) == 0.5
    with deadline.request_scope(0.49):
        assert deadline.spawn_timeout(120.0) is None
    with deadline.request_scope(45.0):
        clock["t"] = 50.0
        assert deadline.spawn_timeout(120.0) is None


# ----------------------------------------------------------------------
# T2 _run_abc (L2)
# ----------------------------------------------------------------------


def test_run_abc_clamps_timeout(monkeypatch, clock) -> None:
    """kills: `timeout=t` replaced by `timeout=timeout` in _run_abc."""
    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    spawns = _Spawns(monkeypatch)
    with deadline.request_scope(3.0):
        _run_abc("cec a b", timeout=120.0)
    assert spawns.timeouts == [3.0]


def test_run_abc_not_started_when_budget_spent(monkeypatch, clock) -> None:
    """kills: the None check removed; the not-started message changed."""
    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    spawns = _Spawns(monkeypatch)
    with deadline.request_scope(0.3):
        with pytest.raises(ABCTimeoutError) as exc:
            _run_abc("cec a b", timeout=120.0)
    assert spawns.timeouts == []
    assert str(exc.value) == (
        "ABC was not started for 'cec': this request had less than 0.5s of its time budget left"
    )


def test_run_abc_recomputes_after_resolve(monkeypatch, clock) -> None:
    """kills: spawn_timeout moved before _resolve_abc() (a cold resolve can eat the budget)."""

    def slow_resolve():
        clock["t"] = 9.7
        return "/fake/abc"

    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", slow_resolve)
    spawns = _Spawns(monkeypatch)
    with deadline.request_scope(10.0):
        with pytest.raises(ABCTimeoutError, match="was not started"):
            _run_abc("cec a b", timeout=120.0)
    assert spawns.timeouts == []


def test_run_abc_timeout_message_clamped_vs_not(monkeypatch, clock) -> None:
    """kills: clamped-branch condition reversed; the unclamped message changed."""
    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    _Spawns(monkeypatch, _raise_timeout)
    with deadline.request_scope(3.0):
        with pytest.raises(ABCTimeoutError) as exc:
            _run_abc("cec a b", timeout=120.0)
    assert str(exc.value) == f"ABC invocation timed out after 3.0s running 'cec' {_BUDGET_MSG}"
    # No scope: the original message, word for word.
    with pytest.raises(ABCTimeoutError) as exc:
        _run_abc("cec a b", timeout=5.0)
    assert str(exc.value) == "ABC invocation timed out after 5.0s running 'cec'"
    # Scope with more time left than the timeout: also unchanged.
    with deadline.request_scope(200.0):
        with pytest.raises(ABCTimeoutError) as exc:
            _run_abc("cec a b", timeout=5.0)
    assert str(exc.value) == "ABC invocation timed out after 5.0s running 'cec'"


# ----------------------------------------------------------------------
# T3 _resolve_abc (L1)
# ----------------------------------------------------------------------


def test_resolve_abc_clamped_and_not_started(monkeypatch, clock) -> None:
    """kills: clamp removed from _resolve_abc; the None check removed; failure caching the path."""
    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    spawns = _Spawns(monkeypatch, lambda t: _Result(stdout="/fake/abc\n"))
    with deadline.request_scope(4.0):
        assert _resolve_abc() == "/fake/abc"
    assert spawns.timeouts == [4.0]

    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    spawns.timeouts.clear()
    with deadline.request_scope(0.2):
        with pytest.raises(ABCTimeoutError) as exc:
            _resolve_abc()
    assert isinstance(exc.value, ABCInconclusiveError)
    assert spawns.timeouts == []
    assert str(exc.value) == (
        "the ABC locator was not run: this request had less than 0.5s of its time budget left"
    )
    assert abc_bridge_module._abc_path is None


def test_resolve_abc_clamped_timeout_is_abc_timeout_and_not_cached(monkeypatch, clock) -> None:
    """kills: ABCTimeoutError downgraded; _abc_path written on failure."""
    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    spawns = _Spawns(monkeypatch, _raise_timeout)
    with deadline.request_scope(3.0):
        with pytest.raises(ABCTimeoutError) as exc:
            _resolve_abc()
    assert "after 3.0s" in str(exc.value) and _BUDGET_MSG in str(exc.value)
    assert abc_bridge_module._abc_path is None

    # Next call, no scope: spawns again with the full limit and caches.
    spawns.action = lambda t: _Result(stdout="/fake/abc\n")
    spawns.timeouts.clear()
    assert _resolve_abc() == "/fake/abc"
    assert spawns.timeouts == [abc_bridge_module._RESOLVE_TIMEOUT]
    assert abc_bridge_module._abc_path == "/fake/abc"


def test_resolve_abc_unclamped_timeout_stays_plain_bridge_error(monkeypatch, clock) -> None:
    """kills: `t <` turned into `<=` (t == limit must stay plain); the full-limit timeout turned into an inconclusive error."""
    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    _Spawns(monkeypatch, _raise_timeout)
    with deadline.request_scope(1000.0):
        with pytest.raises(ABCBridgeError, match="timed out resolving") as exc:
            _resolve_abc()
    assert not isinstance(exc.value, ABCInconclusiveError)


# ----------------------------------------------------------------------
# T4 _run_abc_synthesis (L3)
# ----------------------------------------------------------------------


@pytest.fixture
def temp_dirs(monkeypatch):
    """Counts `tempfile.TemporaryDirectory` constructions."""
    import tempfile

    real = tempfile.TemporaryDirectory
    made: list = []

    def spy(*a, **k):
        made.append(1)
        return real(*a, **k)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", spy)
    return made


def _view(tmp_path):
    return extract_combinational_view(_design(tmp_path, _CHAIN_V), "free_pi")


_SYNTH_NOT_STARTED = (
    "ABC synthesis was not started: this request had less than 0.5s of its time budget left"
)


def test_synthesis_not_started_before_any_work(monkeypatch, clock, temp_dirs, tmp_path) -> None:
    """kills: the first (function-top) check removed -- the temp dir / BLIF write would run."""
    view = _view(tmp_path)
    monkeypatch.setattr(abc_synth_module, "_resolve_abc", lambda: "/fake/abc")
    wrote: list = []
    monkeypatch.setattr(abc_synth_module, "write_blif", lambda *a, **k: wrote.append(1))
    spawns = _Spawns(monkeypatch)
    with deadline.request_scope(0.3):
        with pytest.raises(_SynthError) as exc:
            _run_abc_synthesis(view, None, 120.0)
    assert str(exc.value) == _SYNTH_NOT_STARTED
    assert spawns.timeouts == [] and temp_dirs == [] and wrote == []


def test_synthesis_recomputes_after_resolve(monkeypatch, clock, temp_dirs, tmp_path) -> None:
    """kills: the post-resolve recompute removed."""
    view = _view(tmp_path)

    def slow_resolve():
        clock["t"] = 9.8
        return "/fake/abc"

    monkeypatch.setattr(abc_synth_module, "_resolve_abc", slow_resolve)
    spawns = _Spawns(monkeypatch)
    with deadline.request_scope(10.0):
        with pytest.raises(_SynthError) as exc:
            _run_abc_synthesis(view, None, 120.0)
    assert str(exc.value) == _SYNTH_NOT_STARTED
    assert temp_dirs == [1]  # the first check passed, so the temp dir was made
    assert spawns.timeouts == []


def test_synthesis_clamps_timeout_and_messages(monkeypatch, clock, tmp_path) -> None:
    """kills: `timeout=t` replaced by the original; clamped-branch condition reversed; unclamped message changed."""
    view = _view(tmp_path)
    monkeypatch.setattr(abc_synth_module, "_resolve_abc", lambda: "/fake/abc")
    spawns = _Spawns(monkeypatch, _raise_timeout)
    with deadline.request_scope(3.0):
        with pytest.raises(_SynthError) as exc:
            _run_abc_synthesis(view, None, 120.0)
    assert spawns.timeouts == [3.0]
    assert str(exc.value) == f"ABC synthesis timed out after 3.0s {_BUDGET_MSG}"
    with pytest.raises(_SynthError) as exc:
        _run_abc_synthesis(view, None, 5.0)
    assert str(exc.value) == "ABC synthesis timed out after 5.0s"
    assert spawns.timeouts[-1] == 5.0


# ----------------------------------------------------------------------
# T5 is_constant: the second cec is not started once the first used up the budget
# ----------------------------------------------------------------------


def test_is_constant_second_cec_not_started(monkeypatch, clock, tmp_path) -> None:
    """kills: _run_abc not re-checking the budget per spawn."""
    design = _design(tmp_path, _CONST_V)
    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")

    def first_cec(t):
        clock["t"] = 9.8
        return _Result(stdout="Networks are NOT EQUIVALENT.  Time = 0.00 sec\n")

    spawns = _Spawns(monkeypatch, first_cec)
    with deadline.request_scope(10.0):
        with pytest.raises(ABCTimeoutError, match="was not started"):
            is_constant(design, NetBit("n1", None))
    assert len(spawns.timeouts) == 1


# ----------------------------------------------------------------------
# T7 optimize_depth with the budget already spent
# ----------------------------------------------------------------------


def test_optimize_depth_with_spent_budget_returns_original(monkeypatch, clock, tmp_path) -> None:
    """kills: a spent budget escaping optimize_depth as an exception or a changed design."""
    design = _design(tmp_path, _CHAIN_V)
    monkeypatch.setattr(abc_synth_module, "_resolve_abc", lambda: "/fake/abc")
    spawns = _Spawns(monkeypatch)
    with deadline.request_scope(1.0):
        clock["t"] = 5.0
        result = optimize_depth(design)
    assert result.design is design
    assert result.changed is False
    assert result.failure is not None and "was not started" in result.failure
    assert spawns.timeouts == []
