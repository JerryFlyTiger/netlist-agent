"""Unit tests for netlist_agent/deadline.py and its cli.run wiring (batch 21a).

Must not import `netlist_agent.router`: this file is exported to the public
repo, where the router is a stub. All time is a fake clock."""

from __future__ import annotations

import io
import re

import pytest

import netlist_agent.cli as cli_module
from netlist_agent import deadline
from netlist_agent.cli import Config, GenerationConfig
from netlist_agent.cli import run as cli_run

_RESPONSE_RE = re.compile(r"#RESPONSE (\d+)\n(.*?)\n#END \1\n", re.DOTALL)
_DUMMY_CONFIG = Config(provider="openai", openai=None, anthropic=None, generation=GenerationConfig(0.2, 4096))


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(deadline, "_monotonic", lambda: now["t"])
    return now


def test_no_scope_is_inert(clock) -> None:
    """kills: a default state that is not "no scope" (e.g. a leaked scope from another test)."""
    assert deadline.remaining() is None
    deadline.check()
    deadline.extend(100.0)
    assert deadline.clamp(5.0) == 5.0
    assert deadline.remaining() is None


def test_remaining_and_scope_exit(clock) -> None:
    """kills: remaining() not reading the clock via _monotonic; scope not restored on exit."""
    clock["t"] = 10.0
    with deadline.request_scope(45.0):
        assert deadline.remaining() == 45.0
        clock["t"] = 20.0
        assert deadline.remaining() == 35.0
        clock["t"] = 60.0
        assert deadline.remaining() == -5.0  # not clamped to 0
    assert deadline.remaining() is None


def test_scope_restored_on_exception(clock) -> None:
    """kills: restoring the state outside a finally."""
    with pytest.raises(ValueError):
        with deadline.request_scope(45.0):
            raise ValueError("x")
    assert deadline.remaining() is None


def test_check_boundary(clock) -> None:
    """kills: `< 0` instead of `<= 0` (expiry exactly at the deadline must raise); check never raising;
    wrong budget on the exception."""
    with deadline.request_scope(45.0):
        clock["t"] = 44.999
        deadline.check()
        clock["t"] = 45.0
        with pytest.raises(deadline.DeadlineExceeded) as ei:
            deadline.check()
        assert ei.value.budget == 45.0
        clock["t"] = 46.0
        with pytest.raises(deadline.DeadlineExceeded):
            deadline.check()


def test_deadline_exceeded_escapes_except_exception() -> None:
    """kills: DeadlineExceeded deriving from Exception (cli.py/llm client `except Exception` would eat it)."""
    assert not issubclass(deadline.DeadlineExceeded, Exception)
    with pytest.raises(deadline.DeadlineExceeded):
        try:
            raise deadline.DeadlineExceeded(1.0)
        except Exception:  # noqa: BLE001
            pytest.fail("except Exception caught DeadlineExceeded")


def test_extend_lengthens_never_shortens(clock) -> None:
    """kills: extend() shortening; extend() not updating the budget field."""
    with deadline.request_scope(45.0):
        deadline.extend(10.0)
        assert deadline.remaining() == 45.0
        deadline.extend(270.0)
        assert deadline.remaining() == 270.0
        clock["t"] = 270.0
        with pytest.raises(deadline.DeadlineExceeded) as ei:
            deadline.check()
        assert ei.value.budget == 270.0


def test_extend_counts_from_scope_start(clock) -> None:
    """kills: extend() counting from now instead of from the scope start."""
    with deadline.request_scope(45.0):
        clock["t"] = 30.0
        deadline.extend(270.0)
        assert deadline.remaining() == 240.0


def test_extend_is_capped_by_outer_scope(clock) -> None:
    """kills: inner extend() escaping the outer deadline; budget field not following the cap."""
    with deadline.request_scope(100.0):
        with deadline.request_scope(45.0):
            deadline.extend(270.0)
            assert deadline.remaining() == 100.0
            clock["t"] = 100.0
            with pytest.raises(deadline.DeadlineExceeded) as ei:
                deadline.check()
            assert ei.value.budget == 100.0


def test_nested_scope_takes_min_and_restores_outer(clock) -> None:
    """kills: inner scope replacing instead of min-ing; outer not restored after leaving the inner."""
    with deadline.request_scope(45.0):
        clock["t"] = 10.0
        with deadline.request_scope(300.0):
            assert deadline.remaining() == 35.0
        assert deadline.remaining() == 35.0
        with deadline.request_scope(5.0):
            assert deadline.remaining() == 5.0
        assert deadline.remaining() == 35.0
        deadline.extend(270.0)  # the outer one is still extendable
        assert deadline.remaining() == 260.0


def test_disabled_behaviour(clock) -> None:
    """kills: any of the four functions ignoring disabled(); scope opened inside disabled() taking effect."""
    with deadline.request_scope(1.0):
        clock["t"] = 5.0  # long expired
        with deadline.disabled():
            assert deadline.remaining() is None
            deadline.check()
            deadline.extend(10.0)
            assert deadline.clamp(7.0) == 7.0
            with deadline.request_scope(1.0):
                assert deadline.remaining() is None
                deadline.check()
        assert deadline.remaining() == -4.0
        with pytest.raises(deadline.DeadlineExceeded):
            deadline.check()


def test_disabled_restored_on_exception(clock) -> None:
    """kills: disabled() restoring outside a finally."""
    with pytest.raises(ValueError):
        with deadline.disabled():
            raise ValueError("x")
    with deadline.request_scope(1.0):
        assert deadline.remaining() == 1.0


def test_clamp(clock) -> None:
    """kills: clamp() not taking the min; negative remaining leaking out."""
    with deadline.request_scope(10.0):
        clock["t"] = 7.0
        assert deadline.clamp(5.0) == 3.0
        assert deadline.clamp(1.0) == 1.0
        clock["t"] = 12.0
        assert deadline.clamp(5.0) == 0.0


@pytest.mark.parametrize("budget,text", [(45.0, "45"), (270.0, "270")])
def test_incomplete_message_wording(budget, text) -> None:
    """kills: wording drift (verbatim string pinned)."""
    msg = deadline.incomplete_message(deadline.DeadlineExceeded(budget))
    assert msg == f"Incomplete: this request hit its {text}-second time budget before finishing, so it has no answer."


def _run_lines(monkeypatch, tmp_path, fake, lines):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "handle_request", fake)
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=lambda s, t: "FB")
    session.close()
    return stdout.getvalue()


_BEGIN = "This is the beginning of a new testcase. The case name is dlcase."


def test_cli_opens_a_fresh_scope_per_line(monkeypatch, tmp_path, clock) -> None:
    """kills: one scope shared across lines (second line would see an expired deadline)."""
    seen: list = []

    def fake(session, text, fallback):
        seen.append(deadline.remaining())
        clock["t"] += 100.0
        return "ok"

    out = _run_lines(monkeypatch, tmp_path, fake, [_BEGIN, "a", "b"])
    assert seen == [45.0, 45.0]
    assert len(_RESPONSE_RE.findall(out)) == 3


def test_cli_turns_deadline_exceeded_into_incomplete_and_carries_on(monkeypatch, tmp_path, clock) -> None:
    """kills: cli not catching DeadlineExceeded (BaseException escapes `except Exception`); losing #END or the next line."""

    def fake(session, text, fallback):
        if text == "slow":
            raise deadline.DeadlineExceeded(45.0)
        return "fine"

    out = _run_lines(monkeypatch, tmp_path, fake, [_BEGIN, "slow", "next"])
    responses = _RESPONSE_RE.findall(out)
    assert [int(r[0]) for r in responses] == [1, 2, 3]
    assert responses[1][1] == deadline.incomplete_message(deadline.DeadlineExceeded(45.0))
    assert responses[2][1] == "fine"


def test_cli_respond_runs_outside_the_scope(monkeypatch, tmp_path, clock) -> None:
    """kills: respond() inside the scope (an expiring deadline could then cut off the log/#END)."""
    seen: list = []
    real_respond = cli_module.respond

    def spy(session, body, stdout=None):
        seen.append(deadline.remaining())
        return real_respond(session, body, stdout=stdout)

    monkeypatch.setattr(cli_module, "respond", spy)
    _run_lines(monkeypatch, tmp_path, lambda s, t, f: "ok", [_BEGIN, "a"])
    assert seen == [None, None]


def test_extend_counts_from_scope_start_with_large_clock(clock) -> None:
    """kills: A10/A17/A18 (extend() measuring from 0 or from "now" instead of the scope start).
    With a scope that starts at clock 0, "from 0" and "from the scope start" coincide, so A17/A18
    need a large absolute clock. The clock moves between opening the scope and the extend so that
    "from now" lands somewhere else too (A10 is also killed by `test_extend_counts_from_scope_start`)."""
    clock["t"] = 1000.0
    with deadline.request_scope(45.0):
        clock["t"] = 1030.0
        deadline.extend(270.0)
        assert deadline.remaining() == 240.0
        clock["t"] = 1270.0
        with pytest.raises(deadline.DeadlineExceeded) as ei:
            deadline.check()
        assert ei.value.budget == 270.0


def test_nested_extend_with_large_clock(clock) -> None:
    """kills: A17/A19 in the nested case (extend inside an inner scope, outer cap not binding).
    A18 only touches the outermost scope's start, which a nested extend never reads."""
    clock["t"] = 1000.0
    with deadline.request_scope(300.0):
        clock["t"] = 1010.0
        with deadline.request_scope(45.0):
            deadline.extend(100.0)
            assert deadline.remaining() == 100.0


def test_capped_extend_budget_field_with_large_clock(clock) -> None:
    """kills: A20 (budget field of a capped extend not reflecting the outer cap), also A17/A19."""
    clock["t"] = 1000.0
    with deadline.request_scope(100.0):
        clock["t"] = 1020.0
        with deadline.request_scope(45.0):
            deadline.extend(270.0)
            clock["t"] = 1100.0
            with pytest.raises(deadline.DeadlineExceeded) as ei:
                deadline.check()
            assert ei.value.budget == 80.0


def test_nested_budget_field_when_outer_is_tighter(clock) -> None:
    """kills: A21 (including the `else deadline` variant): the inner scope reporting its own
    budget instead of the effective time left when the tighter outer deadline fires."""
    with deadline.request_scope(45.0):
        clock["t"] = 10.0
        with deadline.request_scope(300.0):
            clock["t"] = 45.0
            with pytest.raises(deadline.DeadlineExceeded) as ei:
                deadline.check()
            assert ei.value.budget == 35.0


def test_t1_current_budget(clock) -> None:
    """SPEC_21c T1. kills: current_budget() returning a constant, ignoring extend(), or not honouring disabled()."""
    assert deadline.current_budget() is None
    with deadline.request_scope(45.0):
        assert deadline.current_budget() == 45.0
        with deadline.disabled():
            assert deadline.current_budget() is None
        assert deadline.current_budget() == 45.0
        deadline.extend(270.0)
        assert deadline.current_budget() == 270.0
    assert deadline.current_budget() is None
