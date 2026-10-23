"""Per-request wall-clock deadline (core, plus the ABC leaf clamp helpers).

`cli.run` opens one `request_scope` per request line; code that can run long
calls `check()` / `clamp()` against it. Since 21b the ABC subprocess leaves
(`abc_bridge._resolve_abc` / `_run_abc`, `abc_synth._run_abc_synthesis`) clamp their
timeouts with `spawn_timeout()`. Since 21d the read-only analysis loops call `check()` too;
the exact list of checkpoint sites is pinned by the checkpoint whitelist tests (private repo).

No signal, thread or subprocess is used: the deadline is cooperative. State
lives in one immutable object in a single `ContextVar`, so leaving a scope
(normally or by exception) restores exactly what was there before.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import time
from dataclasses import dataclass
from typing import Iterator, Optional

# Analysis questions get 60s in QA A89; 15s are left for the constraint check,
# the log and `#END`, same split as ENABLE_HOLD_TIME_BUDGET.
ANALYSIS_BUDGET = 45.0
# Transformation requests get 300s; 30s are left for the same tail work.
TRANSFORM_BUDGET = 270.0

# Indirection so tests can monkeypatch the clock.
_monotonic = time.monotonic


class DeadlineExceeded(BaseException):
    """The request ran out of its time budget.

    Derives from BaseException on purpose: cli.py and llm/client.py both have
    `except Exception` guards that would otherwise swallow it and turn it into
    an "Internal error" or a tool-result string."""

    def __init__(self, budget: float) -> None:
        super().__init__(budget)
        self.budget = budget


@dataclass(frozen=True)
class _State:
    start: float
    deadline: float
    budget: float
    cap: float


# Sentinel: deadlines switched off (see `disabled()`).
_DISABLED = _State(0.0, math.inf, math.inf, math.inf)

_state: contextvars.ContextVar[Optional[_State]] = contextvars.ContextVar("netlist_agent_deadline", default=None)


def _active() -> Optional[_State]:
    st = _state.get()
    if st is None or st is _DISABLED:
        return None
    return st


@contextlib.contextmanager
def request_scope(budget: float) -> Iterator[None]:
    outer = _state.get()
    if outer is _DISABLED:
        yield
        return
    now = _monotonic()
    if outer is None:
        st = _State(now, now + budget, budget, math.inf)
    else:
        deadline = min(outer.deadline, now + budget)
        eff = budget if deadline == now + budget else deadline - now
        st = _State(now, deadline, eff, outer.deadline)
    token = _state.set(st)
    try:
        yield
    finally:
        _state.reset(token)


def extend(budget: float) -> None:
    """Lengthen the current scope to `budget` seconds from its start. Never shortens;
    never goes past the enclosing scope's deadline."""
    st = _active()
    if st is None:
        return
    wanted = st.start + budget
    if wanted <= st.deadline:
        return
    if wanted <= st.cap:
        new = _State(st.start, wanted, budget, st.cap)
    else:
        new = _State(st.start, st.cap, st.cap - st.start, st.cap)
    _state.set(new)


def remaining() -> Optional[float]:
    st = _active()
    if st is None:
        return None
    return st.deadline - _monotonic()


def current_budget() -> Optional[float]:
    """The active scope's budget in seconds (after any extend()), or None with no scope or inside disabled()."""
    st = _active()
    if st is None:
        return None
    return st.budget


def check() -> None:
    st = _active()
    if st is not None and st.deadline - _monotonic() <= 0:
        raise DeadlineExceeded(st.budget)


def clamp(timeout: float) -> float:
    left = remaining()
    if left is None:
        return timeout
    return min(timeout, max(0.0, left))


# An ABC subprocess is not started with less than this many seconds left
# (the same 0.5s as 20a's enable_hold.ABC_MIN_REMAINING).
MIN_SPAWN_REMAINING = 0.5


def spawn_timeout(timeout: float) -> Optional[float]:
    """The timeout to give a subprocess whose own limit is `timeout`, or None when
    the request has less than MIN_SPAWN_REMAINING seconds left (do not start it).
    With no active scope (or under `disabled()`) returns `timeout` itself."""
    left = remaining()
    if left is None:
        return timeout
    if left < MIN_SPAWN_REMAINING:
        return None
    return min(timeout, left)


@contextlib.contextmanager
def disabled() -> Iterator[None]:
    token = _state.set(_DISABLED)
    try:
        yield
    finally:
        _state.reset(token)


def incomplete_message(exc: DeadlineExceeded) -> str:
    return f"Incomplete: this request hit its {exc.budget:g}-second time budget before finishing, so it has no answer."
