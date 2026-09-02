"""Stub for the private rule-based router.

This is NOT the real router. The private repo's `netlist_agent/router.py` is
a 4,000+ line, ~95-handler regex dispatcher (`route()` + `handle_request()`)
that answers most of the contest's request phrasings deterministically,
without an LLM call. It is not included in this public export -- see the
project README for why.

What IS here is copied verbatim from the private router: the four symbols
`netlist_agent/cli.py` imports (`BEGIN_RE`, `Fallback`, `_extract_log_filename`,
`handle_request`), unchanged in behaviour/signature, plus four small,
non-routing utility functions (`_BASIS_MAP`, `_normalize_op_args`,
`_rename_gate_instance`, `_resolve_write_path`) that
`netlist_agent/llm/tools_schema.py` also imports from the router module --
these are generic helpers (argument normalization, gate renaming, output
path resolution), not part of the 95-handler dispatch table, so copying them
here does not reintroduce any of the router's own request-matching logic.

`handle_request` here does no regex matching at all: every request goes
straight to `fallback` (the LLM-backed path `netlist_agent/cli.py` wires up
via `build_llm_fallback`). If `fallback` is None, it returns a message
explaining that this public build has no rule-based router and needs an LLM
configured.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Optional

from netlist_agent.ir import Design, GateType, NetBit
from netlist_agent.netref import netbit_token as _netbit_token
from netlist_agent.session import Session

Fallback = Callable[[Session, str], str]

BEGIN_RE = re.compile(
    r"this is the beginning of (?:a new testcase\.\s*the case name is|testcase)\s*"
    r"['\"]?(\w+)['\"]?\.?",
    re.IGNORECASE,
)

_LOG_FILENAME_RE = re.compile(r"\binto\s+([\w.\-]+\.log)\b", re.IGNORECASE)


def _extract_log_filename(text: str) -> Optional[str]:
    """Pull an explicit log filename out of a "...into <name>.log" clause,
    if the begin-testcase line names one; otherwise None (caller falls back
    to "<case_name>.log")."""
    m = _LOG_FILENAME_RE.search(text)
    return m.group(1) if m else None

def _normalize_op_args(*args: object) -> tuple:
    """Turn a transform's positional (non-`design`) arguments into a stable,
    order-independent tuple that two calls with the "same" arguments always
    compare equal on -- used to tell an actual rerun of one recorded
    operation (`Session.last_op_kind`/`last_op_args`) apart from a
    different call that merely shares the same transform function (see
    `Session.last_op_args`'s docstring).

    Deliberately NOT `repr(args)`: a `set`/`list` argument's iteration order
    is not guaranteed stable across two otherwise-identical calls (a caller
    building a gate-name scope or sink list from a `set`/dict, as the
    private router's handlers do, gets a different iteration order each
    time), so comparing raw `repr()` text would call
    two equal-but-differently-ordered calls unequal -- a false NEGATIVE,
    which is the safe direction (see below), but pointlessly so, since a
    real normalization is no harder here. Each element is normalized
    in turn:

      * `NetBit` -> its "name"/"name[bit]" token (`netbit_token`) -- two
        `NetBit`s naming the same net-bit are the same argument even if
        they're different object instances.
      * `GateType` -> its `.name` -- likewise, comparable across instances.
      * `list`/`set`/`frozenset`/`tuple` -> a tuple of its elements,
        normalized the same way, THEN sorted -- order-independent.
      * anything else (`str`, `int`, `bool`, `None`, ...) -> passed through
        unchanged; already a stable, comparable value.

    An argument type this function doesn't know how to normalize sorts and
    compares by nothing more than its own `==` -- if that ever produces a
    wrong "equal", the caller only gains a spurious rerun disclosure, never
    a wrong count. The mechanism this function feeds is deliberately biased
    that direction throughout: prefer a false negative (missing a rerun
    disclosure) over a false positive (falsely claiming one), so an
    unrecognized type here is not a correctness bug, only a missed
    disclosure -- extend the branches below if one shows up.
    """
    return tuple(_normalize_op_arg(a) for a in args)


def _normalize_op_arg(value: object) -> object:
    if isinstance(value, NetBit):
        return ("netbit", _netbit_token(value))
    if isinstance(value, GateType):
        return ("gatetype", value.name)
    if isinstance(value, (list, set, frozenset, tuple)):
        return tuple(sorted(_normalize_op_arg(v) for v in value))
    return value

def _rename_gate_instance(design: Design, old_name: str, new_name: str) -> None:
    """Rename a gate INSTANCE (not a net) -- ir.py only exposes
    `Design.rename_signal` for nets, so this is implemented locally.
    Mutating `Gate.inst_name` in place doesn't change `design.gates`'
    length, so the lazily-built `_gate_index` cache (keyed by the OLD name)
    would otherwise go stale without ever being rebuilt (its only staleness
    signal is a length mismatch) -- clearing it forces a rebuild on next use.

    The duplicate-name guard below lives here rather than in the router
    handler because this function has two callers with independent handler
    layers -- `_h_rename_gate` in this module and the LLM tool registry
    (`llm/tools_schema.py`) -- and both had the same measured bug: renaming
    a gate to an inst_name that already exists silently produced two gates
    sharing one `inst_name` (confirmed via `Counter` showing a count of 2),
    corrupting the design without any error surfacing. Guarding here fixes
    both call sites at once, mirroring `Design.rename_signal`'s own
    same-shaped guard for signals (ir.py).
    """
    gate = next((g for g in design.gates if g.inst_name == old_name), None)
    if gate is None:
        raise KeyError(f"no such gate: {old_name!r}")
    if new_name != old_name and any(g.inst_name == new_name for g in design.gates):
        raise ValueError(f"gate name already in use: {new_name!r}")
    gate.inst_name = new_name
    design._gate_index = {}

_BASIS_MAP = {"AND": "and_not", "NAND": "nand_not", "NOR": "nor_not"}


def _resolve_write_path(session: Session, filename: str) -> str:
    if os.path.dirname(filename):
        return filename
    if session.load_dir:
        return os.path.join(session.load_dir, filename)
    return filename

_NO_ROUTER_MESSAGE = (
    "This public build does not include the private rule-based router; "
    "every request is answered by the LLM fallback. No LLM fallback is "
    "configured for this run (no API key / client set up), so this request "
    "could not be answered. See the README for how to configure an LLM "
    "provider."
)


def handle_request(session: Session, text: str, fallback: Fallback) -> str:
    """No rule matching in this build -- always defers to `fallback`."""
    stripped = text.strip()
    if fallback is None:
        return _NO_ROUTER_MESSAGE
    return fallback(session, stripped)
