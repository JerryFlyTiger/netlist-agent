"""Tool registry for the LLM fallback: one entry per netlist capability the
rule-based router (router.py) already exposes via its ~90 regex patterns.

Each entry pairs a JSON-schema tool spec (name/description/parameters, the
shape both the OpenAI and Anthropic tool-calling APIs want) with a plain
Python callable of the form `(session, **kwargs) -> JSON-serializable
result`. Net/gate references arrive from the LLM as plain strings (e.g.
"n6[3]", "g0") -- `netlist_agent.netref` (shared with router.py) does the
bit-select parsing.

router.py's own handlers are tightly coupled to `re.Match` objects (they read
capture groups positionally), so they are not reusable as tool callables
directly. The functions below instead call the same underlying
analysis.py/graph.py/transform.py/abc_bridge.py primitives router.py's
handlers call -- except for a handful of genuinely reusable, match-object-free
helpers already factored out in router.py (`_resolve_write_path`,
`_rename_gate_instance`, `_BASIS_MAP`), which are imported and reused as-is.

Every callable raises `ToolError` (or lets a `KeyError`/`ValueError` with an
already-clear message propagate) on a bad reference -- an unknown gate/signal
name, an out-of-range bit, etc. `client.py` catches these and feeds the
message back to the model as a tool-result error rather than crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from netlist_agent.abc_bridge import are_equivalent, check_symmetry, is_constant, verify_equivalence
from netlist_agent.abc_synth import (
    BASIS_GATE_NAMES,
    optimize_cone_depth,
    optimize_cone_gate_count,
    optimize_depth,
    optimize_gate_count,
)
from netlist_agent.analysis import (
    dffs_on_clock,
    direct_fanout,
    fanin_cone_size,
    fanout_count,
    find_floating_signals,
    gate_count_by_type,
    gates_by_name_substring,
    gates_of_type,
    is_cut_signal_bits,
    largest_fanin_cone,
    list_primary_inputs,
    list_primary_outputs,
    max_fanout_among,
    max_fanout_pi,
    po_cone_sizes,
    primary_input_bit_count,
    primary_input_port_count,
    primary_output_bit_count,
    primary_output_port_count,
)
from netlist_agent.boolean_function import derive_boolean_function
from netlist_agent.graph import DffPin, NetlistGraph
from netlist_agent.ir import Const, DFF_PIN_ORDER, Design, Gate, GateType, NetBit, OUTPUT_PIN, Pin, POSITIONAL_PIN_ORDER
from netlist_agent.netref import (
    netbit_sort_key,
    netbit_token,
    resolve_bit as _resolve_bit_raw,
    resolve_bits as _resolve_bits_raw,
    signal_name_only,
)
from netlist_agent.parser import parse_verilog
from netlist_agent.property_check import check_asserted_only_when
from netlist_agent.router import _BASIS_MAP, _normalize_op_args, _rename_gate_instance, _resolve_write_path
from netlist_agent.session import Session
from netlist_agent.signal_pair_search import SUPPORTED_OPS, find_pair_for_op
from netlist_agent.transform import (
    balance_depth_to_sinks,
    collapse_double_inverters,
    collapse_inverter_buffer_chains,
    deduplicate_gates,
    insert_buffer_per_load,
    limit_fanout,
    limit_fanout_net,
    remap_to_basis,
    remove_dangling_gates,
    replace_buf_with_and,
    simplify_constant_inputs,
)
from netlist_agent.writer import write_verilog

_LIST_LIMIT = 200
_GATE_TYPE_VALUES = [gt.value for gt in GateType]
_BASIS_VALUES = ["and", "nand", "nor"]
_DEPTH_BASIS_VALUES = sorted(BASIS_GATE_NAMES)


class ToolError(Exception):
    """Raised by a tool callable on bad/unresolvable input (unknown gate or
    signal name, no design loaded, etc.) -- caught by client.py and turned
    into a tool-result error string for the model to see and react to,
    rather than propagating as a crash."""


# ----------------------------------------------------------------------
# Small shared helpers
# ----------------------------------------------------------------------


def _design(session: Session) -> Design:
    if session.current_design is None:
        raise ToolError("No design is currently loaded. Call load_design first.")
    return session.current_design


def _graph(session: Session) -> NetlistGraph:
    return NetlistGraph(_design(session))


def _resolve_bit(design: Design, token: str) -> NetBit:
    """Resolve `token` to exactly one net-bit on `design`, for tool
    callables whose operation is semantically about a single net-bit --
    raises `ToolError` (not a bare `NetRefError`/`ValueError`) so the
    message reaches the model as a tool-result error rather than a crash.
    Catches plain `ValueError` too, not just `NetRefError`: `resolve_bit`
    itself first calls `parse_net`, which can raise a bare `ValueError` on a
    token that isn't even syntactically "name"/"name[bit]" shaped."""
    try:
        return _resolve_bit_raw(design, token)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def _resolve_bits(design: Design, token: str) -> list[NetBit]:
    """Resolve `token` to the list of net-bits it refers to (a bit-selected
    token -> that one bit; a bare signal name -> every bit of that signal),
    for tool callables whose operation is semantically about a whole
    signal. Raises `ToolError` on an unresolvable OR unparseable reference
    (see `_resolve_bit`'s docstring on why this catches plain `ValueError`,
    not just `NetRefError`)."""
    try:
        return _resolve_bits_raw(design, token)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def _require_gate(design: Design, name: str) -> Gate:
    for gate in design.gates:
        if gate.inst_name == name:
            return gate
    raise ToolError(f"no such gate: {name!r}")


def _cap(items: list[str]) -> dict[str, Any]:
    return {"count": len(items), "items": items[:_LIST_LIMIT], "truncated": len(items) > _LIST_LIMIT}


def _pin_json(v: Pin) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, Const):
        return "1'b0" if v == Const.ZERO else "1'b1"
    return netbit_token(v)


def _gate_json(gate: Gate) -> dict[str, Any]:
    order = DFF_PIN_ORDER if gate.gate_type == GateType.DFF else POSITIONAL_PIN_ORDER[gate.gate_type]
    return {
        "name": gate.inst_name,
        "type": gate.gate_type.value,
        "pins": {pin: _pin_json(gate.pins.get(pin)) for pin in order},
    }


def _gate_type(token: str) -> GateType:
    try:
        return GateType(token.lower())
    except ValueError as exc:
        raise ToolError(f"unknown gate type {token!r}; choose one of {_GATE_TYPE_VALUES}") from exc


_BASIS_LONG_FORM_ALIASES = {"AND_NOT": "AND", "NAND_NOT": "NAND", "NOR_NOT": "NOR"}


def _basis(token: str) -> str:
    key = token.strip().upper()
    key = _BASIS_LONG_FORM_ALIASES.get(key, key)
    if key not in _BASIS_MAP:
        raise ToolError(
            f"unknown basis {token!r}; choose one of {_BASIS_VALUES} "
            f"(the long forms {sorted(_BASIS_LONG_FORM_ALIASES)} are accepted as aliases; note that the "
            f"do_optimize_* tools use those long forms instead, and 'and_or_not' exists only there)"
        )
    return _BASIS_MAP[key]


def _depth_opt_basis(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    key = token.strip().lower()
    if key not in BASIS_GATE_NAMES:
        raise ToolError(f"unknown basis {token!r}; choose one of {_DEPTH_BASIS_VALUES}")
    return key


# ----------------------------------------------------------------------
# Load / write
# ----------------------------------------------------------------------


def load_design(session: Session, filename: str, directory: str) -> dict[str, Any]:
    path = os.path.join(directory, filename)
    session.current_design = parse_verilog(path)
    session.original_snapshot = parse_verilog(path)
    session.load_dir = directory
    session.load_filename = filename
    # Same reset as router._h_load, and for the same reason -- see its
    # comment. Without this, a design loaded via this tool inherits
    # whatever "last operation" counters (and, now, rerun-disclosure
    # fields) the PREVIOUS design left behind.
    session.last_op_count = None
    session.last_op_kind = None
    session.last_op_args = None
    session.last_gate_delta = {}
    return {"module_name": session.current_design.module_name, "path": path}


def set_testcase(session: Session, case_name: str, log_filename: Optional[str] = None) -> dict[str, Any]:
    """Record the testcase's case name (and open its log file) if this
    hasn't already happened -- the same "first one wins" guard router.py's
    own begin-testcase handler uses (`_h_begin`), so calling this after the
    case name was already recognized some other way is a harmless no-op.
    Exists as an LLM-callable escape hatch for testcase-begin phrasings the
    rule-based router's `BEGIN_RE` doesn't recognize at all -- any responses
    already emitted this testcase (buffered in `session.pending_log`) are
    flushed to the log as soon as it opens.

    `log_filename`, if given, must be a bare filename (no directory
    component) -- it is handed straight to `open()` by `Session.start()`,
    so anything else would let a hostile/confused model write outside the
    working directory. Rejected with an error dict (not an exception, so
    the model gets a clear, retryable signal) rather than silently
    accepted or silently dropped."""
    if session.case_name is not None:
        return {"case_name": session.case_name, "log_path": session.log_path}
    if log_filename is not None:
        if (
            not log_filename
            or log_filename in (".", "..")
            or "/" in log_filename
            or (os.sep != "/" and os.sep in log_filename)
            or (os.altsep is not None and os.altsep in log_filename)
        ):
            return {
                "error": (
                    f"invalid log_filename {log_filename!r}: must be a bare filename with no directory "
                    "component (no '/', no '.', no '..')"
                )
            }
    session.start(case_name, log_filename)
    return {"case_name": session.case_name, "log_path": session.log_path}


def write_design(session: Session, filename: str) -> dict[str, Any]:
    design = _design(session)
    path = _resolve_write_path(session, filename)
    write_verilog(design, path)
    return {"path": path}


# ----------------------------------------------------------------------
# Counting / listing
# ----------------------------------------------------------------------


def count_gates_by_type(session: Session) -> dict[str, Any]:
    counts = gate_count_by_type(_design(session))
    by_type = {gt.name: n for gt, n in counts.items()}
    return {"total": sum(counts.values()), "by_type": by_type}


def count_primary_ports(session: Session) -> dict[str, Any]:
    design = _design(session)
    return {
        "primary_input_ports": primary_input_port_count(design),
        "primary_output_ports": primary_output_port_count(design),
        "primary_input_bits": primary_input_bit_count(design),
        "primary_output_bits": primary_output_bit_count(design),
    }


def list_primary_ports(session: Session, direction: str) -> dict[str, Any]:
    design = _design(session)
    if direction not in ("input", "output"):
        raise ToolError("direction must be 'input' or 'output'")
    infos = list_primary_inputs(design) if direction == "input" else list_primary_outputs(design)
    return _cap([f"{p.name} ({p.width} bit{'s' if p.width != 1 else ''})" for p in infos])


def list_gates_of_type(session: Session, gate_type: str) -> dict[str, Any]:
    design = _design(session)
    gt = _gate_type(gate_type)
    gates = [_gate_json(g) for g in gates_of_type(design, gt)]
    return {"count": len(gates), "gates": gates[:_LIST_LIMIT], "truncated": len(gates) > _LIST_LIMIT}


def find_gates_by_name(session: Session, substring: str, gate_type: Optional[str] = None) -> dict[str, Any]:
    """Find gates whose instance name contains `substring` as a literal
    substring (not a regex/glob), optionally restricted to one gate type.
    Also records the match as the "last found gates" set, so a later
    do_replace_buf_with_and call may omit gate_names to reuse it."""
    design = _design(session)
    gt = _gate_type(gate_type) if gate_type is not None else None
    names = sorted(g.inst_name for g in gates_by_name_substring(design, substring, gt))
    session.last_query_gate_names = names
    return _cap(names)


def get_gate_info(session: Session, gate: str) -> dict[str, Any]:
    design = _design(session)
    return _gate_json(_require_gate(design, gate))


def list_dffs_on_clock(session: Session, clock: str) -> dict[str, Any]:
    design = _design(session)
    # `dffs_on_clock` matches by net NAME only (ignoring bit index), so a
    # bit-selected `clock` (e.g. "n0[0]") is resolved purely for validation
    # (unknown-signal / out-of-range errors) and then the bare name is used
    # for the actual lookup -- previously an unknown or wrongly-bit-selected
    # clock silently returned an empty list (0 DFFs) with no indication
    # anything was wrong.
    _resolve_bits(design, clock)
    clock_name = signal_name_only(clock)
    names = sorted(g.inst_name for g in dffs_on_clock(design, clock_name))
    return _cap(names)


def check_dffs_same_clock_domain(session: Session, dff_names: list[str]) -> dict[str, Any]:
    """Compare the clock (CK) nets of two or more named DFF instances --
    the reverse direction of list_dffs_on_clock. Raises ToolError if any
    name doesn't resolve to a gate, isn't a DFF, or has an unconnected CK
    pin, rather than silently guessing."""
    design = _design(session)
    if len(dff_names) < 2:
        raise ToolError("need at least two DFF instance names to compare clock domains")
    by_name = {g.inst_name: g for g in design.gates}
    clocks: dict[str, str] = {}
    for name in dff_names:
        gate = by_name.get(name)
        if gate is None:
            raise ToolError(f"no such gate: {name!r}")
        if gate.gate_type != GateType.DFF:
            raise ToolError(f"{name} is not a DFF (it is a {gate.gate_type.value.upper()})")
        ck = gate.pins.get("CK")
        if not isinstance(ck, NetBit):
            raise ToolError(f"{name}'s clock (CK) pin is not connected to any net")
        clocks[name] = ck.name
    return {"same_domain": len(set(clocks.values())) == 1, "clocks": clocks}


def list_gates_with_constant_input(
    session: Session, gate_type: Optional[str] = None, value: Optional[int] = None
) -> dict[str, Any]:
    """List 2-input gates with at least one input tied to a Boolean constant.
    `gate_type` restricts to one gate type (e.g. "nand"); `value` (0 or 1)
    restricts to that specific constant. Both default to unrestricted."""
    design = _design(session)
    if value is not None and value not in (0, 1):
        raise ToolError("value must be 0 or 1")
    gates = gates_of_type(design, _gate_type(gate_type)) if gate_type is not None else design.gates
    want = None if value is None else (Const.ZERO if value == 0 else Const.ONE)

    def _has_const(g: Gate) -> bool:
        i0, i1 = g.pins.get("I0"), g.pins.get("I1")
        if want is None:
            return isinstance(i0, Const) or isinstance(i1, Const)
        return i0 == want or i1 == want

    names = sorted(g.inst_name for g in gates if _has_const(g))
    return _cap(names)


def get_direct_pi_po_connections(session: Session) -> dict[str, Any]:
    """Net-bits wired straight from a primary input to a primary output with
    zero intervening gates (length-0 paths)."""
    graph = _graph(session)
    zero = [nb for nb in graph.po_bits if nb in graph.pi_bits]
    return _cap([netbit_token(nb) for nb in sorted(zero, key=netbit_sort_key)])


# ----------------------------------------------------------------------
# Fanin / fanout
# ----------------------------------------------------------------------


def get_gate_direct_fanout(session: Session, gate: str) -> dict[str, Any]:
    graph = _graph(session)
    g = _require_gate(graph.design, gate)
    out_nb = g.pins.get(OUTPUT_PIN[g.gate_type])
    if not isinstance(out_nb, NetBit):
        return {"count": 0, "gates": [], "truncated": False, "drives_primary_output": False}
    loads = direct_fanout(graph, out_nb)
    names = sorted({l.gate.inst_name for l in loads if l.kind == "gate" and l.gate is not None})
    result = _cap(names)
    result["gates"] = result.pop("items")
    result["drives_primary_output"] = any(l.kind == "po" for l in loads)
    return result


def get_gate_fanout_count(session: Session, gate: str) -> dict[str, Any]:
    graph = _graph(session)
    g = _require_gate(graph.design, gate)
    out_nb = g.pins.get(OUTPUT_PIN[g.gate_type])
    if not isinstance(out_nb, NetBit):
        return {"count": 0}
    loads = direct_fanout(graph, out_nb)
    return {"count": len({l.gate.inst_name for l in loads if l.kind == "gate" and l.gate is not None})}


def get_net_fanout(session: Session, net: str) -> dict[str, Any]:
    graph = _graph(session)
    nb = _resolve_bit(graph.design, net)
    count = fanout_count(graph, nb)
    loads = direct_fanout(graph, nb)
    names = sorted({l.gate.inst_name for l in loads if l.kind == "gate" and l.gate is not None})
    return {"count": count, "gates": names}


def get_max_fanout_of_signal(session: Session, signal: str) -> dict[str, Any]:
    design = _design(session)
    graph = _graph(session)
    # Bare signal name -> maximum fanout over every bit of the signal
    # (unchanged whole-bus semantics); a bit-selected token narrows the
    # search down to that one bit only.
    netbits = _resolve_bits(design, signal)
    nb, count = max_fanout_among(graph, netbits)
    return {"net": None if nb is None else netbit_token(nb), "count": count}


def get_max_fanout_primary_input(session: Session) -> dict[str, Any]:
    graph = _graph(session)
    nb, count = max_fanout_pi(graph)
    return {"net": None if nb is None else netbit_token(nb), "count": count}


def get_gates_connected_to_signal(session: Session, signal: str) -> dict[str, Any]:
    design = _design(session)
    graph = _graph(session)
    netbits = _resolve_bits(design, signal)
    names: set[str] = set()
    for nb in netbits:
        loads = direct_fanout(graph, nb)
        names |= {l.gate.inst_name for l in loads if l.kind == "gate" and l.gate is not None}
    return _cap(sorted(names))


# ----------------------------------------------------------------------
# Depth
# ----------------------------------------------------------------------


def get_depth_of_cone(session: Session, net: str) -> dict[str, Any]:
    graph = _graph(session)
    return {"depth": graph.depth_to_sink(_resolve_bit(graph.design, net))}


def get_depth_between(session: Session, source: str, target: str) -> dict[str, Any]:
    graph = _graph(session)
    result = graph.depth_between(_resolve_bit(graph.design, source), _resolve_bit(graph.design, target))
    if result is None:
        return {"depth": None, "path": None}
    depth, path = result
    return {"depth": depth, "path": [g.inst_name for g in path]}


def get_max_design_depth(session: Session) -> dict[str, Any]:
    return {"depth": _graph(session).max_design_depth()}


def get_max_reg_to_reg_depth(session: Session) -> dict[str, Any]:
    return {"depth": _graph(session).max_reg_to_reg_depth()}


def get_max_pi_to_dff_d_depth(session: Session) -> dict[str, Any]:
    return {"depth": _graph(session).max_pi_to_dff_d_depth()}


def check_gate_on_max_depth_path(session: Session, gate: str) -> dict[str, Any]:
    """Whether a gate lies on any maximum-depth path of the whole design --
    thin wrapper over `graph.depth_through_gate`/`graph.max_design_depth`,
    the same two primitives `router._h_gate_on_max_depth_path` calls."""
    graph = _graph(session)
    g = _require_gate(graph.design, gate)
    if g.gate_type == GateType.DFF:
        return {"on_max_depth_path": False, "depth_through_gate": None, "max_design_depth": graph.max_design_depth(), "note": "gate is a DFF (sequential boundary)"}
    depth = graph.depth_through_gate(gate)
    max_depth = graph.max_design_depth()
    return {
        "on_max_depth_path": depth is not None and depth == max_depth,
        "depth_through_gate": depth,
        "max_design_depth": max_depth,
    }


def count_outputs_over_depth(session: Session, threshold: int) -> dict[str, Any]:
    graph = _graph(session)
    depths = graph.per_output_depths()
    return {"count": sum(1 for v in depths.values() if v > threshold)}


def list_primary_outputs_over_cone_size(session: Session, threshold: int) -> dict[str, Any]:
    """Primary outputs (only -- unlike count_outputs_over_depth, no DFF D
    pins) whose fanin logic cone has strictly more than `threshold` gates."""
    graph = _graph(session)
    sizes = po_cone_sizes(graph)
    over = sorted((nb for nb, size in sizes.items() if size > threshold), key=netbit_sort_key)
    return _cap([netbit_token(nb) for nb in over])


# ----------------------------------------------------------------------
# Path existence / counting / enumeration / cuts
# ----------------------------------------------------------------------

# Keys added to check_path_exists/count_paths/enumerate_paths' result ONLY
# when `avoid` is given (see `_resolve_avoid` and each function's
# docstring) -- named as module-level constants and threaded through the
# TOOL_SCHEMA descriptions below rather than hand-written in prose there,
# so a rename can't silently leave the description describing a key that
# no longer exists (the do_replace_buf_with_and description is this
# project's on-the-books example of that drift).
_EXISTS_IGNORING_AVOID_KEY = "exists_ignoring_avoid"
_COUNT_IGNORING_AVOID_KEY = "count_ignoring_avoid"
_AVOID_RESOLVED_TO_KEY = "avoid_resolved_to"


def _resolve_avoid(design: Design, avoid: Optional[str]) -> tuple[Optional[NetBit], Optional[str]]:
    """Resolve the `avoid` parameter shared by check_path_exists/count_paths/
    enumerate_paths. Net resolution is tried FIRST, exactly as before this
    helper existed; only when that fails does this fall back to treating
    `avoid` as a gate instance name and avoiding that gate's output net-bit
    instead (every gate in this IR has exactly one output pin, keyed by
    `ir.OUTPUT_PIN`, so "the" output is unambiguous). That ordering means
    every `avoid` value that already resolved as a net keeps resolving the
    same way -- this is a strict widening of what's accepted, not a change
    to any input that already worked, and if a net and a gate happen to
    share a name the net wins. Returns `(avoid_netbit, avoid_resolved_to)`;
    `avoid_resolved_to` is None unless the gate fallback actually fired, in
    which case it names which net the gate's output resolved to. Raises the
    same `ToolError` `_resolve_bit` would have raised when `avoid` is
    neither a net nor a gate."""
    if avoid is None:
        return None, None
    try:
        return _resolve_bit(design, avoid), None
    except ToolError:
        for gate in design.gates:
            if gate.inst_name == avoid:
                out_val = gate.pins.get(OUTPUT_PIN[gate.gate_type])
                if isinstance(out_val, NetBit):
                    # No `!r` on the gate name: the repr's closing quote runs
                    # straight into the possessive `'s` and renders as
                    # `gate 'g2''s output n12`, which reads as a typo to
                    # whoever (or whatever) is meant to act on it.
                    return out_val, f"gate {gate.inst_name}'s output {netbit_token(out_val)}"
                break
        raise


def check_path_exists(session: Session, source: str, target: str, avoid: Optional[str] = None) -> dict[str, Any]:
    """`exists` answers the question as asked (with `avoid` excluded from the
    graph, if given). When `avoid` is given, an extra `exists_ignoring_avoid`
    key reports the same query with `avoid` NOT excluded, so a `false`
    `exists` is never ambiguous between "avoid blocks every path" and "there
    is no path between source and target at all" -- `exists_ignoring_avoid:
    false` alongside `exists: false` means the second (avoid isn't the
    reason), `exists_ignoring_avoid: true` alongside `exists: false` means
    the first (avoid blocks every path). With no `avoid`, the result has
    only `exists`, unchanged from before this key existed."""
    graph = _graph(session)
    source_nb, target_nb = _resolve_bit(graph.design, source), _resolve_bit(graph.design, target)
    avoid_nb, avoid_resolved_to = _resolve_avoid(graph.design, avoid)
    result: dict[str, Any] = {"exists": graph.path_exists(source_nb, target_nb, avoid=avoid_nb)}
    if avoid is not None:
        result[_EXISTS_IGNORING_AVOID_KEY] = graph.path_exists(source_nb, target_nb, avoid=None)
        if avoid_resolved_to is not None:
            result[_AVOID_RESOLVED_TO_KEY] = avoid_resolved_to
    return result


def count_paths(session: Session, source: str, target: str, avoid: Optional[str] = None) -> dict[str, Any]:
    """`count` answers the question as asked (with `avoid` excluded from the
    graph, if given). When `avoid` is given, an extra `count_ignoring_avoid`
    key reports the same count with `avoid` NOT excluded -- the same
    disambiguation `check_path_exists` documents, in counting form: `count:
    0` next to `count_ignoring_avoid: 0` means there was never a path
    regardless of `avoid`; `count: 0` next to a nonzero
    `count_ignoring_avoid` means `avoid` blocks every one of them. With no
    `avoid`, the result has only `count`, unchanged from before this key
    existed."""
    graph = _graph(session)
    source_nb, target_nb = _resolve_bit(graph.design, source), _resolve_bit(graph.design, target)
    avoid_nb, avoid_resolved_to = _resolve_avoid(graph.design, avoid)
    result: dict[str, Any] = {"count": graph.path_count(source_nb, target_nb, avoid=avoid_nb)}
    if avoid is not None:
        result[_COUNT_IGNORING_AVOID_KEY] = graph.path_count(source_nb, target_nb, avoid=None)
        if avoid_resolved_to is not None:
            result[_AVOID_RESOLVED_TO_KEY] = avoid_resolved_to
    return result


def enumerate_paths(
    session: Session, source: str, target: str, avoid: Optional[str] = None, max_results: int = 50
) -> dict[str, Any]:
    """Enumerate paths (as ordered lists of gate instance names; an empty
    list means a direct wire, depth 0) from `source` to `target`. `count` is
    the TRUE total number of paths (via a fast DP, not full enumeration);
    `paths` is capped at `max_results` (hard ceiling 500) to stay tractable
    -- see `truncated`. When `avoid` is given, an extra `count_ignoring_avoid`
    key reports the same true total with `avoid` NOT excluded -- the same
    disambiguation `check_path_exists` documents, in counting form. With no
    `avoid`, the result is unchanged from before this key existed."""
    graph = _graph(session)
    source_nb, target_nb = _resolve_bit(graph.design, source), _resolve_bit(graph.design, target)
    avoid_nb, avoid_resolved_to = _resolve_avoid(graph.design, avoid)
    cap = max(1, min(int(max_results), 500))
    total = graph.path_count(source_nb, target_nb, avoid=avoid_nb)
    paths: list[list[str]] = []
    for path in graph.enumerate_paths(source_nb, target_nb, avoid=avoid_nb):
        if len(paths) >= cap:
            break
        paths.append([g.inst_name for g in path])
    result: dict[str, Any] = {"count": total, "paths": paths, "truncated": total > len(paths)}
    if avoid is not None:
        result[_COUNT_IGNORING_AVOID_KEY] = graph.path_count(source_nb, target_nb, avoid=None)
        if avoid_resolved_to is not None:
            result[_AVOID_RESOLVED_TO_KEY] = avoid_resolved_to
    return result


def get_reg_to_reg_path_stats(session: Session) -> dict[str, Any]:
    """Whole-design register-to-register combinational path count -- thin
    wrapper over `graph.reg_to_reg_path_stats`, the same primitive
    `router._h_reg_to_reg_paths` calls. `combinational_path_count` is the
    true total (can be in the millions on real designs, hence no attempt to
    list every path here); `direct_wire_count` (a DFF's Q wired straight
    into a DFF's D pin, zero gates in between) is reported separately and is
    NOT included in `combinational_path_count`."""
    graph = _graph(session)
    stats = graph.reg_to_reg_path_stats()
    return {
        "combinational_path_count": stats.combinational_path_count,
        "direct_wire_count": stats.direct_wire_count,
        "direct_wire_examples": [
            {"source_dff": src, "net": netbit_token(nb), "sink_dff": dst}
            for src, nb, dst in stats.direct_wire_examples
        ],
    }


def get_cut_nets_between(session: Session, source: str, target: str) -> dict[str, Any]:
    graph = _graph(session)
    result = graph.cut_nets_between(_resolve_bit(graph.design, source), _resolve_bit(graph.design, target))
    if not result.path_exists:
        return {"path_exists": False, "cut_nets": []}
    return {
        "path_exists": True,
        "cut_nets": [netbit_token(nb) for nb in sorted(result.cut_nets, key=netbit_sort_key)],
    }


def check_is_cut_signal(session: Session, signal: str) -> dict[str, Any]:
    graph = _graph(session)
    # Bare name -> ANY bit of the signal (is_cut_signal_bits' existing "any
    # bit" semantics); a bit-selected token narrows the ANY down to just
    # that one bit.
    netbits = _resolve_bits(graph.design, signal)
    return {"is_cut": is_cut_signal_bits(graph, netbits)}


_FLOATING_COUNT_KEY = "floating_input_nets_plus_unconnected_output_ports_count"
_FLOATING_EXTRA_KEY = "additional_findings_not_counted_above"


def check_floating_signals(session: Session) -> dict[str, Any]:
    """Thin wrapper over `analysis.find_floating_signals` -- the same
    function `router._h_floating_signals` calls. The returned shape is
    deliberately self-explaining (see `FloatingSignalsResult`'s docstring for
    why the count is narrower than the full set of findings): the count key
    names what it counts, and the items that make up that count are kept in
    a separate `counted_in_that_number` sub-object from the
    `additional_findings_not_counted_above` sub-object -- so a count of 0
    sitting next to a non-empty additional-findings list is never
    self-contradictory, it is two different questions answered at once."""
    res = find_floating_signals(_graph(session))
    # F6: sort the NetBits by (name, bit) NUMERICALLY (netbit_sort_key)
    # before rendering to tokens -- a plain string sort of the rendered
    # tokens puts "n17[10]" before "n17[2]".
    counted = {
            "floating_input_nets_referenced_but_undriven": [
                netbit_token(nb)
                for nb in sorted(res.floating_input_nets_referenced_but_undriven, key=netbit_sort_key)
            ],
            "unconnected_output_ports_undriven": [
                netbit_token(nb) for nb in sorted(res.unconnected_output_ports_undriven, key=netbit_sort_key)
            ],
    }
    not_counted = {
            "declared_input_ports_completely_unused": [
                netbit_token(nb) for nb in sorted(res.declared_input_ports_completely_unused, key=netbit_sort_key)
            ],
            "unconnected_gate_input_pins": sorted(
                f"{g.inst_name}.{pin}" for g, pin in res.unconnected_gate_input_pins
            ),
            "dangling_gate_outputs_never_consumed": sorted(
                g.inst_name for g in res.dangling_gate_outputs_never_consumed
            ),
            "dead_internal_wire_bits": [
                netbit_token(nb) for nb in sorted(res.dead_internal_wire_bits, key=netbit_sort_key)
            ],
    }
    # Built FROM the two groups rather than written next to them. This
    # sentence is data the model reads, and a note that contradicts the
    # payload is worse than no note: it actively argues for the wrong
    # answer. Prose has no mutation coverage -- inverting the hand-written
    # version of this to say every category WAS counted left the whole
    # suite green -- so the claim is derived from the structure it
    # describes and cannot drift from it.
    note = (
        f"{_FLOATING_COUNT_KEY} uses the strict definition of the corpus's own 'Check if there are any "
        f"floating inputs or unconnected output ports' question: it counts "
        f"{' and '.join(counted)} and nothing else. The categories under "
        f"{_FLOATING_EXTRA_KEY} ({', '.join(not_counted)}) are real structural observations worth "
        f"reporting, but are NOT part of that count."
    )
    return {
        _FLOATING_COUNT_KEY: res.headline_count,
        "counted_in_that_number": counted,
        _FLOATING_EXTRA_KEY: not_counted,
        "note": note,
    }


# ----------------------------------------------------------------------
# Cones
# ----------------------------------------------------------------------


def get_fanin_cone_size(session: Session, net: str) -> dict[str, Any]:
    graph = _graph(session)
    return {"size": fanin_cone_size(graph, _resolve_bit(graph.design, net))}


def get_fanin_cone_gates(session: Session, net: str) -> dict[str, Any]:
    graph = _graph(session)
    return _cap(sorted(graph.backward_reachable_gates(_resolve_bit(graph.design, net))))


def get_fanout_cone_gates(session: Session, net: str) -> dict[str, Any]:
    graph = _graph(session)
    return _cap(sorted(graph.forward_reachable_gates(_resolve_bit(graph.design, net))))


def get_largest_fanin_cone(session: Session) -> dict[str, Any]:
    graph = _graph(session)
    nb, size = largest_fanin_cone(graph)
    return {"net": None if nb is None else netbit_token(nb), "size": size}


def get_cone_gate_type_breakdown(session: Session, net: str) -> dict[str, Any]:
    """`by_type` is empty when `net`'s fanin cone genuinely contains no
    gates (e.g. it is driven directly by a primary input) -- `cone_gates`
    makes that explicit (0) rather than leaving an empty `by_type` dict
    that reads the same as "this call failed"."""
    design = _design(session)
    graph = _graph(session)
    names = graph.backward_reachable_gates(_resolve_bit(design, net))
    counts: dict[str, int] = {}
    for g in design.gates:
        if g.inst_name in names:
            counts[g.gate_type.name] = counts.get(g.gate_type.name, 0) + 1
    return {"net": net, "cone_gates": len(names), "by_type": counts}


def get_shared_fanin_gates(session: Session, net_a: str, net_b: str) -> dict[str, Any]:
    graph = _graph(session)
    shared = graph.backward_reachable_gates(_resolve_bit(graph.design, net_a)) & graph.backward_reachable_gates(
        _resolve_bit(graph.design, net_b)
    )
    return _cap(sorted(shared))


# ----------------------------------------------------------------------
# Boolean-semantic (abc_bridge)
# ----------------------------------------------------------------------


def check_equivalence_to_snapshot(session: Session) -> dict[str, Any]:
    if session.original_snapshot is None:
        raise ToolError("No originally-loaded snapshot is available (no design has been loaded yet).")
    result = verify_equivalence(session.original_snapshot, _design(session))
    return {"equivalent": result.equivalent, "detail": result.detail}


def check_signal_equivalence(session: Session, net_a: str, net_b: str) -> dict[str, Any]:
    design = _design(session)
    return {"equivalent": are_equivalent(design, _resolve_bit(design, net_a), _resolve_bit(design, net_b))}


def check_symmetry_tool(session: Session, output: str, input_a: str, input_b: str) -> dict[str, Any]:
    design = _design(session)
    return {
        "symmetric": check_symmetry(
            design, _resolve_bit(design, output), _resolve_bit(design, input_a), _resolve_bit(design, input_b)
        )
    }


def get_constant_value(session: Session, net: str) -> dict[str, Any]:
    design = _design(session)
    val = is_constant(design, _resolve_bit(design, net))
    return {"is_constant": val is not None, "value": None if val is None else str(val.value)}


def check_property_asserted_only_when(session: Session, signal: str, condition: str) -> dict[str, Any]:
    """"For output <signal>, verify that it is asserted only when <condition>,
    and provide a counterexample if this is not true." `condition` is one or
    more "<net> is <0|1|high|low>" literals joined by "and"/"or" (an optional
    leading "both" is tolerated), e.g. "req is 1 and busy is 0". See
    `netlist_agent.property_check` for the free_pi (DFF-boundary)
    approximation this makes and why a returned counterexample may pin an
    unreachable register state."""
    design = _design(session)
    try:
        result = check_asserted_only_when(design, signal, condition)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "holds": result.holds,
        "assignment": result.assignment,
        "caveat": result.caveat,
        "detail": result.detail,
    }


def find_signal_pair_for_operator(session: Session, target: str, op: str) -> dict[str, Any]:
    """"Does there exist a pair of signals (a, b) already in the netlist such
    that OP(a, b) is equivalent to target?" -- bit-parallel-simulation-backed
    search over `signal_pair_search.find_pair_for_op`, formally verified
    before being reported. `a == b` is allowed; neither may be `target`
    itself."""
    design = _design(session)
    target_nb = _resolve_bit(design, target)
    try:
        result = find_pair_for_op(design, target_nb, op)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "found": result.found,
        "pair": list(result.pair) if result.pair is not None else None,
        "explanation": result.explanation,
        "stats": result.stats,
    }


def get_boolean_function(session: Session, net: str) -> dict[str, Any]:
    """"Derive the Boolean equation for output X in terms of its primary
    inputs" / "What Boolean function does X compute?" -- thin wrapper over
    `boolean_function.derive_boolean_function` (see its module docstring
    for the three cases: X directly driven by a DFF's Q, combinational with
    a DFF.Q in its support, or purely PI-only). No SOP minimization is
    attempted; `explanation` is the same natural-language rendering
    router.py's own handler returns.

    `expressible_in_pis_only` is about TARGET (`net`) itself: always False
    when `is_dff_q` is True (a DFF's own Q output is sequential state, never
    a PI-only combinational function of it, regardless of its D pin's own
    support). `next_state_expressible_in_pis_only` is set iff `is_dff_q` and
    is about the D PIN's own next-state function instead (None otherwise, or
    when the D pin is unconnected)."""
    design = _design(session)
    try:
        result = derive_boolean_function(design, net)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "target": result.target,
        "is_dff_q": result.is_dff_q,
        "dff_inst": result.dff_inst,
        "root": result.root,
        "cone_gate_count": result.cone_gate_count,
        "support": result.support,
        "support_pi": result.support_pi,
        "support_dffq": result.support_dffq,
        "support_other": result.support_other,
        "expressible_in_pis_only": result.expressible_in_pis_only,
        "next_state_expressible_in_pis_only": result.next_state_expressible_in_pis_only,
        "expression_lines": result.expression_lines,
        "truncated": result.truncated,
        "onset_count": result.onset_count,
        "total_count": result.total_count,
        "onset_minterms": result.onset_minterms,
        "offset_minterms": result.offset_minterms,
        "verified": result.verified,
        "caveat": result.caveat,
        "explanation": result.explanation,
    }


# ----------------------------------------------------------------------
# Session bookkeeping (read-only)
# ----------------------------------------------------------------------


_UNAFFECTED_BY_THIS_TURNS_TOOL_CALLS_KEY = "unaffected_by_this_turns_tool_calls"


def get_last_operation_summary(session: Session) -> dict[str, Any]:
    """Read back the counters router.py's own handlers already maintain for
    "how many gates did the last operation touch?"-style follow-ups --
    WITHOUT rerunning anything. If an earlier tool call (or an earlier
    rule-routed request this same session) already reported a count, this
    is how to retrieve it again: calling the original do_* tool a second
    time repeats its mutation and produces a NEW, different count, not the
    same answer over again.

    These counters describe the operation from the user's PREVIOUS request
    that last updated them -- deliberately NOT refreshed by any tool call
    made THIS turn, including a do_* tool this same turn already called.
    Reading `last_op_count` unchanged after running a mutating tool this
    turn is therefore correct, not stale: it is still answering "what did
    the earlier request do", not "what did this turn's tool call do" (that
    call's own result already reports its own count under its own key)."""
    return {
        "last_op_count": session.last_op_count,
        "last_gate_delta": {gt.name: n for gt, n in session.last_gate_delta.items()},
        "last_query_gate_names": list(session.last_query_gate_names),
        "last_floating_count": session.last_floating_count,
        "last_enable_hold_count": session.last_enable_hold_count,
        "functional_change_ops": session.functional_change_ops,
        _UNAFFECTED_BY_THIS_TURNS_TOOL_CALLS_KEY: True,
    }


# ----------------------------------------------------------------------
# Rename
# ----------------------------------------------------------------------


def rename_gate(session: Session, old_name: str, new_name: str) -> dict[str, Any]:
    design = _design(session)
    try:
        _rename_gate_instance(design, old_name, new_name)
    except KeyError as exc:
        raise ToolError(str(exc)) from exc
    return {"renamed": True, "old_name": old_name, "new_name": new_name}


def rename_signal(session: Session, old_name: str, new_name: str) -> dict[str, Any]:
    design = _design(session)
    try:
        design.rename_signal(old_name, new_name)
    except (KeyError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    return {"renamed": True, "old_name": old_name, "new_name": new_name}


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
#
# Rerun-of-a-prior-operation disclosure (measured:
# experiments/count_question_reruns_2026-08-29/REPORT.md). `session.last_op_kind`
# names the transform function that produced `session.last_op_count` --
# router.py sets both together, whether the count came from an EARLIER
# rule-routed request in this session or an earlier call to one of these
# same do_* tools. A mutating do_* tool below has no way to know, on its
# own, whether the model is asking about the operation it is CURRENTLY
# running or one from earlier -- but it DOES know when it is re-running the
# exact same transform `last_op_kind` already reported a count for, and
# that is precisely the traced failure: the model read the correct earlier
# count via get_last_operation_summary, called the do_* tool again anyway
# (a NEW operation, e.g. a second dedup pass legitimately finding fewer or
# zero further merges), and reported the rerun's number to a question about
# the first. When `last_op_kind` names a DIFFERENT transform, nothing is
# added: this tool has no basis to guess what that other operation's count
# means here, and guessing would manufacture a new kind of wrong answer
# instead of fixing this one.
#
# `last_op_kind` naming the same transform function is NOT, by itself,
# enough -- `remap_to_basis` alone backs 6 differently-scoped rule-routed
# handlers (whole-design, cone-restricted, type-restricted, the XOR/XNOR
# sugar handlers...), so comparing only the function name flagged a
# same-function-different-scope call (e.g. a whole-design "and_not" remap
# immediately after a rule-routed XOR-only "nand_not" remap) as a "rerun"
# of a scope it never touched, with `previously_reported_count` naming a
# number from that unrelated scope. `session.last_op_args` (see its
# docstring) carries the actual arguments the recorded call ran with,
# normalized so two calls with the "same" arguments compare equal even if
# a `set`/`list` argument iterated in a different order -- comparing that
# too, not just the function name, is what fixes this.
#
# Batch 2026-08-29 (experiments/refuse_rerun_2026-08-29/PROTOCOL.md): the
# disclosure above stopped the WRONG NUMBER but not the unasked mutation --
# 6 of 18 measured rows still deleted 691 gates to answer a question that
# only asked "how many were merged?", just now reporting the recorded count
# alongside the deletion. `_refuse_rerun` (below) upgrades this from
# disclosure to a default refusal: when the same condition holds, the
# mutating do_* tool performs NO mutation at all and hands back
# `previously_reported_count` instead -- the number the model actually
# wanted, in the freshest tool result, with no netlist edit attached to it.
# This is not a hard block: `last_op_kind` is only ever updated by the rule
# layer (never by a do_* tool call itself, see `_run_and_track`'s own
# comment), so an intervening tool-layer mutation this same turn leaves it
# stale -- a model that legitimately wants a SECOND dedup pass after making
# other changes would be refused for a reason that is no longer true. Each
# of the 11 mutating tools below therefore takes an explicit `force: bool =
# False` override; passing `force=True` always runs the transform, no
# matter what `last_op_kind`/`last_op_args` say.
_RERUN_OF_PRIOR_OPERATION_KEY = "rerun_of_a_previously_reported_operation"
_PREVIOUSLY_REPORTED_COUNT_KEY = "previously_reported_count"
_MUTATION_PERFORMED_KEY = "mutation_performed"
_REFUSED_REASON_KEY = "refused_reason"
_FORCE_OVERRIDE_PARAM_KEY = "override_with"
_FORCE_PARAM_NAME = "force"
_RERUN_NOTE_TEXT = (
    f"If this call would re-run a transform whose count was already reported earlier this session (by "
    f"an earlier request or an earlier call to this same tool) with the SAME effective arguments, it "
    f"performs NO mutation by default and instead returns {{{_MUTATION_PERFORMED_KEY!r}: False, "
    f"{_PREVIOUSLY_REPORTED_COUNT_KEY!r}: <the earlier count>, {_REFUSED_REASON_KEY!r}: <why>, "
    f"{_FORCE_OVERRIDE_PARAM_KEY!r}: {_FORCE_PARAM_NAME!r}}} -- use {_PREVIOUSLY_REPORTED_COUNT_KEY!r} to "
    f"answer a question about that earlier operation; do not treat a refusal as an error. Pass "
    f"{_FORCE_PARAM_NAME}=true to run the transform anyway (e.g. a deliberate second pass); the result "
    f"then reports what THIS call itself did, plus (if it does turn out to repeat the same earlier "
    f"operation) {_RERUN_OF_PRIOR_OPERATION_KEY!r} and {_PREVIOUSLY_REPORTED_COUNT_KEY!r} again, this "
    f"time for reference rather than in place of running it."
)


def _is_rerun_of_recorded_op(session: Session, fn: Callable[..., Any], *args: object) -> bool:
    """True exactly when `session.last_op_kind` already names this SAME
    transform (`fn.__name__`) AND `session.last_op_args` (normalized via
    `_normalize_op_args`, same as `*args` here) already equals THIS call's
    own normalized `*args` -- see the module comment above this section for
    why the function name alone is too coarse (`remap_to_basis` backs 6
    differently-scoped handlers). `*args` must be exactly the positional
    arguments this call is about to pass (or already passed) to `fn`
    itself, minus `design` -- same convention `router._run_and_track`/
    `_run_and_track_bits` use for what they record into `last_op_args` in
    the first place, so the two sides are comparing like with like.

    False whenever either comparison doesn't hold -- including the first
    mutating call of a session, where `last_op_kind`/`last_op_args` are
    still `None`. A normalization this function (or `_normalize_op_args`)
    doesn't recognize an argument type for can only ever produce a missed
    match, never a wrong one: an args tuple that fails to compare equal to
    an actual rerun's just means this returns False, same as a genuinely
    different call would -- see `_normalize_op_args`'s docstring. That
    asymmetry is deliberate: a false negative here costs the model one
    missed hint (or, now, one avoidable refusal); a false positive would
    refuse -- or hand back a `previously_reported_count` for -- a
    DIFFERENT operation, inviting exactly the wrong-answer failure this
    mechanism exists to prevent. An uncertain match must resolve to "not a
    rerun", never the other way."""
    if session.last_op_kind != fn.__name__:
        return False
    return session.last_op_args == _normalize_op_args(*args)


def _rerun_conflict_fields(session: Session, fn: Callable[..., Any], *args: object) -> dict[str, Any]:
    """Disclosure fields for a mutating do_* tool's result on a call that
    actually ran (either because `_is_rerun_of_recorded_op` was False, or
    because it was True and the caller passed `force=True` to run anyway):
    present ONLY when `_is_rerun_of_recorded_op(session, fn, *args)` is
    True. Empty dict otherwise, so a do_* tool's return shape is UNCHANGED
    on every call this condition doesn't apply to.

    This call's OWN count is not echoed under a new key here -- every
    caller already returns it under its own existing key (`merged`,
    `replaced`, `removed`, ...), so duplicating it would only add a second,
    differently-named source of truth for the same number."""
    if not _is_rerun_of_recorded_op(session, fn, *args):
        return {}
    return {
        _RERUN_OF_PRIOR_OPERATION_KEY: True,
        # Stated positively rather than left to the absence of the refusal's
        # `False`: a cold read pointed out that "this key is missing" was
        # carrying the whole distinction between "refused, nothing ran" and
        # "forced, it ran". Semantics by absence is a bad thing to ask a
        # model to notice, so both paths now say which happened outright.
        _MUTATION_PERFORMED_KEY: True,
        _PREVIOUSLY_REPORTED_COUNT_KEY: session.last_op_count,
    }


def _force_requested(force: object) -> bool:
    """Whether a mutating tool's `force` argument really means "run it anyway".

    NOT a plain truth test, and the difference is the whole point. Tool-call
    arguments reach these functions as `fn(session, **json.loads(raw))` (see
    `llm/client.py`) with no schema-driven type coercion in between, so a
    provider that serialises booleans as JSON strings hands this the string
    `"false"` -- which is truthy in Python. A cold read reproduced it: an
    argument that says "do not force" silently bypassed the refusal and let
    the mutation run. That is the worst available direction for this
    particular flag to fail in.

    So only values that unambiguously say yes count as yes, and ANYTHING
    unrecognised -- a bare string, a number, None, a dict -- means "do not
    force", i.e. the tool refuses. A wrongly-refused call costs the model one
    round and tells it exactly which parameter to set; a wrongly-forced one
    edits the user's netlist. The asymmetry is deliberate."""
    if force is True:
        return True
    if isinstance(force, str):
        return force.strip().lower() in {"true", "yes", "1"}
    return False


def _refuse_rerun(session: Session, fn: Callable[..., Any], *args: object, force: bool) -> Optional[dict[str, Any]]:
    """Gate for a mutating do_* tool, called BEFORE it performs any
    mutation. Returns `None` when the call should proceed as normal (either
    `force` is True, or `_is_rerun_of_recorded_op` is False -- i.e. this is
    not a detected rerun at all); the caller runs the transform exactly as
    it did before this mechanism existed. Returns a refusal dict when
    `force` is False AND `_is_rerun_of_recorded_op` is True -- the caller
    must `return` that dict immediately and perform NO mutation, not even
    a partial one.

    The refusal dict is a DIFFERENT shape from `_rerun_conflict_fields`'s
    disclosure dict (both name `_PREVIOUSLY_REPORTED_COUNT_KEY`, but the
    refusal is the tool's ENTIRE result, not fields merged into a normal
    one -- there is no `merged`/`replaced`/`removed` key to merge into,
    because nothing ran):

      * `_MUTATION_PERFORMED_KEY`: False -- unambiguous "nothing changed"
        flag, checkable without knowing this mechanism exists at all.
      * `_PREVIOUSLY_REPORTED_COUNT_KEY`: `session.last_op_count` -- the
        number the model almost always actually wants, landing in the
        freshest tool result instead of requiring a separate
        get_last_operation_summary call.
      * `_REFUSED_REASON_KEY`: human-readable explanation.
      * `_FORCE_OVERRIDE_PARAM_KEY`: the name of the parameter (always
        `_FORCE_PARAM_NAME`, `"force"`, the same across all 11 mutating
        tools) that runs the transform anyway.

    Not a hard block: `force=True` always returns `None` regardless of
    `_is_rerun_of_recorded_op`, so a model that deliberately wants to
    re-run an operation (e.g. a genuine second dedup pass after other
    changes made new duplicates possible) always has a way through."""
    if _force_requested(force):
        return None
    if not _is_rerun_of_recorded_op(session, fn, *args):
        return None
    return {
        _MUTATION_PERFORMED_KEY: False,
        _PREVIOUSLY_REPORTED_COUNT_KEY: session.last_op_count,
        _REFUSED_REASON_KEY: (
            f"this call would re-run {fn.__name__!r} with the same arguments as an operation already "
            f"reported earlier this session -- refused by default so a question does not silently edit "
            f"the netlist again; the earlier count is {_PREVIOUSLY_REPORTED_COUNT_KEY!r} above. Pass "
            f"{_FORCE_PARAM_NAME}=true to run it anyway."
        ),
        _FORCE_OVERRIDE_PARAM_KEY: _FORCE_PARAM_NAME,
        _RERUN_OF_PRIOR_OPERATION_KEY: True,
    }


def do_limit_fanout_global(session: Session, max_fanout: int, force: bool = False) -> dict[str, Any]:
    refusal = _refuse_rerun(session, limit_fanout, max_fanout, force=force)
    if refusal is not None:
        return refusal
    design = _design(session)
    try:
        result = limit_fanout(design, max_fanout)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {"buffers_added": result, **_rerun_conflict_fields(session, limit_fanout, max_fanout)}


def do_limit_fanout_net(session: Session, net: str, max_fanout: int, force: bool = False) -> dict[str, Any]:
    design = _design(session)
    # A whole-signal operation, via `_resolve_bits` rather than `_resolve_bit`:
    # a bare name on a multi-bit signal now caps EVERY bit of the signal
    # independently, instead of erroring. The earlier bit-select-required
    # error (still correct for the read-side queries) turned out to invite a
    # worse failure mode here -- a model retrying against the error message's
    # own example (net[0]) would fix exactly one over-limit bit, then report
    # the whole net's fanout as capped, silently leaving the rest of the bus
    # over the limit. A bit-selected token (`net[3]`) still narrows to that
    # one bit only, same as before. `bits_processed` reports how many bits
    # were actually resolved and SWEPT (checked against the cap); that is
    # NOT the same as how many bits were actually MODIFIED (already-within-
    # cap bits are swept but contribute 0 buffers) -- `bits_modified` reports
    # the latter, so the model can tell "we checked everything" from "we
    # changed everything" instead of conflating them via `buffers_added`
    # alone (a `buffers_added` of 0 is otherwise ambiguous between "nothing
    # needed to change" and a silently-failed op).
    netbits = _resolve_bits(design, net)
    refusal = _refuse_rerun(session, limit_fanout_net, netbits, max_fanout, force=force)
    if refusal is not None:
        return refusal
    try:
        per_bit = [limit_fanout_net(design, nb.name, max_fanout, nb.bit) for nb in netbits]
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "buffers_added": sum(per_bit),
        "bits_processed": len(netbits),
        "bits_modified": sum(1 for c in per_bit if c > 0),
        **_rerun_conflict_fields(session, limit_fanout_net, netbits, max_fanout),
    }


def do_insert_buffer_per_load(session: Session, net: str, force: bool = False) -> dict[str, Any]:
    design = _design(session)
    # See `do_limit_fanout_net`'s comment: same whole-signal widening, same
    # `bits_processed` (swept) vs. `bits_modified` (actually got a buffer)
    # distinction.
    netbits = _resolve_bits(design, net)
    refusal = _refuse_rerun(session, insert_buffer_per_load, netbits, force=force)
    if refusal is not None:
        return refusal
    per_bit = [insert_buffer_per_load(design, nb.name, nb.bit) for nb in netbits]
    return {
        "buffers_added": sum(per_bit),
        "bits_processed": len(netbits),
        "bits_modified": sum(1 for c in per_bit if c > 0),
        **_rerun_conflict_fields(session, insert_buffer_per_load, netbits),
    }


def do_balance_depth_to_sinks(session: Session, source: str, sinks: list[str], force: bool = False) -> dict[str, Any]:
    """Insert buffers so the depth from `source` to every net in `sinks` is
    the same (the max of their existing depths -- depth is only ever added,
    never removed). Equivalence-preserving. For a pure fanout tree this is
    the exact minimum-buffer solution; for a reconvergent DAG (some node
    reached by more than one path from `source`) the result is a valid
    balancing but not guaranteed minimal -- see `is_tree` in the result."""
    design = _design(session)
    src = _resolve_bit(design, source)
    sink_nbs = [_resolve_bit(design, s) for s in sinks]
    refusal = _refuse_rerun(session, balance_depth_to_sinks, src, sink_nbs, force=force)
    if refusal is not None:
        return refusal
    try:
        result = balance_depth_to_sinks(design, src, sink_nbs)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "buffers_added": result.buffers_added,
        "target_depth": result.target_depth,
        "already_balanced": result.already_balanced,
        "is_tree": result.is_tree,
        **_rerun_conflict_fields(session, balance_depth_to_sinks, src, sink_nbs),
    }


def do_replace_buf_with_and(
    session: Session, ctrl_net: str, gate_names: Optional[list[str]] = None, force: bool = False
) -> dict[str, Any]:
    """Rewrite BUF gates in place into 2-input AND gates: each keeps its
    original input on I0 and gets `ctrl_net` wired to I1; the output net is
    left completely untouched. This is a deliberate FUNCTIONAL change, not
    an equivalence-preserving transform. If `gate_names` is omitted, targets
    whatever the most recent find_gates_by_name call found.

    Three different things can put a 0 in `replaced`, and the payload has to
    tell them apart -- an earlier version of this docstring claimed there
    were only two, which was false: `transform.replace_buf_with_and` also
    skips a gate whose OWN output net is `ctrl` (it would be a direct
    combinational self-loop), and that gate IS a BUF. The rule-routed path
    has always reported those skips (`router._h_...` passes
    `skipped_self_loop`); the tool did not pass the list at all, so the one
    explanation the model could read was the one that could be wrong.

      * `buf_candidates_in_scope: 0` -- the scope was understood and holds
        no BUF gates (or is empty). Not a failure.
      * `skipped_self_loop` non-empty -- those named gates are BUFs, and
        were left alone because ctrl_net is their own output.
      * neither -- every named gate resolved to something other than a BUF.
    """
    design = _design(session)
    ctrl = _resolve_bit(design, ctrl_net)
    if gate_names is not None:
        names = list(gate_names)
        scope = "explicit gate_names list"
    else:
        names = list(session.last_query_gate_names)
        scope = "gate names from the most recent find_gates_by_name query"
    by_name = {g.inst_name: g for g in design.gates}
    buf_candidates_in_scope = sum(
        1 for n in names if by_name.get(n) is not None and by_name[n].gate_type == GateType.BUF
    )
    skipped_self_loop: list[str] = []
    # Both the refusal check and the disclosure fields are computed BEFORE
    # the call below, against `skipped_self_loop` while it's still the empty
    # list it's about to be passed in as -- `replace_buf_with_and` mutates it
    # in place as its own OUTPUT, and router.py's `_run_and_track` (which
    # this compares against) normalizes that same argument at the same
    # pre-call point, for the same reason: an output parameter's post-call
    # contents describe what the call just did, not what request identifies
    # it, so comparing post-call would compare two different things that
    # happen to share a name. This also means the refusal check -- which
    # must run before ANY mutation -- naturally sees the same pre-call
    # `skipped_self_loop` the disclosure check does, so no separate ordering
    # question was introduced by adding it.
    refusal = _refuse_rerun(session, replace_buf_with_and, names, ctrl, skipped_self_loop, force=force)
    if refusal is not None:
        return refusal
    rerun_fields = _rerun_conflict_fields(session, replace_buf_with_and, names, ctrl, skipped_self_loop)
    try:
        replaced = replace_buf_with_and(design, names, ctrl, skipped_self_loop)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "skipped_self_loop": sorted(skipped_self_loop),
        "replaced": replaced,
        "scope": scope,
        "names_in_scope": len(names),
        "buf_candidates_in_scope": buf_candidates_in_scope,
        **rerun_fields,
    }


def do_remove_dangling_gates(session: Session, force: bool = False) -> dict[str, Any]:
    refusal = _refuse_rerun(session, remove_dangling_gates, force=force)
    if refusal is not None:
        return refusal
    return {
        "removed": remove_dangling_gates(_design(session)),
        **_rerun_conflict_fields(session, remove_dangling_gates),
    }


def do_deduplicate_gates(session: Session, force: bool = False) -> dict[str, Any]:
    refusal = _refuse_rerun(session, deduplicate_gates, force=force)
    if refusal is not None:
        return refusal
    return {
        "merged": deduplicate_gates(_design(session)),
        **_rerun_conflict_fields(session, deduplicate_gates),
    }


def do_collapse_double_inverters(session: Session, force: bool = False) -> dict[str, Any]:
    refusal = _refuse_rerun(session, collapse_double_inverters, force=force)
    if refusal is not None:
        return refusal
    return {
        "collapsed": collapse_double_inverters(_design(session)),
        **_rerun_conflict_fields(session, collapse_double_inverters),
    }


def do_collapse_inverter_buffer_chains(session: Session, force: bool = False) -> dict[str, Any]:
    """Splice out every BUF directly fed by a NOT's output (BUF(NOT(x)) ==
    NOT(x)), leaving a single inverter -- equivalence-preserving, unlike
    do_replace_buf_with_and."""
    refusal = _refuse_rerun(session, collapse_inverter_buffer_chains, force=force)
    if refusal is not None:
        return refusal
    return {
        "collapsed": collapse_inverter_buffer_chains(_design(session)),
        **_rerun_conflict_fields(session, collapse_inverter_buffer_chains),
    }


def do_simplify_constant_inputs(
    session: Session, gate_types: Optional[list[str]] = None, force: bool = False
) -> dict[str, Any]:
    design = _design(session)
    gts = {_gate_type(t) for t in gate_types} if gate_types else None
    refusal = _refuse_rerun(session, simplify_constant_inputs, gts, force=force)
    if refusal is not None:
        return refusal
    return {
        "simplified": simplify_constant_inputs(design, gts),
        **_rerun_conflict_fields(session, simplify_constant_inputs, gts),
    }


def do_remap_to_basis(
    session: Session,
    basis: str,
    gate_type: Optional[str] = None,
    cone_root: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Replace gates with an equivalent subcircuit built only from `basis`
    ("and", "nand", or "nor", each paired with NOT). Restrict the sweep with
    `gate_type` (only replace gates of that source type, e.g. "xor") and/or
    `cone_root` (only replace gates in that net's fanin cone); with neither,
    the whole design is remapped. `replaced: 0` with `gates_in_scope: 0` is
    the legitimate answer for an empty cone/type restriction -- it means the
    scope was understood and is simply empty, not that the call failed; do
    not widen the scope (e.g. drop `cone_root`) on a zero unless that was
    actually asked for."""
    design = _design(session)
    basis_key = _basis(basis)
    only_gates: Optional[set[str]] = None
    scope_parts: list[str] = []
    if cone_root is not None:
        graph = _graph(session)
        only_gates = graph.backward_reachable_gates(_resolve_bit(design, cone_root))
        scope_parts.append(f"fanin cone of {cone_root}")
    if gate_type is not None:
        gt = _gate_type(gate_type)
        type_names = {g.inst_name for g in design.gates if g.gate_type == gt}
        only_gates = type_names if only_gates is None else (only_gates & type_names)
        scope_parts.append(f"gates of type {gate_type}")
    scope = " and ".join(scope_parts) if scope_parts else "whole design"
    gates_in_scope = len(only_gates) if only_gates is not None else len(design.gates)
    # router.py's own rule-routed handlers call `remap_to_basis` with
    # VARIABLE arity: `(design, basis)` for a whole-design remap, versus
    # `(design, basis, only_gates)` for any restricted one -- omitting the
    # trailing `None` entirely rather than passing it explicitly. Matched
    # here (instead of always passing `(basis_key, only_gates)`, `only_gates`
    # possibly `None`) so a whole-design tool call's arity actually lines up
    # with a whole-design rule-routed call's for comparison -- passing an
    # explicit trailing `None` would make the two calls compare as DIFFERENT
    # argument tuples even though they mean the same "no restriction". Built
    # BEFORE the mutating call below (not after, as originally): the refusal
    # check needs these exact args and must run before any mutation.
    rerun_args = (basis_key,) if only_gates is None else (basis_key, only_gates)
    refusal = _refuse_rerun(session, remap_to_basis, *rerun_args, force=force)
    if refusal is not None:
        return refusal
    try:
        replaced = remap_to_basis(design, basis_key, only_gates)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "replaced": replaced,
        "scope": scope,
        "gates_in_scope": gates_in_scope,
        **_rerun_conflict_fields(session, remap_to_basis, *rerun_args),
    }


def do_optimize_depth(session: Session, basis: Optional[str] = None) -> dict[str, Any]:
    """Whole-design ABC-backed depth optimization (see netlist_agent.abc_synth).
    Mutates the session's current design in place only if a genuinely
    equivalence-preserving, strictly-better (or same-depth/fewer-gates)
    result was found; otherwise leaves it untouched."""
    design = _design(session)
    result = optimize_depth(design, _depth_opt_basis(basis))
    session.current_design = result.design
    return {
        "changed": result.changed,
        "depth_before": result.depth_before,
        "depth_after": result.depth_after,
        "note": result.note,
    }


def do_optimize_cone_depth(session: Session, net: str, basis: Optional[str] = None) -> dict[str, Any]:
    """Cone-restricted ABC-backed depth optimization: re-synthesizes only
    `net`'s fanin cone, leaving the rest of the design untouched."""
    design = _design(session)
    result = optimize_cone_depth(design, _resolve_bit(design, net), _depth_opt_basis(basis))
    session.current_design = result.design
    return {
        "changed": result.changed,
        "depth_before": result.depth_before,
        "depth_after": result.depth_after,
        "note": result.note,
    }


def do_optimize_gate_count(session: Session, basis: Optional[str] = None, max_depth: Optional[int] = None) -> dict[str, Any]:
    """Whole-design ABC-backed gate-count (area) optimization: tries a small
    set of candidate resynthesis scripts and commits whichever verified-
    equivalent candidate has the smallest gate count, honoring `max_depth`
    (if given) as a HARD ceiling on the resulting maximum design depth --
    a candidate that would exceed it is rejected outright, never accepted
    with a note. Only actually changes the design if a genuinely smaller,
    verified-equivalent result was found under those constraints; otherwise
    reports the design unchanged."""
    design = _design(session)
    result = optimize_gate_count(design, _depth_opt_basis(basis), max_depth)
    session.current_design = result.design
    return {
        "changed": result.changed,
        "gates_before": result.gates_before,
        "gates_after": result.gates_after,
        "depth_before": result.depth_before,
        "depth_after": result.depth_after,
        "note": result.note,
    }


def do_optimize_cone_gate_count(
    session: Session, net: str, basis: Optional[str] = None, max_depth: Optional[int] = None
) -> dict[str, Any]:
    """Cone-restricted ABC-backed gate-count (area) optimization: the
    `do_optimize_gate_count` counterpart restricted to `net`'s fanin cone,
    leaving every other gate in the design untouched. `max_depth` (if given)
    is a HARD ceiling on the resulting depth of `net`'s own cone, not the
    whole design's."""
    design = _design(session)
    result = optimize_cone_gate_count(design, _resolve_bit(design, net), _depth_opt_basis(basis), max_depth)
    session.current_design = result.design
    return {
        "changed": result.changed,
        "gates_before": result.gates_before,
        "gates_after": result.gates_after,
        "depth_before": result.depth_before,
        "depth_after": result.depth_after,
        "note": result.note,
    }


# ----------------------------------------------------------------------
# Schema + registry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


def _schema(properties: dict[str, dict[str, Any]], required: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _s(desc: str, enum: Optional[list[str]] = None) -> dict[str, Any]:
    d: dict[str, Any] = {"type": "string", "description": desc}
    if enum is not None:
        d["enum"] = enum
    return d


def _i(desc: str) -> dict[str, Any]:
    return {"type": "integer", "description": desc}


def _sa(desc: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": desc}


def _b(desc: str) -> dict[str, Any]:
    return {"type": "boolean", "description": desc}


_FORCE_DESC = (
    f"Optional, defaults to false. If this call would otherwise be refused as a rerun of an operation "
    f"already reported earlier this session (see this tool's description), pass true to run it anyway. "
    f"Has no effect when this call is not a detected rerun."
)

_NET_DESC = (
    "A net reference that must resolve to exactly ONE net-bit: a bare name (e.g. 'clk') for a "
    "scalar (1-bit) signal, or an explicit bit-select (e.g. 'n6[3]') for a multi-bit signal -- a "
    "bare name on a multi-bit signal is REJECTED as ambiguous (which bit is meant?), it is not a "
    "shorthand for 'the whole bus' or 'bit 0'."
)
_NET_OR_WHOLE_SIGNAL_DESC = (
    "A net/signal reference. A bit-select (e.g. 'n6[3]') targets that one bit only; a bare signal "
    "name (e.g. 'n6') targets EVERY bit of that (possibly multi-bit) signal, not just bit 0."
)
_GATE_DESC = "A gate instance name, e.g. 'g0'."

TOOL_SCHEMA: list[ToolSpec] = [
    ToolSpec(
        "load_design",
        "Load (or reload) a gate-level Verilog design from a file, replacing the current design and "
        "capturing a fresh original-load snapshot for later equivalence checks.",
        _schema(
            {"filename": _s("Verilog filename, e.g. 'test01.v'."), "directory": _s("Directory containing the file.")},
            ["filename", "directory"],
        ),
    ),
    ToolSpec(
        "write_design",
        "Write the current (possibly transformed) design out to a Verilog file.",
        _schema({"filename": _s("Output filename.")}, ["filename"]),
    ),
    ToolSpec(
        "set_testcase",
        "Record the current testcase's case name (and open its log file) if this hasn't already happened. "
        "Call this if the request stream's opening line named a testcase but the rule-based router didn't "
        "recognize the phrasing -- any responses already emitted this testcase are preserved and flushed to "
        "the log once it opens. A no-op if the case name was already recognized some other way.",
        _schema(
            {
                "case_name": _s("The testcase's case name."),
                "log_filename": _s("Explicit log filename, if the request named one (optional; defaults to '<case_name>.log')."),
            },
            ["case_name"],
        ),
    ),
    ToolSpec(
        "count_gates_by_type",
        "Get the total gate count and a breakdown by gate type (AND/OR/NAND/NOR/XOR/XNOR/NOT/BUF/DFF).",
        _schema({}),
    ),
    ToolSpec(
        "count_primary_ports",
        "Get the number of primary input/output ports and their total bit widths.",
        _schema({}),
    ),
    ToolSpec(
        "list_primary_ports",
        "List primary input or output ports with their bit widths.",
        _schema({"direction": _s("'input' or 'output'.", ["input", "output"])}, ["direction"]),
    ),
    ToolSpec(
        "list_gates_of_type",
        "List every gate instance of a given gate type, with its pin connections.",
        _schema({"gate_type": _s("Gate type.", _GATE_TYPE_VALUES)}, ["gate_type"]),
    ),
    ToolSpec(
        "get_gate_info",
        "Look up a gate instance by name: its type and pin connections.",
        _schema({"gate": _s(_GATE_DESC)}, ["gate"]),
    ),
    ToolSpec(
        "find_gates_by_name",
        "Find gates whose instance name contains a given substring (literal substring match, not "
        "glob/regex), optionally restricted to one gate type. Also records the match as the 'last found "
        "gates' set for a later do_replace_buf_with_and call that omits gate_names.",
        _schema(
            {
                "substring": _s("Substring to search for in gate instance names."),
                "gate_type": _s("Restrict to gates of this type (optional).", _GATE_TYPE_VALUES),
            },
            ["substring"],
        ),
    ),
    ToolSpec(
        "list_dffs_on_clock",
        "List every DFF (flip-flop) instance whose clock pin is driven by a named clock net.",
        _schema(
            {
                "clock": _s(
                    "Name of the clock signal (a bit-select, e.g. 'clk[0]', is accepted for "
                    "validation but matching is always by net NAME alone, ignoring bit index)."
                )
            },
            ["clock"],
        ),
    ),
    ToolSpec(
        "check_dffs_same_clock_domain",
        "Compare two or more DFF instances' clock (CK) nets and report whether they're all on the same "
        "clock domain, along with each DFF's resolved clock net name.",
        _schema({"dff_names": _sa("Two or more DFF instance names to compare.")}, ["dff_names"]),
    ),
    ToolSpec(
        "list_gates_with_constant_input",
        "List 2-input gates with at least one input tied to a constant 0/1, optionally restricted to one "
        "gate type and/or one constant value.",
        _schema(
            {
                "gate_type": _s("Restrict to this gate type (optional).", _GATE_TYPE_VALUES),
                "value": _i("Restrict to this constant value, 0 or 1 (optional)."),
            }
        ),
    ),
    ToolSpec(
        "get_direct_pi_po_connections",
        "List net-bits wired directly from a primary input to a primary output with zero gates in between.",
        _schema({}),
    ),
    ToolSpec(
        "get_gate_direct_fanout",
        "List the gates directly driven by a gate's output, and whether that output also drives a primary output.",
        _schema({"gate": _s(_GATE_DESC)}, ["gate"]),
    ),
    ToolSpec(
        "get_gate_fanout_count",
        "Count how many gates a gate's output directly drives.",
        _schema({"gate": _s(_GATE_DESC)}, ["gate"]),
    ),
    ToolSpec(
        "get_net_fanout",
        "Get the fanout count and the list of gates directly driven by a net (works for any net: primary input, "
        "internal signal, or gate output).",
        _schema({"net": _s(_NET_DESC)}, ["net"]),
    ),
    ToolSpec(
        "get_max_fanout_of_signal",
        "Find the bit of a (possibly multi-bit) named signal with the highest fanout, and that fanout count. "
        "A bit-selected `signal` narrows the search to that one bit only (its own fanout).",
        _schema({"signal": _s(_NET_OR_WHOLE_SIGNAL_DESC)}, ["signal"]),
    ),
    ToolSpec(
        "get_max_fanout_primary_input",
        "Find which primary input net-bit has the highest fanout in the design, and its count.",
        _schema({}),
    ),
    ToolSpec(
        "get_gates_connected_to_signal",
        "List every gate directly driven by a named signal (every bit of it, if bare) or by one "
        "specific bit of it (if bit-selected).",
        _schema({"signal": _s(_NET_OR_WHOLE_SIGNAL_DESC)}, ["signal"]),
    ),
    ToolSpec(
        "get_depth_of_cone",
        "Get the maximum logic depth (longest chain of gates) in a net's fanin cone.",
        _schema({"net": _s(_NET_DESC)}, ["net"]),
    ),
    ToolSpec(
        "get_depth_between",
        "Get the longest combinational path depth (and the gate sequence) from a source net to a target net; "
        "depth is null if no path exists.",
        _schema({"source": _s(_NET_DESC), "target": _s(_NET_DESC)}, ["source", "target"]),
    ),
    ToolSpec("get_max_design_depth", "Get the maximum combinational logic depth anywhere in the whole design.", _schema({})),
    ToolSpec(
        "get_max_reg_to_reg_depth",
        "Get the maximum combinational depth on any register-to-register (DFF.Q to DFF.D) path.",
        _schema({}),
    ),
    ToolSpec(
        "get_max_pi_to_dff_d_depth",
        "Get the maximum combinational depth from any primary input to any DFF's D pin.",
        _schema({}),
    ),
    ToolSpec(
        "check_gate_on_max_depth_path",
        "Check whether a gate lies on any maximum-depth combinational path of the whole design.",
        _schema({"gate": _s(_GATE_DESC)}, ["gate"]),
    ),
    ToolSpec(
        "count_outputs_over_depth",
        "Count primary outputs (and DFF D pins) whose fanin logic depth exceeds a threshold.",
        _schema({"threshold": _i("Depth threshold.")}, ["threshold"]),
    ),
    ToolSpec(
        "list_primary_outputs_over_cone_size",
        "List primary outputs (port-level PO net-bits only, no DFF D pins) whose fanin logic cone contains "
        "strictly more than threshold gates; 'count' in the result also answers 'how many'.",
        _schema({"threshold": _i("Gate-count threshold.")}, ["threshold"]),
    ),
    ToolSpec(
        "check_path_exists",
        "Check whether a combinational path exists from a source net to a target net, optionally avoiding "
        "(excluding) one intermediate net or gate from the graph. When 'avoid' is given, the result also "
        f"has {_EXISTS_IGNORING_AVOID_KEY!r}: the same check with 'avoid' NOT excluded, so "
        f"{{'exists': False, {_EXISTS_IGNORING_AVOID_KEY!r}: False}} means there was never a path between "
        f"source and target (avoid is not why), while {{'exists': False, {_EXISTS_IGNORING_AVOID_KEY!r}: "
        f"True}} means avoid blocks every path that exists. Without 'avoid' the result has only 'exists'.",
        _schema(
            {
                "source": _s(_NET_DESC),
                "target": _s(_NET_DESC),
                "avoid": _s(
                    "Net OR gate instance name to exclude from the graph (optional); a gate name resolves "
                    f"to that gate's output net, reported back in {_AVOID_RESOLVED_TO_KEY!r}."
                ),
            },
            ["source", "target"],
        ),
    ),
    ToolSpec(
        "count_paths",
        "Count the number of distinct combinational paths from a source net to a target net, optionally "
        "avoiding one intermediate net or gate. When 'avoid' is given, the result also has "
        f"{_COUNT_IGNORING_AVOID_KEY!r}: the same count with 'avoid' NOT excluded, so "
        f"{{'count': 0, {_COUNT_IGNORING_AVOID_KEY!r}: 0}} means there was never a path (avoid is not why), "
        f"while a nonzero {_COUNT_IGNORING_AVOID_KEY!r} alongside 'count': 0 means avoid blocks every one "
        "of them. Without 'avoid' the result has only 'count'.",
        _schema(
            {
                "source": _s(_NET_DESC),
                "target": _s(_NET_DESC),
                "avoid": _s(
                    "Net OR gate instance name to exclude from the graph (optional); a gate name resolves "
                    f"to that gate's output net, reported back in {_AVOID_RESOLVED_TO_KEY!r}."
                ),
            },
            ["source", "target"],
        ),
    ),
    ToolSpec(
        "enumerate_paths",
        "List the actual gate-sequence paths from a source net to a target net (capped at max_results; "
        f"'count' is always the true total). When 'avoid' is given (net or gate name), the result also has "
        f"{_COUNT_IGNORING_AVOID_KEY!r}: the true total with 'avoid' NOT excluded, same disambiguation as "
        "check_path_exists. Without 'avoid' the result is unchanged.",
        _schema(
            {
                "source": _s(_NET_DESC),
                "target": _s(_NET_DESC),
                "avoid": _s(
                    "Net OR gate instance name to exclude from the graph (optional); a gate name resolves "
                    f"to that gate's output net, reported back in {_AVOID_RESOLVED_TO_KEY!r}."
                ),
                "max_results": _i("Maximum number of paths to return (default 50, hard cap 500)."),
            },
            ["source", "target"],
        ),
    ),
    ToolSpec(
        "get_reg_to_reg_path_stats",
        "Count register-to-register combinational paths (DFF.Q through combinational logic to DFF.D) across "
        "the whole design: the true total (can be in the millions), plus the zero-gate direct-wire-connection "
        "count (DFF.Q wired straight into a DFF.D, reported separately, not included in the total) with a few "
        "examples.",
        _schema({}),
    ),
    ToolSpec(
        "get_cut_nets_between",
        "Find every net whose removal would disconnect a source net from a target net (articulation points).",
        _schema({"source": _s(_NET_DESC), "target": _s(_NET_DESC)}, ["source", "target"]),
    ),
    ToolSpec(
        "check_is_cut_signal",
        "Check whether a named signal is a cut (removing it disconnects some primary-input/primary-output "
        "pair). A bare signal name checks ANY bit of the signal; a bit-selected token checks that one bit only.",
        _schema({"signal": _s(_NET_OR_WHOLE_SIGNAL_DESC)}, ["signal"]),
    ),
    ToolSpec(
        "check_floating_signals",
        "Sweep the design for floating inputs (undriven nets read as a gate input) and unconnected output "
        "ports (declared PO with no driver), plus related structural checks (unused input ports, unconnected "
        "gate input pins, dangling never-consumed gate outputs, dead internal wire declarations). Returns "
        f"{{{_FLOATING_COUNT_KEY!r}, 'counted_in_that_number', "
        f"{_FLOATING_EXTRA_KEY!r}, 'note'}}: the count is ONLY floating inputs + unconnected "
        "output ports (the strict definition of this question); every other structural finding is reported "
        f"in {_FLOATING_EXTRA_KEY!r} and is NOT part of the count, so a count of 0 next to a "
        "non-empty additional-findings list is not a contradiction.",
        _schema({}),
    ),
    ToolSpec(
        "get_fanin_cone_size",
        "Count the gates in a net's transitive fanin (logic) cone.",
        _schema({"net": _s(_NET_DESC)}, ["net"]),
    ),
    ToolSpec(
        "get_fanin_cone_gates",
        "List the gate instances in a net's transitive fanin (logic) cone.",
        _schema({"net": _s(_NET_DESC)}, ["net"]),
    ),
    ToolSpec(
        "get_fanout_cone_gates",
        "List the gate instances in a net's transitive fanout cone.",
        _schema({"net": _s(_NET_DESC)}, ["net"]),
    ),
    ToolSpec(
        "get_largest_fanin_cone",
        "Find which output (primary output or DFF D pin) has the largest fanin cone, and its gate count.",
        _schema({}),
    ),
    ToolSpec(
        "get_cone_gate_type_breakdown",
        "Get a gate-type breakdown (counts per type) of a net's fanin cone. Returns "
        "{'net', 'cone_gates', 'by_type'}: 'cone_gates' is the total gate count in the cone (0 is a "
        "legitimate answer, e.g. the net is driven directly by a primary input -- not a failure), and "
        "'by_type' is empty exactly when 'cone_gates' is 0.",
        _schema({"net": _s(_NET_DESC)}, ["net"]),
    ),
    ToolSpec(
        "get_shared_fanin_gates",
        "List the gates shared between the fanin cones of two nets.",
        _schema({"net_a": _s(_NET_DESC), "net_b": _s(_NET_DESC)}, ["net_a", "net_b"]),
    ),
    ToolSpec(
        "check_equivalence_to_snapshot",
        "Check whether the current (possibly transformed) design is still functionally equivalent to the "
        "design as originally loaded from disk.",
        _schema({}),
    ),
    ToolSpec(
        "check_signal_equivalence",
        "Check whether two internal signals of the current design compute the identical Boolean function.",
        _schema({"net_a": _s(_NET_DESC), "net_b": _s(_NET_DESC)}, ["net_a", "net_b"]),
    ),
    ToolSpec(
        "check_symmetry_tool",
        "Check whether an output's function is symmetric (positive symmetry only) with respect to two named inputs.",
        _schema(
            {"output": _s(_NET_DESC), "input_a": _s(_NET_DESC), "input_b": _s(_NET_DESC)},
            ["output", "input_a", "input_b"],
        ),
    ),
    ToolSpec(
        "get_constant_value",
        "Check whether a net is provably constant (0 or 1) across every input/state assignment.",
        _schema({"net": _s(_NET_DESC)}, ["net"]),
    ),
    ToolSpec(
        "check_property_asserted_only_when",
        "Verify that a named output is asserted (1) only when a given condition holds, across every "
        "input and register state; if the property fails, returns a concrete counterexample assignment.",
        _schema(
            {
                "signal": _s(_NET_DESC),
                "condition": _s(
                    "One or more '<net> is <0|1|high|low>' literals joined by 'and'/'or' (an optional "
                    "leading 'both' is tolerated), e.g. 'req is 1 and busy is 0'."
                ),
            },
            ["signal", "condition"],
        ),
    ),
    ToolSpec(
        "find_signal_pair_for_operator",
        "Search for a pair of signals (a, b) already present in the design (a == b allowed; neither may "
        "be the target itself) such that OP(a, b) is functionally equivalent to a named target net. Uses "
        "bit-parallel random simulation to filter candidates on large designs, then formally verifies each "
        "surviving candidate before reporting it.",
        _schema(
            {"target": _s(_NET_DESC), "op": _s("Two-input Boolean operator.", list(SUPPORTED_OPS))},
            ["target", "op"],
        ),
    ),
    ToolSpec(
        "get_boolean_function",
        "Derive the Boolean-function description of a named output net: either its direct combinational "
        "equation, or -- when the target is a DFF's Q output (a registered/sequential bit) -- its D-pin "
        "next-state function instead, with an honest verdict on whether a primary-input-only equation "
        "exists at all (it never does when the target is itself a DFF.Q, and not always even for a "
        "combinational target -- its real support may include DFF.Q pseudo-inputs). No SOP minimization is "
        "performed; the expression is rendered structurally over the actual support, self-verified against "
        "exhaustive netlist simulation.",
        _schema({"net": _s(_NET_DESC)}, ["net"]),
    ),
    ToolSpec(
        "get_last_operation_summary",
        "Read back the counts the MOST RECENT operations in this session already reported (gates added/"
        "removed/merged/eliminated by the last transform, the last found-gates list, the last floating-"
        "signal count, the last enable/hold count, and the running functional-change-op total) WITHOUT "
        "rerunning anything. Use this to answer a follow-up like 'how many gates did that merge?' -- do "
        "NOT call the original do_*/check_* tool again just to recover a number it already returned, "
        "since calling it again repeats its effect (e.g. re-running a merge/dedup mutates the design "
        "again and reports a different, wrong count for what 'just happened'). A null field means NO "
        "such operation has run yet this session -- it does NOT mean the operation ran and found zero; "
        f"an operation that found zero reports 0. Do not quote a null as a count. {_UNAFFECTED_BY_THIS_TURNS_TOOL_CALLS_KEY!r} "
        f"is always true: these counters describe the user's PREVIOUS request, and are deliberately not "
        f"refreshed by any tool call made THIS turn (including a do_* tool this same turn already called) "
        f"-- seeing them unchanged right after running a mutating tool is correct, not stale.",
        _schema({}),
    ),
    ToolSpec(
        "rename_gate",
        "Rename a gate instance.",
        _schema({"old_name": _s("Current gate instance name."), "new_name": _s("New gate instance name.")}, ["old_name", "new_name"]),
    ),
    ToolSpec(
        "rename_signal",
        "Rename a signal (net/wire/port), updating every gate pin and port that references it.",
        _schema({"old_name": _s("Current signal name."), "new_name": _s("New signal name.")}, ["old_name", "new_name"]),
    ),
    ToolSpec(
        "do_limit_fanout_global",
        "Insert BUF gates wherever needed so that no net in the whole design drives more than max_fanout "
        "loads. " + _RERUN_NOTE_TEXT,
        _schema({"max_fanout": _i("Maximum allowed fanout."), "force": _b(_FORCE_DESC)}, ["max_fanout"]),
    ),
    ToolSpec(
        "do_limit_fanout_net",
        "Insert BUF gates so that the named net drives at most max_fanout loads. If net is a bare "
        "multi-bit signal name (no bit-select), every bit of that signal is capped independently. "
        "The result's bits_processed reports how many bits were swept (checked against the cap); "
        "bits_modified reports how many of those actually got a buffer (already-within-cap bits are "
        "swept but not modified, so buffers_added of 0 with bits_processed > 0 means every bit was "
        "already within the cap, not that the operation failed). " + _RERUN_NOTE_TEXT,
        _schema(
            {"net": _s(_NET_DESC), "max_fanout": _i("Maximum allowed fanout."), "force": _b(_FORCE_DESC)},
            ["net", "max_fanout"],
        ),
    ),
    ToolSpec(
        "do_insert_buffer_per_load",
        "Insert one dedicated BUF gate per existing load of a named net. If net is a bare multi-bit "
        "signal name (no bit-select), this is done for every bit of that signal. The result's "
        "bits_processed reports how many bits were swept; bits_modified reports how many of those "
        "actually got a buffer (a bit with no gate loads is swept but not modified). " + _RERUN_NOTE_TEXT,
        _schema({"net": _s(_NET_DESC), "force": _b(_FORCE_DESC)}, ["net"]),
    ),
    ToolSpec(
        "do_balance_depth_to_sinks",
        "Insert buffers so the logic depth from source to every net in sinks is equal (the max of their "
        "existing depths -- depth is only ever added, never removed). Equivalence-preserving. Exact "
        "minimum-buffer for a pure fanout tree; for a reconvergent DAG the result is valid but not "
        "guaranteed minimal (see the returned is_tree flag). " + _RERUN_NOTE_TEXT,
        _schema(
            {
                "source": _s(_NET_DESC),
                "sinks": _sa("Net references to balance the depth to, e.g. ['B', 'C', 'D']."),
                "force": _b(_FORCE_DESC),
            },
            ["source", "sinks"],
        ),
    ),
    ToolSpec(
        "do_replace_buf_with_and",
        "Rewrite BUF gates in place into 2-input AND gates: each keeps its original input on I0 and gets "
        "ctrl_net wired to I1; the output net is left untouched (a functional change, not an "
        "equivalence-preserving transform). If gate_names is omitted, targets whatever the most recent "
        "find_gates_by_name call found. Returns {'replaced', 'scope', 'names_in_scope', "
        "'buf_candidates_in_scope', 'skipped_self_loop'}: 'buf_candidates_in_scope' is how many of the "
        "named gates actually resolved to a BUF BEFORE rewriting -- 'replaced: 0' paired with "
        "'buf_candidates_in_scope: 0' means the requested scope was understood and genuinely has no BUF "
        "gates in it, not that the call failed. 'skipped_self_loop' lists named gates that ARE BUFs but "
        "were left alone because ctrl_net is that gate's own output net (rewiring it would make a direct "
        "combinational self-loop), so 'replaced: 0' with a non-empty 'skipped_self_loop' is a third, "
        "distinct outcome -- not 'no BUFs here' and not a failure. " + _RERUN_NOTE_TEXT,
        _schema(
            {
                "ctrl_net": _s(_NET_DESC),
                "gate_names": _sa(
                    "Exact gate instance names to rewrite (optional; defaults to the most recent "
                    "find_gates_by_name result)."
                ),
                "force": _b(_FORCE_DESC),
            },
            ["ctrl_net"],
        ),
    ),
    ToolSpec(
        "do_remove_dangling_gates",
        "Remove every gate that cannot reach a primary output or a DFF pin (dead logic sweep). " + _RERUN_NOTE_TEXT,
        _schema({"force": _b(_FORCE_DESC)}),
    ),
    ToolSpec(
        "do_deduplicate_gates",
        "Merge structurally duplicate gates (same type, same inputs) into one survivor each. " + _RERUN_NOTE_TEXT,
        _schema({"force": _b(_FORCE_DESC)}),
    ),
    ToolSpec(
        "do_collapse_double_inverters",
        "Splice out back-to-back inverter (NOT-NOT) pairs. " + _RERUN_NOTE_TEXT,
        _schema({"force": _b(_FORCE_DESC)}),
    ),
    ToolSpec(
        "do_collapse_inverter_buffer_chains",
        "Splice out every BUF directly fed by a NOT gate's output (BUF(NOT(x)) == NOT(x)), leaving a single "
        "inverter -- functionally equivalence-preserving, unlike do_replace_buf_with_and. " + _RERUN_NOTE_TEXT,
        _schema({"force": _b(_FORCE_DESC)}),
    ),
    ToolSpec(
        "do_simplify_constant_inputs",
        "Fold 2-input gates that have a constant-tied input, per their truth table, optionally restricted to a "
        "list of gate types. " + _RERUN_NOTE_TEXT,
        _schema({"gate_types": _sa("Restrict to these gate types (optional)."), "force": _b(_FORCE_DESC)}),
    ),
    ToolSpec(
        "do_remap_to_basis",
        "Replace gates with equivalent subcircuits built only from a given gate basis ('and', 'nand', or 'nor', "
        "each combined with NOT), optionally restricted to one source gate type and/or one net's fanin cone. "
        "NOTE: this tool's 'basis' vocabulary ('and'/'nand'/'nor', also accepting the long forms "
        "'and_not'/'nand_not'/'nor_not') is DIFFERENT from the one used by do_optimize_depth/do_optimize_cone_depth/"
        "do_optimize_gate_count/do_optimize_cone_gate_count ('and_not'/'and_or_not'/'nand_not'/'nor_not') -- 'and_or_not' "
        "has no equivalent here. Returns {'replaced', 'scope', 'gates_in_scope'}: 'gates_in_scope' is how many gates "
        "the restriction (cone/type) actually matched BEFORE remapping -- 'replaced: 0' paired with 'gates_in_scope: 0' "
        "means the requested scope was understood and is genuinely empty, not that the call failed. " + _RERUN_NOTE_TEXT,
        _schema(
            {
                "basis": _s("Target basis.", _BASIS_VALUES),
                "gate_type": _s("Restrict to gates of this source type (optional).", _GATE_TYPE_VALUES),
                "cone_root": _s("Restrict to this net's fanin cone (optional)."),
                "force": _b(_FORCE_DESC),
            },
            ["basis"],
        ),
    ),
    ToolSpec(
        "do_optimize_depth",
        "Reduce the design's maximum combinational logic depth (critical path) via ABC-backed logic "
        "restructuring, preserving functional equivalence exactly. Optionally restricts the resynthesized "
        "logic to a fixed gate basis. Only actually changes the design if a verified-equivalent, "
        "genuinely-better result was found; otherwise reports the design as already optimal and leaves it "
        "untouched.",
        _schema({"basis": _s("Restrict resynthesized logic to this gate basis: one of 'and_not', 'and_or_not', 'nand_not', 'nor_not' (optional; unrestricted if omitted). NOT the same vocabulary as do_remap_to_basis's 'basis' -- see that tool's description.", _DEPTH_BASIS_VALUES)}),
    ),
    ToolSpec(
        "do_optimize_cone_depth",
        "Reduce the maximum combinational depth of one net's fanin cone via ABC-backed logic restructuring, "
        "leaving every other gate in the design untouched. Optionally restricts the resynthesized cone to a "
        "fixed gate basis. Only actually changes the design if a verified-equivalent, genuinely-better result "
        "was found; otherwise reports the cone as already optimal and leaves the design untouched.",
        _schema(
            {
                "net": _s(_NET_DESC),
                "basis": _s("Restrict the resynthesized cone to this gate basis: one of 'and_not', 'and_or_not', 'nand_not', 'nor_not' (optional; unrestricted if omitted). NOT the same vocabulary as do_remap_to_basis's 'basis' -- see that tool's description.", _DEPTH_BASIS_VALUES),
            },
            ["net"],
        ),
    ),
    ToolSpec(
        "do_optimize_gate_count",
        "Reduce the design's total gate count via ABC-backed logic restructuring, preserving functional "
        "equivalence exactly. Optionally restricts the resynthesized logic to a fixed gate basis, and "
        "optionally enforces a hard maximum on the resulting maximum combinational depth (a candidate that "
        "would exceed it is rejected, never accepted). Only actually changes the design if a genuinely "
        "smaller, verified-equivalent result was found; otherwise reports the design as unchanged.",
        _schema(
            {
                "basis": _s("Restrict resynthesized logic to this gate basis: one of 'and_not', 'and_or_not', 'nand_not', 'nor_not' (optional; unrestricted if omitted). NOT the same vocabulary as do_remap_to_basis's 'basis' -- see that tool's description.", _DEPTH_BASIS_VALUES),
                "max_depth": _i("Hard ceiling on the resulting maximum design depth (optional; no cap if omitted)."),
            }
        ),
    ),
    ToolSpec(
        "do_optimize_cone_gate_count",
        "Reduce the total gate count via ABC-backed logic restructuring of one net's fanin cone, leaving "
        "every other gate in the design untouched. Optionally restricts the resynthesized cone to a fixed "
        "gate basis, and optionally enforces a hard maximum on the resulting depth of that cone (a candidate "
        "that would exceed it is rejected, never accepted). Only actually changes the design if a genuinely "
        "smaller, verified-equivalent result was found; otherwise reports the design as unchanged.",
        _schema(
            {
                "net": _s(_NET_DESC),
                "basis": _s("Restrict the resynthesized cone to this gate basis: one of 'and_not', 'and_or_not', 'nand_not', 'nor_not' (optional; unrestricted if omitted). NOT the same vocabulary as do_remap_to_basis's 'basis' -- see that tool's description.", _DEPTH_BASIS_VALUES),
                "max_depth": _i("Hard ceiling on the resulting depth of this net's cone (optional; no cap if omitted)."),
            },
            ["net"],
        ),
    ),
]

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "load_design": load_design,
    "write_design": write_design,
    "set_testcase": set_testcase,
    "count_gates_by_type": count_gates_by_type,
    "count_primary_ports": count_primary_ports,
    "list_primary_ports": list_primary_ports,
    "list_gates_of_type": list_gates_of_type,
    "get_gate_info": get_gate_info,
    "list_dffs_on_clock": list_dffs_on_clock,
    "check_dffs_same_clock_domain": check_dffs_same_clock_domain,
    "list_gates_with_constant_input": list_gates_with_constant_input,
    "get_direct_pi_po_connections": get_direct_pi_po_connections,
    "get_gate_direct_fanout": get_gate_direct_fanout,
    "get_gate_fanout_count": get_gate_fanout_count,
    "get_net_fanout": get_net_fanout,
    "get_max_fanout_of_signal": get_max_fanout_of_signal,
    "get_max_fanout_primary_input": get_max_fanout_primary_input,
    "get_gates_connected_to_signal": get_gates_connected_to_signal,
    "get_depth_of_cone": get_depth_of_cone,
    "get_depth_between": get_depth_between,
    "get_max_design_depth": get_max_design_depth,
    "get_max_reg_to_reg_depth": get_max_reg_to_reg_depth,
    "get_max_pi_to_dff_d_depth": get_max_pi_to_dff_d_depth,
    "check_gate_on_max_depth_path": check_gate_on_max_depth_path,
    "count_outputs_over_depth": count_outputs_over_depth,
    "list_primary_outputs_over_cone_size": list_primary_outputs_over_cone_size,
    "check_path_exists": check_path_exists,
    "count_paths": count_paths,
    "enumerate_paths": enumerate_paths,
    "get_reg_to_reg_path_stats": get_reg_to_reg_path_stats,
    "get_cut_nets_between": get_cut_nets_between,
    "check_is_cut_signal": check_is_cut_signal,
    "check_floating_signals": check_floating_signals,
    "get_fanin_cone_size": get_fanin_cone_size,
    "get_fanin_cone_gates": get_fanin_cone_gates,
    "get_fanout_cone_gates": get_fanout_cone_gates,
    "get_largest_fanin_cone": get_largest_fanin_cone,
    "get_cone_gate_type_breakdown": get_cone_gate_type_breakdown,
    "get_shared_fanin_gates": get_shared_fanin_gates,
    "check_equivalence_to_snapshot": check_equivalence_to_snapshot,
    "check_signal_equivalence": check_signal_equivalence,
    "check_symmetry_tool": check_symmetry_tool,
    "get_constant_value": get_constant_value,
    "check_property_asserted_only_when": check_property_asserted_only_when,
    "find_signal_pair_for_operator": find_signal_pair_for_operator,
    "get_boolean_function": get_boolean_function,
    "get_last_operation_summary": get_last_operation_summary,
    "rename_gate": rename_gate,
    "rename_signal": rename_signal,
    "do_limit_fanout_global": do_limit_fanout_global,
    "do_limit_fanout_net": do_limit_fanout_net,
    "do_insert_buffer_per_load": do_insert_buffer_per_load,
    "do_balance_depth_to_sinks": do_balance_depth_to_sinks,
    "find_gates_by_name": find_gates_by_name,
    "do_replace_buf_with_and": do_replace_buf_with_and,
    "do_remove_dangling_gates": do_remove_dangling_gates,
    "do_deduplicate_gates": do_deduplicate_gates,
    "do_collapse_double_inverters": do_collapse_double_inverters,
    "do_collapse_inverter_buffer_chains": do_collapse_inverter_buffer_chains,
    "do_simplify_constant_inputs": do_simplify_constant_inputs,
    "do_remap_to_basis": do_remap_to_basis,
    "do_optimize_depth": do_optimize_depth,
    "do_optimize_cone_depth": do_optimize_cone_depth,
    "do_optimize_gate_count": do_optimize_gate_count,
    "do_optimize_cone_gate_count": do_optimize_cone_gate_count,
}

assert {t.name for t in TOOL_SCHEMA} == set(TOOL_REGISTRY), "TOOL_SCHEMA/TOOL_REGISTRY name mismatch"
