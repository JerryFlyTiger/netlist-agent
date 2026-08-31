"""Unit tests for netlist_agent/llm/tools_schema.py: schema well-formedness,
plus a representative slice of the tool registry called directly against a
small synthetic Design (mirroring tests/test_router.py's fixture and
ground-truth-via-direct-call style for the ABC-backed checks), and error
behavior on bad gate/signal/net references.
"""

from __future__ import annotations

import inspect
import json
import typing
from typing import Any

import pytest

from netlist_agent.abc_bridge import are_equivalent, check_symmetry, is_constant
from netlist_agent.analysis import find_floating_signals
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import Const, Design, Direction, Gate, GateType, NetBit, Port, Signal, design_fingerprint
from netlist_agent.llm.tools_schema import (
    TOOL_REGISTRY,
    TOOL_SCHEMA,
    ToolError,
    _FORCE_OVERRIDE_PARAM_KEY,
    _FORCE_PARAM_NAME,
    _MUTATION_PERFORMED_KEY,
    _PREVIOUSLY_REPORTED_COUNT_KEY,
    _REFUSED_REASON_KEY,
    _RERUN_OF_PRIOR_OPERATION_KEY,
    _UNAFFECTED_BY_THIS_TURNS_TOOL_CALLS_KEY,
)
from netlist_agent.netref import netbit_token
from netlist_agent.parser import parse_verilog
from netlist_agent import router as _router_module
from netlist_agent.router import handle_request

_HAS_REAL_ROUTER = hasattr(_router_module, "route")
from netlist_agent.session import Session
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


def _nb(name: str, bit: int | None = None) -> NetBit:
    return NetBit(name, bit)


def _build_design() -> Design:
    """Same fixture shape as tests/test_router.py's `_build_design`: a small
    synthetic design with buses, constants, DFFs, and enough combinational
    structure to exercise fanin/fanout/depth/path/cone/ABC queries."""
    design = Design(module_name="top")
    design.signals["n0"] = Signal(name="n0", msb=1, lsb=0, direction=Direction.INPUT)
    for name in ("n2", "clk", "rn"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    for name in ("n20", "n21", "n26"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.OUTPUT)
    for name in ("n10", "n11", "n12", "n13", "n14", "n15", "n23", "n25"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)

    port_order = ["n0", "n2", "clk", "rn", "n20", "n21", "n26"]
    design.ports = [Port(name=n, direction=design.signals[n].direction) for n in port_order]

    design.gates = [
        Gate("g0", GateType.AND, {"O": _nb("n10"), "I0": _nb("n0", 0), "I1": _nb("n0", 1)}),
        Gate("g1", GateType.NAND, {"O": _nb("n11"), "I0": _nb("n10"), "I1": _nb("n2")}),
        Gate("g2", GateType.NOT, {"O": _nb("n12"), "I0": _nb("n10")}),
        Gate("g3", GateType.NOT, {"O": _nb("n13"), "I0": _nb("n12")}),
        Gate("g4", GateType.BUF, {"O": _nb("n20"), "I0": _nb("n13")}),
        Gate("g5", GateType.AND, {"O": _nb("n14"), "I0": _nb("n10"), "I1": Const.ZERO}),
        Gate("g6", GateType.NAND, {"O": _nb("n15"), "I0": _nb("n2"), "I1": Const.ONE}),
        Gate("g12", GateType.AND, {"O": _nb("n25"), "I0": _nb("n0", 0), "I1": _nb("n0", 1)}),
        Gate("g13", GateType.BUF, {"O": _nb("n26"), "I0": _nb("n25")}),
        Gate(
            "dff0",
            GateType.DFF,
            {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("n11"), "Q": _nb("n23")},
        ),
    ]
    # n21 is left with no driver at all deliberately, so it participates as a
    # PO with a zero fanin cone (get_largest_fanin_cone/get_fanin_cone_size
    # exercise this edge without an extra gate).
    design.build_indices()
    return design


def _fixture_path(tmp_path) -> str:
    path = str(tmp_path / "fixture.v")
    write_verilog(_build_design(), path)
    return path


def _new_session(tmp_path) -> Session:
    path = _fixture_path(tmp_path)
    session = Session()
    session.current_design = parse_verilog(path)
    session.original_snapshot = parse_verilog(path)
    session.load_dir = str(tmp_path)
    return session


# ----------------------------------------------------------------------
# Schema well-formedness
# ----------------------------------------------------------------------


def test_schema_names_match_registry() -> None:
    schema_names = {t.name for t in TOOL_SCHEMA}
    assert schema_names == set(TOOL_REGISTRY)
    assert len(TOOL_SCHEMA) == len(schema_names), "duplicate tool name in TOOL_SCHEMA"


@pytest.mark.parametrize("spec", TOOL_SCHEMA, ids=[t.name for t in TOOL_SCHEMA])
def test_schema_entry_well_formed(spec) -> None:
    assert spec.name and isinstance(spec.name, str)
    assert spec.description and isinstance(spec.description, str)
    params = spec.parameters
    assert params.get("type") == "object"
    assert isinstance(params.get("properties"), dict)
    required = params.get("required", [])
    assert isinstance(required, list)
    for r in required:
        assert r in params["properties"], f"{spec.name}: required field {r!r} not in properties"
    for prop_name, prop_schema in params["properties"].items():
        assert isinstance(prop_schema, dict)
        assert "type" in prop_schema
        assert "description" in prop_schema
    assert callable(TOOL_REGISTRY[spec.name])


# `session` is the one implicit parameter every tool callable takes that is
# never part of its ToolSpec.parameters -- client.py injects it, the model
# never supplies it. See tools_schema.py's module docstring: every callable
# has the form `(session, **kwargs) -> JSON-serializable result`.
_INJECTED_PARAMS = {"session"}

# Every non-injected tool parameter in tools_schema.py is one of these three
# resolved-annotation shapes (checked exhaustively against the module's own
# source by hand, not assumed): bare `str`/`int`, `list[str]`, and
# `Optional[...]` around either, plus the mutating tools' `bool` force flag.
# There is no `float`/nested-Optional usage, so this map is narrow rather
# than a general typing-to-JSON-schema translator -- a parameter shape this
# doesn't recognize should fail loudly (see the `else` branch below) rather
# than silently pass.
_PY_SCALAR_TO_JSON_TYPE = {str: "string", int: "integer", bool: "boolean"}


def _expected_json_type(annotation: object) -> str:
    """Reduce a RESOLVED (not the postponed-string form -- see the
    `typing.get_type_hints` call below) Python annotation to the single JSON
    schema "type" string tools_schema.py's `_s`/`_i`/`_sa` helpers can
    produce: `Optional[T]` unwraps to T's type (these tools use a `None`
    default to signal "the model may omit this", not a JSON-schema nullable
    type -- `_schema()` never emits one), and `list[T]` becomes "array"
    regardless of T (every list parameter here is `list[str]`, and `_sa()`
    only ever produces `{"type": "array", "items": {"type": "string"}}`).
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        assert len(non_none) == 1, f"unsupported Optional/Union shape: {annotation!r}"
        return _expected_json_type(non_none[0])
    if origin is list:
        return "array"
    if annotation in _PY_SCALAR_TO_JSON_TYPE:
        return _PY_SCALAR_TO_JSON_TYPE[annotation]
    raise AssertionError(
        f"no JSON-type mapping registered for Python annotation {annotation!r} -- "
        "either _PY_SCALAR_TO_JSON_TYPE/the list/Optional handling above needs "
        "widening, or this parameter's annotation is a typo"
    )


@pytest.mark.parametrize("spec", TOOL_SCHEMA, ids=[t.name for t in TOOL_SCHEMA])
def test_schema_parameters_match_callable_signature(spec) -> None:
    """The LLM builds tool-call arguments strictly from ToolSpec.parameters
    (JSON schema): a name or required/optional mismatch against the actual
    callable's signature means a tool call that schema-validates cleanly
    still raises TypeError at execution time -- caught by client.py and
    turned into an opaque `{"error": ...}` the model has to somehow recover
    from, rather than a crash anyone would notice. No automated test
    exercised this path before, so nothing here caught a mismatch by
    accident; this test checks it directly, tool by tool."""
    fn = TOOL_REGISTRY[spec.name]
    sig = inspect.signature(fn)
    func_params = list(sig.parameters.values())

    assert func_params and func_params[0].name in _INJECTED_PARAMS, (
        f"{spec.name}: first parameter must be the injected 'session' param, "
        f"got {func_params[0].name if func_params else '<none>'}"
    )

    var_positional = [p.name for p in func_params if p.kind == inspect.Parameter.VAR_POSITIONAL]
    assert not var_positional, f"{spec.name}: *args ({var_positional}) defeats schema/signature comparison"
    var_keyword = [p.name for p in func_params if p.kind == inspect.Parameter.VAR_KEYWORD]
    assert not var_keyword, f"{spec.name}: **kwargs ({var_keyword}) defeats schema/signature comparison"

    non_injected = [p for p in func_params if p.name not in _INJECTED_PARAMS]
    func_names = {p.name for p in non_injected}
    func_required = {p.name for p in non_injected if p.default is inspect.Parameter.empty}

    schema_props = set(spec.parameters.get("properties", {}).keys())
    schema_required = set(spec.parameters.get("required", []))

    extra_in_schema = schema_props - func_names
    assert not extra_in_schema, (
        f"{spec.name}: schema declares parameter(s) {sorted(extra_in_schema)} the callable doesn't accept "
        "-- a model-issued tool call using them would TypeError"
    )
    extra_in_func = func_names - schema_props
    assert not extra_in_func, (
        f"{spec.name}: callable accepts parameter(s) {sorted(extra_in_func)} the schema doesn't declare "
        "-- the model can never supply them"
    )

    missing_required = func_required - schema_required
    assert not missing_required, (
        f"{spec.name}: callable requires (no default) parameter(s) {sorted(missing_required)} that the "
        "schema doesn't mark required -- a model-issued call omitting them would TypeError"
    )
    over_required = schema_required - func_required
    assert not over_required, (
        f"{spec.name}: schema marks parameter(s) {sorted(over_required)} required but the callable has a "
        "default for them"
    )

    # The comparisons above only ever look at parameter NAMES -- a schema
    # declaring "threshold" as a string while the callable takes `threshold:
    # int` passes every check so far, then TypeErrors (or worse, silently
    # coerces) the first time the model sends a numeric-looking string. This
    # was checked once by hand while writing this test (0 mismatches) but
    # never turned into an assertion, so it could regress unnoticed the next
    # time a parameter's annotation changed without its schema entry
    # following -- `typing.get_type_hints` (not `p.annotation` from the
    # `inspect.signature` above) is required here specifically because
    # tools_schema.py has `from __future__ import annotations`, which makes
    # every annotation a postponed, unevaluated STRING ("Optional[str]") --
    # comparing that string against the schema's own JSON string would look
    # like it works and then silently compare nothing meaningful.
    hints = typing.get_type_hints(fn)
    type_mismatches = []
    for name in sorted(schema_props):
        prop_schema = spec.parameters["properties"][name]
        schema_type = prop_schema.get("type")
        expected_type = _expected_json_type(hints[name])
        if schema_type != expected_type:
            type_mismatches.append((name, schema_type, expected_type))
    assert not type_mismatches, (
        f"{spec.name}: schema/annotation type mismatch, (param, schema \"type\", type the callable's own "
        f"annotation implies): {type_mismatches}"
    )


# ----------------------------------------------------------------------
# Whole-registry smoke test: every tool in TOOL_REGISTRY called once through
# the registry (not the underlying analysis.py/transform.py primitives
# directly) with a legal argument set, checking the call succeeds and the
# result is JSON-serializable. Before this, schema/signature consistency was
# checked for all 65 tools (above), but actual return-shape/error behavior
# was only ever exercised for a representative slice -- 12 tools had zero
# coverage via TOOL_REGISTRY at all (grep confirmed 0 hits for each).
# ----------------------------------------------------------------------


def _build_two_dff_design() -> Design:
    """Minimal two-DFF design for check_dffs_same_clock_domain, which raises
    ToolError on fewer than two DFF instances -- the main `_build_design()`
    fixture above only has one (dff0), so that tool needs its own fixture."""
    design = Design(module_name="top")
    for name in ("clk", "rn", "d0", "d1"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    for name in ("q0", "q1"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports = [
        Port("clk", Direction.INPUT),
        Port("rn", Direction.INPUT),
        Port("d0", Direction.INPUT),
        Port("d1", Direction.INPUT),
        Port("q0", Direction.OUTPUT),
        Port("q1", Direction.OUTPUT),
    ]
    design.gates = [
        Gate("dff0", GateType.DFF, {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("d0"), "Q": _nb("q0")}),
        Gate("dff1", GateType.DFF, {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("d1"), "Q": _nb("q1")}),
    ]
    design.build_indices()
    return design


def _new_two_dff_session(tmp_path) -> Session:
    path = str(tmp_path / "two_dff.v")
    write_verilog(_build_two_dff_design(), path)
    session = Session()
    session.current_design = parse_verilog(path)
    session.original_snapshot = parse_verilog(path)
    session.load_dir = str(tmp_path)
    return session


# Value to supply for each required parameter NAME, looked up generically
# across all 65 tools -- every value below is chosen to be valid against
# `_build_design()` (the main fixture): "n10" is a scalar internal net,
# "n0[0]" -> "n20" is an actual combinational path (g0 -> n10 -> g2 -> n12 ->
# g3 -> n13 -> g4 -> n20), "n10"/"n11" share gate g0 in their fanin cones,
# etc. "directory" is deliberately absent -- it is tmp_path-dependent, so
# `_args_for_spec` fills it in per-test instead of from this static table.
_PARAM_VALUES: dict[str, Any] = {
    "filename": "fixture.v",
    "case_name": "probe_case",
    "direction": "input",
    "gate_type": "and",
    "gate": "g0",
    "substring": "g0",
    "clock": "clk",
    "dff_names": ["dff0", "dff1"],
    "net": "n10",
    "signal": "n10",
    "source": "n0[0]",
    "target": "n20",
    "threshold": 0,
    "net_a": "n10",
    "net_b": "n11",
    "output": "n11",
    "input_a": "n0[0]",
    "input_b": "n2",
    "condition": "n2 is 1",
    "op": "AND",
    "old_name": "g0",
    "new_name": "g0_renamed",
    "max_fanout": 2,
    "sinks": ["n12"],
    "ctrl_net": "n2",
    "basis": "and",
}

# Per-tool overrides where the generic table above doesn't fit: rename_signal
# needs old_name/new_name to be SIGNAL names (the generic table's are gate
# names, for rename_gate), do_balance_depth_to_sinks needs a source whose
# cone actually contains its sink, and check_property_asserted_only_when
# reads more naturally against an actual primary output.
_TOOL_ARG_OVERRIDES: dict[str, dict[str, Any]] = {
    "rename_signal": {"old_name": "n14", "new_name": "n14_renamed"},
    "do_balance_depth_to_sinks": {"source": "n10", "sinks": ["n12"]},
    "check_property_asserted_only_when": {"signal": "n20", "condition": "n2 is 1"},
}


def _args_for_spec(spec, tmp_path) -> dict[str, Any]:
    overrides = _TOOL_ARG_OVERRIDES.get(spec.name, {})
    args: dict[str, Any] = {}
    for name in spec.parameters.get("required", []):
        if name in overrides:
            args[name] = overrides[name]
        elif name == "directory":
            args[name] = str(tmp_path)
        else:
            args[name] = _PARAM_VALUES[name]
    return args


@pytest.mark.parametrize("spec", TOOL_SCHEMA, ids=[t.name for t in TOOL_SCHEMA])
def test_tool_registry_call_returns_serializable_dict(spec, tmp_path, monkeypatch) -> None:
    """Every tool in TOOL_REGISTRY, called once through the registry (not
    the underlying analysis.py/transform.py primitive directly) with a
    legal argument set built from the schema's own declared required
    parameters. Confirms two things no other test in this file checks for
    the full 65: the call actually succeeds, and the successful result is a
    JSON-serializable dict -- client.py serializes every tool result before
    feeding it back to the model, so a shape that survives Python but not
    json.dumps is broken in production regardless of what any other test
    thinks of it."""
    # set_testcase opens its log file relative to the CWD (Session.start()),
    # same as tests/test_router.py's begin-testcase tests -- chdir into
    # tmp_path so that doesn't drop a "probe_case.log" into the repo root.
    monkeypatch.chdir(tmp_path)
    if spec.name == "check_dffs_same_clock_domain":
        session = _new_two_dff_session(tmp_path)
    else:
        session = _new_session(tmp_path)
    args = _args_for_spec(spec, tmp_path)
    fn = TOOL_REGISTRY[spec.name]
    result = fn(session, **args)
    assert isinstance(result, dict), f"{spec.name}: result is a {type(result).__name__}, not a dict"
    json.dumps(result)  # must not raise -- this is what actually reaches the model


# ----------------------------------------------------------------------
# Load / write
# ----------------------------------------------------------------------


def test_load_and_write_design(tmp_path) -> None:
    session = Session()
    path = _fixture_path(tmp_path)
    result = TOOL_REGISTRY["load_design"](session, filename="fixture.v", directory=str(tmp_path))
    assert result["module_name"] == "top"
    assert session.current_design is not None
    assert session.original_snapshot is not None

    out = TOOL_REGISTRY["write_design"](session, filename="out.v")
    assert out["path"] == str(tmp_path / "out.v")
    reparsed = parse_verilog(out["path"])
    assert reparsed.module_name == "top"


def test_tool_without_loaded_design_raises() -> None:
    session = Session()
    with pytest.raises(ToolError):
        TOOL_REGISTRY["count_gates_by_type"](session)


# ----------------------------------------------------------------------
# Counting / listing
# ----------------------------------------------------------------------


def test_count_gates_by_type(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["count_gates_by_type"](session)
    assert result["total"] == len(session.current_design.gates)
    assert result["by_type"]["AND"] == 3
    assert result["by_type"]["DFF"] == 1


def test_count_primary_ports(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["count_primary_ports"](session)
    assert result["primary_input_ports"] == 4
    assert result["primary_output_ports"] == 3
    assert result["primary_input_bits"] == 5  # n0 is 2 bits + n2/clk/rn


def test_list_primary_ports(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["list_primary_ports"](session, direction="input")
    assert result["count"] == 4
    assert any("n0 (2 bits)" == item for item in result["items"])
    with pytest.raises(ToolError):
        TOOL_REGISTRY["list_primary_ports"](session, direction="sideways")


def test_list_gates_of_type(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["list_gates_of_type"](session, gate_type="and")
    assert result["count"] == 3
    names = {g["name"] for g in result["gates"]}
    assert names == {"g0", "g5", "g12"}
    with pytest.raises(ToolError):
        TOOL_REGISTRY["list_gates_of_type"](session, gate_type="not-a-gate-type")


def test_get_gate_info(tmp_path) -> None:
    session = _new_session(tmp_path)
    info = TOOL_REGISTRY["get_gate_info"](session, gate="g0")
    assert info["type"] == "and"
    assert info["pins"]["I0"] == "n0[0]"
    assert info["pins"]["I1"] == "n0[1]"
    with pytest.raises(ToolError):
        TOOL_REGISTRY["get_gate_info"](session, gate="no_such_gate")


def test_list_gates_with_constant_input(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["list_gates_with_constant_input"](session, gate_type="nand", value=1)
    assert result["items"] == ["g6"]
    result_any = TOOL_REGISTRY["list_gates_with_constant_input"](session)
    assert set(result_any["items"]) == {"g5", "g6"}


# ----------------------------------------------------------------------
# Fanin / fanout
# ----------------------------------------------------------------------


def test_gate_direct_fanout_and_count(tmp_path) -> None:
    session = _new_session(tmp_path)
    fanout = TOOL_REGISTRY["get_gate_direct_fanout"](session, gate="g0")
    assert set(fanout["gates"]) == {"g1", "g2", "g5"}
    assert fanout["drives_primary_output"] is False
    count = TOOL_REGISTRY["get_gate_fanout_count"](session, gate="g0")
    assert count["count"] == 3
    with pytest.raises(ToolError):
        TOOL_REGISTRY["get_gate_fanout_count"](session, gate="ghost")


def test_get_net_fanout(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["get_net_fanout"](session, net="n10")
    assert result["count"] == 3
    assert set(result["gates"]) == {"g1", "g2", "g5"}
    with pytest.raises(ToolError):
        TOOL_REGISTRY["get_net_fanout"](session, net="not[a[net")


def test_get_max_fanout_of_signal_and_pi(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["get_max_fanout_of_signal"](session, signal="n10")
    assert result["net"] == "n10"
    assert result["count"] == 3
    with pytest.raises(ToolError):
        TOOL_REGISTRY["get_max_fanout_of_signal"](session, signal="no_such_signal")

    pi_result = TOOL_REGISTRY["get_max_fanout_primary_input"](session)
    # n0[0], n0[1], and n2 all tie at fanout 2 -- which one wins is an
    # iteration-order artifact of a set, not asserted here.
    assert pi_result["net"] in {"n0[0]", "n0[1]", "n2"}
    assert pi_result["count"] == 2


def test_list_dffs_on_clock(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["list_dffs_on_clock"](session, clock="clk")
    assert result["items"] == ["dff0"]


def test_get_gates_connected_to_signal(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["get_gates_connected_to_signal"](session, signal="n10")
    assert set(result["items"]) == {"g1", "g2", "g5"}
    with pytest.raises(ToolError):
        TOOL_REGISTRY["get_gates_connected_to_signal"](session, signal="ghost_signal")


# ----------------------------------------------------------------------
# Depth
# ----------------------------------------------------------------------


def test_depth_tools(tmp_path) -> None:
    session = _new_session(tmp_path)
    # n20 <- g4(BUF) <- g3(NOT) <- g2(NOT) <- g0(AND) <- n0[0]/n0[1]: depth 4.
    assert TOOL_REGISTRY["get_depth_of_cone"](session, net="n20")["depth"] == 4
    between = TOOL_REGISTRY["get_depth_between"](session, source="n0[0]", target="n20")
    assert between["depth"] == 4
    assert between["path"] == ["g0", "g2", "g3", "g4"]
    assert TOOL_REGISTRY["get_max_design_depth"](session)["depth"] >= 4
    assert TOOL_REGISTRY["count_outputs_over_depth"](session, threshold=2)["count"] >= 1

    unreachable = TOOL_REGISTRY["get_depth_between"](session, source="n21", target="n20")
    assert unreachable["depth"] is None
    assert unreachable["path"] is None


def test_check_gate_on_max_depth_path(tmp_path) -> None:
    # g0 -> g2 -> g3 -> g4 -> n20 is depth 4, the design's max (see
    # test_depth_tools), so g0 must be reported as on a max-depth path.
    session = _new_session(tmp_path)
    graph = NetlistGraph(session.current_design)
    max_depth = graph.max_design_depth()
    result = TOOL_REGISTRY["check_gate_on_max_depth_path"](session, gate="g0")
    assert result["on_max_depth_path"] is True
    assert result["depth_through_gate"] == graph.depth_through_gate("g0")
    assert result["max_design_depth"] == max_depth

    # g6's longest path (n2/const -> g6 -> nothing further) is far shorter
    # than the design's max, so it must NOT be reported as on a max-depth
    # path.
    result_g6 = TOOL_REGISTRY["check_gate_on_max_depth_path"](session, gate="g6")
    assert result_g6["on_max_depth_path"] is False

    with pytest.raises(ToolError):
        TOOL_REGISTRY["check_gate_on_max_depth_path"](session, gate="no_such_gate")


def test_check_gate_on_max_depth_path_dff_short_circuit(tmp_path) -> None:
    # A DFF is a sequential boundary, never part of a combinational
    # max-depth path -- this exercises the DFF branch that F4 found had
    # zero test coverage (deleting it left every existing test green).
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["check_gate_on_max_depth_path"](session, gate="dff0")
    assert result["on_max_depth_path"] is False
    assert result["depth_through_gate"] is None
    assert result["note"] == "gate is a DFF (sequential boundary)"


# ----------------------------------------------------------------------
# Path existence / counting / enumeration / cuts
# ----------------------------------------------------------------------


def test_path_tools(tmp_path) -> None:
    session = _new_session(tmp_path)
    assert TOOL_REGISTRY["check_path_exists"](session, source="n0[0]", target="n20")["exists"] is True
    assert TOOL_REGISTRY["check_path_exists"](session, source="n2", target="n26")["exists"] is False

    counted = TOOL_REGISTRY["count_paths"](session, source="n0[0]", target="n20")
    assert counted["count"] == 1

    enumerated = TOOL_REGISTRY["enumerate_paths"](session, source="n0[0]", target="n20", max_results=10)
    assert enumerated["count"] == 1
    assert enumerated["paths"] == [["g0", "g2", "g3", "g4"]]
    assert enumerated["truncated"] is False


def test_path_tools_avoid_absent_result_shape_unchanged(tmp_path) -> None:
    """Test 1: without `avoid`, the three path tools' result key sets are
    exactly what they were before `avoid` gained the disambiguation keys --
    no new key leaks in when it isn't relevant."""
    session = _new_session(tmp_path)
    assert set(TOOL_REGISTRY["check_path_exists"](session, source="n0[0]", target="n20")) == {"exists"}
    assert set(TOOL_REGISTRY["count_paths"](session, source="n0[0]", target="n20")) == {"count"}
    assert set(TOOL_REGISTRY["enumerate_paths"](session, source="n0[0]", target="n20", max_results=10)) == {
        "count",
        "paths",
        "truncated",
    }


def test_check_path_exists_avoid_no_path_at_all(tmp_path) -> None:
    """Test 2: `avoid` given, but source and target were never connected in
    the first place -- `exists` and `exists_ignoring_avoid` are both False,
    so a reader can tell `avoid` isn't why."""
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["check_path_exists"](session, source="n2", target="n26", avoid="clk")
    assert result["exists"] is False
    assert result["exists_ignoring_avoid"] is False


def test_check_path_exists_avoid_blocks_the_only_path(tmp_path) -> None:
    """Test 3: `avoid` given, and a path DOES exist without it but `avoid`
    sits on that path -- `exists` is False while `exists_ignoring_avoid` is
    True. This is the distinction rows 2/36 of the held-out run could not
    make: the whole point of the fix."""
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["check_path_exists"](session, source="n0[0]", target="n20", avoid="n12")
    assert result["exists"] is False
    assert result["exists_ignoring_avoid"] is True


def test_check_path_exists_avoid_accepts_gate_name(tmp_path) -> None:
    """Test 4: `avoid` given as a gate instance name resolves to that gate's
    output net and gives the same `exists`/`exists_ignoring_avoid` verdict
    as passing that net directly; `avoid_resolved_to` names the resolution."""
    session = _new_session(tmp_path)
    by_gate = TOOL_REGISTRY["check_path_exists"](session, source="n0[0]", target="n20", avoid="g2")
    by_net = TOOL_REGISTRY["check_path_exists"](session, source="n0[0]", target="n20", avoid="n12")
    assert by_gate["exists"] == by_net["exists"] is False
    assert by_gate["exists_ignoring_avoid"] == by_net["exists_ignoring_avoid"] is True
    assert by_gate["avoid_resolved_to"] == "gate g2's output n12"
    assert "avoid_resolved_to" not in by_net


def test_check_path_exists_avoid_unresolvable_token_raises_original_error(tmp_path) -> None:
    """Test 5: `avoid` that is neither a net nor a gate still raises the
    same error `_resolve_bit` always raised (net resolution attempted, gate
    fallback attempted, both fail, original error re-raised unchanged)."""
    session = _new_session(tmp_path)
    with pytest.raises(ToolError) as direct_exc:
        TOOL_REGISTRY["check_path_exists"](session, source="n0[0]", target="n20", avoid="nope")
    # cross-check against calling resolution directly on the same bad token
    from netlist_agent.llm.tools_schema import _resolve_bit

    with pytest.raises(ToolError) as via_resolve_bit:
        _resolve_bit(session.current_design, "nope")
    assert str(direct_exc.value) == str(via_resolve_bit.value)


def test_check_path_exists_avoid_net_and_gate_same_name_net_wins(tmp_path) -> None:
    """Test 6: when a net and a gate instance share a name, net resolution
    (tried first, unchanged from before this fix) wins -- the gate fallback
    never fires and no `avoid_resolved_to` appears."""
    design = Design(module_name="samename")
    design.signals["a"] = Signal(name="a", msb=None, lsb=None, direction=Direction.INPUT)
    design.signals["g0"] = Signal(name="g0", msb=None, lsb=None, direction=Direction.INTERNAL)
    design.signals["z"] = Signal(name="z", msb=None, lsb=None, direction=Direction.INTERNAL)
    design.signals["out"] = Signal(name="out", msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports = [Port(name=n, direction=design.signals[n].direction) for n in ("a", "out")]
    design.gates = [
        # A gate instance also named "g0", wired off to the side (its
        # output "z" is not on the a->out path at all).
        Gate("g0", GateType.NOT, {"O": _nb("z"), "I0": _nb("a")}),
        Gate("g1", GateType.BUF, {"O": _nb("g0"), "I0": _nb("a")}),
        Gate("g2", GateType.BUF, {"O": _nb("out"), "I0": _nb("g0")}),
    ]
    design.build_indices()
    session = Session()
    session.current_design = design
    session.original_snapshot = design
    result = TOOL_REGISTRY["check_path_exists"](session, source="a", target="out", avoid="g0")
    # If the gate fallback had fired, avoid would resolve to "z" (gate g0's
    # output), which isn't on the a->out path, so exists would stay True.
    assert result["exists"] is False
    assert "avoid_resolved_to" not in result


def test_count_paths_avoid_disambiguation(tmp_path) -> None:
    """Test 7 (count_paths): same no-path-at-all vs. avoid-blocks-everything
    distinction as check_path_exists, in counting form."""
    session = _new_session(tmp_path)
    no_path = TOOL_REGISTRY["count_paths"](session, source="n2", target="n26", avoid="clk")
    assert no_path["count"] == 0
    assert no_path["count_ignoring_avoid"] == 0

    blocked = TOOL_REGISTRY["count_paths"](session, source="n0[0]", target="n20", avoid="n12")
    assert blocked["count"] == 0
    assert blocked["count_ignoring_avoid"] == 1


def test_enumerate_paths_avoid_disambiguation(tmp_path) -> None:
    """Test 7 (enumerate_paths): same distinction, plus the gate-name form
    of `avoid`."""
    session = _new_session(tmp_path)
    no_path = TOOL_REGISTRY["enumerate_paths"](session, source="n2", target="n26", avoid="clk")
    assert no_path["count"] == 0
    assert no_path["count_ignoring_avoid"] == 0

    blocked = TOOL_REGISTRY["enumerate_paths"](session, source="n0[0]", target="n20", avoid="g2")
    assert blocked["count"] == 0
    assert blocked["count_ignoring_avoid"] == 1
    assert blocked["avoid_resolved_to"] == "gate g2's output n12"


@pytest.mark.parametrize(
    "bad_pins",
    [
        # Output pin absent from `pins` entirely -- `.get()` returns None.
        {"I0": _nb("a")},
        # Output pin present but holding a Const. `Pin` is
        # Union[NetBit, Const, None], and parser.py turns 1'b0/1'b1 into
        # Const.ZERO/Const.ONE, so this is a shape the IR really carries --
        # not a hypothetical. It is the one that matters: `None` is
        # rejected by any falsy check, a Const is not, so a guard weakened
        # to `out_val is not None` would let Const through to
        # netbit_token() and raise AttributeError instead of ToolError.
        {"O": Const.ZERO, "I0": _nb("a")},
    ],
    ids=["output-pin-missing", "output-pin-is-a-const"],
)
def test_avoid_gate_fallback_ignores_a_gate_with_no_netbit_output(bad_pins) -> None:
    """The `isinstance(out_val, NetBit)` guard in `_resolve_avoid`. A cold
    read found this branch had no coverage at all: every fixture design
    wires each gate's output pin to a real NetBit, so nothing exercised the
    case the guard exists for. Driven directly rather than through a parsed
    netlist, because the parser cannot produce the missing-pin shape --
    which is exactly why the guard would rot unnoticed.

    Both non-NetBit shapes are covered deliberately. A second cold read
    showed that the missing-pin case ALONE does not pin the guard: weaken
    it to `out_val is not None` and that case still passes, because None
    fails both tests. Only the Const case separates "rejects None" from
    "rejects anything that is not a NetBit", which is what the guard says.

    The contract being pinned: a gate whose output pin does not hold a
    NetBit is NOT a usable `avoid` target, so resolution falls through to
    the original "no such signal" error instead of handing `netbit_token`
    something it cannot render.
    """
    design = Design(module_name="badout")
    design.signals["a"] = Signal(name="a", msb=None, lsb=None, direction=Direction.INPUT)
    design.signals["out"] = Signal(name="out", msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports = [Port(name=n, direction=design.signals[n].direction) for n in ("a", "out")]
    design.gates = [
        Gate("g1", GateType.BUF, {"O": _nb("out"), "I0": _nb("a")}),
        Gate("broken", GateType.BUF, dict(bad_pins)),
    ]
    design.build_indices()
    session = Session()
    session.current_design = design
    session.original_snapshot = design
    with pytest.raises(ToolError) as exc:
        TOOL_REGISTRY["check_path_exists"](session, source="a", target="out", avoid="broken")
    from netlist_agent.llm.tools_schema import _resolve_bit

    with pytest.raises(ToolError) as as_a_net:
        _resolve_bit(design, "broken")
    assert str(exc.value) == str(as_a_net.value)


def test_path_tool_descriptions_render_the_example_dicts_correctly() -> None:
    """The three descriptions teach the model how to read a zero, by showing
    literal result dicts. Those examples are built by f-string interpolation
    with `{{`/`}}` escapes, and a cold read found one continuation line had
    lost its `f` prefix -- so `}}` stopped being an escape and the model was
    shown `{'exists': False, 'exists_ignoring_avoid': True}}`, with a stray
    brace, in the one sentence the whole fix exists to communicate.

    Nothing else in the suite looks at description *content*
    (`test_schema_entry_well_formed` only checks it is a non-empty string),
    so the malformed text was invisible. This asserts the braces balance
    and that each key name reaches the text through the constant.
    """
    from netlist_agent.llm.tools_schema import (
        _AVOID_RESOLVED_TO_KEY,
        _COUNT_IGNORING_AVOID_KEY,
        _EXISTS_IGNORING_AVOID_KEY,
    )

    by_name = {spec.name: spec for spec in TOOL_SCHEMA}
    expected_key = {
        "check_path_exists": _EXISTS_IGNORING_AVOID_KEY,
        "count_paths": _COUNT_IGNORING_AVOID_KEY,
        "enumerate_paths": _COUNT_IGNORING_AVOID_KEY,
    }
    for name, key in expected_key.items():
        spec = by_name[name]
        text = spec.description + json.dumps(spec.parameters)
        assert key in text, f"{name} never names {key}"
        assert _AVOID_RESOLVED_TO_KEY in text, f"{name} never names {_AVOID_RESOLVED_TO_KEY}"
        assert text.count("{") == text.count("}"), f"{name}'s text has unbalanced braces"
        assert "}}" not in spec.description, f"{name}'s description has an unescaped {{{{ or }}}}"


def test_cut_tools(tmp_path) -> None:
    session = _new_session(tmp_path)
    cuts = TOOL_REGISTRY["get_cut_nets_between"](session, source="n0[0]", target="n20")
    assert cuts["path_exists"] is True
    assert "n10" in cuts["cut_nets"]

    assert TOOL_REGISTRY["check_is_cut_signal"](session, signal="n10")["is_cut"] is True
    assert TOOL_REGISTRY["check_is_cut_signal"](session, signal="n2")["is_cut"] is False


def test_direct_pi_po_connections_empty(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["get_direct_pi_po_connections"](session)
    assert result["items"] == []


# ----------------------------------------------------------------------
# Cones
# ----------------------------------------------------------------------


def test_cone_tools(tmp_path) -> None:
    session = _new_session(tmp_path)
    assert TOOL_REGISTRY["get_fanin_cone_size"](session, net="n20")["size"] == 4
    gates = TOOL_REGISTRY["get_fanin_cone_gates"](session, net="n20")
    assert set(gates["items"]) == {"g0", "g2", "g3", "g4"}

    fanout_cone = TOOL_REGISTRY["get_fanout_cone_gates"](session, net="n0[0]")
    assert "g0" in fanout_cone["items"]

    largest = TOOL_REGISTRY["get_largest_fanin_cone"](session)
    assert largest["size"] >= 4

    breakdown = TOOL_REGISTRY["get_cone_gate_type_breakdown"](session, net="n20")
    assert breakdown["net"] == "n20"
    assert breakdown["cone_gates"] == 4
    assert breakdown["by_type"] == {"AND": 1, "NOT": 2, "BUF": 1}

    shared = TOOL_REGISTRY["get_shared_fanin_gates"](session, net_a="n20", net_b="n26")
    assert shared["items"] == []


def test_cone_gate_type_breakdown_empty_cone_is_not_an_error(tmp_path) -> None:
    """n0[0] is a primary input bit: nothing drives it, so its fanin cone is
    genuinely empty. `cone_gates` must say so explicitly (0), not collapse
    to the same bare-`{}` shape a failed lookup would also produce."""
    session = _new_session(tmp_path)
    breakdown = TOOL_REGISTRY["get_cone_gate_type_breakdown"](session, net="n0[0]")
    assert breakdown["net"] == "n0[0]"
    assert breakdown["cone_gates"] == 0
    assert breakdown["by_type"] == {}


# ----------------------------------------------------------------------
# ABC-backed (Boolean semantic) tools -- ground truth via direct call
# ----------------------------------------------------------------------


def test_check_equivalence_to_snapshot(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["check_equivalence_to_snapshot"](session)
    assert result["equivalent"] is True

    TOOL_REGISTRY["rename_gate"](session, old_name="g0", new_name="g0_renamed")
    result_after = TOOL_REGISTRY["check_equivalence_to_snapshot"](session)
    assert result_after["equivalent"] is True  # a pure rename never changes function


def test_check_signal_equivalence(tmp_path) -> None:
    session = _new_session(tmp_path)
    expected = are_equivalent(session.current_design, NetBit("n10", None), NetBit("n25", None))
    result = TOOL_REGISTRY["check_signal_equivalence"](session, net_a="n10", net_b="n25")
    assert result["equivalent"] == expected


def test_check_symmetry_tool(tmp_path) -> None:
    session = _new_session(tmp_path)
    expected = check_symmetry(session.current_design, NetBit("n10", None), NetBit("n0", 0), NetBit("n0", 1))
    result = TOOL_REGISTRY["check_symmetry_tool"](session, output="n10", input_a="n0[0]", input_b="n0[1]")
    assert result["symmetric"] == expected


def test_get_constant_value(tmp_path) -> None:
    session = _new_session(tmp_path)
    expected = is_constant(session.current_design, NetBit("n14", None))
    result = TOOL_REGISTRY["get_constant_value"](session, net="n14")
    assert result["is_constant"] == (expected is not None)
    if expected is not None:
        assert result["value"] == str(expected.value)


# ----------------------------------------------------------------------
# Rename
# ----------------------------------------------------------------------


def test_rename_gate_and_signal(tmp_path) -> None:
    session = _new_session(tmp_path)
    TOOL_REGISTRY["rename_gate"](session, old_name="g0", new_name="g0_new")
    assert TOOL_REGISTRY["get_gate_info"](session, gate="g0_new")["type"] == "and"
    with pytest.raises(ToolError):
        TOOL_REGISTRY["rename_gate"](session, old_name="ghost", new_name="whatever")

    TOOL_REGISTRY["rename_signal"](session, old_name="n26", new_name="n26_new")
    assert "n26_new" in session.current_design.signals
    with pytest.raises(ToolError):
        TOOL_REGISTRY["rename_signal"](session, old_name="ghost_signal", new_name="whatever")


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------


def test_transform_tools(tmp_path) -> None:
    session = _new_session(tmp_path)
    assert TOOL_REGISTRY["do_collapse_double_inverters"](session)["collapsed"] == 1
    # g2/g3 (NOT-NOT) collapsed; n20 now fed by g4(BUF) reading n10 directly (via g0).
    assert TOOL_REGISTRY["get_depth_of_cone"](session, net="n20")["depth"] == 2


def test_dedup_and_dangling(tmp_path) -> None:
    session = _new_session(tmp_path)
    dedup = TOOL_REGISTRY["do_deduplicate_gates"](session)
    assert dedup["merged"] == 1  # g0 and g12 are structural duplicates (both AND(n0[0], n0[1]))

    removed = TOOL_REGISTRY["do_remove_dangling_gates"](session)
    assert removed["removed"] >= 0


def test_simplify_constant_inputs(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["do_simplify_constant_inputs"](session, gate_types=["nand"])
    assert result["simplified"] == 1
    assert TOOL_REGISTRY["get_gate_info"](session, gate="g6")["type"] == "not"


def test_remap_to_basis(tmp_path) -> None:
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["do_remap_to_basis"](session, basis="nand", gate_type="nand")
    # g1 and g6 (both NAND) are already in the nand/not basis -- nothing to
    # replace, but the scope itself is non-empty (both were matched and
    # simply needed no work).
    assert result["replaced"] == 0
    assert result["gates_in_scope"] == 2
    assert result["scope"] == "gates of type nand"
    with pytest.raises(ToolError):
        TOOL_REGISTRY["do_remap_to_basis"](session, basis="xor")


def test_remap_to_basis_empty_scope_is_a_legitimate_zero(tmp_path) -> None:
    """cone_root=n0[0] (a primary input bit -- empty fanin cone) must report
    `gates_in_scope: 0` alongside `replaced: 0`, distinguishing "your scope
    was understood and is empty" from a silent no-op/failure."""
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["do_remap_to_basis"](session, basis="nand", cone_root="n0[0]")
    assert result["replaced"] == 0
    assert result["gates_in_scope"] == 0
    assert result["scope"] == "fanin cone of n0[0]"


def test_limit_fanout_and_insert_buffer(tmp_path) -> None:
    session = _new_session(tmp_path)
    added = TOOL_REGISTRY["do_limit_fanout_net"](session, net="n10", max_fanout=2)
    assert added["buffers_added"] >= 1
    assert TOOL_REGISTRY["get_net_fanout"](session, net="n10")["count"] <= 2

    session2 = _new_session(tmp_path)
    global_added = TOOL_REGISTRY["do_limit_fanout_global"](session2, max_fanout=2)
    assert global_added["buffers_added"] >= 1

    session3 = _new_session(tmp_path)
    per_load = TOOL_REGISTRY["do_insert_buffer_per_load"](session3, net="n10")
    assert per_load["buffers_added"] == 3


# ----------------------------------------------------------------------
# ABC-backed depth optimization
# ----------------------------------------------------------------------


def test_optimize_depth_tool_contract(tmp_path) -> None:
    session = _new_session(tmp_path)
    depth_before = TOOL_REGISTRY["get_max_design_depth"](session)["depth"]

    result = TOOL_REGISTRY["do_optimize_depth"](session)
    assert set(result) == {"changed", "depth_before", "depth_after", "note"}
    assert result["depth_before"] == depth_before
    assert result["depth_after"] <= depth_before
    assert isinstance(result["changed"], bool)
    assert isinstance(result["note"], str) and result["note"]
    # session.current_design was updated in place to whatever the tool returned.
    assert TOOL_REGISTRY["get_max_design_depth"](session)["depth"] == result["depth_after"]


def test_optimize_depth_tool_rejects_unknown_basis(tmp_path) -> None:
    session = _new_session(tmp_path)
    with pytest.raises(ToolError):
        TOOL_REGISTRY["do_optimize_depth"](session, basis="xor_not")


def test_optimize_cone_depth_tool_contract(tmp_path) -> None:
    session = _new_session(tmp_path)
    depth_before = TOOL_REGISTRY["get_depth_of_cone"](session, net="n20")["depth"]

    result = TOOL_REGISTRY["do_optimize_cone_depth"](session, net="n20", basis="and_not")
    assert set(result) == {"changed", "depth_before", "depth_after", "note"}
    assert result["depth_before"] == depth_before
    assert result["depth_after"] <= depth_before
    assert TOOL_REGISTRY["get_depth_of_cone"](session, net="n20")["depth"] == result["depth_after"]


def test_optimize_cone_depth_tool_rejects_bad_net_and_basis(tmp_path) -> None:
    session = _new_session(tmp_path)
    with pytest.raises(ToolError):
        TOOL_REGISTRY["do_optimize_cone_depth"](session, net="not a net!!", basis="and_not")
    with pytest.raises(ToolError):
        TOOL_REGISTRY["do_optimize_cone_depth"](session, net="n20", basis="xor_not")


# ----------------------------------------------------------------------
# find_gates_by_name -> do_replace_buf_with_and (P6: this is the one call
# path that reaches replace_buf_with_and's `gate.gate_type != GateType.BUF`
# guard with a mixed-type name list -- find_gates_by_name has no gate_type
# filter here, so it's on this path (unlike router.py's regex handlers,
# which always pre-filter to BUF via gates_by_name_substring) that a non-BUF
# gate name can actually reach replace_buf_with_and.
# ----------------------------------------------------------------------


def test_find_gates_by_name_then_replace_buf_with_and_via_tools(tmp_path) -> None:
    session = _new_session(tmp_path)
    # "g1" (a NAND, not a BUF) and "g13" (a BUF) both contain "1" in their name.
    TOOL_REGISTRY["find_gates_by_name"](session, substring="1")
    assert "g1" in session.last_query_gate_names
    assert "g13" in session.last_query_gate_names

    result = TOOL_REGISTRY["do_replace_buf_with_and"](session, ctrl_net="n2")
    # Only g13 (the actual BUF) is rewritten; g1 (a NAND) is left untouched.
    assert result["replaced"] == 1
    types = {g.inst_name: g.gate_type for g in session.current_design.gates}
    assert types["g13"] == GateType.AND
    assert types["g1"] == GateType.NAND


# ----------------------------------------------------------------------
# Reg-to-reg path stats / floating signals (F5: these three wrappers had
# only structural (schema-shape) coverage before this fix, never an actual
# TOOL_REGISTRY call)
# ----------------------------------------------------------------------


def _build_reg_to_reg_design() -> Design:
    """A three-DFF design with BOTH kinds of reg-to-reg connection at once,
    so a test asserting on it can't pass with a stub that always returns
    zero: dffA.Q feeds dffB.D through one combinational gate (g0), and
    dffB.Q wires STRAIGHT into dffC.D with no gate in between at all."""
    design = Design(module_name="top")
    for name in ("clk", "rn"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    for name in ("qa", "db"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
    design.ports = [Port("clk", Direction.INPUT), Port("rn", Direction.INPUT)]
    design.gates = [
        Gate(
            "dffA",
            GateType.DFF,
            {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": Const.ZERO, "Q": _nb("qa")},
        ),
        Gate("g0", GateType.NOT, {"O": _nb("db"), "I0": _nb("qa")}),
        Gate(
            "dffB",
            GateType.DFF,
            {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("db"), "Q": _nb("qb")},
        ),
        Gate(
            "dffC",
            GateType.DFF,
            {"RN": _nb("rn"), "SN": Const.ONE, "CK": _nb("clk"), "D": _nb("qb"), "Q": _nb("qc")},
        ),
    ]
    for name in ("qb", "qc"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
    design.build_indices()
    return design


def test_get_reg_to_reg_path_stats() -> None:
    """A stub that ignores `graph` and always returns 0/0/[] would pass
    against `_new_session`'s single-DFF fixture (it genuinely has no
    reg-to-reg connections) -- this design has one gated path (dffA -> g0 ->
    dffB) and one zero-gate direct wire (dffB -> dffC), so both counts must
    come back nonzero and a stub can't fake it."""
    design = _build_reg_to_reg_design()
    graph = NetlistGraph(design)
    expected = graph.reg_to_reg_path_stats()
    assert expected.combinational_path_count == 1
    assert expected.direct_wire_count == 1

    session = Session()
    session.current_design = design
    result = TOOL_REGISTRY["get_reg_to_reg_path_stats"](session)
    assert result["combinational_path_count"] == 1
    assert result["direct_wire_count"] == 1
    assert len(result["direct_wire_examples"]) == 1
    assert result["direct_wire_examples"][0]["source_dff"] == "dffB"
    assert result["direct_wire_examples"][0]["sink_dff"] == "dffC"


def test_get_reg_to_reg_path_stats_zero_paths(tmp_path) -> None:
    """The single-DFF fixture (`_new_session`) genuinely has zero reg-to-reg
    connections at all (only one DFF, so no DFF-to-DFF pair exists) -- kept
    as the "zero also reported correctly" case, not the only case (see
    `test_get_reg_to_reg_path_stats` above for the nonzero case)."""
    session = _new_session(tmp_path)
    graph = NetlistGraph(session.current_design)
    expected = graph.reg_to_reg_path_stats()
    result = TOOL_REGISTRY["get_reg_to_reg_path_stats"](session)
    assert result["combinational_path_count"] == expected.combinational_path_count == 0
    assert result["direct_wire_count"] == expected.direct_wire_count == 0
    assert result["direct_wire_examples"] == expected.direct_wire_examples == []


def test_check_floating_signals(tmp_path) -> None:
    session = _new_session(tmp_path)
    graph = NetlistGraph(session.current_design)
    expected = find_floating_signals(graph)
    result = TOOL_REGISTRY["check_floating_signals"](session)
    assert result["floating_input_nets_plus_unconnected_output_ports_count"] == expected.headline_count
    counted = result["counted_in_that_number"]
    assert set(counted["floating_input_nets_referenced_but_undriven"]) == {
        netbit_token(nb) for nb in expected.floating_input_nets_referenced_but_undriven
    }
    assert set(counted["unconnected_output_ports_undriven"]) == {
        netbit_token(nb) for nb in expected.unconnected_output_ports_undriven
    }


def _build_floating_order_design() -> Design:
    """D: same-name multi-bit signals whose defective bits are {1, 2, 10,
    20} for all three list-valued fields at once -- a plain string sort of
    the rendered "name[bit]" tokens would put "[10]"/"[20]" before "[2]";
    this pins down that `check_floating_signals` (the LLM tool wrapper, one
    layer above `analysis.find_floating_signals`) reports true numeric
    order via an exact list `==`, not `set()` (see `test_check_floating_signals`
    above, which only checks set membership and can't observe order at all)."""
    order_bits = (1, 2, 10, 20)
    design = Design(module_name="top")

    design.signals["float17"] = Signal(name="float17", msb=20, lsb=1, direction=Direction.INTERNAL)
    design.signals["pi17"] = Signal(name="pi17", msb=20, lsb=1, direction=Direction.INPUT)
    design.signals["po17"] = Signal(name="po17", msb=20, lsb=1, direction=Direction.OUTPUT)
    design.ports = [Port("pi17", Direction.INPUT), Port("po17", Direction.OUTPUT)]

    gates: list[Gate] = []
    # float17: reference exactly bits {1, 2, 10, 20} as undriven gate inputs.
    for i, bit in enumerate(order_bits):
        sink = f"floatsink{i}"
        design.signals[sink] = Signal(name=sink, msb=None, lsb=None, direction=Direction.INTERNAL)
        gates.append(Gate(f"gf{i}", GateType.BUF, {"O": NetBit(sink), "I0": NetBit("float17", bit)}))

    # pi17: consume every bit EXCEPT {1, 2, 10, 20}, so only those are unused.
    consumer = 0
    for bit in range(1, 21):
        if bit in order_bits:
            continue
        sink = f"piused{consumer}"
        design.signals[sink] = Signal(name=sink, msb=None, lsb=None, direction=Direction.INTERNAL)
        gates.append(Gate(f"gp{consumer}", GateType.BUF, {"O": NetBit(sink), "I0": NetBit("pi17", bit)}))
        consumer += 1

    # po17: drive every bit EXCEPT {1, 2, 10, 20}, so only those are undriven.
    for bit in range(1, 21):
        if bit in order_bits:
            continue
        src = f"posrc{bit}"
        design.signals[src] = Signal(name=src, msb=None, lsb=None, direction=Direction.INPUT)
        design.ports.append(Port(src, Direction.INPUT))
        gates.append(Gate(f"gd{bit}", GateType.BUF, {"O": NetBit("po17", bit), "I0": NetBit(src)}))

    design.gates = gates
    design.build_indices()
    return design


def test_check_floating_signals_numeric_ordering() -> None:
    session = Session()
    session.current_design = _build_floating_order_design()
    result = TOOL_REGISTRY["check_floating_signals"](session)
    counted = result["counted_in_that_number"]
    additional = result["additional_findings_not_counted_above"]
    expected_tokens = ["float17[1]", "float17[2]", "float17[10]", "float17[20]"]
    assert counted["floating_input_nets_referenced_but_undriven"] == expected_tokens
    assert additional["declared_input_ports_completely_unused"] == [
        "pi17[1]", "pi17[2]", "pi17[10]", "pi17[20]",
    ]
    assert counted["unconnected_output_ports_undriven"] == [
        "po17[1]", "po17[2]", "po17[10]", "po17[20]",
    ]


# ----------------------------------------------------------------------
# _expected_json_type's own failure branches
# ----------------------------------------------------------------------
# The comparison above is only as trustworthy as this helper's refusal to
# guess. Its two failure branches -- an unsupported multi-arm Union, and an
# annotation with no registered mapping -- are unreachable from TOOL_SCHEMA
# as it stands, because no tool parameter is currently shaped like `dict`,
# `bool` or `Union[str, int]`. That left "it fails loudly rather than
# silently passing" as a claim resting on reading the code, which is the
# one form of evidence this project has repeatedly found insufficient. The
# branches are cheap to exercise directly, so they are.


def test_expected_json_type_refuses_an_unmapped_annotation() -> None:
    with pytest.raises(AssertionError, match="no JSON-type mapping registered"):
        _expected_json_type(dict)


def test_expected_json_type_refuses_a_multi_arm_union() -> None:
    with pytest.raises(AssertionError, match="unsupported Optional/Union shape"):
        _expected_json_type(typing.Union[str, int])


def test_expected_json_type_still_unwraps_a_real_optional() -> None:
    """Control: the guards above must not be refusing everything. Optional
    is the shape the tools actually use, and it has to keep resolving."""
    assert _expected_json_type(typing.Optional[str]) == "string"
    assert _expected_json_type(typing.Optional[int]) == "integer"
    assert _expected_json_type(list[str]) == "array"


# ----------------------------------------------------------------------
# get_last_operation_summary: reads router.py's own bookkeeping back
# without rerunning the operation that produced it (measured gap: see
# experiments/heldout_fallback_2026-08-28/REPORT.md, traces_p08/p21/p34).
# ----------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_ROUTER, reason="requires the private rule-based router (not present in this public export)")
def test_get_last_operation_summary_after_rule_routed_mutation(tmp_path) -> None:
    """A rule-routed request (dedup merge) records `session.last_op_count`;
    the read-only tool must return that exact count WITHOUT rerunning
    anything -- the design's fingerprint must be identical before and after
    the tool call, unlike calling do_deduplicate_gates a second time, which
    would merge nothing further but still mutate/rebuild state."""
    design = Design(module_name="dup")
    for name in ("a", "b"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    design.signals["y"] = Signal(name="y", msb=None, lsb=None, direction=Direction.OUTPUT)
    for name in ("m1", "m2"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
    design.ports = [Port("a", Direction.INPUT), Port("b", Direction.INPUT), Port("y", Direction.OUTPUT)]
    design.gates = [
        Gate("g0", GateType.AND, {"O": _nb("m1"), "I0": _nb("a"), "I1": _nb("b")}),
        Gate("g1", GateType.AND, {"O": _nb("m2"), "I0": _nb("a"), "I1": _nb("b")}),
        Gate("g2", GateType.OR, {"O": _nb("y"), "I0": _nb("m1"), "I1": _nb("m2")}),
    ]
    design.build_indices()

    session = Session()
    session.current_design = design

    def _no_fallback(s: Session, t: str) -> str:
        raise AssertionError("fallback must not run: this request is rule-routed")

    body = handle_request(
        session,
        "Find and merge all gate pairs in the design that are functionally equivalent (produce the same "
        "function). Make sure nothing changes functionally.",
        _no_fallback,
    )
    assert session.last_op_count == 1
    assert str(session.last_op_count) in body

    fp_before = design_fingerprint(session.current_design)
    summary = TOOL_REGISTRY["get_last_operation_summary"](session)
    fp_after = design_fingerprint(session.current_design)

    assert summary["last_op_count"] == 1
    assert fp_before == fp_after


def _dup_gate_design() -> Design:
    """Same fixture `test_get_last_operation_summary_after_rule_routed_mutation`
    builds inline: two structurally-identical AND gates (g0, g1) feeding an
    OR -- exactly one duplicate pair, so a rule-routed dedup merges 1 and a
    SECOND dedup pass over the now-deduped design finds 0 further merges."""
    design = Design(module_name="dup")
    for name in ("a", "b"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    design.signals["y"] = Signal(name="y", msb=None, lsb=None, direction=Direction.OUTPUT)
    for name in ("m1", "m2"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INTERNAL)
    design.ports = [Port("a", Direction.INPUT), Port("b", Direction.INPUT), Port("y", Direction.OUTPUT)]
    design.gates = [
        Gate("g0", GateType.AND, {"O": _nb("m1"), "I0": _nb("a"), "I1": _nb("b")}),
        Gate("g1", GateType.AND, {"O": _nb("m2"), "I0": _nb("a"), "I1": _nb("b")}),
        Gate("g2", GateType.OR, {"O": _nb("y"), "I0": _nb("m1"), "I1": _nb("m2")}),
    ]
    design.build_indices()
    return design


# ----------------------------------------------------------------------
# Refuse-by-default on a detected rerun (measured:
# experiments/count_question_reruns_2026-08-29/REPORT.md -- a model asked
# for a past merge count, re-ran do_deduplicate_gates instead, and reported
# the rerun's own, different, count, deleting 691 gates in the corpus
# measurement to answer a question that only asked for a count.
# experiments/refuse_rerun_2026-08-29/PROTOCOL.md upgrades the disclosure
# that fix originally shipped into a default REFUSAL: the mutating tool
# performs no mutation at all on a detected rerun, unless force=True.)
# ----------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_ROUTER, reason="requires the private rule-based router (not present in this public export)")
def test_do_deduplicate_gates_refuses_a_rerun_by_default_and_performs_no_mutation(tmp_path) -> None:
    """Rule-routed dedup records last_op_count=1/last_op_kind="deduplicate_gates".
    Calling the do_deduplicate_gates TOOL again with the same (empty) args is
    refused by default: no 'merged' key (nothing ran), the previously
    recorded count comes back under _PREVIOUSLY_REPORTED_COUNT_KEY, and the
    design's fingerprint is untouched."""
    session = Session()
    session.current_design = _dup_gate_design()

    def _no_fallback(s: Session, t: str) -> str:
        raise AssertionError("fallback must not run: this request is rule-routed")

    handle_request(
        session,
        "Find and merge all gate pairs in the design that are functionally equivalent (produce the same "
        "function). Make sure nothing changes functionally.",
        _no_fallback,
    )
    assert session.last_op_count == 1
    assert session.last_op_kind == "deduplicate_gates"
    fp_before = design_fingerprint(session.current_design)

    result = TOOL_REGISTRY["do_deduplicate_gates"](session)
    fp_after = design_fingerprint(session.current_design)

    assert "merged" not in result  # no mutation was performed
    assert result[_MUTATION_PERFORMED_KEY] is False
    assert result[_PREVIOUSLY_REPORTED_COUNT_KEY] == 1
    assert result[_RERUN_OF_PRIOR_OPERATION_KEY] is True
    assert _REFUSED_REASON_KEY in result and result[_REFUSED_REASON_KEY]
    assert result[_FORCE_OVERRIDE_PARAM_KEY] == _FORCE_PARAM_NAME == "force"
    assert fp_before == fp_after


def test_do_deduplicate_gates_force_true_actually_runs_and_changes_the_design(tmp_path) -> None:
    """force=True bypasses the refusal above: it must actually run the
    transform (mutating the design) AND still carry the disclosure fields --
    force runs the operation, it does not hide that it was a rerun. Uses a
    session.last_op_kind/last_op_args/last_op_count injected directly
    (rather than run via a first rule-routed dedup) so the design still HAS
    a genuine duplicate pair left to merge -- otherwise a real rerun would
    legitimately find 0 further merges and 'the design changed' would be
    untestable here, as it is in the sibling refusal test above."""
    session = Session()
    session.current_design = _dup_gate_design()
    session.last_op_kind = "deduplicate_gates"
    session.last_op_args = ()
    session.last_op_count = 999  # a fake earlier count, never actually run
    fp_before = design_fingerprint(session.current_design)

    result = TOOL_REGISTRY["do_deduplicate_gates"](session, force=True)
    fp_after = design_fingerprint(session.current_design)

    assert result["merged"] == 1  # the one real duplicate pair actually got merged
    # Stated positively on BOTH paths. It used to be absent here, so "did it
    # run?" was answered by a missing key; a cold read called that out and it
    # now says so outright.
    assert result[_MUTATION_PERFORMED_KEY] is True
    assert result[_RERUN_OF_PRIOR_OPERATION_KEY] is True
    assert result[_PREVIOUSLY_REPORTED_COUNT_KEY] == 999
    assert fp_before != fp_after


@pytest.mark.parametrize(
    "force_value",
    [False, "false", "False", "0", "no", "", None, 0, 1, 2.0, {}, ["true"]],
    ids=lambda v: f"force={v!r}",
)
def test_non_boolean_force_values_do_not_bypass_the_refusal(force_value) -> None:
    """`force` is not a plain truth test, and this is the reason.

    Tool arguments arrive as `fn(session, **json.loads(raw))` with no
    schema-driven coercion (llm/client.py), so a provider that serialises
    booleans as JSON strings hands this the string "false" -- truthy in
    Python. A cold read reproduced exactly that: `force="false"` ran the
    mutation. An argument whose plain meaning is "do not force" must never
    force, and neither must anything else unrecognised: the wrong direction
    here edits the user's netlist, while the wrong direction the other way
    costs one round and names the parameter to set.

    `1` and `2.0` are in this list deliberately. They are conventionally
    truthy, and they still must not force -- only the spellings a caller
    could plausibly MEAN as the boolean true do.
    """
    session = Session()
    session.current_design = _dup_gate_design()
    session.last_op_kind = "deduplicate_gates"
    session.last_op_args = ()
    session.last_op_count = 999
    fp_before = design_fingerprint(session.current_design)

    result = TOOL_REGISTRY["do_deduplicate_gates"](session, force=force_value)

    assert result[_MUTATION_PERFORMED_KEY] is False
    assert "merged" not in result
    assert result[_PREVIOUSLY_REPORTED_COUNT_KEY] == 999
    assert design_fingerprint(session.current_design) == fp_before


@pytest.mark.parametrize("force_value", [True, "true", "TRUE", " True ", "yes", "1"],
                         ids=lambda v: f"force={v!r}")
def test_spellings_of_true_do_bypass_the_refusal(force_value) -> None:
    """The other half of the asymmetry: a caller that plainly means true --
    including the JSON-string spellings the defect above came from -- still
    gets through, so the strictness does not turn into an unusable flag."""
    session = Session()
    session.current_design = _dup_gate_design()
    session.last_op_kind = "deduplicate_gates"
    session.last_op_args = ()
    session.last_op_count = 999
    fp_before = design_fingerprint(session.current_design)

    result = TOOL_REGISTRY["do_deduplicate_gates"](session, force=force_value)

    assert result["merged"] == 1
    assert result[_MUTATION_PERFORMED_KEY] is True
    assert design_fingerprint(session.current_design) != fp_before


@pytest.mark.skipif(not _HAS_REAL_ROUTER, reason="requires the private rule-based router (not present in this public export)")
def test_a_different_kind_of_operation_is_not_flagged_as_a_rerun(tmp_path) -> None:
    """Honesty guard: last_op_kind names deduplicate_gates (from a
    rule-routed dedup), but the tool call that follows runs a DIFFERENT
    transform (remove_dangling_gates) -- the new fields must NOT appear,
    since this tool has no basis to claim ITS count is a rerun of dedup's."""
    session = Session()
    session.current_design = _dup_gate_design()

    def _no_fallback(s: Session, t: str) -> str:
        raise AssertionError("fallback must not run: this request is rule-routed")

    handle_request(
        session,
        "Find and merge all gate pairs in the design that are functionally equivalent (produce the same "
        "function). Make sure nothing changes functionally.",
        _no_fallback,
    )
    assert session.last_op_kind == "deduplicate_gates"

    result = TOOL_REGISTRY["do_remove_dangling_gates"](session)
    assert _RERUN_OF_PRIOR_OPERATION_KEY not in result
    assert _PREVIOUSLY_REPORTED_COUNT_KEY not in result


def test_no_prior_recorded_operation_leaves_the_return_shape_unchanged(tmp_path) -> None:
    """A fresh session (last_op_kind is still None, the initial value) must
    get exactly the pre-existing keys back -- no new keys appear just
    because a mutating tool happened to run."""
    session = _new_session(tmp_path)
    assert session.last_op_kind is None
    result = TOOL_REGISTRY["do_deduplicate_gates"](session)
    assert set(result.keys()) == {"merged"}


@pytest.mark.skipif(not _HAS_REAL_ROUTER, reason="requires the private rule-based router (not present in this public export)")
def test_get_last_operation_summary_reports_unaffected_by_this_turns_tool_calls(tmp_path) -> None:
    """get_last_operation_summary's own new key is always True, and its
    last_op_count keeps reporting the RULE-ROUTED request's count even
    after the model's own do_deduplicate_gates tool call this turn already
    re-ran the transform (pinning "deliberately not refreshed by this
    turn's tool calls")."""
    session = Session()
    session.current_design = _dup_gate_design()

    def _no_fallback(s: Session, t: str) -> str:
        raise AssertionError("fallback must not run: this request is rule-routed")

    handle_request(
        session,
        "Find and merge all gate pairs in the design that are functionally equivalent (produce the same "
        "function). Make sure nothing changes functionally.",
        _no_fallback,
    )
    summary_before = TOOL_REGISTRY["get_last_operation_summary"](session)
    assert summary_before[_UNAFFECTED_BY_THIS_TURNS_TOOL_CALLS_KEY] is True
    assert summary_before["last_op_count"] == 1

    TOOL_REGISTRY["do_deduplicate_gates"](session)  # the model's own attempted re-run, refused this turn

    summary_after = TOOL_REGISTRY["get_last_operation_summary"](session)
    assert summary_after[_UNAFFECTED_BY_THIS_TURNS_TOOL_CALLS_KEY] is True
    assert summary_after["last_op_count"] == 1  # still the earlier request's count, unmoved


def test_rerun_conflict_descriptions_render_the_keys_correctly() -> None:
    """Same discipline as test_path_tool_descriptions_render_the_example_dicts_correctly:
    every do_* description touched by this fix names both new keys through
    the module constant (not a hand-typed literal), and the descriptions'
    braces stay balanced."""
    by_name = {spec.name: spec for spec in TOOL_SCHEMA}
    rerun_tools = [
        "do_limit_fanout_global",
        "do_limit_fanout_net",
        "do_insert_buffer_per_load",
        "do_balance_depth_to_sinks",
        "do_replace_buf_with_and",
        "do_remove_dangling_gates",
        "do_deduplicate_gates",
        "do_collapse_double_inverters",
        "do_collapse_inverter_buffer_chains",
        "do_simplify_constant_inputs",
        "do_remap_to_basis",
    ]
    for name in rerun_tools:
        spec = by_name[name]
        text = spec.description + json.dumps(spec.parameters)
        assert _RERUN_OF_PRIOR_OPERATION_KEY in text, f"{name} never names {_RERUN_OF_PRIOR_OPERATION_KEY}"
        assert _PREVIOUSLY_REPORTED_COUNT_KEY in text, f"{name} never names {_PREVIOUSLY_REPORTED_COUNT_KEY}"
        assert _MUTATION_PERFORMED_KEY in text, f"{name} never names {_MUTATION_PERFORMED_KEY}"
        assert _REFUSED_REASON_KEY in text, f"{name} never names {_REFUSED_REASON_KEY}"
        assert _FORCE_OVERRIDE_PARAM_KEY in text, f"{name} never names {_FORCE_OVERRIDE_PARAM_KEY}"
        assert "force" in spec.parameters["properties"], f"{name} has no 'force' override parameter in its schema"
        assert spec.parameters["properties"]["force"]["type"] == "boolean", f"{name}'s force param is not boolean"
        assert text.count("{") == text.count("}"), f"{name}'s text has unbalanced braces"
        assert "}}" not in spec.description, f"{name}'s description has an unescaped {{{{ or }}}}"

    summary_spec = by_name["get_last_operation_summary"]
    summary_text = summary_spec.description + json.dumps(summary_spec.parameters)
    assert _UNAFFECTED_BY_THIS_TURNS_TOOL_CALLS_KEY in summary_text
    assert summary_text.count("{") == summary_text.count("}")
    assert "}}" not in summary_spec.description


def _xor_design() -> Design:
    """A tiny design with one XOR gate (scoped-remap target) and one NAND
    gate (whole-design-remap target), so a scoped "XOR -> 4-NAND" remap and
    a subsequent unrestricted "and_not" remap are genuinely two DIFFERENT
    requests (different basis, different scope) that happen to share the
    same transform function -- the defect-1 repro."""
    design = Design(module_name="xor")
    for name in ("a", "b"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.INPUT)
    for name in ("xo", "nd"):
        design.signals[name] = Signal(name=name, msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports = [
        Port("a", Direction.INPUT),
        Port("b", Direction.INPUT),
        Port("xo", Direction.OUTPUT),
        Port("nd", Direction.OUTPUT),
    ]
    design.gates = [
        Gate("gx", GateType.XOR, {"O": _nb("xo"), "I0": _nb("a"), "I1": _nb("b")}),
        Gate("gn", GateType.NAND, {"O": _nb("nd"), "I0": _nb("a"), "I1": _nb("b")}),
    ]
    design.build_indices()
    return design


@pytest.mark.skipif(not _HAS_REAL_ROUTER, reason="requires the private rule-based router (not present in this public export)")
def test_remap_to_basis_same_fn_different_basis_and_scope_is_not_flagged_as_a_rerun(tmp_path) -> None:
    """Defect-1 repro: a rule-routed "convert every XOR gate to 4-NAND"
    scopes `remap_to_basis` to just the XOR gate(s) with basis "nand_not".
    A later do_remap_to_basis(basis="and_not") call with NO scope restriction
    shares `remap_to_basis` as its transform function but is neither the
    same basis nor the same scope -- comparing `last_op_kind` (the function
    name) alone used to flag this as a "rerun" and hand back an unrelated
    `previously_reported_count`; comparing `last_op_args` too must not."""
    session = Session()
    session.current_design = _xor_design()

    def _no_fallback(s: Session, t: str) -> str:
        raise AssertionError("fallback must not run: this request is rule-routed")

    handle_request(session, "Convert all XOR gates to an equivalent 4-nand circuit.", _no_fallback)
    assert session.last_op_kind == "remap_to_basis"
    assert session.last_op_count == 1  # the one XOR gate

    result = TOOL_REGISTRY["do_remap_to_basis"](session, basis="and")
    assert result["scope"] == "whole design"
    assert _RERUN_OF_PRIOR_OPERATION_KEY not in result, result
    assert _PREVIOUSLY_REPORTED_COUNT_KEY not in result, result


# ----------------------------------------------------------------------
# Wiring coverage for all 11 mutating do_* tools (cold-read mutation gap:
# deleting a **_rerun_conflict_fields(session, <fn>, <args>) expansion, or
# passing the wrong fn/args, left every existing test green -- the schema
# text test above checks tool DESCRIPTIONS, a different piece of source
# from the handler body, so it cannot catch a wiring mistake in the body).
# ----------------------------------------------------------------------


def _build_rerun_wiring_cases(session: Session) -> list[tuple]:
    """One case per mutating do_* tool: (tool_name, kwargs, transform_fn,
    raw_args) where `raw_args` are exactly the (non-design) positional
    arguments that tool call is expected to pass to `transform_fn` --
    mirroring each do_* function's own argument construction."""
    from netlist_agent.llm.tools_schema import _resolve_bit, _resolve_bits

    design = session.current_design
    assert design is not None
    n10 = _resolve_bits(design, "n10")
    n2 = _resolve_bit(design, "n2")
    n20 = _resolve_bit(design, "n20")
    return [
        ("do_limit_fanout_global", {"max_fanout": 5}, limit_fanout, (5,)),
        ("do_limit_fanout_net", {"net": "n10", "max_fanout": 3}, limit_fanout_net, (n10, 3)),
        ("do_insert_buffer_per_load", {"net": "n10"}, insert_buffer_per_load, (n10,)),
        ("do_balance_depth_to_sinks", {"source": "n10", "sinks": ["n20"]}, balance_depth_to_sinks, (n10[0], [n20])),
        (
            "do_replace_buf_with_and",
            {"ctrl_net": "n2", "gate_names": ["g4"]},
            replace_buf_with_and,
            (["g4"], n2, []),
        ),
        ("do_remove_dangling_gates", {}, remove_dangling_gates, ()),
        ("do_deduplicate_gates", {}, deduplicate_gates, ()),
        ("do_collapse_double_inverters", {}, collapse_double_inverters, ()),
        ("do_collapse_inverter_buffer_chains", {}, collapse_inverter_buffer_chains, ()),
        ("do_simplify_constant_inputs", {"gate_types": ["and"]}, simplify_constant_inputs, ({GateType.AND},)),
        ("do_remap_to_basis", {"basis": "and"}, remap_to_basis, ("and_not",)),
    ]


@pytest.mark.parametrize("case_index", range(11))
def test_all_eleven_mutating_tools_refuse_the_correctly_wired_rerun_and_perform_no_mutation(
    tmp_path, case_index: int
) -> None:
    """Same wiring coverage as before (a wrong fn/args passed to
    _refuse_rerun/_rerun_conflict_fields would leave this green), now
    checking the REFUSAL shape (default, no force) rather than the old
    disclosure-while-mutating shape: no mutation key from the underlying
    transform, the full refusal dict, and the design's fingerprint
    untouched."""
    from netlist_agent.router import _normalize_op_args

    session = _new_session(tmp_path)
    cases = _build_rerun_wiring_cases(session)
    tool_name, kwargs, fn, raw_args = cases[case_index]

    SENTINEL_COUNT = 424242
    session.last_op_kind = fn.__name__
    session.last_op_args = _normalize_op_args(*raw_args)
    session.last_op_count = SENTINEL_COUNT
    fp_before = design_fingerprint(session.current_design)

    result = TOOL_REGISTRY[tool_name](session, **kwargs)
    fp_after = design_fingerprint(session.current_design)

    assert result.get(_MUTATION_PERFORMED_KEY) is False, (tool_name, result)
    assert result.get(_RERUN_OF_PRIOR_OPERATION_KEY) is True, (tool_name, result)
    assert result.get(_PREVIOUSLY_REPORTED_COUNT_KEY) == SENTINEL_COUNT, (tool_name, result)
    assert result.get(_REFUSED_REASON_KEY), (tool_name, result)
    assert result.get(_FORCE_OVERRIDE_PARAM_KEY) == _FORCE_PARAM_NAME, (tool_name, result)
    assert fp_before == fp_after, (tool_name, "design changed despite a refused rerun")


@pytest.mark.parametrize("case_index", range(11))
def test_all_eleven_mutating_tools_force_true_bypasses_the_refusal(tmp_path, case_index: int) -> None:
    """The same 11 wired-rerun scenarios with force=True: the refusal must
    NOT fire (no _MUTATION_PERFORMED_KEY: False), and the disclosure fields
    from the actual rerun must still be present."""
    from netlist_agent.router import _normalize_op_args

    session = _new_session(tmp_path)
    cases = _build_rerun_wiring_cases(session)
    tool_name, kwargs, fn, raw_args = cases[case_index]

    SENTINEL_COUNT = 424242
    session.last_op_kind = fn.__name__
    session.last_op_args = _normalize_op_args(*raw_args)
    session.last_op_count = SENTINEL_COUNT

    result = TOOL_REGISTRY[tool_name](session, force=True, **kwargs)
    assert result[_MUTATION_PERFORMED_KEY] is True, (tool_name, result)
    assert result.get(_RERUN_OF_PRIOR_OPERATION_KEY) is True, (tool_name, result)
    assert result.get(_PREVIOUSLY_REPORTED_COUNT_KEY) == SENTINEL_COUNT, (tool_name, result)


# ----------------------------------------------------------------------
# Defect 2: loading a new design must reset every "last operation" field,
# on BOTH load paths (router._h_load and llm.tools_schema.load_design) --
# otherwise a design's genuinely FIRST mutating operation can come back
# tagged as a rerun of an operation that ran on a design already replaced.
# ----------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_ROUTER, reason="requires the private rule-based router (not present in this public export)")
def test_h_load_resets_last_operation_state(tmp_path) -> None:
    from netlist_agent.router import handle_request as _handle_request

    session = _new_session(tmp_path)

    def _no_fallback(s: Session, t: str) -> str:
        raise AssertionError("fallback must not run: this request is rule-routed")

    result1 = TOOL_REGISTRY["do_deduplicate_gates"](session)
    assert _RERUN_OF_PRIOR_OPERATION_KEY not in result1  # first dedup, nothing to flag against
    session.last_op_kind = "deduplicate_gates"
    session.last_op_args = ()
    session.last_op_count = 7
    assert session.last_op_kind is not None

    path2 = str(tmp_path / "second.v")
    write_verilog(_dup_gate_design(), path2)
    reply = _handle_request(session, f"Please load the design from {path2}.", _no_fallback)
    assert "second" in reply or "Loaded" in reply

    assert session.last_op_kind is None
    assert session.last_op_args is None
    assert session.last_op_count is None
    assert session.last_gate_delta == {}

    # And the design's genuinely first dedup pass must not be flagged either.
    result2 = TOOL_REGISTRY["do_deduplicate_gates"](session)
    assert result2["merged"] == 1  # the one duplicate pair in _dup_gate_design
    assert _RERUN_OF_PRIOR_OPERATION_KEY not in result2, result2


def test_tool_layer_load_design_resets_last_operation_state(tmp_path) -> None:
    from netlist_agent.llm.tools_schema import load_design

    session = _new_session(tmp_path)
    session.last_op_kind = "deduplicate_gates"
    session.last_op_args = ()
    session.last_op_count = 7
    session.last_gate_delta = {GateType.AND: -1}

    path2 = str(tmp_path / "second.v")
    write_verilog(_dup_gate_design(), path2)
    load_design(session, "second.v", str(tmp_path))

    assert session.last_op_kind is None
    assert session.last_op_args is None
    assert session.last_op_count is None
    assert session.last_gate_delta == {}

    result = TOOL_REGISTRY["do_deduplicate_gates"](session)
    assert result["merged"] == 1
    assert _RERUN_OF_PRIOR_OPERATION_KEY not in result, result


# ----------------------------------------------------------------------
# End-to-end (rule-routed, then a do_* tool rerun) coverage for the two
# `session.last_op_args` write sites nothing else here reaches:
# `_run_and_track_bits` (bus-wide sweeps) and `_h_balance_depth`.
# ----------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_ROUTER, reason="requires the private rule-based router (not present in this public export)")
def test_end_to_end_rerun_flag_via_run_and_track_bits(tmp_path) -> None:
    """Rule-routed "insert buffers on signal n10 so each load..." runs via
    `_run_and_track_bits`, then the do_insert_buffer_per_load TOOL is asked
    to rerun the identical whole-signal request -- must be REFUSED (no
    mutation), reporting the earlier count instead."""
    session = _new_session(tmp_path)

    def _no_fallback(s: Session, t: str) -> str:
        raise AssertionError("fallback must not run: this request is rule-routed")

    handle_request(session, "Insert a buf gate on signal n10 so that each load gets its own dedicated buffer.", _no_fallback)
    assert session.last_op_kind == "insert_buffer_per_load"
    first_count = session.last_op_count
    assert first_count is not None and first_count > 0

    result = TOOL_REGISTRY["do_insert_buffer_per_load"](session, net="n10")
    assert result[_MUTATION_PERFORMED_KEY] is False
    assert result[_RERUN_OF_PRIOR_OPERATION_KEY] is True
    assert result[_PREVIOUSLY_REPORTED_COUNT_KEY] == first_count


@pytest.mark.skipif(not _HAS_REAL_ROUTER, reason="requires the private rule-based router (not present in this public export)")
def test_end_to_end_rerun_flag_via_h_balance_depth(tmp_path) -> None:
    """Rule-routed "balance the depth from n10 to {n20}" runs via
    `_h_balance_depth`, then the do_balance_depth_to_sinks TOOL is asked to
    rerun the identical (source, sinks) request -- must be REFUSED (no
    mutation), reporting the earlier count instead."""
    session = _new_session(tmp_path)

    def _no_fallback(s: Session, t: str) -> str:
        raise AssertionError("fallback must not run: this request is rule-routed")

    handle_request(session, "Balance the depth from n10 to {n20}.", _no_fallback)
    assert session.last_op_kind == "balance_depth_to_sinks"
    first_count = session.last_op_count
    assert first_count is not None

    result = TOOL_REGISTRY["do_balance_depth_to_sinks"](session, source="n10", sinks=["n20"])
    assert result[_MUTATION_PERFORMED_KEY] is False
    assert result[_RERUN_OF_PRIOR_OPERATION_KEY] is True
    assert result[_PREVIOUSLY_REPORTED_COUNT_KEY] == first_count


def test_remap_to_basis_scope_is_the_intersection_of_cone_and_type(tmp_path) -> None:
    """`cone_root` and `gate_type` together must report the INTERSECTION as
    the scope, not either one alone.

    Neither of the two scope tests covers this combination, which a cold read
    pointed out: each passes one restriction only, so a `gates_in_scope` that
    ignored the second would still look right in both.
    """
    session = _new_session(tmp_path)
    design = session.current_design
    assert design is not None
    cone = set(TOOL_REGISTRY["get_fanin_cone_gates"](session, net="n20")["items"])
    ands_everywhere = {g.inst_name for g in design.gates if g.gate_type is GateType.AND}
    expected = len(cone & ands_everywhere)
    # The point of the test is lost if either restriction alone gives the
    # same number, so the fixture is asserted to make them differ.
    assert expected != len(cone) and expected != len(ands_everywhere), (
        f"fixture makes the intersection trivial: cone={len(cone)} ands={len(ands_everywhere)} both={expected}"
    )

    result = TOOL_REGISTRY["do_remap_to_basis"](session, basis="nand", gate_type="and", cone_root="n20")

    assert result["gates_in_scope"] == expected
    assert "fanin cone of n20" in result["scope"] and "type and" in result["scope"]


def test_floating_signals_count_excludes_dead_wires_even_when_there_are_many() -> None:
    """The count must stay narrow while the not-counted lists are FULL.

    A mutation run is why this exists. `test_check_floating_signals` does
    assert the count against the engine's own headline, and that assertion
    has teeth -- but only for categories the shared fixture happens to
    populate. `dead_internal_wire_bits` is empty there, so widening the count
    by exactly that category changed nothing and no test noticed.

    It is also the category that broke in production: on a corpus design the
    tool returned a count of 0 beside 34 dead wire bits, and two of three
    held-out requests reported "35" and "38" instead of 0
    (experiments/heldout_fallback_2026-08-28/). The fixture was empty in
    precisely the place the failure lived.
    """
    design = _build_design()
    for i in range(5):  # declared, undriven, unread: dead by definition
        design.signals[f"dead{i}"] = Signal(name=f"dead{i}", msb=None, lsb=None, direction=Direction.INTERNAL)
    design.build_indices()
    session = Session()
    session.current_design = design

    result = TOOL_REGISTRY["check_floating_signals"](session)
    counted = result["counted_in_that_number"]
    extra = result["additional_findings_not_counted_above"]

    assert len(extra["dead_internal_wire_bits"]) >= 5, extra
    assert result["floating_input_nets_plus_unconnected_output_ports_count"] == len(
        counted["floating_input_nets_referenced_but_undriven"]
    ) + len(counted["unconnected_output_ports_undriven"])
    # And the whole point: a large not-counted list does not move the count.
    baseline = Session()
    baseline.current_design = _build_design()
    assert (
        result["floating_input_nets_plus_unconnected_output_ports_count"]
        == TOOL_REGISTRY["check_floating_signals"](baseline)[
            "floating_input_nets_plus_unconnected_output_ports_count"
        ]
    )


def test_replace_buf_with_and_scope_counts_distinguish_named_from_buf(tmp_path) -> None:
    """`names_in_scope` counts what was asked for; `buf_candidates_in_scope`
    counts what could actually be rewritten. Naming three gates of which one
    is a BUF has to show up as 3 and 1 -- otherwise `replaced: 0` cannot be
    told apart from "the scope was empty", which is the distinction the
    field was added for. Dropping the BUF filter survived a mutation run
    before this test existed.
    """
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["do_replace_buf_with_and"](
        session, ctrl_net="n2", gate_names=["g4", "g0", "g1"]  # g4 is the BUF; g0/g1 are AND/NAND
    )
    assert result["names_in_scope"] == 3
    assert result["buf_candidates_in_scope"] == 1
    assert result["replaced"] == 1


def test_floating_signals_note_is_derived_from_the_payload_it_describes() -> None:
    """The note is data the model reads, and prose has no mutation coverage.

    Inverting the hand-written version of this sentence -- so it told the
    model that every category WAS included in the count -- left all 256
    tests green. It is now built from the two groups it describes, and this
    pins that: every counted category is named on the counted side, every
    not-counted category on the other, so the sentence cannot drift from the
    structure the way a hand-maintained one silently did.
    """
    session = Session()
    session.current_design = _build_design()
    result = TOOL_REGISTRY["check_floating_signals"](session)
    note = result["note"]

    counted = list(result["counted_in_that_number"])
    not_counted = list(result["additional_findings_not_counted_above"])
    assert counted and not_counted
    head, _, tail = note.partition("are NOT part of that count")
    assert tail == "."
    for key in counted:
        assert key in head, f"{key} is counted but the note never names it"
    for key in not_counted:
        assert key in head, f"{key} is excluded but the note never names it"
    # and the two are on the right sides of the sentence
    counts_clause, _, excludes_clause = head.partition("are real structural observations")
    for key in counted:
        assert key in counts_clause.split("The categories under")[0]
    for key in not_counted:
        assert key not in counts_clause.split("The categories under")[0]


def test_replace_buf_with_and_reports_a_buf_skipped_for_a_self_loop(tmp_path) -> None:
    """`replaced: 0` has three causes, and this is the one the payload used
    to deny existed.

    g4 is a BUF whose output net is n20. Asking to rewrite it with
    ctrl_net=n20 would wire its own output back to its input, so
    transform.replace_buf_with_and skips it -- a real BUF, not rewritten.
    The tool never passed the primitive's `skipped_self_loop` list, so the
    payload said `buf_candidates_in_scope: 1, replaced: 0`, which its own
    description explained as "every named gate resolved to something other
    than a BUF". That explanation was false, and the rule-routed path has
    always reported these skips. Found by a cold read of this batch's diff.
    """
    session = _new_session(tmp_path)
    result = TOOL_REGISTRY["do_replace_buf_with_and"](session, ctrl_net="n20", gate_names=["g4"])

    assert result["replaced"] == 0
    assert result["buf_candidates_in_scope"] == 1  # it IS a BUF
    assert result["skipped_self_loop"] == ["g4"]  # and this is why it was left alone
    assert session.current_design is not None
    assert {g.inst_name: g.gate_type for g in session.current_design.gates}["g4"] is GateType.BUF


def test_replace_buf_with_and_scope_string_names_which_list_it_used(tmp_path) -> None:
    """The two ways of choosing gates must not describe themselves the same
    way. Forcing both branches to emit one string left all 245 tests green.
    """
    explicit = TOOL_REGISTRY["do_replace_buf_with_and"](
        _new_session(tmp_path), ctrl_net="n2", gate_names=["g4"]
    )
    session = _new_session(tmp_path)
    TOOL_REGISTRY["find_gates_by_name"](session, substring="g4")
    remembered = TOOL_REGISTRY["do_replace_buf_with_and"](session, ctrl_net="n2")

    assert explicit["scope"] != remembered["scope"]
    assert "gate_names" in explicit["scope"]
    assert "find_gates_by_name" in remembered["scope"]
    assert remembered["names_in_scope"] >= 1, "the remembered-query branch found nothing to act on"


def test_remap_to_basis_accepts_the_long_form_basis_names(tmp_path) -> None:
    """`do_remap_to_basis` takes {and,nand,nor}; the four do_optimize_* tools
    take {and_not,and_or_not,nand_not,nor_not}. Disjoint vocabularies under
    one parameter name. The aliases make a long-form guess work here rather
    than error; deleting the alias table left all 245 tests green.
    """
    short = TOOL_REGISTRY["do_remap_to_basis"](_new_session(tmp_path), basis="nand", gate_type="and")
    long = TOOL_REGISTRY["do_remap_to_basis"](_new_session(tmp_path), basis="nand_not", gate_type="and")
    assert short == long

    with pytest.raises(ToolError) as exc:
        TOOL_REGISTRY["do_remap_to_basis"](_new_session(tmp_path), basis="and_or_not")
    # and_or_not is real, but only for the do_optimize_* family -- the error
    # has to point at that rather than just listing three words.
    assert "and_or_not" in str(exc.value) and "do_optimize" in str(exc.value)
