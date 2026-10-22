"""Unit tests for netlist_agent/llm/client.py's provider-agnostic agentic
tool-calling loop, driven entirely against small fake/mock SDK client
objects -- no network access or API key is ever touched. OpenAI gets the
full case matrix (single tool call, multi-round, bad/unknown tool calls,
iteration-cap enforcement); Anthropic gets one lighter end-to-end pass
confirming the same loop behavior through its adapter.
"""

from __future__ import annotations

import json
import types
from typing import Any, Callable, Optional

import pytest

from netlist_agent import deadline
from netlist_agent.ir import Design, Direction, Gate, GateType, NetBit, Port, Signal
from netlist_agent.llm.client import DEFAULT_MAX_ITERATIONS, LLMClient, ToolCallRequest, _INCOMPLETE_MESSAGE, _execute_tool
from netlist_agent.llm.tools_schema import TOOL_REGISTRY
from netlist_agent.session import Session

RespondFn = Callable[[int, list], dict]


def _tiny_design() -> Design:
    design = Design(module_name="tiny")
    design.signals["a"] = Signal(name="a", msb=None, lsb=None, direction=Direction.INPUT)
    design.signals["b"] = Signal(name="b", msb=None, lsb=None, direction=Direction.INPUT)
    design.signals["y"] = Signal(name="y", msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports = [Port("a", Direction.INPUT), Port("b", Direction.INPUT), Port("y", Direction.OUTPUT)]
    design.gates = [Gate("g0", GateType.AND, {"O": NetBit("y"), "I0": NetBit("a"), "I1": NetBit("b")})]
    design.build_indices()
    return design


def _session() -> Session:
    session = Session()
    session.current_design = _tiny_design()
    session.original_snapshot = _tiny_design()
    return session


# ----------------------------------------------------------------------
# Fake OpenAI-shaped client
# ----------------------------------------------------------------------


class FakeOpenAIClient:
    """Stands in for `openai.OpenAI()`: exposes `.chat.completions.create(...)`
    returning an object shaped like the real SDK's response (`.choices[0].message`
    with `.content` / `.tool_calls`), scripted via `respond_fn(round, messages)`."""

    def __init__(self, respond_fn: RespondFn) -> None:
        self.respond_fn = respond_fn
        self.rounds = 0
        self.timeouts: list = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(
        self, model: str, messages: list, tools: list, temperature: float, max_tokens: int, timeout: Any = None
    ) -> Any:
        self.rounds += 1
        self.timeouts.append(timeout)
        step = self.respond_fn(self.rounds, messages)
        tool_calls = None
        if step.get("tool_calls"):
            tool_calls = [
                types.SimpleNamespace(
                    id=tc["id"],
                    function=types.SimpleNamespace(name=tc["name"], arguments=tc.get("raw_arguments", json.dumps(tc.get("arguments", {})))),
                )
                for tc in step["tool_calls"]
            ]
        message = types.SimpleNamespace(content=step.get("text"), tool_calls=tool_calls)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _openai_client(script: list[dict]) -> FakeOpenAIClient:
    """A FakeOpenAIClient driven by a fixed list of per-round steps; the last
    step repeats forever once the list is exhausted (handy for the
    iteration-cap test, which needs the model to "never" stop on its own)."""

    def respond(round_no: int, messages: list) -> dict:
        idx = min(round_no - 1, len(script) - 1)
        return script[idx]

    return FakeOpenAIClient(respond)


# ----------------------------------------------------------------------
# Fake OpenAI SDK message exposing `model_dump()`, e.g. Gemini's
# OpenAI-compat endpoint, which stuffs non-standard fields (like a
# thought-signature `extra_content`) onto each tool call that must survive
# an assistant-message round-trip or the follow-up turn gets rejected.
# ----------------------------------------------------------------------


class _ModelDumpMessage:
    """A minimal stand-in for such a message: `.content` / `.tool_calls`
    attributes for the adapter's existing (non-model_dump) reading logic,
    plus a `model_dump(exclude_none=...)` method whose returned dict carries
    an `extra_content` field per tool call that only a full round-trip
    (rather than hand-reconstructing the message) would preserve."""

    def __init__(self, content: Optional[str], tool_calls: list[dict]) -> None:
        self.content = content
        self.tool_calls = [
            types.SimpleNamespace(
                id=tc["id"],
                function=types.SimpleNamespace(
                    name=tc["name"], arguments=tc.get("raw_arguments", json.dumps(tc.get("arguments", {})))
                ),
            )
            for tc in tool_calls
        ]
        self._tool_calls_raw = tool_calls
        self.model_dump_calls: list[dict] = []

    def model_dump(self, exclude_none: bool = False) -> dict:
        self.model_dump_calls.append({"exclude_none": exclude_none})
        dumped: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc.get("raw_arguments", json.dumps(tc.get("arguments", {}))),
                    },
                    "extra_content": {"thought_signature": f"sig-{tc['id']}"},
                }
                for tc in self._tool_calls_raw
            ],
        }
        if not exclude_none:
            dumped["content"] = self.content
        return dumped


def test_openai_model_dump_round_trip_preserves_extra_fields() -> None:
    messages_by_round: list[_ModelDumpMessage] = []
    captured_messages: list[list[dict]] = []

    class _ModelDumpClient:
        def __init__(self) -> None:
            self.rounds = 0
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def _create(self, model: str, messages: list, tools: list, temperature: float, max_tokens: int) -> Any:
            self.rounds += 1
            captured_messages.append(list(messages))
            if self.rounds == 1:
                msg = _ModelDumpMessage(
                    content=None, tool_calls=[{"id": "call_1", "name": "count_gates_by_type", "arguments": {}}]
                )
            else:
                msg = _ModelDumpMessage(content="The design has 1 gate.", tool_calls=[])
            messages_by_round.append(msg)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    client = _ModelDumpClient()
    llm = LLMClient(provider="openai", client=client, model="fake-model")
    answer = llm.answer(_session(), "how many gates are there really")
    assert answer == "The design has 1 gate."

    # model_dump() was called exactly once for round 1's message, with
    # exclude_none=True.
    assert messages_by_round[0].model_dump_calls == [{"exclude_none": True}]

    # Round 2's request history includes the assistant message built from
    # round 1's model_dump() output: role/content survived (content key
    # present as None via setdefault), and the tool call's extra_content
    # round-tripped whole -- only possible via the model_dump branch, not
    # the hand-reconstructed fallback.
    history_after_round1 = captured_messages[1]
    assistant_entries = [m for m in history_after_round1 if m.get("role") == "assistant"]
    assert len(assistant_entries) == 1
    assistant_msg = assistant_entries[0]
    assert assistant_msg["role"] == "assistant"
    assert "content" in assistant_msg and assistant_msg["content"] is None
    assert assistant_msg["tool_calls"][0]["extra_content"] == {"thought_signature": "sig-call_1"}

    tool_entries = [m for m in history_after_round1 if m.get("role") == "tool"]
    assert len(tool_entries) == 1
    assert tool_entries[0]["tool_call_id"] == assistant_msg["tool_calls"][0]["id"]


# ----------------------------------------------------------------------
# (a) single tool call then final answer
# ----------------------------------------------------------------------


def test_openai_single_tool_call_then_final_answer() -> None:
    client = _openai_client(
        [
            {"tool_calls": [{"id": "call_1", "name": "count_gates_by_type", "arguments": {}}]},
            {"text": "The design has 1 gate."},
        ]
    )
    llm = LLMClient(provider="openai", client=client, model="fake-model")
    answer = llm.answer(_session(), "how many gates are there really")
    assert answer == "The design has 1 gate."
    assert client.rounds == 2


# ----------------------------------------------------------------------
# (b) sequence of several tool calls across multiple rounds
# ----------------------------------------------------------------------


def test_openai_multi_round_tool_calls() -> None:
    client = _openai_client(
        [
            {"tool_calls": [{"id": "c1", "name": "count_gates_by_type", "arguments": {}}]},
            {"tool_calls": [{"id": "c2", "name": "get_gate_info", "arguments": {"gate": "g0"}}]},
            {"tool_calls": [{"id": "c3", "name": "get_net_fanout", "arguments": {"net": "a"}}]},
            {"text": "g0 is an AND gate; a fans out to it."},
        ]
    )
    llm = LLMClient(provider="openai", client=client, model="fake-model")
    answer = llm.answer(_session(), "tell me about g0 and a")
    assert answer == "g0 is an AND gate; a fans out to it."
    assert client.rounds == 4


# ----------------------------------------------------------------------
# (c) unknown tool / bad arguments -> clear error fed back, no crash
# ----------------------------------------------------------------------


def test_openai_unknown_tool_feeds_back_error() -> None:
    seen_tool_results: list[str] = []

    def respond(round_no: int, messages: list) -> dict:
        if round_no == 1:
            return {"tool_calls": [{"id": "c1", "name": "not_a_real_tool", "arguments": {}}]}
        # Capture what the model "saw" as the tool result before finalizing.
        last = messages[-1]
        seen_tool_results.append(last["content"])
        return {"text": "That tool does not exist; here is a fallback answer."}

    client = FakeOpenAIClient(respond)
    llm = LLMClient(provider="openai", client=client, model="fake-model")
    answer = llm.answer(_session(), "call something bogus")
    assert answer == "That tool does not exist; here is a fallback answer."
    assert len(seen_tool_results) == 1
    assert "unknown tool" in json.loads(seen_tool_results[0])["error"]


def test_openai_bad_arguments_feeds_back_error_not_crash() -> None:
    def respond(round_no: int, messages: list) -> dict:
        if round_no == 1:
            # get_gate_info requires `gate`; this passes a wrong kwarg name.
            return {"tool_calls": [{"id": "c1", "name": "get_gate_info", "arguments": {"gate_name": "g0"}}]}
        return {"text": "Recovered from the bad call."}

    client = FakeOpenAIClient(respond)
    llm = LLMClient(provider="openai", client=client, model="fake-model")
    answer = llm.answer(_session(), "look up a gate wrong")
    assert answer == "Recovered from the bad call."


def test_openai_unparseable_json_arguments_feeds_back_error() -> None:
    def respond(round_no: int, messages: list) -> dict:
        if round_no == 1:
            return {"tool_calls": [{"id": "c1", "name": "get_gate_info", "raw_arguments": "{not valid json"}]}
        last = messages[-1]
        assert "could not parse tool call arguments" in json.loads(last["content"])["error"]
        return {"text": "Handled the parse error."}

    client = FakeOpenAIClient(respond)
    llm = LLMClient(provider="openai", client=client, model="fake-model")
    answer = llm.answer(_session(), "send malformed json args")
    assert answer == "Handled the parse error."


def test_tool_raising_exception_feeds_back_error_not_crash() -> None:
    def respond(round_no: int, messages: list) -> dict:
        if round_no == 1:
            return {"tool_calls": [{"id": "c1", "name": "get_gate_info", "arguments": {"gate": "no_such_gate"}}]}
        last = messages[-1]
        assert "no such gate" in json.loads(last["content"])["error"]
        return {"text": "That gate does not exist."}

    client = FakeOpenAIClient(respond)
    llm = LLMClient(provider="openai", client=client, model="fake-model")
    answer = llm.answer(_session(), "look up a nonexistent gate")
    assert answer == "That gate does not exist."


# ----------------------------------------------------------------------
# (d) iteration cap actually stops an infinitely-tool-calling model
# ----------------------------------------------------------------------


def test_iteration_cap_stops_infinite_tool_calling() -> None:
    def respond(round_no: int, messages: list) -> dict:
        return {"tool_calls": [{"id": f"c{round_no}", "name": "count_gates_by_type", "arguments": {}}]}

    client = FakeOpenAIClient(respond)
    llm = LLMClient(provider="openai", client=client, model="fake-model", max_iterations=5)
    answer = llm.answer(_session(), "just keep calling tools forever")
    assert client.rounds == 5
    assert answer == _INCOMPLETE_MESSAGE


def test_iteration_cap_uses_default_when_unset() -> None:
    def respond(round_no: int, messages: list) -> dict:
        return {"tool_calls": [{"id": f"c{round_no}", "name": "count_gates_by_type", "arguments": {}}]}

    client = FakeOpenAIClient(respond)
    llm = LLMClient(provider="openai", client=client, model="fake-model")
    llm.answer(_session(), "loop forever")
    assert client.rounds == DEFAULT_MAX_ITERATIONS


def test_iteration_cap_returns_last_seen_text_if_any() -> None:
    """If the model never stops calling tools but did emit some running
    commentary text along the way, the cap-hit fallback prefers that over the
    generic incomplete message."""

    def respond(round_no: int, messages: list) -> dict:
        return {"text": f"still working, round {round_no}", "tool_calls": [{"id": f"c{round_no}", "name": "count_gates_by_type", "arguments": {}}]}

    client = FakeOpenAIClient(respond)
    llm = LLMClient(provider="openai", client=client, model="fake-model", max_iterations=3)
    answer = llm.answer(_session(), "keep narrating and calling tools")
    assert answer == "still working, round 3"


# ----------------------------------------------------------------------
# Anthropic: lighter pass confirming the same loop behavior
# ----------------------------------------------------------------------


class FakeAnthropicClient:
    """Stands in for `anthropic.Anthropic()`: exposes `.messages.create(...)`
    returning an object shaped like the real SDK's response (`.content`, a
    list of text/tool_use blocks)."""

    def __init__(self, respond_fn: RespondFn) -> None:
        self.respond_fn = respond_fn
        self.rounds = 0
        self.messages = types.SimpleNamespace(create=self._create)
        self.last_system: Optional[str] = None

    def _create(self, model: str, system: str, messages: list, tools: list, temperature: float, max_tokens: int) -> Any:
        self.rounds += 1
        self.last_system = system
        step = self.respond_fn(self.rounds, messages)
        blocks = []
        if step.get("text"):
            blocks.append(types.SimpleNamespace(type="text", text=step["text"]))
        for tc in step.get("tool_calls", []):
            blocks.append(types.SimpleNamespace(type="tool_use", id=tc["id"], name=tc["name"], input=tc.get("arguments", {})))
        stop_reason = "tool_use" if step.get("tool_calls") else "end_turn"
        return types.SimpleNamespace(content=blocks, stop_reason=stop_reason)


def test_anthropic_single_tool_call_then_final_answer() -> None:
    def respond(round_no: int, messages: list) -> dict:
        if round_no == 1:
            return {"tool_calls": [{"id": "toolu_1", "name": "count_gates_by_type", "arguments": {}}]}
        return {"text": "The design has 1 gate."}

    client = FakeAnthropicClient(respond)
    llm = LLMClient(provider="anthropic", client=client, model="fake-claude")
    answer = llm.answer(_session(), "how many gates, anthropic edition")
    assert answer == "The design has 1 gate."
    assert client.rounds == 2
    assert "gate-level Verilog" in client.last_system


def test_anthropic_unknown_tool_feeds_back_error() -> None:
    def respond(round_no: int, messages: list) -> dict:
        if round_no == 1:
            return {"tool_calls": [{"id": "toolu_1", "name": "nonexistent_tool", "arguments": {}}]}
        last = messages[-1]
        result_content = last["content"][0]["content"]
        assert "unknown tool" in json.loads(result_content)["error"]
        return {"text": "Handled it."}

    client = FakeAnthropicClient(respond)
    llm = LLMClient(provider="anthropic", client=client, model="fake-claude")
    answer = llm.answer(_session(), "call something bogus, anthropic edition")
    assert answer == "Handled it."


def test_anthropic_iteration_cap() -> None:
    def respond(round_no: int, messages: list) -> dict:
        return {"tool_calls": [{"id": f"toolu_{round_no}", "name": "count_gates_by_type", "arguments": {}}]}

    client = FakeAnthropicClient(respond)
    llm = LLMClient(provider="anthropic", client=client, model="fake-claude", max_iterations=4)
    answer = llm.answer(_session(), "loop forever, anthropic edition")
    assert client.rounds == 4
    assert answer == _INCOMPLETE_MESSAGE


# ----------------------------------------------------------------------
# (e) optional trace hook (`on_event`): tool_call / final / max_iterations
# events, and hook exceptions never break the loop
# ----------------------------------------------------------------------


def test_on_event_default_none_does_not_affect_behavior() -> None:
    client = _openai_client(
        [
            {"tool_calls": [{"id": "call_1", "name": "count_gates_by_type", "arguments": {}}]},
            {"text": "The design has 1 gate."},
        ]
    )
    llm = LLMClient(provider="openai", client=client, model="fake-model")
    assert llm.on_event is None
    answer = llm.answer(_session(), "how many gates are there really")
    assert answer == "The design has 1 gate."


def test_on_event_receives_tool_call_and_final_events() -> None:
    events: list[dict] = []
    client = _openai_client(
        [
            {"tool_calls": [{"id": "call_1", "name": "count_gates_by_type", "arguments": {}}]},
            {"text": "The design has 1 gate."},
        ]
    )
    llm = LLMClient(provider="openai", client=client, model="fake-model", on_event=events.append)
    answer = llm.answer(_session(), "how many gates are there really")
    assert answer == "The design has 1 gate."

    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    final_events = [e for e in events if e["type"] == "final"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0]["name"] == "count_gates_by_type"
    assert tool_call_events[0]["arguments"] == {}
    assert tool_call_events[0]["iteration"] == 0
    assert isinstance(tool_call_events[0]["result"], str)
    assert len(final_events) == 1
    assert final_events[0]["text"] == "The design has 1 gate."


def test_on_event_truncates_tool_result_to_500_chars(monkeypatch) -> None:
    from netlist_agent.llm import client as client_module

    def _long_result_tool(session, **kwargs) -> str:
        return "x" * 1000

    # Register a throwaway tool that actually produces a >500-char JSON
    # result, so the assertion below exercises the truncation itself rather
    # than merely tolerating it.
    monkeypatch.setitem(client_module.TOOL_REGISTRY, "big_output_tool", _long_result_tool)

    events: list[dict] = []

    def respond(round_no: int, messages: list) -> dict:
        if round_no == 1:
            return {"tool_calls": [{"id": "c1", "name": "big_output_tool", "arguments": {}}]}
        return {"text": "done"}

    client = FakeOpenAIClient(respond)
    llm = LLMClient(provider="openai", client=client, model="fake-model", on_event=events.append)
    llm.answer(_session(), "trigger a long tool result")
    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_call_events) == 1
    assert len(tool_call_events[0]["result"]) == 500


def test_on_event_receives_max_iterations_event() -> None:
    events: list[dict] = []

    def respond(round_no: int, messages: list) -> dict:
        return {"tool_calls": [{"id": f"c{round_no}", "name": "count_gates_by_type", "arguments": {}}]}

    client = FakeOpenAIClient(respond)
    llm = LLMClient(provider="openai", client=client, model="fake-model", max_iterations=3, on_event=events.append)
    answer = llm.answer(_session(), "just keep calling tools forever")
    assert answer == _INCOMPLETE_MESSAGE
    assert any(e["type"] == "max_iterations" for e in events)


def test_on_event_exception_does_not_break_the_loop() -> None:
    def bad_hook(event: dict) -> None:
        raise RuntimeError("boom")

    client = _openai_client(
        [
            {"tool_calls": [{"id": "call_1", "name": "count_gates_by_type", "arguments": {}}]},
            {"text": "The design has 1 gate."},
        ]
    )
    llm = LLMClient(provider="openai", client=client, model="fake-model", on_event=bad_hook)
    answer = llm.answer(_session(), "how many gates are there really")
    assert answer == "The design has 1 gate."


# ----------------------------------------------------------------------
# Request deadline: mutating tools get the transform budget (batch 21b)
# ----------------------------------------------------------------------

_NON_MUTATING_TOOLS = frozenset(
    {
        "check_dffs_same_clock_domain",
        "check_equivalence_to_snapshot",
        "check_floating_signals",
        "check_gate_on_max_depth_path",
        "check_is_cut_signal",
        "check_path_exists",
        "check_property_asserted_only_when",
        "check_signal_equivalence",
        "check_symmetry_tool",
        "count_gates_by_type",
        "count_outputs_over_depth",
        "count_paths",
        "count_primary_ports",
        "enumerate_paths",
        "find_gates_by_name",
        "find_signal_pair_for_operator",
        "get_boolean_function",
        "get_cone_gate_type_breakdown",
        "get_constant_value",
        "get_cut_nets_between",
        "get_depth_between",
        "get_depth_of_cone",
        "get_direct_pi_po_connections",
        "get_fanin_cone_gates",
        "get_fanin_cone_size",
        "get_fanout_cone_gates",
        "get_gate_direct_fanout",
        "get_gate_fanout_count",
        "get_gate_info",
        "get_gates_connected_to_signal",
        "get_largest_fanin_cone",
        "get_last_operation_summary",
        "get_max_design_depth",
        "get_max_fanout_of_signal",
        "get_max_fanout_primary_input",
        "get_max_pi_to_dff_d_depth",
        "get_max_reg_to_reg_depth",
        "get_net_fanout",
        "get_reg_to_reg_path_stats",
        "get_shared_fanin_gates",
        "list_dffs_on_clock",
        "list_gates_of_type",
        "list_gates_with_constant_input",
        "list_primary_outputs_over_cone_size",
        "list_primary_ports",
        "load_design",
        "set_testcase",
        "write_design",
    }
)


@pytest.fixture
def _fake_clock(monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(deadline, "_monotonic", lambda: now["t"])
    return now


def _remaining_seen_by(monkeypatch, name: str) -> float:
    seen: list = []

    def tool(session, **kwargs):
        seen.append(deadline.remaining())
        return {"ok": True}

    monkeypatch.setitem(TOOL_REGISTRY, name, tool)
    with deadline.request_scope(deadline.ANALYSIS_BUDGET):
        out = _execute_tool(_session(), ToolCallRequest(id="1", name=name, arguments={}))
    assert json.loads(out) == {"ok": True}
    return seen[0]


@pytest.mark.parametrize("name", ["do_fake_transform", "rename_fake"])
def test_execute_tool_extends_for_mutating_prefixes(monkeypatch, _fake_clock, name) -> None:
    """kills: no extend() in _execute_tool; a prefix (`do_` / `rename_`) dropped from the tuple."""
    assert _remaining_seen_by(monkeypatch, name) == deadline.TRANSFORM_BUDGET


_REAL_MUTATING_TOOLS = sorted(k for k in TOOL_REGISTRY if k.startswith(("do_", "rename_")))


def test_real_tool_name_lists_are_not_empty() -> None:
    """A zero-length parametrization passes silently; fail here instead."""
    assert len(_NON_MUTATING_TOOLS) > 0
    assert len(_REAL_MUTATING_TOOLS) > 0


@pytest.mark.parametrize("name", sorted(_NON_MUTATING_TOOLS))
def test_execute_tool_does_not_extend_other_tools(monkeypatch, _fake_clock, name) -> None:
    """kills: extend() applied to every tool (every real non-mutating registry name)."""
    assert _remaining_seen_by(monkeypatch, name) == deadline.ANALYSIS_BUDGET


@pytest.mark.parametrize("name", _REAL_MUTATING_TOOLS)
def test_execute_tool_extends_for_every_real_mutating_tool(monkeypatch, _fake_clock, name) -> None:
    """kills: a real do_*/rename_* tool that does not get the transform budget."""
    assert _remaining_seen_by(monkeypatch, name) == deadline.TRANSFORM_BUDGET


def test_execute_tool_unknown_do_tool_does_not_extend(_fake_clock) -> None:
    """kills: extend() placed before the TOOL_REGISTRY lookup."""
    with deadline.request_scope(deadline.ANALYSIS_BUDGET):
        out = _execute_tool(_session(), ToolCallRequest(id="1", name="do_not_a_tool", arguments={}))
        left = deadline.remaining()
    assert "unknown tool" in json.loads(out)["error"]
    assert left == deadline.ANALYSIS_BUDGET


def test_execute_tool_parse_error_does_not_extend(_fake_clock) -> None:
    """kills: extend() placed before the parse_error check."""
    with deadline.request_scope(deadline.ANALYSIS_BUDGET):
        out = _execute_tool(_session(), ToolCallRequest(id="1", name="do_optimize_depth", arguments={}, parse_error="x"))
        left = deadline.remaining()
    assert "could not parse" in json.loads(out)["error"]
    assert left == deadline.ANALYSIS_BUDGET


def test_registry_names_without_mutating_prefix_are_pinned() -> None:
    """A new tool whose name does not start with do_/rename_ fails here, so someone decides on
    purpose whether it needs the transform budget (rename it, or add it to the set below)."""
    actual = frozenset(k for k in TOOL_REGISTRY if not k.startswith(("do_", "rename_")))
    assert actual == _NON_MUTATING_TOOLS


# ----------------------------------------------------------------------
# SPEC_21c: request deadline in the LLM loop (retries, per-request timeout, Incomplete exit)
# ----------------------------------------------------------------------

import netlist_agent.llm.client as _client_module  # noqa: E402
from netlist_agent.llm.client import _is_retryable  # noqa: E402

_NO_TIMEOUT = object()  # the fake's default: "the caller did not pass timeout= at all"


class _StatusError(Exception):
    """A fake SDK error: just a status_code, optionally a response with headers."""

    def __init__(self, status_code: int, headers: Optional[dict] = None) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        if headers is not None:
            self.response = types.SimpleNamespace(headers=headers)


class APIConnectionError(Exception):
    """Same class name as the SDKs' (the retry check matches by name, not by import)."""


class APITimeoutError(APIConnectionError):
    pass


class _ScriptedClient:
    """A fake SDK client of either provider. Each create() call: records whether/what timeout= was
    passed, runs on_call(), then plays steps[i] (the last repeats): an exception is raised,
    a dict is turned into a response ({"text"} / {"tool_calls": [...]})."""

    def __init__(self, provider: str, steps: list, on_call: Optional[Callable[[], None]] = None) -> None:
        self.steps = steps
        self.on_call = on_call
        self.calls = 0
        self.timeouts: list = []
        if provider == "openai":
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._openai_create))
        else:
            self.messages = types.SimpleNamespace(create=self._anthropic_create)
        self.provider = provider

    def _step(self, timeout: Any) -> Any:
        self.timeouts.append(timeout)
        i = self.calls
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        step = self.steps[min(i, len(self.steps) - 1)]
        if isinstance(step, BaseException):
            raise step
        return step

    def _openai_create(self, model, messages, tools, temperature, max_tokens, timeout=_NO_TIMEOUT) -> Any:
        step = self._step(timeout)
        tool_calls = None
        if step.get("tool_calls"):
            tool_calls = [
                types.SimpleNamespace(
                    id=tc["id"], function=types.SimpleNamespace(name=tc["name"], arguments=json.dumps(tc.get("arguments", {})))
                )
                for tc in step["tool_calls"]
            ]
        message = types.SimpleNamespace(content=step.get("text"), tool_calls=tool_calls)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    def _anthropic_create(self, model, system, messages, tools, temperature, max_tokens, timeout=_NO_TIMEOUT) -> Any:
        step = self._step(timeout)
        blocks = []
        if step.get("text"):
            blocks.append(types.SimpleNamespace(type="text", text=step["text"]))
        for tc in step.get("tool_calls", []):
            blocks.append(types.SimpleNamespace(type="tool_use", id=tc["id"], name=tc["name"], input=tc.get("arguments", {})))
        return types.SimpleNamespace(content=blocks, stop_reason="tool_use" if step.get("tool_calls") else "end_turn")


class _Env:
    def __init__(self, clock: dict) -> None:
        self.clock = clock
        self.sleeps: list = []
        self.events: list = []


@pytest.fixture
def _env(monkeypatch):
    """Fake clock + a `_sleep` that records the delay and advances the fake clock.
    No test in this section may ever really sleep."""
    clock = {"t": 0.0}
    env = _Env(clock)
    monkeypatch.setattr(deadline, "_monotonic", lambda: clock["t"])

    def fake_sleep(delay: float) -> None:
        env.sleeps.append(delay)
        clock["t"] += delay

    monkeypatch.setattr(_client_module, "_sleep", fake_sleep)
    return env


def _llm(client: _ScriptedClient, env: _Env) -> LLMClient:
    return LLMClient(provider=client.provider, client=client, model="fake-model", on_event=env.events.append)


_OK = {"text": "done"}


def _incomplete(budget: float) -> str:
    return deadline.incomplete_message(deadline.DeadlineExceeded(budget))


def _cls(name: str, base: type = Exception) -> type:
    return type(name, (base,), {})


def test_t2_is_retryable_by_name_and_status() -> None:
    """SPEC_21c T2 (fake classes). kills: dropped APIConnectionError name branch / MRO walk; wrong status set."""
    assert _is_retryable(APIConnectionError())
    assert _is_retryable(APITimeoutError())
    # the name must be equal to, not merely contain, "APIConnectionError" (F9)
    assert not _is_retryable(_cls("FooAPIConnectionErrorBar")())
    for status in (408, 409, 429, 500, 503):
        assert _is_retryable(_StatusError(status)), status
    for status in (400, 401, 403, 404, 422):
        assert not _is_retryable(_StatusError(status)), status
    assert not _is_retryable(ValueError("x"))


@pytest.mark.parametrize("sdk", ["openai", "anthropic"])
def test_t2_is_retryable_real_sdk_classes(sdk: str) -> None:
    """SPEC_21c T2 (real SDK classes, built with __new__). Pins that both SDKs really use these class names."""
    mod = pytest.importorskip(sdk)
    for name, status, expected in (
        ("RateLimitError", 429, True),
        ("InternalServerError", 500, True),
        ("BadRequestError", 400, False),
    ):
        cls = getattr(mod, name)
        exc = cls.__new__(cls)
        exc.status_code = status
        assert _is_retryable(exc) is expected, (sdk, name)
    for name in ("APIConnectionError", "APITimeoutError"):
        cls = getattr(mod, name)
        assert _is_retryable(cls.__new__(cls)) is True, (sdk, name)


def test_t3_timeout_is_remaining_minus_margin(_env) -> None:
    """SPEC_21c T3. kills: wrong margin; timeout computed once outside the loop; timeout not passed."""
    client = _ScriptedClient(
        "openai",
        [{"tool_calls": [{"id": "c1", "name": "count_gates_by_type", "arguments": {}}]}, _OK],
        on_call=lambda: _env.clock.__setitem__("t", _env.clock["t"] + (10.0 if client.calls == 1 else 0.0)),
    )
    with deadline.request_scope(45.0):
        assert _llm(client, _env).answer(_session(), "q") == "done"
    # round 1 sees 45 - 2; the call itself takes 10 s; round 2 sees 45 - 10 - 2.
    assert client.timeouts == [43.0, 33.0]


def test_t4_threshold_exactly_3_sends_below_does_not(_env) -> None:
    """SPEC_21c T4. kills: `<` -> `<=`; missing pre-send check; wrong Incomplete text."""
    client = _ScriptedClient("openai", [_OK])
    with deadline.request_scope(45.0):
        _env.clock["t"] = 42.0
        assert _llm(client, _env).answer(_session(), "q") == "done"
    assert client.timeouts == [1.0]

    client = _ScriptedClient("openai", [_OK])
    _env.clock["t"] = 0.0  # a new scope starts at the clock's current reading
    with deadline.request_scope(45.0):
        _env.clock["t"] = 42.01
        out = _llm(client, _env).answer(_session(), "q")
    assert client.calls == 0
    assert out == _incomplete(45)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_t5_no_scope_passes_no_timeout_kwarg(provider: str) -> None:
    """SPEC_21c T5. kills: adapter always passing timeout=None (or a value) outside any scope."""
    client = _ScriptedClient(provider, [_OK])
    assert LLMClient(provider=provider, client=client, model="m").answer(_session(), "q") == "done"
    assert client.timeouts == [_NO_TIMEOUT]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_t5_scope_passes_timeout_kwarg(provider: str, _env) -> None:
    """SPEC_21c T5 (positive control). kills: an adapter that never forwards timeout= even inside a scope."""
    client = _ScriptedClient(provider, [_OK])
    with deadline.request_scope(45.0):
        _llm(client, _env).answer(_session(), "q")
    assert client.timeouts == [43.0]


def test_t6_retry_then_success(_env) -> None:
    """SPEC_21c T6. kills: wrong first delay (2 ** (attempt+1)); wrong/missing retry event; no retry."""
    client = _ScriptedClient("openai", [_StatusError(429), _OK])
    assert _llm(client, _env).answer(_session(), "q") == "done"
    assert client.calls == 2
    assert _env.sleeps == [0.5]
    retries = [e for e in _env.events if e["type"] == "retry"]
    assert retries == [{"type": "retry", "attempt": 1, "error": "_StatusError", "delay": 0.5}]


def test_t7_retry_cut_by_deadline(_env) -> None:
    """SPEC_21c T7. kills: missing `left - delay` check; sleeping before checking."""
    client = _ScriptedClient(
        "openai", [_StatusError(503)], on_call=lambda: _env.clock.__setitem__("t", _env.clock["t"] + 1.0)
    )
    with deadline.request_scope(5.0):
        out = _llm(client, _env).answer(_session(), "q")
    # send 1: left 5.0 -> timeout 3.0, fails with 4.0 left, delay 0.5, 3.5 >= 3 -> sleep, 3.5 left.
    # send 2: timeout 1.5, fails with 2.5 left, delay 1.0, 1.5 < 3 -> no sleep.
    assert client.calls == 2
    assert client.timeouts == [3.0, 1.5]
    assert _env.sleeps == [0.5]
    assert out == _incomplete(5)
    assert len([e for e in _env.events if e["type"] == "deadline"]) == 1
    assert len([e for e in _env.events if e["type"] == "retry"]) == 1


def test_t8_retry_cap_without_scope(_env) -> None:
    """SPEC_21c T8. kills: `>=` -> `>` on the cap; backoff cap/base changes; raising a new exception."""
    errors: list = []

    def make() -> _StatusError:
        e = _StatusError(429)
        errors.append(e)
        return e

    client = _ScriptedClient("openai", [_OK])

    def create(*args, **kwargs):
        client.calls += 1
        raise make()

    client.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    with pytest.raises(_StatusError) as info:
        _llm(client, _env).answer(_session(), "q")
    assert client.calls == 6
    assert _env.sleeps == [0.5, 1.0, 2.0, 4.0, 8.0]
    assert len(errors) == 6
    assert info.value is errors[-1]


def test_t9_non_retryable_raises_immediately(_env) -> None:
    """SPEC_21c T9. kills: retrying 4xx; swallowing/replacing the original exception."""
    err = _StatusError(400)
    client = _ScriptedClient("openai", [err])
    with pytest.raises(_StatusError) as info:
        _llm(client, _env).answer(_session(), "q")
    assert info.value is err
    assert client.calls == 1
    assert _env.sleeps == []


@pytest.mark.parametrize(
    "header, expected_delay",
    [("3", 3.0), ("Wed, 21 Oct 2026 07:28:00 GMT", 0.5), (None, 0.5)],
)
def test_t10_retry_after_header(_env, header, expected_delay) -> None:
    """SPEC_21c T10. kills: `max` -> `min`; Retry-After ignored; HTTP-date crashing the parser;
    missing .response not tolerated (header None means: exception has no response attribute)."""
    err = _StatusError(429) if header is None else _StatusError(429, {"retry-after": header})
    client = _ScriptedClient("openai", [err, _OK])
    assert _llm(client, _env).answer(_session(), "q") == "done"
    assert _env.sleeps == [expected_delay]


def test_t10_retry_after_over_60_does_not_retry(_env) -> None:
    """SPEC_21c T10. kills: missing `> 60 -> do not retry` branch."""
    err = _StatusError(429, {"retry-after": "100"})
    client = _ScriptedClient("openai", [err, _OK])
    with pytest.raises(_StatusError) as info:
        _llm(client, _env).answer(_session(), "q")
    assert info.value is err
    assert client.calls == 1
    assert _env.sleeps == []


def test_t11_tool_raising_deadline_exceeded_is_incomplete(_env, monkeypatch) -> None:
    """SPEC_21c T11. kills: no outer `except DeadlineExceeded` in answer(); no deadline event; event budget hard-coded."""

    def tool(session, **kwargs):
        raise deadline.DeadlineExceeded(37)

    monkeypatch.setitem(TOOL_REGISTRY, "fake_slow_tool", tool)
    client = _ScriptedClient("openai", [{"tool_calls": [{"id": "c1", "name": "fake_slow_tool", "arguments": {}}]}, _OK])
    with deadline.request_scope(45.0):
        out = _llm(client, _env).answer(_session(), "q")
    # budget 37 != the scope's 45: the exit must report the exception's budget, not a constant.
    assert out == _incomplete(37)
    assert client.calls == 1
    assert [e for e in _env.events if e["type"] == "deadline"] == [{"type": "deadline", "budget": 37}]


def test_t13_timeout_recomputed_after_extend(_env, monkeypatch) -> None:
    """SPEC_21c T13. kills: `left` computed once outside the loop; extend() not reflected in the next timeout."""

    def tool(session, **kwargs):
        _env.clock["t"] += 5.0
        return {"ok": True}

    monkeypatch.setitem(TOOL_REGISTRY, "do_fake_slow", tool)
    client = _ScriptedClient("openai", [{"tool_calls": [{"id": "c1", "name": "do_fake_slow", "arguments": {}}]}, _OK])
    with deadline.request_scope(45.0):
        assert _llm(client, _env).answer(_session(), "q") == "done"
    # clock starts at 0; do_ tool extends 45 -> 270 and burns 5 s: round 2 timeout = 270 - 5 - 2.
    assert client.timeouts == [43.0, 270.0 - 5.0 - 2.0]


def test_t14_undo_prefix_is_not_a_do_tool(_fake_clock, monkeypatch) -> None:
    """SPEC_21c T14 (SPEC_21b knife). kills: startswith -> `in` (`"do_" in "undo_fake"`)."""
    assert _remaining_seen_by(monkeypatch, "undo_fake") <= deadline.ANALYSIS_BUDGET


def _advance(env: _Env, seconds: float) -> Callable[[], None]:
    return lambda: env.clock.__setitem__("t", env.clock["t"] + seconds)


def test_c1_failure_that_burned_the_timeout_does_not_sleep(_env) -> None:
    """Round-1 C1. kills: dropping the post-failure re-read of remaining() (client.py `left = deadline.remaining()`)."""
    client = _ScriptedClient("openai", [APITimeoutError()], on_call=_advance(_env, 43.0))
    with deadline.request_scope(45.0):
        out = _llm(client, _env).answer(_session(), "q")
    assert client.calls == 1
    assert _env.sleeps == []
    assert out == _incomplete(45)
    assert [e for e in _env.events if e["type"] == "retry"] == []


def test_c2_retry_allowed_when_exactly_at_threshold(_env) -> None:
    """Round-1 C2. kills: `left - delay < 3` -> `<= 3`."""
    client = _ScriptedClient("openai", [_StatusError(503), _OK], on_call=_advance(_env, 1.5))
    with deadline.request_scope(5.0):
        out = _llm(client, _env).answer(_session(), "q")
    # send 1: timeout 3.0, fails with 3.5 left; delay 0.5 -> 3.0 left, exactly the threshold -> sleep.
    # send 2: left 3.0 -> timeout 1.0 (the fake advances 1.5 more, harmless).
    assert out == "done"
    assert _env.sleeps == [0.5]
    assert client.timeouts == [3.0, 1.0]


def test_c3_retry_after_never_shortens_the_backoff(_env) -> None:
    """Round-1 C3. kills: `delay = max(delay, ra)` -> `delay = ra`."""
    steps = [_StatusError(429), _StatusError(429, {"retry-after": "0.1"}), _OK]
    client = _ScriptedClient("openai", steps)
    assert _llm(client, _env).answer(_session(), "q") == "done"
    assert _env.sleeps == [0.5, 1.0]


def test_c4_retry_event_is_emitted_before_sleeping(_env, monkeypatch) -> None:
    """Round-1 C4. kills: swapping `_emit(retry)` and `_sleep(delay)`."""
    log: list = []
    monkeypatch.setattr(_client_module, "_sleep", lambda delay: log.append(("sleep", delay)))
    client = _ScriptedClient("openai", [_StatusError(429), _OK])
    llm = LLMClient(provider="openai", client=client, model="m", on_event=lambda e: log.append((e["type"], e.get("delay"))))
    assert llm.answer(_session(), "q") == "done"
    assert log.index(("retry", 0.5)) < log.index(("sleep", 0.5))


def test_c5_pre_send_exit_reports_the_scope_budget(_env) -> None:
    """Round-1 C5. kills: pre-send `DeadlineExceeded(current_budget())` -> a hard-coded 45."""
    client = _ScriptedClient("openai", [_OK])
    with deadline.request_scope(30.0):
        _env.clock["t"] = 28.0
        out = _llm(client, _env).answer(_session(), "q")
    assert client.calls == 0
    assert out == _incomplete(30)
