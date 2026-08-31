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

from netlist_agent.ir import Design, Direction, Gate, GateType, NetBit, Port, Signal
from netlist_agent.llm.client import DEFAULT_MAX_ITERATIONS, LLMClient, _INCOMPLETE_MESSAGE
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
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, model: str, messages: list, tools: list, temperature: float, max_tokens: int) -> Any:
        self.rounds += 1
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
