"""Provider-agnostic agentic tool-calling client: drives an "ask a question,
call tools, get a final text answer" turn against either the OpenAI or
Anthropic SDK, grounded in `netlist_agent.llm.tools_schema`'s tool registry.

The actual OpenAI/Anthropic SDK client object is a constructor parameter
(`LLMClient.client`), so tests substitute a small fake/mock object matching
just enough of the real SDK's shape -- no network access, no API key, ever
required to exercise this module.

Each round: send the accumulated message history + tool schema to the model;
if it returns tool call(s), execute each against the registry (catching any
exception and feeding a JSON error string back as that tool's result) and
loop; if it returns final text with no tool calls, that is the answer. A
bounded `max_iterations` is a safety net against a model that keeps calling
tools forever -- not something that should fire in normal operation. If it
does fire, the best partial text seen so far (if any) is returned, else a
clear "could not complete" message -- never a hang or a crash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from netlist_agent.llm.tools_schema import TOOL_REGISTRY, TOOL_SCHEMA, ToolSpec
from netlist_agent.session import Session

DEFAULT_MAX_ITERATIONS = 8
_INCOMPLETE_MESSAGE = (
    "I was unable to finish gathering the information needed to answer this within the "
    "allotted number of tool-call rounds. Please try rephrasing the request or breaking it "
    "into smaller steps."
)


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool call the model asked for, normalized across providers."""

    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: Optional[str] = None


@dataclass(frozen=True)
class ModelTurn:
    """One round's result: either final text (tool_calls empty) or one or
    more tool calls to execute before the next round."""

    text: Optional[str]
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


def _execute_tool(session: Session, call: ToolCallRequest) -> str:
    """Run one tool call against the registry, returning a JSON string --
    either the tool's JSON-serializable result, or `{"error": "..."}` if the
    tool raised, the tool name is unknown, or the arguments could not be
    parsed/bound. Never raises."""
    if call.parse_error is not None:
        return json.dumps({"error": f"could not parse tool call arguments as JSON: {call.parse_error}"})
    fn = TOOL_REGISTRY.get(call.name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {call.name!r}"})
    try:
        result = fn(session, **call.arguments)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any tool failure becomes a tool-result error
        return json.dumps({"error": str(exc)})
    try:
        return json.dumps(result)
    except TypeError as exc:
        return json.dumps({"error": f"tool {call.name!r} returned a non-JSON-serializable result: {exc}"})


def _openai_tools(schema: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in schema
    ]


def _anthropic_tools(schema: list[ToolSpec]) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in schema]


class OpenAIAdapter:
    """Talks to an OpenAI-SDK-shaped client: `client.chat.completions.create(...)`
    returning an object with `.choices[0].message.content` / `.tool_calls`.
    """

    def __init__(self, client: Any, model: str, temperature: float, max_output_tokens: int) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._tools = _openai_tools(TOOL_SCHEMA)

    def initial_history(self, system_prompt: str, user_message: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def request(self, history: list[dict[str, Any]]) -> ModelTurn:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            tools=self._tools,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        message = response.choices[0].message
        raw_tool_calls = list(getattr(message, "tool_calls", None) or [])

        if hasattr(message, "model_dump"):
            # Round-trip the full SDK message so provider-specific extra fields
            # survive; Gemini's OpenAI-compat endpoint rejects follow-up turns
            # whose tool calls lack their `extra_content` thought_signature.
            assistant_msg = message.model_dump(exclude_none=True)
            assistant_msg["role"] = "assistant"
            assistant_msg.setdefault("content", None)
        else:
            assistant_msg = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in raw_tool_calls
                ]
                or None,
            }
        history.append(assistant_msg)

        tool_calls: list[ToolCallRequest] = []
        for tc in raw_tool_calls:
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
                parse_error = None
            except json.JSONDecodeError as exc:
                args = {}
                parse_error = str(exc)
            tool_calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args, parse_error=parse_error))

        return ModelTurn(text=message.content, tool_calls=tool_calls)

    def add_tool_results(
        self, history: list[dict[str, Any]], tool_calls: list[ToolCallRequest], results: dict[str, str]
    ) -> None:
        for call in tool_calls:
            history.append({"role": "tool", "tool_call_id": call.id, "content": results[call.id]})


class AnthropicAdapter:
    """Talks to an Anthropic-SDK-shaped client: `client.messages.create(...)`
    returning an object with `.content` (a list of text/tool_use blocks) and
    `.stop_reason`.
    """

    def __init__(self, client: Any, model: str, temperature: float, max_output_tokens: int) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._tools = _anthropic_tools(TOOL_SCHEMA)
        self._system_prompt = ""

    def initial_history(self, system_prompt: str, user_message: str) -> list[dict[str, Any]]:
        self._system_prompt = system_prompt
        return [{"role": "user", "content": user_message}]

    def request(self, history: list[dict[str, Any]]) -> ModelTurn:
        response = self.client.messages.create(
            model=self.model,
            system=self._system_prompt,
            messages=history,
            tools=self._tools,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        blocks = list(response.content)
        history.append({"role": "assistant", "content": blocks})

        text_parts = [b.text for b in blocks if getattr(b, "type", None) == "text"]
        tool_calls = [
            ToolCallRequest(id=b.id, name=b.name, arguments=dict(b.input))
            for b in blocks
            if getattr(b, "type", None) == "tool_use"
        ]
        text = "\n".join(text_parts) if text_parts else None
        return ModelTurn(text=text, tool_calls=tool_calls)

    def add_tool_results(
        self, history: list[dict[str, Any]], tool_calls: list[ToolCallRequest], results: dict[str, str]
    ) -> None:
        content = [{"type": "tool_result", "tool_use_id": call.id, "content": results[call.id]} for call in tool_calls]
        history.append({"role": "user", "content": content})


_ADAPTERS: dict[str, Callable[[Any, str, float, int], Any]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}


def build_system_prompt(session: Session) -> str:
    """Small, bounded current-state context -- never the whole netlist text
    (some designs are 100k+ gates) -- so the model can reason about what is
    askable right now without guessing at structure it hasn't looked up."""
    design = session.current_design
    if design is None:
        state = "No design is currently loaded."
    else:
        from netlist_agent.analysis import gate_count_by_type, primary_input_port_count, primary_output_port_count

        total = sum(gate_count_by_type(design).values())
        state = (
            f"A design named {design.module_name!r} is currently loaded, with {total} gate(s) total, "
            f"{primary_input_port_count(design)} primary input port(s), and "
            f"{primary_output_port_count(design)} primary output port(s)."
        )
    return (
        "You are the fallback natural-language interpreter for a gate-level Verilog netlist "
        "exploration and transformation agent. A separate rule-based router already handles common, "
        "recognized phrasings of requests; you are invoked only when a request did not match any of "
        "its patterns, so it may be novel phrasing of a familiar capability, or something genuinely "
        "unusual.\n\n"
        "You MUST call the provided tools to inspect or modify the netlist -- never guess at gate "
        "names, signal names, counts, depths, or structure; every one of those must come from a tool "
        "result. When you have a complete, definitive answer (or have finished performing a requested "
        "transform), reply with a short, direct, plain-text final answer and stop calling tools -- do "
        "not narrate intermediate steps.\n\n"
        f"Current state: {state}"
    )


@dataclass
class LLMClient:
    provider: str
    client: Any
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 4096
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    on_event: Optional[Callable[[dict], None]] = None

    def __post_init__(self) -> None:
        if self.provider not in _ADAPTERS:
            raise ValueError(f"unsupported provider {self.provider!r}; choose one of {sorted(_ADAPTERS)}")

    def _emit(self, event: dict) -> None:
        """Best-effort trace hook dispatch: a caller's `on_event` must never
        be able to break the agentic loop, so any exception it raises is
        swallowed here."""
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:  # noqa: BLE001 -- deliberately broad: tracing must never break the loop
            pass

    def answer(self, session: Session, user_message: str) -> str:
        adapter = _ADAPTERS[self.provider](self.client, self.model, self.temperature, self.max_output_tokens)
        history = adapter.initial_history(build_system_prompt(session), user_message)

        last_text: Optional[str] = None
        for iteration in range(self.max_iterations):
            turn = adapter.request(history)
            if turn.text:
                last_text = turn.text
            if not turn.tool_calls:
                final_text = turn.text if turn.text is not None else (last_text or "")
                self._emit({"type": "final", "text": final_text})
                return final_text
            results = {call.id: _execute_tool(session, call) for call in turn.tool_calls}
            for call in turn.tool_calls:
                self._emit(
                    {
                        "type": "tool_call",
                        "iteration": iteration,
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": results[call.id][:500],
                    }
                )
            adapter.add_tool_results(history, turn.tool_calls, results)

        self._emit({"type": "max_iterations"})
        return last_text or _INCOMPLETE_MESSAGE
