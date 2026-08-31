"""Config parsing unit tests, plus end-to-end integration tests that run
netlist_agent.cli.run() against real Alpha_Testcase corpus files (feeding a
real prompt.txt's lines as stdin, cwd set to Alpha_Testcase/ so its
testcase/testNN/ relative paths resolve). Only a representative sample of
the 40 real testcases is run through this expensive full pipeline: test01
(trivial, no dff, load+write only), test04 (no dff, a fanin-depth query),
test21 (has dff instances, exercises the buffer-insertion path), and test17
(larger, has dff instances and an internal-signal-equivalence ABC check) --
covering both small/no-dff and larger/with-dff cases without running all 40
through ABC-backed checks.
"""

from __future__ import annotations

import io
import os
import re
import types

import pytest
import yaml

from netlist_agent.cli import (
    Config,
    ConfigError,
    GenerationConfig,
    ProviderConfig,
    _build_provider_client,
    _resolve_api_key,
    build_llm_fallback,
    load_config,
    run,
)
from netlist_agent.parser import parse_verilog
from netlist_agent.session import Session

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA_ROOT = os.path.join(REPO_ROOT, "Alpha_Testcase")

_VALID_CONFIG = {
    "provider": "openai",
    "openai": {"api_key": "sk-test", "model": "gpt-4o-mini"},
    "anthropic": {"api_key": "ant-test", "model": "claude-haiku-4-5"},
    "generation": {"temperature": 0.2, "max_output_tokens": 4096},
}


def _write_config(tmp_path, data: dict) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


# ----------------------------------------------------------------------
# Config parsing
# ----------------------------------------------------------------------


def test_load_config_valid(tmp_path) -> None:
    path = _write_config(tmp_path, _VALID_CONFIG)
    config = load_config(path)
    assert config.provider == "openai"
    assert config.openai.api_key == "sk-test"
    assert config.openai.model == "gpt-4o-mini"
    assert config.generation.temperature == 0.2
    assert config.generation.max_output_tokens == 4096


def test_load_config_anthropic_provider(tmp_path) -> None:
    data = dict(_VALID_CONFIG)
    data["provider"] = "anthropic"
    path = _write_config(tmp_path, data)
    config = load_config(path)
    assert config.provider == "anthropic"
    assert config.anthropic.model == "claude-haiku-4-5"


def test_load_config_missing_file(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "does_not_exist.yaml"))


def test_load_config_malformed_yaml(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("provider: [this is: not, valid")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_load_config_missing_provider_section(tmp_path) -> None:
    data = {"provider": "openai"}
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_invalid_provider_value(tmp_path) -> None:
    data = dict(_VALID_CONFIG)
    data["provider"] = "not-a-real-provider"
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_not_a_mapping(tmp_path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_load_config_base_url_optional(tmp_path) -> None:
    path = _write_config(tmp_path, _VALID_CONFIG)
    config = load_config(path)
    assert config.openai.base_url is None


def test_load_config_parses_base_url(tmp_path) -> None:
    data = dict(_VALID_CONFIG)
    data["openai"] = dict(data["openai"], base_url="https://example.invalid/v1beta/openai/")
    path = _write_config(tmp_path, data)
    config = load_config(path)
    assert config.openai.base_url == "https://example.invalid/v1beta/openai/"


def test_load_config_gemini_style_no_anthropic_section(tmp_path) -> None:
    data = {
        "provider": "openai",
        "openai": {
            "api_key": "env:SOME_TEST_VAR_NAME",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-2.0-flash",
        },
        "generation": {"temperature": 0.2, "max_output_tokens": 4096},
    }
    path = _write_config(tmp_path, data)
    config = load_config(path)
    assert config.anthropic is None
    assert config.openai.api_key == "env:SOME_TEST_VAR_NAME"
    assert config.openai.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


# ----------------------------------------------------------------------
# api_key `env:VARNAME` indirection
# ----------------------------------------------------------------------


def test_resolve_api_key_literal_value_used_as_is() -> None:
    assert _resolve_api_key("sk-literal-value") == "sk-literal-value"


def test_resolve_api_key_env_indirection(monkeypatch) -> None:
    monkeypatch.setenv("NETLIST_AGENT_TEST_KEY", "the-real-secret")
    assert _resolve_api_key("env:NETLIST_AGENT_TEST_KEY") == "the-real-secret"


def test_resolve_api_key_env_var_unset_raises_without_leaking_value(monkeypatch) -> None:
    monkeypatch.delenv("NETLIST_AGENT_TEST_KEY_UNSET", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        _resolve_api_key("env:NETLIST_AGENT_TEST_KEY_UNSET")
    assert "NETLIST_AGENT_TEST_KEY_UNSET" in str(exc_info.value)


def test_resolve_api_key_env_var_empty_string_raises(monkeypatch) -> None:
    monkeypatch.setenv("NETLIST_AGENT_TEST_KEY_EMPTY", "")
    with pytest.raises(ConfigError):
        _resolve_api_key("env:NETLIST_AGENT_TEST_KEY_EMPTY")


# ----------------------------------------------------------------------
# _build_provider_client: base_url + max_retries pass-through
# ----------------------------------------------------------------------


def test_build_provider_client_openai_passes_base_url_and_max_retries(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    _build_provider_client("openai", "sk-test", base_url="https://example.invalid/v1beta/openai/")
    assert captured["api_key"] == "sk-test"
    assert captured["base_url"] == "https://example.invalid/v1beta/openai/"
    assert captured["max_retries"] == 5


def test_build_provider_client_openai_no_base_url_when_unset(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    _build_provider_client("openai", "sk-test")
    assert "base_url" not in captured
    assert captured["max_retries"] == 5


def test_build_provider_client_openai_resolves_env_api_key(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("NETLIST_AGENT_TEST_KEY", "resolved-secret")
    _build_provider_client("openai", "env:NETLIST_AGENT_TEST_KEY")
    assert captured["api_key"] == "resolved-secret"


def test_build_provider_client_anthropic_passes_base_url(monkeypatch) -> None:
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    _build_provider_client("anthropic", "ant-test", base_url="https://example.invalid/")
    assert captured["api_key"] == "ant-test"
    assert captured["base_url"] == "https://example.invalid/"


# ----------------------------------------------------------------------
# build_llm_fallback: threads config's base_url into _build_provider_client
# and threads on_event through to LLMClient
# ----------------------------------------------------------------------


class _FakeSDKClient:
    """Stands in for the real openai.OpenAI()/anthropic.Anthropic() client
    that `_build_provider_client` would normally return: exposes just enough
    of the OpenAI-shaped surface (`.chat.completions.create`) for `LLMClient`
    to complete one no-tool-calls round and return final text."""

    def __init__(self, answer_text: str) -> None:
        self.answer_text = answer_text
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, model: str, messages: list, tools: list, temperature: float, max_tokens: int):
        message = types.SimpleNamespace(content=self.answer_text, tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def test_build_llm_fallback_passes_base_url_to_provider_client(monkeypatch) -> None:
    import netlist_agent.cli as cli_module

    captured: dict = {}

    def _fake_build_provider_client(provider, api_key, base_url=None):
        captured["provider"] = provider
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return _FakeSDKClient("fallback answer")

    monkeypatch.setattr(cli_module, "_build_provider_client", _fake_build_provider_client)

    config = Config(
        provider="openai",
        openai=ProviderConfig(
            api_key="sk-test", model="gpt-4o-mini", base_url="https://example.invalid/v1beta/openai/"
        ),
        anthropic=None,
        generation=GenerationConfig(temperature=0.2, max_output_tokens=4096),
    )

    build_llm_fallback(config)

    assert captured["provider"] == "openai"
    assert captured["api_key"] == "sk-test"
    assert captured["base_url"] == "https://example.invalid/v1beta/openai/"


def test_build_llm_fallback_threads_on_event_into_llm_client(monkeypatch) -> None:
    import netlist_agent.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_build_provider_client",
        lambda provider, api_key, base_url=None: _FakeSDKClient("fallback answer"),
    )

    config = Config(
        provider="openai",
        openai=ProviderConfig(api_key="sk-test", model="gpt-4o-mini"),
        anthropic=None,
        generation=GenerationConfig(temperature=0.2, max_output_tokens=4096),
    )

    events: list[dict] = []
    fallback = build_llm_fallback(config, on_event=events.append)
    answer = fallback(Session(), "some fallback question")

    assert answer == "fallback answer"
    assert any(e["type"] == "final" and e["text"] == "fallback answer" for e in events)


# ----------------------------------------------------------------------
# End-to-end integration against real testcases
# ----------------------------------------------------------------------

_RESPONSE_RE = re.compile(r"#RESPONSE (\d+)\n(.*?)\n#END \1\n", re.DOTALL)


def _run_testcase(name: str, monkeypatch) -> tuple[str, list[str]]:
    monkeypatch.chdir(ALPHA_ROOT)
    prompt_path = os.path.join("testcase", name, "prompt.txt")
    with open(prompt_path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]

    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    config = Config(
        provider="openai", openai=None, anthropic=None, generation=GenerationConfig(0.2, 4096)
    )
    run(config, stdin=stdin, stdout=stdout)

    log_path = f"{name}.log"
    assert os.path.exists(log_path)
    with open(log_path) as f:
        log_content = f.read()
    assert log_content == stdout.getvalue()
    os.remove(log_path)

    # Router handlers for cone/path-enumeration queries may write their own
    # "<case_name>_<stub>.txt" side files into cwd (ALPHA_ROOT) -- sweep up
    # anything matching before returning so this test leaves no stray state.
    for extra in os.listdir(ALPHA_ROOT):
        if extra.startswith(f"{name}_") and extra.endswith(".txt"):
            os.remove(os.path.join(ALPHA_ROOT, extra))

    return stdout.getvalue(), lines


@pytest.mark.parametrize("name", ["test01", "test04", "test21"])
@pytest.mark.skip(reason='requires the private rule-based router (not present in this public export) to produce deterministic no-LLM-call responses')
def test_cli_end_to_end_small(name: str, monkeypatch, tmp_path) -> None:
    output, lines = _run_testcase(name, monkeypatch)
    responses = _RESPONSE_RE.findall(output)
    assert len(responses) == len(lines)
    ids = [int(r[0]) for r in responses]
    assert ids == list(range(1, len(lines) + 1))

    out_v = os.path.join(ALPHA_ROOT, "testcase", name, f"{name}_out.v")
    assert os.path.exists(out_v)
    reparsed = parse_verilog(out_v)
    assert reparsed.gates
    os.remove(out_v)


@pytest.mark.skip(reason='requires the private rule-based router (not present in this public export) to produce deterministic no-LLM-call responses')
def test_cli_end_to_end_with_dff_larger(monkeypatch) -> None:
    output, lines = _run_testcase("test17", monkeypatch)
    responses = _RESPONSE_RE.findall(output)
    assert len(responses) == len(lines)

    out_v = os.path.join(ALPHA_ROOT, "testcase", "test17", "test17_out.v")
    assert os.path.exists(out_v)
    reparsed = parse_verilog(out_v)
    assert reparsed.gates
    os.remove(out_v)
