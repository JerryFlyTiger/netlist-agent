"""Entry point: `python -m netlist_agent.cli -config <path>`.

Reads stdin one line at a time (one request per line, per spec) until EOF,
dispatching each through `router.handle_request` and emitting the response
via `io_protocol`. Blank lines are skipped entirely (treated as not being a
request at all, not even an empty one) -- they don't consume a response id.

The very first non-blank line is expected to be the "beginning of a new
testcase" framing; it is detected specially here (via `router.BEGIN_RE`, the
same pattern `router.py` itself tolerates defensively if seen again) so the
`Session`/log file can be initialized before response 1 is emitted, exactly
matching the spec's "the very first response ... is response 1".
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional, TextIO

import yaml

from netlist_agent.io_protocol import respond
from netlist_agent.router import BEGIN_RE, Fallback, _extract_log_filename, handle_request
from netlist_agent.session import Session


_MAX_CONSECUTIVE_DECODE_ERRORS = 100


class ConfigError(Exception):
    """Raised when `-config`'s file is missing or structurally malformed."""


@dataclass
class ProviderConfig:
    api_key: str
    model: str
    base_url: Optional[str] = None


@dataclass
class GenerationConfig:
    temperature: float
    max_output_tokens: int


@dataclass
class Config:
    provider: str
    openai: Optional[ProviderConfig]
    anthropic: Optional[ProviderConfig]
    generation: GenerationConfig


def _parse_provider(raw: Any, name: str) -> Optional[ProviderConfig]:
    if raw is None:
        return None
    if not isinstance(raw, dict) or "api_key" not in raw or "model" not in raw:
        raise ConfigError(f"{name!r} section must be a mapping with 'api_key' and 'model' keys")
    base_url = raw.get("base_url")
    return ProviderConfig(
        api_key=str(raw["api_key"]), model=str(raw["model"]), base_url=str(base_url) if base_url is not None else None
    )


def load_config(path: str) -> Config:
    """Parse and structurally validate a `-config` YAML file. This stage
    only parses/stores the config -- no LLM client is built yet (that's an
    explicitly future stage); anything the rule-based router doesn't
    recognize falls through to a stub (see `not_yet_understood` below).
    """
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except OSError as exc:
        raise ConfigError(f"could not read config file {path!r}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {path!r} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path!r} must contain a YAML mapping at the top level")

    provider = raw.get("provider")
    if provider not in ("openai", "anthropic"):
        raise ConfigError("config 'provider' must be either 'openai' or 'anthropic'")

    openai_cfg = _parse_provider(raw.get("openai"), "openai")
    anthropic_cfg = _parse_provider(raw.get("anthropic"), "anthropic")
    selected = openai_cfg if provider == "openai" else anthropic_cfg
    if selected is None:
        raise ConfigError(f"provider is {provider!r} but no matching {provider!r} section was found")

    gen_raw = raw.get("generation") or {}
    if not isinstance(gen_raw, dict):
        raise ConfigError("config 'generation' section must be a mapping")
    try:
        generation = GenerationConfig(
            temperature=float(gen_raw.get("temperature", 0.2)),
            max_output_tokens=int(gen_raw.get("max_output_tokens", 4096)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config 'generation' section has a non-numeric field: {exc}") from exc

    return Config(provider=provider, openai=openai_cfg, anthropic=anthropic_cfg, generation=generation)


def _harden_text_stream(stream: Any) -> None:
    """Best-effort: make `stream` never raise on a byte/character it can't
    translate, by reconfiguring it (if it supports that -- real `stdin`/
    `stdout` do) to replace undecodable/unencodable data with U+FFFD/`?`
    rather than raising. This is deliberately best-effort, not a hard
    requirement: `io.StringIO`/`io.TextIOWrapper` around a `BytesIO` (both
    common in tests, and the latter one test here uses on purpose) have no
    `reconfigure` method at all, and callers are free to hand `run()` any
    text-like stream. When reconfiguration isn't available or itself fails,
    this silently does nothing and the caller-level safety nets (see
    `run()`'s manual iteration and the per-line exception guard) are what
    keep a single bad line from taking down the whole testcase.

    After this succeeds, an undecodable input byte becomes U+FFFD and an
    unencodable output character becomes `?` -- the affected request line
    still flows all the way through routing/fallback and gets a normal,
    well-formed response; it just may not roundtrip byte-for-byte."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(errors="replace")
    except Exception:  # noqa: BLE001 -- best effort only
        pass


def _internal_error_body(exc: Exception) -> str:
    return f"Internal error while handling this request ({type(exc).__name__}: {exc}); skipping to the next line."


def _undecodable_input_body(exc: UnicodeDecodeError) -> str:
    return (
        f"Internal error while reading this request line (UnicodeDecodeError: {exc}); "
        "the line containing the undecodable bytes has been skipped and processing continues "
        "with the next line."
    )


def _decode_error_backstop_body(count: int) -> str:
    return (
        f"Internal error: {count} consecutive request lines in a row failed to decode "
        "(UnicodeDecodeError); the input stream appears unable to advance past this point, "
        "so this testcase ends here."
    )


def not_yet_understood(session: Session, text: str) -> str:
    """Fallback hook used only when no real LLM-backed fallback is supplied
    (e.g. a test that doesn't care about fallback behavior at all). Real
    usage always goes through `build_llm_fallback` below -- `router.handle_request`
    takes a fallback as a plain callable parameter rather than hard-wiring one
    internally, so this and the real thing are interchangeable.
    """
    return "This request is not yet understood by the rule-based router."


def _resolve_api_key(raw_key: str) -> str:
    """Resolve an `api_key` config value: `env:VARNAME` indirects through an
    environment variable (resolved lazily here, not at parse time, so config
    files can be loaded/validated without the variable being set yet); any
    other value is used literally. Never logs or echoes the resolved key --
    only the variable *name* may appear in the error message."""
    if raw_key.startswith("env:"):
        var_name = raw_key[len("env:") :]
        value = os.environ.get(var_name)
        if not value:
            raise ConfigError(f"environment variable {var_name!r} is not set (required by api_key: 'env:{var_name}')")
        return value
    return raw_key


def _build_provider_client(provider: str, api_key: str, base_url: Optional[str] = None) -> Any:
    """Construct the real OpenAI/Anthropic SDK client. Imported lazily here
    (rather than at module level) so that merely importing `cli` -- as every
    test importing this module does -- never requires either SDK package to
    be importable in isolation, and never touches the network."""
    resolved_key = _resolve_api_key(api_key)
    if provider == "openai":
        import openai

        kwargs: dict[str, Any] = {"api_key": resolved_key, "max_retries": 5}
        if base_url is not None:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)
    if provider == "anthropic":
        import anthropic

        kwargs = {"api_key": resolved_key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        return anthropic.Anthropic(**kwargs)
    raise ConfigError(f"unsupported provider {provider!r}")


def build_llm_fallback(config: Config, on_event: Optional[Callable[[dict], None]] = None) -> Fallback:
    """Build the real LLM-backed fallback from a parsed `Config`: a real
    provider SDK client plus `netlist_agent.llm.client.LLMClient`, closing over
    both so `run()` can hand it to `router.handle_request` as a plain
    `Fallback` callable. `on_event`, if given, is threaded through to
    `LLMClient` as its trace hook (see `llm/client.py`)."""
    from netlist_agent.llm.client import LLMClient

    provider_cfg = config.openai if config.provider == "openai" else config.anthropic
    if provider_cfg is None:
        raise ConfigError(f"no configuration section found for provider {config.provider!r}")

    sdk_client = _build_provider_client(config.provider, provider_cfg.api_key, provider_cfg.base_url)
    llm_client = LLMClient(
        provider=config.provider,
        client=sdk_client,
        model=provider_cfg.model,
        temperature=config.generation.temperature,
        max_output_tokens=config.generation.max_output_tokens,
        on_event=on_event,
    )

    def fallback(session: Session, text: str) -> str:
        return llm_client.answer(session, text)

    return fallback


def _lazy_llm_fallback(config: Config) -> Fallback:
    """A `Fallback` that defers building the real LLM client (and thus
    validating `config`'s provider section) until the very first time it is
    actually invoked. This lets `run()` default to real LLM-backed routing
    without forcing every caller (including tests that never exercise the
    fallback path at all) to supply a fully-populated provider config."""
    built: list[Fallback] = []

    def fallback(session: Session, text: str) -> str:
        if not built:
            built.append(build_llm_fallback(config))
        return built[0](session, text)

    return fallback


def run(
    config: Config,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    fallback: Optional[Fallback] = None,
) -> Session:
    session = Session()
    resolved_fallback = fallback if fallback is not None else _lazy_llm_fallback(config)
    _harden_text_stream(stdin)
    _harden_text_stream(stdout)
    # `close()`'s emergency-log warnings can embed a non-ASCII case/design
    # name (e.g. `session.py`'s `_warn()` calls) -- without this, a narrow
    # stderr encoding raised `UnicodeEncodeError` right out of `close()`,
    # measured end to end: a non-ASCII load filename plus an ASCII-encoded
    # `sys.stderr` reproduced it before this line was added, and no longer
    # does after.
    _harden_text_stream(sys.stderr)
    first_line = True
    line_iter = iter(stdin)
    consecutive_decode_errors = 0
    while True:
        try:
            raw_line = next(line_iter)
        except StopIteration:
            break
        except UnicodeDecodeError as exc:
            # Reached only if `stdin` couldn't be hardened above (e.g. it has
            # no `reconfigure` method) and the bytes underneath it are
            # undecodable. Measured (not assumed) behavior of `continue` here,
            # both against `io_protocol.py`'s interactive mode (scorer sends
            # one line, waits for `#END`, sends the next) and batch mode
            # (stdin pre-filled with a whole file/buffer):
            #
            # - Interactive: feeding 5 lines one at a time via os.pipe, with
            #   the 3rd containing an invalid byte, each failing `next()` call
            #   only ever raises for that single line -- `continue` yields
            #   ['line one', 'line two', '<DECODE ERROR>', 'line four',
            #   'line five']: 5 lines in, 5 items out, no id drift. The bad
            #   byte only costs its own line.
            # - Batch: a 20711-byte, 2301-line stdin with one bad byte inside
            #   the first ~8KB chunk. `continue` recovers 1391 lines, hits
            #   exactly 1 decode error, and reaches a normal EOF; `break`
            #   recovers 0 lines. The cost, stated plainly: the underlying
            #   `TextIOWrapper` decodes a whole ~8KB chunk at a time, and the
            #   one failing `next()` silently drops that entire pending chunk
            #   -- NOT merely the part after the bad byte. Measured by walking
            #   the bad byte across a 3000-line stdin: at byte offsets 10,
            #   1000 and 8000 alike, the first line recovered is line 1170,
            #   so ~1170 perfectly valid lines *preceding* the bad byte are
            #   lost with it, and response ids stop lining up 1:1 with input
            #   line numbers. `break` loses every line from the first chunk
            #   onward in this mode, so `continue` still dominates it.
            #
            # `continue` wins in both modes, so that's what happens here, one
            # response per decode error to keep "one line in, one response
            # out" for the interactive-mode case. The one thing `continue`
            # genuinely cannot rule out is a caller-supplied stream that always
            # raises and never advances at all (a real `TextIOWrapper` does
            # not do this -- confirmed above) -- `_MAX_CONSECUTIVE_DECODE_ERRORS`
            # below is the backstop against exactly that.
            consecutive_decode_errors += 1
            if consecutive_decode_errors > _MAX_CONSECUTIVE_DECODE_ERRORS:
                respond(session, _decode_error_backstop_body(consecutive_decode_errors), stdout=stdout)
                break
            respond(session, _undecodable_input_body(exc), stdout=stdout)
            continue
        consecutive_decode_errors = 0
        line = raw_line.rstrip("\n").rstrip("\r")
        if not line.strip():
            continue
        if first_line:
            first_line = False
            m = BEGIN_RE.search(line.strip())
            if m:
                try:
                    session.start(m.group(1), _extract_log_filename(line.strip()))
                    body = f"This is the beginning of testcase {m.group(1)}."
                except Exception as exc:  # noqa: BLE001 -- see the handler guard below
                    body = _internal_error_body(exc)
                respond(session, body, stdout=stdout)
                continue
        # A single request line's handler must never be allowed to take down
        # the whole process -- protocol integrity (every request line gets a
        # well-formed #RESPONSE/#END pair, with ids still incrementing by
        # one) matters more than any single answer, since scoring reads the
        # log. KeyboardInterrupt/SystemExit are deliberately NOT caught here
        # (bare `except Exception`, not a bare `except:`) so the process can
        # still be interrupted/exited normally.
        try:
            body = handle_request(session, line, resolved_fallback)
        except Exception as exc:  # noqa: BLE001 -- intentionally broad, see above
            body = _internal_error_body(exc)
        respond(session, body, stdout=stdout)
    # Unguarded on purpose: `Session.close()`'s docstring now guarantees it
    # never raises OSError (an OSError from its emergency log write, or from
    # the warning about it, is caught and swallowed internally), so nothing
    # here needs a try/except for that. A non-OSError from close() would
    # still be a real bug and is meant to propagate.
    session.close()
    return session


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="netlist-agent")
    parser.add_argument("-config", dest="config_path", required=True)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
