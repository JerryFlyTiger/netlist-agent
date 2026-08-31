"""Coverage for netlist_agent.cli.run()'s per-line exception isolation: a
single request line whose handler raises must not take down the rest of the
testcase (or the process) -- it must still get a well-formed #RESPONSE/#END
pair with the next sequential id, and every following line must be processed
normally. See the P1 finding this addresses: previously `run()`'s dispatch
loop had no exception handling at all, so any handler exception propagated
all the way out of `main()` and killed the whole run, silently dropping every
remaining line/testcase.
"""

from __future__ import annotations

import io
import re
import sys

import pytest

import netlist_agent.cli as cli_module
from netlist_agent.cli import Config, GenerationConfig
from netlist_agent.cli import run as cli_run
from netlist_agent.session import Session

_RESPONSE_RE = re.compile(r"#RESPONSE (\d+)\n(.*?)\n#END \1\n", re.DOTALL)

_DUMMY_CONFIG = Config(provider="openai", openai=None, anthropic=None, generation=GenerationConfig(0.2, 4096))


def _stub_fallback(session: Session, text: str) -> str:
    return f"FALLBACK:{text}"


def test_handler_exception_does_not_kill_the_run(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    real_handle_request = cli_module.handle_request

    def flaky_handle_request(session, text, fallback):
        if "BOOM" in text:
            raise RuntimeError("simulated handler failure")
        return real_handle_request(session, text, fallback)

    monkeypatch.setattr(cli_module, "handle_request", flaky_handle_request)

    lines = [
        "This is the beginning of a new testcase. The case name is boomcase.",
        "trigger a BOOM here please",
        "another line that should still be processed",
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()

    # Must not raise.
    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)

    output = stdout.getvalue()
    responses = _RESPONSE_RE.findall(output)

    # (a) the process didn't die -- we got here at all, and every line got a
    # response, including the one whose handler raised.
    assert len(responses) == len(lines)

    # (b) ids are sequential/continuous despite the exception.
    ids = [int(r[0]) for r in responses]
    assert ids == list(range(1, len(lines) + 1))

    # The failing line's response body says something failed (doesn't crash,
    # doesn't silently look like success).
    assert "error" in responses[1][1].lower() or "fail" in responses[1][1].lower()

    # (c) the line AFTER the failure was processed normally (fell through to
    # the stub fallback, since it doesn't match any router pattern).
    assert responses[2][1] == "FALLBACK:another line that should still be processed"

    assert session.case_name == "boomcase"
    session.close()


def test_keyboard_interrupt_from_a_handler_is_not_swallowed(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    def raising_handle_request(session, text, fallback):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "handle_request", raising_handle_request)

    lines = [
        "This is the beginning of a new testcase. The case name is interruptcase.",
        "some request",
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()

    with pytest.raises(KeyboardInterrupt):
        cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)


def test_undecodable_stdin_byte_does_not_zero_out_the_whole_run(tmp_path, monkeypatch) -> None:
    """T1 / kills: removing the `_harden_text_stream(stdin)` call in `run()`.

    A real `TextIOWrapper` decodes a whole chunk (not line-by-line), so a
    single bad byte anywhere in stdin used to blank out the entire run --
    even the well-formed lines *before* the bad byte never got a response,
    because the decode error happened inside the `for raw_line in stdin`
    iterator before the first line was ever yielded. With stdin hardened to
    `errors="replace"`, the bad byte becomes U+FFFD and every line -- before,
    at, and after it -- still gets a normal response.
    """
    monkeypatch.chdir(tmp_path)

    raw = (
        "This is the beginning of a new testcase. The case name is bytecase.\n"
        "first ordinary request\n"
        "second request with a bad byte: \xff here\n"
        "third ordinary request\n"
    ).encode("utf-8")
    # Corrupt the third line's bytes with an invalid UTF-8 start byte (the
    # `.encode` above can't produce 0xff itself since it's not valid UTF-8
    # input, so splice it in after encoding).
    raw = raw.replace(b"bad byte: \xc3\xbf here", b"bad byte: \xff here")

    stdin = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8")
    stdout = io.StringIO()

    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)

    output = stdout.getvalue()
    responses = _RESPONSE_RE.findall(output)

    # (a) exactly 4 responses, ids 1..4 -- including the lines before AND
    # after the bad byte, which the un-hardened code got zero responses for.
    assert len(responses) == 4
    ids = [int(r[0]) for r in responses]
    assert ids == [1, 2, 3, 4]

    # (b) the line containing the bad byte still made it through the router/
    # fallback -- its response came from the stub fallback and contains the
    # replacement character where the bad byte was.
    assert responses[2][1].startswith("FALLBACK:")
    assert "�" in responses[2][1]

    session.close()


class _NoReconfigureStream:
    """A minimal text-stream stand-in with no `reconfigure` method at all
    (unlike `io.TextIOWrapper`), so `_harden_text_stream` is a no-op on it --
    exercising `run()`'s manual-iteration fallback (A5) instead."""

    def __init__(self, lines: list) -> None:
        self._lines = list(lines)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._lines):
            raise StopIteration
        item = self._lines[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


def test_undecodable_bytes_from_a_stream_without_reconfigure_skip_and_continue(tmp_path, monkeypatch) -> None:
    """T2a / kills: `continue` in `run()`'s manual-iteration `except
    UnicodeDecodeError` branch reverting to `break`.

    Simulates a caller-supplied stdin-like object that has no `reconfigure`
    method (so A2's hardening can't help it) and raises `UnicodeDecodeError`
    partway through iteration, as a real `TextIOWrapper` over invalid bytes
    would -- but, unlike the old T2, followed by a further ordinary line
    that must still get processed. Measured (see `run()`'s comment at the
    `except UnicodeDecodeError` site) behavior for both interactive and batch
    stdin: `continue` past a decode error and keep reading recovers strictly
    more lines than `break`ing out of the loop, in both modes. `run()` must
    not propagate the exception, must emit a response explaining the skipped
    line, and must still process every line after it.
    """
    monkeypatch.chdir(tmp_path)

    decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    stdin = _NoReconfigureStream(
        [
            "This is the beginning of a new testcase. The case name is streamcase.\n",
            "first ordinary request\n",
            decode_error,
            "third ordinary request\n",
        ]
    )
    stdout = io.StringIO()

    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)

    output = stdout.getvalue()
    responses = _RESPONSE_RE.findall(output)

    # No exception propagated (we got here), all 4 lines got a response --
    # including the one AFTER the decode error, which `break` would never
    # reach -- with ids sequential throughout.
    assert len(responses) == 4
    ids = [int(r[0]) for r in responses]
    assert ids == [1, 2, 3, 4]

    assert "decodeerror" in responses[2][1].lower().replace(" ", "")

    # The line after the decode error was processed normally -- proof that
    # `run()` kept reading instead of giving up on the whole testcase.
    assert responses[3][1] == "FALLBACK:third ordinary request"

    session.close()
    assert session.log_file is None


def test_undecodable_bytes_backstop_when_the_stream_never_advances(tmp_path, monkeypatch) -> None:
    """T2b / kills: removing (or raising past its trigger point)
    `_MAX_CONSECUTIVE_DECODE_ERRORS` in `run()`.

    `continue`ing past a decode error (see T2a) is only safe because a real
    `TextIOWrapper` actually advances past the bad bytes on the next read.
    Nothing stops a caller-supplied stream-like object from raising
    `UnicodeDecodeError` forever without ever advancing -- this is the
    backstop against that: after `_MAX_CONSECUTIVE_DECODE_ERRORS` in a row,
    `run()` gives up and ends the testcase rather than looping until the
    (finite, but very large) stream is exhausted.

    The fake stream below still terminates on its own after 5000 errors
    (rather than being genuinely infinite) so that if the backstop were
    removed, this test would fail on the response-count assertion instead of
    hanging.
    """
    monkeypatch.chdir(tmp_path)

    decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    stdin = _NoReconfigureStream([decode_error] * 5000)
    stdout = io.StringIO()

    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)

    output = stdout.getvalue()
    responses = _RESPONSE_RE.findall(output)

    # Exactly, not `<=`: errors 1..N each get their own skip response and the
    # N+1'th trips the backstop, so an off-by-one in the backstop's own
    # comparison (`>` widened to `>=`, ending the testcase one error early)
    # changes this count. A `<=` bound would let that mutation through.
    assert len(responses) == cli_module._MAX_CONSECUTIVE_DECODE_ERRORS + 1
    last_body = responses[-1][1].lower()
    assert "consecutive" in last_body or "limit" in last_body or "unable to advance" in last_body

    session.close()
    assert session.log_file is None


@pytest.mark.parametrize(
    "advancing_line, expected_responses",
    [
        ("an ordinary line\n", 121),
        # A blank line is skipped before it can become a request, so it adds
        # no response of its own (120, not 121) -- but it is still a
        # successful read, i.e. the stream provably advanced, so it must
        # still reset the counter. This case is what pins the reset's
        # *position*: moving it below `run()`'s `if not line.strip():
        # continue` would leave the ordinary-line case above passing while
        # this one trips the backstop early.
        ("\n", 120),
    ],
)
def test_undecodable_bytes_consecutive_counter_resets_on_a_successful_line(
    tmp_path, monkeypatch, advancing_line: str, expected_responses: int
) -> None:
    """T2c: the consecutive-decode-error counter must reset to zero once a
    line is read successfully -- otherwise decode errors from unrelated,
    widely separated parts of a long-running stream would accumulate toward
    the same backstop, ending the testcase over failures that were each
    individually recovered from just fine.

    60 decode errors (under the 100-error backstop on their own), then one
    line that reads successfully, then 60 more decode errors: if the counter
    resets, none of this trips the backstop (and no response is the backstop
    message); if it doesn't reset, the second batch pushes the running total
    past 100 partway through and the backstop fires early.
    """
    monkeypatch.chdir(tmp_path)

    decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    stdin = _NoReconfigureStream([decode_error] * 60 + [advancing_line] + [decode_error] * 60)
    stdout = io.StringIO()

    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)

    output = stdout.getvalue()
    responses = _RESPONSE_RE.findall(output)

    assert len(responses) == expected_responses
    last_body = responses[-1][1].lower()
    assert "consecutive" not in last_body and "unable to advance" not in last_body

    session.close()
    assert session.log_file is None


@pytest.mark.skip(reason='requires the private rule-based router (not present in this public export) to produce the exact rule-routed begin-line response text')
def test_session_start_failure_on_the_begin_line_does_not_kill_the_run(monkeypatch, tmp_path) -> None:
    """T3 / kills: removing the try/except around `session.start(...)` on the
    begin line (A4).

    Before A4, an exception from `Session.start()` (e.g. the log file
    couldn't be opened) propagated straight out of `run()`'s `for` loop and
    killed the whole run -- even though `Session.start()`'s own docstring
    promises that a failed `start()` leaves every session field untouched
    specifically so a later retry (a subsequent begin-like line) can still
    succeed. That promise was worthless if the caller never got a chance to
    retry because the process had already died.
    """
    monkeypatch.chdir(tmp_path)

    real_start = Session.start
    call_count = {"n": 0}

    def flaky_start(self, case_name, log_filename=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated: could not open log file")
        return real_start(self, case_name, log_filename)

    monkeypatch.setattr(Session, "start", flaky_start)

    lines = [
        "This is the beginning of a new testcase. The case name is retrycase.",
        "an ordinary request",
        "This is the beginning of a new testcase. The case name is retrycase.",
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()

    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)

    output = stdout.getvalue()
    responses = _RESPONSE_RE.findall(output)

    assert len(responses) == 3
    ids = [int(r[0]) for r in responses]
    assert ids == [1, 2, 3]
    assert "internal error" in responses[0][1].lower()

    # The session is in its pre-`start()` state right after the failure --
    # proof that `Session.start()`'s "don't touch anything on failure"
    # contract was preserved. But that promise is worthless unless a real
    # retry (a later begin-like line, sent here as the 3rd line) actually
    # succeeds -- it does: the 3rd response is the normal "beginning of
    # testcase" body, not another internal-error one, and `case_name` ends up
    # set for real.
    assert responses[2][1] == "This is the beginning of testcase retrycase."
    assert session.case_name == "retrycase"


def test_stdout_hardened_against_unencodable_characters(monkeypatch, tmp_path) -> None:
    """T4 / kills: removing the `_harden_text_stream(stdout)` call in `run()`.

    stdout here is an `ascii`-encoding `TextIOWrapper`, and the fallback's
    response body contains non-ASCII characters. Without hardening,
    `stdout.write(...)` inside `emit_response` raises `UnicodeEncodeError`.
    `emit_response` now catches that itself (see its own docstring and
    test_io_protocol.py) rather than letting it escape `run()`, so this no
    longer kills the whole run the way it used to -- but the catch doesn't
    make the write succeed: a `TextIOWrapper.write()` that fails to encode
    raises before anything reaches the underlying buffer (checked directly
    against this exact scenario), so response 2's bytes are simply never
    written at all, not even partially. That's still exactly what this test
    catches, just via a different assertion: `len(responses) == 3` below
    fails (2, not 3) because the un-hardened run produces only responses 1
    and 3 on stdout with response 2 missing outright, rather than 3 malformed
    responses or a dead process. With stdout hardened, the unencodable
    characters degrade to `?` instead of raising at all, and every line,
    including the one after the offending response, comes through complete.
    """
    monkeypatch.chdir(tmp_path)

    def unicode_fallback(session: Session, text: str) -> str:
        if "unicode" in text:
            return "answer: café — done"
        return f"FALLBACK:{text}"

    lines = [
        "This is the beginning of a new testcase. The case name is asciicase.",
        "give me the unicode answer",
        "a following ordinary request",
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    raw_out = io.BytesIO()
    stdout = io.TextIOWrapper(raw_out, encoding="ascii", write_through=True)

    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=unicode_fallback)

    stdout.flush()
    decoded = raw_out.getvalue().decode("ascii")
    responses = _RESPONSE_RE.findall(decoded)

    assert len(responses) == 3
    ids = [int(r[0]) for r in responses]
    assert ids == [1, 2, 3]
    # The non-ASCII characters were replaced with '?' rather than raising.
    assert "?" in responses[1][1]
    # The line after the unencodable response was still processed normally.
    assert responses[2][1] == "FALLBACK:a following ordinary request"

    session.close()


def test_session_log_file_opened_with_errors_replace(tmp_path, monkeypatch) -> None:
    """T5 / kills: removing `errors="replace"` from the `open(path, "w", ...)`
    call in `Session.start()`.

    This is a property assertion rather than a behavior assertion: forcing a
    behavioral reproduction would require the log file's *encoding* (locale-
    determined, not something this test can reliably override) to be
    something narrower than UTF-8, which isn't practical to pin down inside
    a test. Asserting the attribute directly still precisely kills the
    mutation this guards against -- a log `write()` later raising
    `UnicodeEncodeError`. That no longer takes down the whole run the way it
    used to (`emit_response` now catches it on the log channel too -- see its
    own docstring and test_io_protocol.py), but it would still silently drop
    that one response from the log entirely, which is exactly the failure
    mode `errors="replace"` exists to avoid: degrading the unencodable
    character to `?` so the response is logged intact instead of lost.
    """
    monkeypatch.chdir(tmp_path)

    session = Session()
    session.start("attrcase")
    assert session.log_file.errors == "replace"
    session.close()


def test_close_write_failure_at_end_of_run_does_not_kill_the_process(monkeypatch, tmp_path, capsys) -> None:
    """Reproduces (with monkeypatch rather than `chmod`, which is a no-op
    for a root-run CI) the exact scenario measured against a real
    subprocess: the case name is never recognized (so `Session.start()`
    never opens a log file), a design gets "loaded" along the way (so
    `close()`'s emergency stem-fallback path fires instead of the plain
    discard-and-warn one), and the emergency `open()` fails. Before this
    batch, that `open()` in `close()` -- called unguarded from `run()`,
    which is itself called unguarded from `main()` -- raised straight out:
    a bare traceback on stderr and a non-zero-ish dead process, even though
    both responses had already been written to stdout complete. This
    asserts the fix: `run()` returns normally, stdout has both responses,
    stderr has a warning naming the real reason. (No separate "no traceback
    on stderr" assertion: if the fix didn't work, `cli_run(...)` itself
    raises and the test errors out before reaching any assertion at all --
    that call not raising is already the discriminating check.)
    """
    monkeypatch.chdir(tmp_path)

    def loading_fallback(session: Session, text: str) -> str:
        session.load_filename = "test09.v"
        return f"FALLBACK:{text}"

    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        if path == "test09.log" and "w" in mode:
            raise OSError(13, "Permission denied")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    lines = [
        "a line that never names a testcase",
        "another line that never names a testcase either",
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()

    # Must not raise -- this is the assertion the real subprocess bug broke.
    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=loading_fallback)

    output = stdout.getvalue()
    responses = _RESPONSE_RE.findall(output)
    assert len(responses) == 2
    ids = [int(r[0]) for r in responses]
    assert ids == [1, 2]

    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "test09.log" in captured.err
    assert "permission denied" in captured.err.lower()
    assert session.log_file is None
    assert not (tmp_path / "test09.log").exists()


def test_stderr_hardened_against_unencodable_characters(monkeypatch, tmp_path) -> None:
    """`run()` now hardens `sys.stderr` the same way it already hardens
    `stdin`/`stdout` (see `test_stdout_hardened_against_unencodable_
    characters` just above). Without it, a warning `close()` prints that
    embeds a non-ASCII case/design name -- reachable via a non-ASCII load
    filename plus a narrow `sys.stderr` encoding, measured directly against
    this repo before the fix -- raises `UnicodeEncodeError` out of
    `print()` inside `session.py`'s `_warn()`, which swallows it and drops
    the warning silently rather than degrading it to `?`. This is a
    discriminating test, not just a "must not raise" one (`_warn`'s own
    swallow already guarantees that regardless of this line): with
    `sys.stderr` hardened, the warning text survives with `?` in place of
    the unencodable characters; with the harden line removed, `sys.stderr`
    ends up with nothing written at all for that message. Measured both
    ways against this exact scenario.
    """
    monkeypatch.chdir(tmp_path)

    def loading_fallback(session: Session, text: str) -> str:
        session.load_filename = "設計.v"  # non-ASCII ("design" in Chinese)
        return f"FALLBACK:{text}"

    # A pre-existing same-stem log file forces close()'s "fallback name
    # already exists" warning, whose message embeds the non-ASCII stem --
    # the plain discard-and-warn message contains no filename at all, so it
    # wouldn't exercise the encoding path this test is about.
    (tmp_path / "設計.log").write_text("preexisting, unrelated content")

    lines = ["a line that never names a testcase"]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()

    raw_err = io.BytesIO()
    narrow_stderr = io.TextIOWrapper(raw_err, encoding="ascii", write_through=True)
    monkeypatch.setattr(sys, "stderr", narrow_stderr)

    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=loading_fallback)

    output = stdout.getvalue()
    responses = _RESPONSE_RE.findall(output)
    assert len(responses) == 1

    narrow_stderr.flush()
    stderr_content = raw_err.getvalue().decode("ascii")
    assert "warning" in stderr_content.lower()
    assert "?" in stderr_content  # the non-ASCII stem, degraded rather than dropped
    assert session.log_file is None


class _FailAfterN:
    """Wraps a real open file handle so its `write()` starts raising
    `OSError` (the full-disk shape) after `fail_after` successful writes --
    same shape as `test_session.py`'s `_FullDisk`, but with a configurable
    threshold so the first few writes can land for real before the channel
    goes bad partway through a testcase."""

    def __init__(self, wrapped, fail_after: int) -> None:
        self._wrapped = wrapped
        self._writes = 0
        self._fail_after = fail_after

    def write(self, text: str) -> int:
        self._writes += 1
        if self._writes > self._fail_after:
            raise OSError(28, "No space left on device")
        return self._wrapped.write(text)

    def flush(self) -> None:
        self._wrapped.flush()

    def close(self) -> None:
        self._wrapped.close()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def test_log_write_failure_partway_through_does_not_drop_any_response_from_stdout(
    monkeypatch, tmp_path, capsys
) -> None:
    """The asymmetry the whole batch is about: the log channel going bad
    partway through a testcase (a full disk, measured) must not cost stdout
    a single response -- every request line still gets a complete
    #RESPONSE/#END pair on stdout with ids allocated and sequential, exactly
    as if nothing had failed."""
    monkeypatch.chdir(tmp_path)
    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if path == "logcase.log" and "w" in mode:
            return _FailAfterN(handle, fail_after=2)
        return handle

    monkeypatch.setattr("builtins.open", fake_open)

    lines = [
        "This is the beginning of a new testcase. The case name is logcase.",
        "request two",
        "request three",
        "request four",
        "request five",
        "request six",
        "request seven",
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()

    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)

    output = stdout.getvalue()
    responses = _RESPONSE_RE.findall(output)
    assert len(responses) == len(lines)
    ids = [int(r[0]) for r in responses]
    assert ids == list(range(1, len(lines) + 1))

    # Responses 1-2 logged fine, 3-7 (5 of them) hit the failing write.
    assert session.log_write_failures == 5
    assert session.stdout_write_failures == 0

    captured = capsys.readouterr()
    # One per-occurrence warning, not five.
    occurrence_lines = [
        line for line in captured.err.splitlines() if line.startswith("warning: failed to write response")
    ]
    assert len(occurrence_lines) == 1
    # The per-occurrence line names the LOG channel, not stdout (stdout
    # never failed here) -- checked on the isolated occurrence line, not the
    # whole captured stderr, because the close() summary line further below
    # mentions both channel labels and would make a whole-string check
    # vacuously pass even with the labels swapped.
    assert "to the testcase log" in occurrence_lines[0]
    assert "to stdout" not in occurrence_lines[0]
    assert "5 response(s) failed to write to the testcase log" in captured.err


def test_stdout_write_failure_does_not_stop_the_log_from_getting_every_response(monkeypatch, tmp_path, capsys) -> None:
    """The mirror case: stdout going bad must not cost the log a single
    response -- per spec 3.3, scoring reads the log, so it's the channel
    that has to keep receiving every response it can regardless of what
    stdout does."""
    monkeypatch.chdir(tmp_path)

    class _FailingStdout:
        def write(self, text: str) -> int:
            raise OSError(28, "No space left on device")

        def flush(self) -> None:
            pass

    lines = [
        "This is the beginning of a new testcase. The case name is stdoutfailcase.",
        "request two",
        "request three",
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = _FailingStdout()

    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)

    log_content = (tmp_path / "stdoutfailcase.log").read_text()
    responses = _RESPONSE_RE.findall(log_content)
    assert len(responses) == len(lines)
    ids = [int(r[0]) for r in responses]
    assert ids == list(range(1, len(lines) + 1))
    assert session.stdout_write_failures == len(lines)
    assert session.log_write_failures == 0

    captured = capsys.readouterr()
    occurrence_lines = [
        line for line in captured.err.splitlines() if line.startswith("warning: failed to write response")
    ]
    assert len(occurrence_lines) == 1
    # The per-occurrence line names the STDOUT channel, not the log (the log
    # never failed here) -- isolated from the close() summary line the same
    # way as the mirror test above, for the same reason.
    assert "to stdout" in occurrence_lines[0]
    assert "to the testcase log" not in occurrence_lines[0]


def test_both_channels_failing_simultaneously_still_completes_the_run(monkeypatch, tmp_path, capsys) -> None:
    """Neither channel's failure is contingent on the other one working --
    with both broken at once, `run()` still returns normally and response
    ids are still allocated sequentially without a gap."""
    monkeypatch.chdir(tmp_path)
    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if path == "bothfailcase.log" and "w" in mode:
            return _FailAfterN(handle, fail_after=0)
        return handle

    monkeypatch.setattr("builtins.open", fake_open)

    class _FailingStdout:
        def write(self, text: str) -> int:
            raise OSError(28, "No space left on device")

        def flush(self) -> None:
            pass

    lines = [
        "This is the beginning of a new testcase. The case name is bothfailcase.",
        "request two",
        "request three",
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = _FailingStdout()

    # Must not raise, despite both channels failing on every response.
    session = cli_run(_DUMMY_CONFIG, stdin=stdin, stdout=stdout, fallback=_stub_fallback)

    assert session.next_response_id == len(lines) + 1
    assert session.stdout_write_failures == len(lines)
    assert session.log_write_failures == len(lines)

    captured = capsys.readouterr()
    assert f"{len(lines)} response(s) failed to write to stdout" in captured.err
    assert f"{len(lines)} response(s) failed to write to the testcase log" in captured.err
