"""Unit tests for netlist_agent/io_protocol.py, in isolation."""

from __future__ import annotations

import io

import pytest

from netlist_agent.io_protocol import emit_response, format_response, respond
from netlist_agent.session import Session


class _FailingStream:
    """A minimal IO[str] stand-in whose `write()`/`flush()` can each be
    configured to start raising `OSError` (the full-disk shape) after a
    given number of successful calls -- `None` means that call never fails.
    The two knobs are independent because `emit_response`'s guard has to
    cover `write()` and `flush()` separately (a full disk can fail either
    one)."""

    def __init__(self, fail_write_after: int | None = None, fail_flush_after: int | None = None) -> None:
        self._buffer = io.StringIO()
        self.writes = 0
        self.flushes = 0
        self._fail_write_after = fail_write_after
        self._fail_flush_after = fail_flush_after

    def write(self, text: str) -> int:
        self.writes += 1
        if self._fail_write_after is not None and self.writes > self._fail_write_after:
            raise OSError(28, "No space left on device")
        return self._buffer.write(text)

    def flush(self) -> None:
        self.flushes += 1
        if self._fail_flush_after is not None and self.flushes > self._fail_flush_after:
            raise OSError(28, "No space left on device")

    def close(self) -> None:
        pass

    def getvalue(self) -> str:
        return self._buffer.getvalue()


class _RaisingStream:
    """Raises a non-`OSError`/`UnicodeEncodeError` exception from `write()`
    -- the kind of internal bug `emit_response`'s guard must NOT swallow."""

    def write(self, text: str) -> int:
        raise RuntimeError("not an I/O failure")

    def flush(self) -> None:
        pass


def test_format_response_exact_wire_format() -> None:
    assert format_response(1, "hello") == "#RESPONSE 1\nhello\n#END 1\n"


def test_format_response_multiline_body() -> None:
    body = "line one\nline two"
    text = format_response(5, body)
    assert text == "#RESPONSE 5\nline one\nline two\n#END 5\n"
    assert text.splitlines()[0] == "#RESPONSE 5"
    assert text.splitlines()[-1] == "#END 5"


def test_respond_writes_to_stdout_and_allocates_id(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("test01")
    out = io.StringIO()

    rid1 = respond(session, "first body", stdout=out)
    rid2 = respond(session, "second body", stdout=out)
    session.close()

    assert rid1 == 1
    assert rid2 == 2
    assert out.getvalue() == "#RESPONSE 1\nfirst body\n#END 1\n#RESPONSE 2\nsecond body\n#END 2\n"


def test_log_file_content_matches_stdout(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("test02")
    out = io.StringIO()

    respond(session, "alpha", stdout=out)
    respond(session, "beta", stdout=out)
    session.close()

    log_content = (tmp_path / "test02.log").read_text()
    assert log_content == out.getvalue()


def test_emit_response_without_log_file_does_not_crash() -> None:
    session = Session()  # never .start()-ed: no log file
    out = io.StringIO()
    emit_response(session, 1, "body", stdout=out)
    assert out.getvalue() == "#RESPONSE 1\nbody\n#END 1\n"


def test_emit_response_stdout_write_failure_does_not_stop_the_log_write() -> None:
    """A broken stdout channel must not cost the log its copy of the same
    response -- scoring reads the log (spec 3.3), so it's the channel that
    has to keep going no matter what stdout does."""
    session = Session()
    log = _FailingStream()
    session.log_file = log
    out = _FailingStream(fail_write_after=0)  # every write raises

    emit_response(session, 1, "body", stdout=out)

    assert out.getvalue() == ""  # nothing landed -- the write really failed
    assert log.getvalue() == "#RESPONSE 1\nbody\n#END 1\n"
    assert session.stdout_write_failures == 1
    assert session.log_write_failures == 0


def test_emit_response_log_write_failure_does_not_stop_the_stdout_write() -> None:
    """The mirror case: a broken log must not cost stdout its copy -- the
    `#END` handshake stdout needs for the NEXT request to even be sent must
    still go through."""
    session = Session()
    session.log_file = _FailingStream(fail_write_after=0)  # every write raises
    out = io.StringIO()

    emit_response(session, 1, "body", stdout=out)

    assert out.getvalue() == "#RESPONSE 1\nbody\n#END 1\n"
    assert session.log_write_failures == 1
    assert session.stdout_write_failures == 0


def test_emit_response_stdout_flush_failure_is_guarded_too() -> None:
    """A half-guard that only wraps `write()` and not `flush()` must fail
    this: `write()` succeeds, `flush()` is what raises."""
    session = Session()
    log = _FailingStream()
    session.log_file = log
    out = _FailingStream(fail_flush_after=0)  # write succeeds, flush raises

    emit_response(session, 1, "body", stdout=out)

    assert out.getvalue() == "#RESPONSE 1\nbody\n#END 1\n"  # write() itself went through
    assert log.getvalue() == "#RESPONSE 1\nbody\n#END 1\n"  # log unaffected
    assert session.stdout_write_failures == 1


def test_emit_response_log_flush_failure_is_guarded_too() -> None:
    """Mirror of the above for the log channel's `flush()`."""
    session = Session()
    session.log_file = _FailingStream(fail_flush_after=0)  # write succeeds, flush raises
    out = io.StringIO()

    emit_response(session, 1, "body", stdout=out)

    assert out.getvalue() == "#RESPONSE 1\nbody\n#END 1\n"
    assert session.log_file.getvalue() == "#RESPONSE 1\nbody\n#END 1\n"  # write() itself went through
    assert session.log_write_failures == 1


def test_emit_response_warns_only_once_per_channel(capsys) -> None:
    """A disk that stays full for the rest of the run must not turn into
    one stderr line per remaining response -- only the first failure on
    each channel warns; later ones are counted silently."""
    session = Session()
    log = _FailingStream(fail_write_after=0)
    session.log_file = log
    out = _FailingStream(fail_write_after=0)

    for response_id in range(1, 6):
        emit_response(session, response_id, f"body{response_id}", stdout=out)

    assert session.stdout_write_failures == 5
    assert session.log_write_failures == 5

    captured = capsys.readouterr()
    # One first-failure warning per channel, not five.
    assert captured.err.count("warning: failed to write response") == 2


def test_emit_response_per_occurrence_warning_names_stdout_when_only_stdout_fails(capsys) -> None:
    """Pins the CONTENT of the per-occurrence warning, not just how many
    there are -- a mutation that reads the wrong channel's counter in the
    "first failure" check, or that swaps the two warning messages' channel
    labels, still produces exactly one warning line here (so a bare count
    assertion wouldn't catch it), but it wouldn't say "stdout"."""
    session = Session()
    session.log_file = _FailingStream()  # never fails
    out = _FailingStream(fail_write_after=0)  # every write raises

    emit_response(session, 1, "body", stdout=out)

    captured = capsys.readouterr()
    assert "to stdout" in captured.err
    assert "to the testcase log" not in captured.err


def test_emit_response_per_occurrence_warning_names_the_log_when_only_the_log_fails(capsys) -> None:
    """Mirror of the above for the log channel."""
    session = Session()
    session.log_file = _FailingStream(fail_write_after=0)  # every write raises
    out = io.StringIO()

    emit_response(session, 1, "body", stdout=out)

    captured = capsys.readouterr()
    assert "to the testcase log" in captured.err
    assert "to stdout" not in captured.err


def test_emit_response_reraises_non_os_error_from_stdout() -> None:
    """The guard is narrowly `OSError`/`UnicodeEncodeError` -- a genuine
    internal bug in the stdout channel must still propagate, not be
    swallowed by an overly broad `except Exception`."""
    session = Session()
    with pytest.raises(RuntimeError):
        emit_response(session, 1, "body", stdout=_RaisingStream())


def test_emit_response_reraises_non_os_error_from_log() -> None:
    """Mirror of the above for the log channel."""
    session = Session()
    session.log_file = _RaisingStream()
    out = io.StringIO()
    with pytest.raises(RuntimeError):
        emit_response(session, 1, "body", stdout=out)
