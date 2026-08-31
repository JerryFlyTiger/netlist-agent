"""Unit tests for netlist_agent/session.py, in isolation from the router/cli."""

from __future__ import annotations

import io
import os
import re
import sys

from netlist_agent.io_protocol import emit_response
from netlist_agent.ir import GateType
from netlist_agent.session import Session

_RESPONSE_RE = re.compile(r"#RESPONSE (\d+)\n(.*?)\n#END \1\n", re.DOTALL)


def test_response_id_starts_at_one_and_increments(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("testX")
    assert session.allocate_response_id() == 1
    assert session.allocate_response_id() == 2
    assert session.allocate_response_id() == 3
    session.close()


def test_start_opens_log_file_named_after_case(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("test07")
    assert session.case_name == "test07"
    assert session.log_path == "test07.log"
    assert os.path.exists("test07.log")
    assert session.log_file is not None
    session.close()
    assert session.log_file is None


def test_snapshot_is_independent_of_mutating_current_design(tmp_path, monkeypatch) -> None:
    from netlist_agent.parser import parse_verilog
    from netlist_agent.transform import remove_dangling_gates

    monkeypatch.chdir(tmp_path)
    src = """
    module top(a, y);
      input a;
      output y;
      wire n1;
      buf g0(n1, a);
      buf g1(y, a);
    endmodule
    """
    path = tmp_path / "t.v"
    path.write_text(src)

    session = Session()
    session.current_design = parse_verilog(str(path))
    session.original_snapshot = parse_verilog(str(path))

    assert len(session.current_design.gates) == 2
    assert len(session.original_snapshot.gates) == 2

    remove_dangling_gates(session.current_design)

    # g0 (n1 = buf(a), n1 unused) is dangling and gets removed from the
    # current design, but the snapshot -- a wholly separate parse -- must be
    # untouched.
    assert len(session.current_design.gates) == 1
    assert len(session.original_snapshot.gates) == 2


def test_close_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("test01")
    session.close()
    session.close()  # must not raise
    assert session.log_file is None


def test_close_summarizes_dropped_writes_when_any_channel_failed(tmp_path, monkeypatch, capsys) -> None:
    """`close()` reports the final per-channel totals of responses that
    `emit_response()` failed to write, one line for the whole testcase --
    not one per failure (see `emit_response`'s own "warns only once" test in
    test_io_protocol.py; this is the summary that's supposed to make up for
    the per-occurrence silence after the first)."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("sumcase")
    session.stdout_write_failures = 2
    session.log_write_failures = 3

    session.close()

    captured = capsys.readouterr()
    assert "2 response(s) failed to write to stdout" in captured.err
    assert "3 response(s) failed to write to the testcase log" in captured.err


def test_close_omits_the_summary_when_nothing_failed(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("nofailcase")

    session.close()

    captured = capsys.readouterr()
    assert "failed to write" not in captured.err


def test_close_failure_summary_is_idempotent(tmp_path, monkeypatch, capsys) -> None:
    """`close()` is documented and tested (`test_close_is_idempotent`) to be
    safe to call twice. The failure-summary line must honor that too -- a
    second `close()` must not print the same summary again, even though the
    counters it's built from are still sitting there non-zero (they stay
    readable on purpose -- see `Session._write_failure_summary_warned`'s
    docstring)."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("idempotentsumcase")
    session.stdout_write_failures = 2
    session.log_write_failures = 3

    session.close()
    first = capsys.readouterr()
    assert first.err.count("failed to write to stdout") == 1

    session.close()  # must not raise, and must not re-warn
    second = capsys.readouterr()
    assert "failed to write to stdout" not in second.err

    # The counters themselves are untouched by printing the summary -- they
    # still reflect the session's whole history, not "since last printed".
    assert session.stdout_write_failures == 2
    assert session.log_write_failures == 3


def test_last_op_bookkeeping_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session = Session()
    assert session.last_op_count is None
    assert session.last_gate_delta == {}
    session.last_op_count = 3
    session.last_gate_delta = {GateType.NAND: 4}
    assert session.last_op_count == 3
    assert session.last_gate_delta[GateType.NAND] == 4


# ----------------------------------------------------------------------
# Log-not-lost: responses emitted before the case name is recognized must
# survive, however (or whether) the case name is eventually recognized.
# ----------------------------------------------------------------------


def test_pending_log_flushed_when_case_name_recognized_late(tmp_path, monkeypatch) -> None:
    """(a) Case name unrecognized on the first line -- simulated here by an
    LLM tool call (`set_testcase`, see llm/tools_schema.py) arriving after
    some responses were already emitted -- must not lose those responses;
    the resulting log contains all of them, #RESPONSE/#END ids sequential
    from 1."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    out = io.StringIO()

    # Two responses emitted before the case name is known at all.
    emit_response(session, 1, "first response, case name still unknown", stdout=out)
    emit_response(session, 2, "second response, case name still unknown", stdout=out)
    assert session.log_file is None
    assert len(session.pending_log) == 2

    # The case name is now recognized (e.g. via the `set_testcase` tool).
    session.start("test_late")
    assert session.log_file is not None
    assert session.pending_log == []

    # A third response emitted normally, after the log is open.
    emit_response(session, 3, "third response, case name now known", stdout=out)
    session.close()

    log_content = (tmp_path / "test_late.log").read_text()
    assert log_content == out.getvalue()

    responses = _RESPONSE_RE.findall(log_content)
    assert len(responses) == 3
    ids = [int(r[0]) for r in responses]
    assert ids == [1, 2, 3]
    assert "first response" in responses[0][1]


def test_begin_line_with_explicit_log_filename(tmp_path, monkeypatch) -> None:
    """(b) A begin-testcase line that names its log file explicitly (e.g.
    "...into case23.log") must produce a log at that name, not
    "<case_name>.log"."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("case23", "case23.log")
    assert session.log_path == "case23.log"
    assert os.path.exists("case23.log")
    session.close()


def test_close_falls_back_to_loaded_design_stem_when_case_name_never_recognized(tmp_path, monkeypatch) -> None:
    """(c) No case name was ever recognized during the whole testcase, but a
    design was loaded at some point -- `close()` must name the log after
    that design's file stem rather than discard the buffered responses."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    out = io.StringIO()

    emit_response(session, 1, "response with no case name ever recognized", stdout=out)
    assert session.log_file is None
    assert len(session.pending_log) == 1

    session.load_filename = "test09.v"

    # Capture the file object `close()` opens internally (the emergency
    # stem-fallback `open(..., errors="replace")` call site) so its `.errors`
    # attribute can be checked below -- `close()` itself sets
    # `session.log_file = None` before returning, so the reference has to be
    # grabbed as `open()` creates it, not read back off `session` afterward.
    opened_files: list = []
    real_open = open

    def spying_open(*args, **kwargs):
        f = real_open(*args, **kwargs)
        opened_files.append(f)
        return f

    monkeypatch.setattr("builtins.open", spying_open)

    session.close()

    assert session.log_path == "test09.log"
    assert os.path.exists("test09.log")
    log_content = (tmp_path / "test09.log").read_text()
    assert log_content == out.getvalue()
    assert len(opened_files) == 1
    assert opened_files[0].errors == "replace"


def test_close_discards_with_warning_when_nothing_to_name_log_after(tmp_path, monkeypatch, capsys) -> None:
    """No case name, no loaded design at all -- close() must not raise, and
    must print a warning to stderr rather than silently discard."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    out = io.StringIO()
    emit_response(session, 1, "orphaned response", stdout=out)

    session.close()  # must not raise

    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert session.log_file is None


def test_close_discard_branch_drains_pending_log_so_it_warns_only_once(tmp_path, monkeypatch, capsys) -> None:
    """F6: the discard branch must clear `pending_log` like the stem-fallback
    branch does. Otherwise the buffered responses stay queued and a second
    `close()` -- which the production path never makes, but which any future
    caller reasonably might -- re-warns about responses already accounted for."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    out = io.StringIO()
    emit_response(session, 1, "orphaned response", stdout=out)

    session.close()
    assert session.pending_log == []
    first = capsys.readouterr()
    assert "warning" in first.err.lower()

    session.close()
    second = capsys.readouterr()
    assert second.err == ""


def test_close_stem_fallback_does_not_truncate_a_preexisting_log(tmp_path, monkeypatch, capsys) -> None:
    """F1: the emergency `<stem>.log` name `close()` synthesizes when the
    case name was never recognized must never overwrite a file that already
    exists there (e.g. a legitimate log some earlier testcase already wrote)
    -- it must pick a non-colliding sibling name instead, and warn."""
    monkeypatch.chdir(tmp_path)
    preexisting_path = tmp_path / "test09.log"
    preexisting_content = "#RESPONSE 1\nsome earlier, unrelated testcase's log content\n#END 1\n"
    preexisting_path.write_text(preexisting_content)

    session = Session()
    out = io.StringIO()
    emit_response(session, 1, "response with no case name ever recognized", stdout=out)
    session.load_filename = "test09.v"

    session.close()

    # The preexisting file must be byte-for-byte untouched.
    assert preexisting_path.read_text() == preexisting_content

    # The pending content must have landed somewhere else instead.
    assert session.log_path != "test09.log"
    assert os.path.exists(session.log_path)
    with open(session.log_path) as f:
        recovered_content = f.read()
    assert recovered_content == out.getvalue()

    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_close_emergency_open_failure_does_not_raise(tmp_path, monkeypatch, capsys) -> None:
    """The emergency stem-fallback `open()` in `close()` used to be
    completely unguarded -- a `PermissionError`/`OSError` there (measured:
    `chmod 0500` on the cwd in a real subprocess) propagated out of
    `close()` and killed the whole process even though every response was
    already on stdout. `close()` must instead warn (with the real reason in
    the message) and return normally, and must not leave `pending_log`
    holding responses nobody will ever flush again."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    out = io.StringIO()
    emit_response(session, 1, "response with no case name ever recognized", stdout=out)
    session.load_filename = "test09.v"

    def failing_open(path, mode="r", *args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr("builtins.open", failing_open)

    session.close()  # must not raise

    assert session.pending_log == []
    assert session.log_file is None
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "test09.log" in captured.err
    assert "permission denied" in captured.err.lower()


def test_close_emergency_write_failure_does_not_raise(tmp_path, monkeypatch, capsys) -> None:
    """Same as above, but the `open()` itself succeeds and the failure hits
    partway through the write loop instead (the full-disk shape: the first
    write goes through for real, the second doesn't -- an unconditional
    "always raise" mock would leave nothing on disk and the "the file really
    was opened" half of this test would be vacuous)."""
    monkeypatch.chdir(tmp_path)
    real_open = open

    class _FullDisk:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped
            self._writes = 0

        def write(self, text: str) -> int:
            self._writes += 1
            if self._writes > 1:
                raise OSError(28, "No space left on device")
            return self._wrapped.write(text)

        def close(self) -> None:
            self._wrapped.close()

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def fake_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if "w" in mode:
            return _FullDisk(handle)
        return handle

    monkeypatch.setattr("builtins.open", fake_open)

    session = Session()
    out = io.StringIO()
    emit_response(session, 1, "first response, no case name ever recognized", stdout=out)
    emit_response(session, 2, "second response, no case name ever recognized", stdout=out)
    session.load_filename = "test09.v"

    session.close()  # must not raise

    assert session.pending_log == []
    assert session.log_file is None
    assert session.log_path is None
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "no space left on device" in captured.err.lower()
    # The partial write really did land on disk -- otherwise "discarded"
    # above would be describing nothing.
    with real_open("test09.log") as f:
        assert "first response" in f.read()


def test_close_final_log_file_close_failure_does_not_raise(tmp_path, monkeypatch, capsys) -> None:
    """The `self.log_file.close()` at the very end of `close()` -- reached
    on the normal path, once the log is already open and fully written --
    was also unguarded. A failure there (e.g. flushing a buffered write to a
    now-full disk, which is exactly what a real file object's `close()` can
    raise) must not raise either, and must still leave `session.log_file`
    `None` so a second `close()` stays a no-op (`test_close_is_idempotent`)."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("test01")

    def failing_close():
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(session.log_file, "close", failing_close)

    session.close()  # must not raise

    assert session.log_file is None
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "no space left on device" in captured.err.lower()

    session.close()  # idempotent even after the close() failure above


class _BrokenStderr:
    """Stands in for `sys.stderr` itself being unwritable (measured: a real
    `print(..., file=sys.stderr)` against an object whose `write()` raises
    `OSError` raises straight out of `print()`). Scoring environments often
    point stderr at the same filesystem as the log, so a disk-full/
    permission failure that breaks the emergency log write can plausibly
    break the warning about it too -- `close()` promises not to raise
    either way, and until now the warning `print()` calls themselves were
    the one place in `close()` that promise wasn't actually enforced."""

    def write(self, _text: str) -> int:
        raise OSError(5, "Input/output error")

    def flush(self) -> None:
        pass


def test_close_name_collision_warning_survives_broken_stderr(tmp_path, monkeypatch) -> None:
    """The THIRD of `close()`'s four warning sites: the emergency log name is
    already taken, so `close()` picks `<stem>.recovered.log` and warns that it
    did. That warning predates this batch and was left as a raw `print()` at
    first -- reverting just this one call site back is invisible to the other
    broken-stderr tests, which never reach it (they need no sibling name).

    The write itself must still succeed here, so this pins that a failing
    warning does not cost the recovered log: the responses land in
    `c2.recovered.log` even though nothing could be printed about it.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c2.log").write_text("a pre-existing log this must not truncate")
    session = Session()
    session.load_filename = "c2.v"
    out = io.StringIO()
    emit_response(session, 1, "orphaned response", stdout=out)

    monkeypatch.setattr(sys, "stderr", _BrokenStderr())

    session.close()  # must not raise even though the collision warning can't print

    assert session.pending_log == []
    assert session.log_file is None
    assert (tmp_path / "c2.log").read_text() == "a pre-existing log this must not truncate"
    assert "orphaned response" in (tmp_path / "c2.recovered.log").read_text()


def test_close_final_close_warning_survives_broken_stderr(tmp_path, monkeypatch) -> None:
    """The FOURTH warning site: the final `self.log_file.close()` fails AND
    the warning about it cannot be printed. Both failures at once is the
    realistic shape, not a contrived one -- a full filesystem breaks the log
    flush and the stderr write for the same reason. `log_file` must still be
    cleared so a second `close()` stays idempotent.
    """
    monkeypatch.chdir(tmp_path)
    session = Session()
    session.start("c3")

    class _UncloseableLog:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def close(self) -> None:
            raise OSError(28, "No space left on device")

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    real_log_file = session.log_file
    session.log_file = _UncloseableLog(real_log_file)

    monkeypatch.setattr(sys, "stderr", _BrokenStderr())

    session.close()  # must not raise: neither the close() nor the warning works

    assert session.log_file is None
    session.close()  # still idempotent afterwards
    real_log_file.close()


def test_close_discard_branch_survives_broken_stderr(tmp_path, monkeypatch) -> None:
    """The plain discard-and-warn branch (no case name, no loaded design)
    must not raise even when the warning `print()` itself fails."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    out = io.StringIO()
    emit_response(session, 1, "orphaned response", stdout=out)

    monkeypatch.setattr(sys, "stderr", _BrokenStderr())

    session.close()  # must not raise even though warning it can't print

    assert session.pending_log == []
    assert session.log_file is None


def test_close_emergency_failure_branch_survives_broken_stderr(tmp_path, monkeypatch) -> None:
    """The emergency-write-failure branch added by this batch calls `_warn`
    from inside an `except OSError` handler that exists specifically to
    keep `close()` from raising -- if that handler's own warning `print()`
    raised, the handler meant to prevent a crash would itself become the
    crash. Must not raise even with a broken stderr."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    out = io.StringIO()
    emit_response(session, 1, "response with no case name ever recognized", stdout=out)
    session.load_filename = "test09.v"

    def failing_open(path, mode="r", *args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr("builtins.open", failing_open)
    monkeypatch.setattr(sys, "stderr", _BrokenStderr())

    session.close()  # must not raise even though warning it can't print

    assert session.pending_log == []
    assert session.log_file is None


def test_close_survives_non_ascii_warning_with_narrow_stderr(tmp_path, monkeypatch) -> None:
    """Reproduces the cold-read finding directly: a non-ASCII load filename
    plus a narrow (`ascii`) `sys.stderr` encoding makes the "fallback log
    name already exists" warning's `print()` raise `UnicodeEncodeError` (a
    `ValueError` subclass, NOT an `OSError`) -- measured against this exact
    scenario before `_warn`'s `except` clause covered it. This test uses a
    real narrow `TextIOWrapper` (not `_BrokenStderr`) specifically so it
    stays sensitive to the exception *type*, unlike the OSError-only tests
    above -- and does NOT go through `cli.py`'s `run()`, so it pins `_warn`
    catching `UnicodeEncodeError` on its own, independent of `run()`
    hardening `sys.stderr` (that's covered separately in
    tests/test_cli_error_resilience.py)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "設計.log").write_text("preexisting, unrelated content")

    session = Session()
    out = io.StringIO()
    emit_response(session, 1, "response with no case name ever recognized", stdout=out)
    session.load_filename = "設計.v"  # non-ASCII

    narrow_stderr = io.TextIOWrapper(io.BytesIO(), encoding="ascii", write_through=True)
    monkeypatch.setattr(sys, "stderr", narrow_stderr)

    session.close()  # must not raise UnicodeEncodeError

    assert session.pending_log == []
    assert session.log_file is None  # closed at the end of close(), as normal
    assert session.log_path is not None
    with open(session.log_path) as f:
        assert "response with no case name ever recognized" in f.read()


# ----------------------------------------------------------------------
# F2: start() must not partially mutate session state on a failed open(),
# and set_testcase (llm/tools_schema.py) must validate log_filename.
# ----------------------------------------------------------------------


def test_start_leaves_session_untouched_on_open_failure_and_retry_succeeds(tmp_path, monkeypatch) -> None:
    """(a) If `start()`'s open() fails, `case_name`/`log_file` must remain
    exactly as they were beforehand (not left in a half-set state that
    permanently defeats the "case name already recognized" guard) -- a
    subsequent retry with good arguments must actually succeed. Uses no
    explicit `log_filename` (so the "fall back to the default name" retry
    doesn't apply and it's the default name itself that's unopenable) --
    that fallback-on-explicit-name path is covered separately below."""
    monkeypatch.chdir(tmp_path)
    session = Session()

    # No explicit log_filename -> the default "<case_name>.log" path is
    # derived from case_name itself, and a case name containing a directory
    # separator makes that default path unopenable.
    bad_case_name = "no_such_subdir/x"
    import pytest

    with pytest.raises(OSError):
        session.start(bad_case_name)

    assert session.case_name is None
    assert session.log_file is None
    assert session.log_path is None

    # Retry with valid arguments must genuinely succeed.
    session.start("test60")
    assert session.case_name == "test60"
    assert session.log_file is not None
    assert session.log_path == "test60.log"
    assert os.path.exists("test60.log")
    session.close()


def test_start_falls_back_to_default_name_when_explicit_log_filename_fails_to_open(tmp_path, monkeypatch, capsys) -> None:
    """`start()` retries against the default "<case_name>.log" name (with a
    stderr warning) if an *explicitly* named log file fails to open."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    bad_path = str(tmp_path / "no_such_subdir" / "x.log")

    session.start("test61", bad_path)

    assert session.case_name == "test61"
    assert session.log_path == "test61.log"
    assert os.path.exists("test61.log")
    assert session.log_file.errors == "replace"
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    session.close()


def test_start_reraises_the_open_error_not_a_broken_stderr_error(tmp_path, monkeypatch) -> None:
    """When an *explicitly* named log file fails to open AND the default-
    name retry also fails AND `sys.stderr` is broken, the exception that
    ultimately reaches the caller must still identify the real `open()`
    failure -- not "stderr couldn't be written to". Before routing the
    "falling back to..." `print()` through `_warn()`, the bare `print(...,
    file=sys.stderr)` between the two `open()` calls raised first (a
    `ValueError`/`OSError` about the broken stream), so the retry `open()`
    right after it was never even reached, and the caller saw a completely
    unrelated error instead of the log-file-couldn't-open one it needs to
    act on."""
    monkeypatch.chdir(tmp_path)
    session = Session()
    bad_path = str(tmp_path / "no_such_subdir" / "x.log")

    monkeypatch.setattr(sys, "stderr", _BrokenStderr())

    real_open = open

    def always_fail_open(path, mode="r", *args, **kwargs):
        if "w" in mode:
            raise OSError(13, "Permission denied: " + path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", always_fail_open)

    try:
        session.start("test61", bad_path)
    except OSError as exc:
        assert exc.errno == 13
        assert "test61.log" in str(exc)  # the *retry's* target, i.e. the real failure
    else:  # pragma: no cover -- the fake open above always fails
        raise AssertionError("expected start() to raise the retry's OSError")

    # Never left in a half-set state (same promise as the non-broken-stderr
    # failure tests above).
    assert session.case_name is None
    assert session.log_file is None


def test_set_testcase_rejects_log_filename_with_path_separator(tmp_path, monkeypatch) -> None:
    """(b) `set_testcase` must reject a `log_filename` containing a path
    separator rather than opening it or crashing."""
    monkeypatch.chdir(tmp_path)
    from netlist_agent.llm.tools_schema import set_testcase

    session = Session()
    result = set_testcase(session, "case1", log_filename="sub/dir/x.log")

    assert "error" in result
    assert session.case_name is None
    assert session.log_file is None


def test_set_testcase_second_call_is_a_noop(tmp_path, monkeypatch) -> None:
    """(c) Calling `set_testcase` a second time (case name already
    recognized) must be a harmless no-op: `log_path` unchanged, existing log
    content not truncated."""
    monkeypatch.chdir(tmp_path)
    from netlist_agent.llm.tools_schema import set_testcase

    session = Session()
    first = set_testcase(session, "case1")
    assert first["case_name"] == "case1"
    assert session.log_file is not None
    session.log_file.write("marker-content\n")
    session.log_file.flush()

    second = set_testcase(session, "case1", log_filename="something_else.log")

    assert second == first
    assert session.log_path == "case1.log"
    with open(session.log_path) as f:
        content = f.read()
    assert "marker-content" in content
    session.close()


def test_start_stays_retryable_when_flushing_buffered_responses_fails(tmp_path, monkeypatch) -> None:
    """`start()` promises that a failure leaves every session field exactly
    as it was, so `_h_begin`'s "case name already recognized" guard still
    lets a later request retry. The *open* failure path has always honored
    that; the `pending_log` flush did not -- `case_name`/`log_path`/
    `log_file` were assigned first, so a write failure (a full disk being
    the obvious way) came back with `case_name` set and permanently blocked
    every retry, trapping all later responses in `pending_log` forever.

    Mutation this kills: move the three `self.<field> = ...` assignments
    back above the flush loop (where they sat before this batch). The retry
    at the end then never happens, because `session.case_name` is already
    "case1".
    """
    monkeypatch.chdir(tmp_path)
    real_open = open
    fail = [True]
    closed: list[bool] = []

    class _FullDisk:
        """A text file that fills up part-way through: the first `write` goes
        through to the real file, the second fails the way a full disk does.

        Writing the first one through for real is the point. A mock that
        refused every write would leave nothing on disk, and then the
        "reopening with mode 'w' truncates whatever the failed attempt left
        behind, so the buffered response cannot end up in the log twice"
        claim below would be asserted against an empty file -- true for the
        wrong reason, and blind to `open(path, "w")` being changed to
        `"a"`.
        """

        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped
            self._writes = 0

        def write(self, text: str) -> int:
            self._writes += 1
            if self._writes > 1:
                raise OSError(28, "No space left on device")
            return self._wrapped.write(text)

        def close(self) -> None:
            closed.append(True)
            self._wrapped.close()

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def fake_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if "w" in mode and fail[0]:
            return _FullDisk(handle)
        return handle

    monkeypatch.setattr("builtins.open", fake_open)

    session = Session()
    # Two buffered responses, so the failure lands *between* them: the first
    # reaches the real file, the second doesn't. That is what puts partial
    # content on disk for the retry to have to truncate away.
    buffered = "#RESPONSE 1\nemitted before the case name was known\n#END 1\n"
    buffered_2 = "#RESPONSE 2\nsecond buffered response\n#END 2\n"
    session.pending_log.extend([buffered, buffered_2])

    try:
        session.start("case1")
    except OSError as exc:
        assert exc.errno == 28
    else:  # pragma: no cover -- the fake open above always fails the write
        raise AssertionError("expected the buffered-response flush to fail")

    # Nothing was touched, so a retry is still possible...
    assert session.case_name is None
    assert session.log_path is None
    assert session.log_file is None
    assert session.pending_log == [buffered, buffered_2]
    # ...and the half-written file was closed rather than leaked.
    assert closed == [True]
    # The failed attempt really did leave content behind -- without this the
    # truncation assertion at the end would be vacuous.
    with real_open("case1.log") as f:
        assert "emitted before the case name was known" in f.read()

    # ...and the retry really does succeed, with each buffered response
    # landing in the log exactly once: mode "w" truncates what the failed
    # attempt left behind. Mutation this pins: `open(path, "w", ...)` ->
    # `open(path, "a", ...)`, which would replay the first response twice.
    fail[0] = False
    session.start("case1")
    assert session.case_name == "case1"
    assert session.pending_log == []
    session.close()

    with real_open("case1.log") as f:
        content = f.read()
    assert content.count("emitted before the case name was known") == 1
    assert content.count("second buffered response") == 1
