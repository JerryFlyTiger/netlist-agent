"""The wire format: for response `<id>` with text `<body>`, print to stdout

    #RESPONSE <id>
    <body>
    #END <id>

and write an identical copy to the testcase's log file. `#END <id>` is always
the last line of a response -- the (hypothetical) evaluation environment
waits for it before sending the next line -- so stdout is explicitly flushed
right after it rather than relying on default buffering.
"""

from __future__ import annotations

import sys
from typing import IO

from netlist_agent.session import Session, _warn


def format_response(response_id: int, body: str) -> str:
    return f"#RESPONSE {response_id}\n{body}\n#END {response_id}\n"


def emit_response(session: Session, response_id: int, body: str, stdout: IO[str] = sys.stdout) -> None:
    """Write one already-numbered response to `stdout` and to the session's
    open log file (if any), in the exact wire format above. If the log file
    isn't open yet (the testcase's case name hasn't been recognized yet),
    the response text is buffered in `session.pending_log` instead of being
    silently dropped -- `Session.start()` flushes it once the log file is
    finally opened.

    Deliberately never raises `OSError` or `UnicodeEncodeError` (same family
    `session.py`'s `_warn` already guards against -- see its docstring for
    why `UnicodeEncodeError` needs naming separately from `OSError`; it's a
    `ValueError` subclass, and `cli.py`'s `run()` only hardens the streams
    it opens itself, not every caller of this function). The two channels
    are guarded independently and neither's failure affects the other: a
    full disk breaking the log write must not cost the stdout side of this
    response (already written -- scoring aside, the `#END` handshake stdout
    needs is intact), and a broken stdout must not stop the log -- per spec
    3.3, scoring reads the log, not stdout, so it's the one channel that
    absolutely must keep getting every later response it can. Each channel
    warns on stderr only the first time it fails this session (see
    `Session.stdout_write_failures`/`log_write_failures`); every failure
    after that is counted but silent, since a disk that stays full for the
    rest of the run would otherwise turn into one stderr line per remaining
    request.

    Writing to an already-`.close()`d stream raises `ValueError` (checked
    against this exact codebase), not `OSError`/`UnicodeEncodeError` -- NOT
    guarded here, on the same reasoning `session.py`'s `_warn` docstring
    already uses for the closed-`sys.stderr` case it also declines to guard:
    checked, this path is unreachable today, so guarding it would just be
    dead code masking a real bug if it ever somehow became reachable. For
    the log channel specifically: `Session.close()` is the only call site
    that ever closes `log_file`, and it unconditionally sets `self.log_file
    = None` right after (outside its own try/except), so a closed-but-still-
    non-`None` `log_file` can't exist for this function's `is not None`
    check to walk into. The stdout channel is weaker, and worth saying so
    plainly: `run()`'s `stdout` is a parameter a caller can override (the
    tests do, constantly), so this rests on a check rather than on the
    structure -- no call site in this codebase closes whatever stream it
    passes, and `main()`, the only production caller, doesn't override the
    default at all."""
    text = format_response(response_id, body)
    try:
        stdout.write(text)
        stdout.flush()
    except (OSError, UnicodeEncodeError) as exc:
        session.stdout_write_failures += 1
        if session.stdout_write_failures == 1:
            _warn(f"warning: failed to write response {response_id} to stdout ({exc})")
    if session.log_file is not None:
        try:
            session.log_file.write(text)
            session.log_file.flush()
        except (OSError, UnicodeEncodeError) as exc:
            session.log_write_failures += 1
            if session.log_write_failures == 1:
                _warn(f"warning: failed to write response {response_id} to the testcase log ({exc})")
    else:
        session.pending_log.append(text)


def respond(session: Session, body: str, stdout: IO[str] = sys.stdout) -> int:
    """Allocate the next response id from `session` and emit it. Returns the
    id used, mostly for tests/callers that want to assert on numbering."""
    response_id = session.allocate_response_id()
    emit_response(session, response_id, body, stdout=stdout)
    return response_id
