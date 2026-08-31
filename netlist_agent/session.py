"""Per-testcase mutable state threaded through the request-handling loop.

One `Session` is created per testcase (on the "beginning of a new testcase"
framing) and lives for the rest of that testcase's stdin lines.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import IO, Optional

from netlist_agent.constraints import StructuralConstraint
from netlist_agent.ir import Design, GateType


def _warn(message: str) -> None:
    """Print a warning to stderr, but never let the warning itself become
    the exception that takes the process down. `close()` uses this for
    every stderr write it makes, because it already promises callers it
    won't raise `OSError` -- and a broken stderr (measured: swapping in a
    stream whose `write()` raises `OSError(5)`) is exactly the kind of I/O
    failure that promise has to survive, not just a failed log `open()`.
    Swallowing an `OSError` costs nothing: if the whole stream can't be
    written to, a traceback couldn't have reached the user through it
    either, so re-raising would only trade a clean warning for an unhandled
    exception, with no gain in what actually gets reported. Also swallows
    `UnicodeEncodeError` -- measured separately: a non-ASCII case/design
    name embedded in one of these messages, plus a narrow `sys.stderr`
    encoding, raises that (a `ValueError` subclass, not an `OSError`) right
    out of `print()`. Unlike the `OSError` case, this one is about *this*
    message, not the whole stream -- `cli.py`'s `run()` hardens `sys.stderr`
    to replace unencodable characters instead, so this is the last-resort
    backstop for callers that don't go through `run()`. `sys.stderr` being
    closed raises `ValueError` too, but that path is NOT guarded here: no
    call site in this codebase ever closes `sys.stderr` (checked), so
    there's nothing here to measure it against."""
    try:
        print(message, file=sys.stderr)
    except (OSError, UnicodeEncodeError):
        pass


@dataclass
class Session:
    case_name: Optional[str] = None

    # `current_design` mutates in place as transforms (router.py) run against
    # it. `original_snapshot` is a wholly separate `Design` -- parsed
    # independently from the same file at load time, never touched again --
    # needed for "verify equivalence to the originally loaded netlist" style
    # requests. Re-parsing the source file a second time (rather than trying
    # to deep-copy `current_design`) is simplest: transforms mutate Design/
    # Gate objects in place, so a snapshot must not share any of that mutable
    # state, and parse_verilog already builds a fully independent Design from
    # scratch.
    current_design: Optional[Design] = None
    original_snapshot: Optional[Design] = None

    # Directory the design was most recently loaded from (relative to the
    # process cwd), used to default a bare output filename (no directory
    # component) on a subsequent write request to the same directory rather
    # than to bare cwd.
    load_dir: Optional[str] = None

    # Filename (not path) of the most recently loaded design, e.g.
    # "test01.v" -- used only as a last-resort log-naming fallback in
    # `close()` when the testcase's case name was never recognized at all.
    load_filename: Optional[str] = None

    log_path: Optional[str] = None
    log_file: Optional[IO[str]] = None

    # Responses emitted before `log_file` exists (case name not yet
    # recognized) are buffered here in wire-format text, in order, rather
    # than silently dropped -- `start()` flushes them to the newly opened
    # log file before returning, and `close()` falls back to a synthesized
    # log name for them if the case name is *never* recognized.
    pending_log: list[str] = field(default_factory=list)

    # The very first response ("this is the beginning of testcase X") is
    # response 1, per spec -- so this counter starts at 1 and is handed out,
    # then incremented, on each allocation (see `allocate_response_id`).
    next_response_id: int = 1

    # Bookkeeping for "how many gates were {added,removed,merged,eliminated}
    # by the {transform} just performed"-style follow-up queries: updated by
    # router.py immediately after every count-returning transform call, then
    # read back (without re-running anything) by the follow-up query.
    last_op_count: Optional[int] = None
    last_gate_delta: dict[GateType, int] = field(default_factory=dict)

    # Name (`fn.__name__`) of the transform function that produced
    # `last_op_count`, set alongside it at every one of `last_op_count`'s
    # write sites (router.py's `_run_and_track`/`_run_and_track_bits`, and
    # `_h_balance_depth`). Exists purely so the LLM tool layer
    # (llm/tools_schema.py) can tell "the model is about to re-run the SAME
    # operation `last_op_count` already recorded" apart from "the model is
    # running a DIFFERENT operation" -- see
    # experiments/count_question_reruns_2026-08-29/REPORT.md, where a model
    # asked for a past merge count re-ran do_deduplicate_gates instead and
    # reported the rerun's (wrong) number. `None` alongside `last_op_count`
    # also `None` means no count-tracked transform has run yet this session.
    last_op_kind: Optional[str] = None

    # Normalized form of the (non-`design`) positional arguments the
    # `last_op_kind` transform was actually called with, set alongside it at
    # the same write sites -- see `router._normalize_op_args` for the
    # normalization (stable, order-independent for list/set arguments; a
    # sorted tuple of tokens rather than the raw objects). Exists so the LLM
    # tool layer can tell an actual RERUN of the same request (same fn, same
    # args) apart from a different request that merely happens to share a
    # transform function -- `remap_to_basis`, for instance, backs 6
    # differently-scoped rule-routed handlers, so `last_op_kind` alone is too
    # coarse (comparing it in isolation flagged an unrelated first-time call
    # as a "rerun" of a scope it never touched). `None` alongside
    # `last_op_kind` also `None` means no count-tracked transform has run
    # yet this session; `None` while `last_op_kind` is set is not otherwise
    # produced by the write sites below (they always pass a real, if empty,
    # args tuple), but is handled the same as any other non-match rather
    # than assumed impossible.
    last_op_args: Optional[tuple] = None

    # Instance names of gates that matched the most recent "find all the
    # gates whose name includes X"-style query -- consumed by a later
    # "replace the found buffers..." follow-up request. Stored as plain name
    # strings, not Gate object references: a transform run between the query
    # and the follow-up can invalidate previously-held Gate references, so
    # names are re-looked-up fresh each time they're used.
    last_query_gate_names: list[str] = field(default_factory=list)

    # Headline floating-input/unconnected-output-port count from the most
    # recent "check for floating inputs/unconnected output ports" query --
    # consumed by a "how many floating signals were found?" follow-up
    # without recomputing anything. `None` (as opposed to `last_op_count`'s
    # docstring-specified transform-count semantics -- deliberately NOT
    # reused here) distinguishes "no such query has run yet this session"
    # from "the last query found exactly zero".
    last_floating_count: Optional[int] = None

    # Headline "flip-flops formally proven to have an enable/hold structure
    # in their D input logic" count (`EnableHoldResult.proven_hold`) from
    # the most recent enable/hold check -- consumed by a "how many
    # flip-flops were found to have enable or hold structures?" follow-up
    # without recomputing anything. A dedicated field rather than reusing
    # `last_op_count` for the same reason `last_floating_count` isn't
    # reused either (see its docstring just above): `last_op_count` is
    # transform-count semantics, and `None` here specifically distinguishes
    # "no such query has run yet this session" from "the last query
    # formally proved exactly zero".
    last_enable_hold_count: Optional[int] = None

    # Running total of gates rewritten by a transform that is a deliberate
    # FUNCTIONAL change (not equivalence-preserving) -- currently only the
    # name-pattern BUF->AND rewrite family. Read by scripts/run_corpus.py to
    # tell "the design is non-equivalent to its input because a functional
    # change was explicitly requested" apart from "the design is
    # non-equivalent because something is broken".
    functional_change_ops: int = 0

    # Hard structural bounds ("max fanout <= 4", "max depth <= 5") established
    # by earlier requests in THIS testcase -- per QA A63, these must keep
    # holding across every later request, even one ("remap the entire
    # design") that has nothing to do with fanout/depth itself, since
    # `Problem_Description/A_20260212.pdf` sec 5 gives the testcase NO credit
    # at all if any hard requirement is ultimately violated. Only requests
    # that recorded an explicit numeric bound append here (see router.py's
    # handlers and its `handle_request` docstring for the single point that
    # re-checks/re-enforces every entry after every request).
    structural_constraints: list[StructuralConstraint] = field(default_factory=list)

    # How many times `io_protocol.emit_response()` has failed to write a
    # response to stdout / to the testcase log, respectively (an `OSError`
    # or `UnicodeEncodeError` raised by that channel's `write()`/`flush()`,
    # swallowed there so one channel going bad -- a full disk is the
    # measured case -- cannot take the other channel's still-good responses
    # down with it). Counted rather than logged one warning per occurrence
    # so a disk that's full for the rest of the run doesn't turn into one
    # stderr line per remaining request; `close()` reads these back to emit
    # a single per-testcase summary line.
    stdout_write_failures: int = 0
    log_write_failures: int = 0

    # Latches once `close()` has emitted the summary line below, so a
    # second `close()` call (an explicit invariant this class promises --
    # see `test_close_is_idempotent`) doesn't print the same summary again.
    # Deliberately a separate flag rather than zeroing the counters after
    # printing: `stdout_write_failures`/`log_write_failures` stay readable
    # by tests/callers after `close()` returns, and zeroing them would also
    # make a *second* close() (idempotent, so it must still be safe to call)
    # silently under-report if new failures could ever occur post-close --
    # they can't today, but the counters' job is to reflect the session's
    # history, not "history since the summary was last printed".
    _write_failure_summary_warned: bool = False

    def start(self, case_name: str, log_filename: Optional[str] = None) -> None:
        """Initialize a fresh testcase: record its name and open the log
        file for writing (truncating any prior contents) -- `log_filename`
        if the request line named one explicitly (e.g. "...into case23.log"),
        otherwise "<case_name>.log". Any responses emitted earlier in this
        testcase, before the case name was recognized, are flushed to the
        newly opened file first so nothing is lost.

        The log file is opened, *and* the buffered responses flushed to it,
        before any session field is touched, so that if either step fails,
        `case_name`/`log_file` are left exactly as
        they were -- otherwise `case_name` would end up set with `log_file`
        still None, permanently defeating the "case name already recognized"
        guards `_h_begin`/`set_testcase` rely on and trapping every
        subsequent response in `pending_log` forever, even on retry with
        good arguments. If an *explicitly* named log file fails to open, one
        retry is made against the default "<case_name>.log" name before
        giving up.

        Opened with `errors="replace"` (encoding left at the locale default,
        so a normal UTF-8 environment sees no behavior change at all): a
        character the log's encoding can't represent becomes `?` instead of
        raising `UnicodeEncodeError` out of a later `write()` -- protocol
        integrity (every request line still gets a well-formed response,
        logged) matters more than one character's fidelity, since scoring
        reads the log."""
        path = log_filename or f"{case_name}.log"
        try:
            log_file = open(path, "w", errors="replace")
        except OSError as exc:
            if log_filename is None:
                raise
            fallback_path = f"{case_name}.log"
            _warn(f"warning: could not open log file {path!r} ({exc}); falling back to {fallback_path!r}")
            path = fallback_path
            log_file = open(path, "w", errors="replace")

        # The pending_log flush is done through the *local* `log_file`, before
        # any session field is assigned, for the same reason the open itself is
        # (see the docstring): a write/flush can fail too (a full disk being
        # the obvious way), and if `case_name` were already set by then,
        # `_h_begin`'s "case name already recognized" guard would refuse every
        # later retry -- the exact trap the open ordering exists to avoid,
        # reintroduced one statement further down. On failure the half-written
        # file is closed rather than leaked; a retry reopens it with mode "w",
        # which truncates, so the partial content cannot survive to be
        # duplicated.
        if self.pending_log:
            try:
                for text in self.pending_log:
                    log_file.write(text)
                log_file.flush()
            except Exception:
                log_file.close()
                raise

        self.case_name = case_name
        self.log_path = path
        self.log_file = log_file
        self.pending_log = []

    def allocate_response_id(self) -> int:
        response_id = self.next_response_id
        self.next_response_id += 1
        return response_id

    def close(self) -> None:
        """... (see module docstring for the general per-testcase framing)

        Deliberately never raises `OSError` (a genuine internal bug --
        anything that isn't `OSError` -- still propagates normally).
        `close()` runs once, after every response for the testcase has
        already been emitted (`cli.py`'s `run()` calls it once, unguarded,
        after the request loop) -- unlike `start()`, there is no later
        request line that could retry it, so re-raising an I/O failure here
        cannot recover anything; it can only turn a testcase that already
        produced complete, correct stdout into a dead process with no
        exit-code-0 and no log. Measured (subprocess, `chmod 0500` on the
        cwd so `open(path, "w")` raises `PermissionError`): before this,
        that took down the whole run with a bare traceback on stderr and
        exit code 1, even though stdout already held every `#RESPONSE`/
        `#END` pair -- the only thing lost was the emergency log write
        itself. `start()`'s asymmetric choice to re-raise is intentional and
        stays that way; do not "unify" the two."""
        # Last resort: the case name was never recognized at all (so
        # `log_file` never got opened) but responses were buffered and a
        # design was loaded at some point -- name the log after that
        # design's file stem rather than discard the buffered responses.
        if self.log_file is None and self.pending_log:
            if self.load_filename:
                stem = os.path.splitext(os.path.basename(self.load_filename))[0]
                candidate = f"{stem}.log"
                path = candidate
                # This is a best-effort emergency fallback name, not one the
                # request stream ever actually asked for -- it must never
                # truncate a file that already exists (e.g. a legitimate log
                # some earlier, unrelated testcase already wrote there).
                # Pick a sibling name instead, warning that we did.
                if os.path.exists(path):
                    path = f"{stem}.recovered.log"
                    n = 1
                    while os.path.exists(path):
                        path = f"{stem}.recovered.{n}.log"
                        n += 1
                    _warn(
                        f"warning: fallback log name {candidate!r} already exists -- writing the recovered "
                        f"testcase log to {path!r} instead"
                    )
                log_file = None
                try:
                    log_file = open(path, "w", errors="replace")
                    for text in self.pending_log:
                        log_file.write(text)
                except OSError as exc:
                    # Partial writes land on disk the same way `start()`'s
                    # flush leaves them (see its docstring) -- close what did
                    # get opened rather than leak the descriptor. A second
                    # failure here (closing itself can fail too) is not
                    # worth a second warning; the one below already says
                    # everything buffered was discarded.
                    if log_file is not None:
                        try:
                            log_file.close()
                        except OSError:
                            pass
                    # `log_path` was never actually set to `path` below when
                    # this fails, so it's left exactly as it entered this
                    # method. In `run()`'s actual usage that's `None` (a
                    # single `close()` call, reached only when `start()`
                    # never succeeded), but this is NOT a class invariant --
                    # a caller that does `start()`, `close()`, buffers more
                    # responses, then `close()`s again would reach here with
                    # a stale non-None `log_path` left over from the first
                    # `close()` (it clears `log_file` but not `log_path`),
                    # and a failure here would leave that unrelated old path
                    # sitting in the field. Not dirtied with a path nothing
                    # from *this* failure was ever written to, either way.
                    _warn(
                        f"warning: could not write the recovered testcase log to {path!r} ({exc}); "
                        f"{len(self.pending_log)} buffered response(s) discarded"
                    )
                else:
                    self.log_path = path
                    self.log_file = log_file
                self.pending_log = []
            else:
                _warn(
                    "warning: testcase log discarded -- no case name was ever recognized and no design was "
                    "ever loaded to name the log after"
                )
                self.pending_log = []
        if self.log_file is not None:
            try:
                self.log_file.close()
            except OSError as exc:
                _warn(f"warning: error closing testcase log {self.log_path!r} ({exc})")
            self.log_file = None
        # One summary line per testcase, not one per dropped write (see the
        # field docstrings on `stdout_write_failures`/`log_write_failures`)
        # -- emitted here rather than at the point of each failure so it
        # reflects the final per-channel totals for the whole testcase. The
        # `_write_failure_summary_warned` latch (not the counters themselves
        # -- see that field's docstring) keeps a second `close()` call from
        # printing this a second time, the same way `close()`'s other
        # branches above are already idempotent.
        if (self.stdout_write_failures or self.log_write_failures) and not self._write_failure_summary_warned:
            _warn(
                f"warning: {self.stdout_write_failures} response(s) failed to write to stdout, "
                f"{self.log_write_failures} response(s) failed to write to the testcase log this session"
            )
            self._write_failure_summary_warned = True
