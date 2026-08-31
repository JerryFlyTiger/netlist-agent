#!/usr/bin/env bash
#
# Drive the agent by hand -- against your own gate-level Verilog, or against a
# real contest testcase if you have Alpha_Testcase/testcase/ (the contest's
# released testcases; not included in this public repo, see the top-level
# README's "Bring your own testcases").
#
# This runs the production entry point -- `python -m netlist_agent.cli
# -config <file>`, reading one request per line from stdin -- exactly as the
# contest's evaluation environment would invoke it. Nothing here is a demo
# shim around the engine; the only thing this script adds is a scratch working
# directory so runs don't scatter output files across the repo.
#
# Works with no corpus:
#   ./demo/run_demo.sh --ask -netlist my.v          # your requests, your netlist
#   ./demo/run_demo.sh --file req.txt -netlist my.v # your request file, your netlist
#
# Needs Alpha_Testcase/testcase/ (the contest's released testcases):
#   ./demo/run_demo.sh                 # default tour: test38 (fanout capping,
#                                      #   inverter collapsing, dead-logic
#                                      #   removal, renaming, const propagation)
#   ./demo/run_demo.sh test30          # any released testcase
#   ./demo/run_demo.sh --list          # what's available
#   ./demo/run_demo.sh --ask test12    # interactive against a named testcase
#   ./demo/run_demo.sh --file my.txt test30  # your request file, named testcase
#   ./demo/run_demo.sh --verify test30 # also prove out.v == in.v with ABC
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
CASE_ROOT="$REPO_ROOT/Alpha_Testcase/testcase"
RUNS_DIR="$SCRIPT_DIR/runs"
DEFAULT_CASE="test38"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
dim()  { printf '\033[2m%s\033[0m\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '3,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

# --- what to run ------------------------------------------------------------
MODE="case"
CASE=""
REQ_FILE=""
NETLIST_PATH=""
VERIFY=0

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)   usage ;;
    --list)      MODE="list"; shift ;;
    --ask)       MODE="ask"; shift ;;
    --verify)    VERIFY=1; shift ;;
    --file)      MODE="file"; REQ_FILE="${2:-}"; [ -n "$REQ_FILE" ] || die "--file needs a path"; shift 2 ;;
    -netlist)    NETLIST_PATH="${2:-}"; [ -n "$NETLIST_PATH" ] || die "-netlist needs a path"; shift 2 ;;
    test*)       CASE="$1"; shift ;;
    *)           die "unknown argument: $1 (try --help)" ;;
  esac
done

if [ -n "$NETLIST_PATH" ] && [ "$MODE" != "ask" ] && [ "$MODE" != "file" ]; then
  die "-netlist is only supported with --ask or --file"
fi

# --- prerequisites ----------------------------------------------------------
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || true)"
  [ -n "$PY" ] || die "no python3 found"
fi
"$PY" -c 'import yaml' 2>/dev/null || die "$PY cannot import yaml. Set up the venv first:
    python3 -m venv .venv && .venv/bin/pip install -e '.[test]'"

# abc_bridge resolves ABC itself (scripts/find_abc.sh); this is only so the
# demo can say up front that a transform request is going to fail, instead of
# letting it fail one request at a time.
ABC_BIN="$("$REPO_ROOT/scripts/find_abc.sh" 2>/dev/null || true)"
if [ -z "$ABC_BIN" ]; then
  dim "warning: no ABC binary found -- analysis requests still work, but every"
  dim "         equivalence check and ABC-backed optimization will fail."
  dim "         Build it with ./scripts/setup_abc.sh"
fi

if [ "$MODE" = "list" ]; then
  if [ ! -d "$CASE_ROOT" ]; then
    die "This public repo does not include the contest testcases. Put the released testcases under Alpha_Testcase/testcase/testNN/, or use --ask -netlist <path> with your own gate-level Verilog file."
  fi
  bold "Released testcases (Alpha_Testcase/testcase/)"
  printf '%-8s %8s %10s  %s\n' CASE REQUESTS "GATE LINES" "FIRST REQUEST"
  for d in "$CASE_ROOT"/test*; do
    name="$(basename "$d")"
    reqs=$(( $(grep -c '' "$d/prompt.txt") - 2 ))   # minus the framing + load lines
    vlines=$(grep -c '' "$d/$name.v")
    first=$(sed -n '3p' "$d/prompt.txt" | cut -c1-56)
    printf '%-8s %8s %10s  %s\n' "$name" "$reqs" "$vlines" "$first"
  done
  exit 0
fi

if [ -n "$NETLIST_PATH" ]; then
  [ -f "$NETLIST_PATH" ] || die "no such file: $NETLIST_PATH"
  CASE="$(basename "$NETLIST_PATH" .v)"
  WORK="$RUNS_DIR/$CASE"
  rm -rf "$WORK"
  mkdir -p "$WORK/testcase/$CASE"
  cp "$NETLIST_PATH" "$WORK/testcase/$CASE/$CASE.v"
else
  CASE="${CASE:-$DEFAULT_CASE}"
  if [ ! -d "$CASE_ROOT" ]; then
    die "This public repo does not include the contest testcases. Put the released testcases under Alpha_Testcase/testcase/testNN/, or use --ask -netlist <path> with your own gate-level Verilog file."
  fi
  [ -d "$CASE_ROOT/$CASE" ] || die "no such testcase: $CASE (try --list)"

  # --- scratch working directory ---------------------------------------------
  # The requests name their design by a path relative to the working directory
  # ("...the file test38.v located in the directory testcase/test38/"), so the
  # run needs a cwd with `testcase/<case>/` under it. The design is *copied*
  # rather than symlinked: an output netlist is written back into the testcase
  # directory (which is what the contest Q&A asks for -- "write all output files
  # to the same testcase directory"), and a symlink would put it in the repo's
  # own Alpha_Testcase/ instead of in this scratch directory.
  WORK="$RUNS_DIR/$CASE"
  rm -rf "$WORK"
  mkdir -p "$WORK/testcase/$CASE"
  cp "$CASE_ROOT/$CASE/$CASE.v" "$CASE_ROOT/$CASE/prompt.txt" "$WORK/testcase/$CASE/"
fi

run_agent() {  # stdin -> the real CLI, inside $WORK
  ( cd "$WORK" && PYTHONPATH="$REPO_ROOT" PYTHONHASHSEED=0 \
      "$PY" -m netlist_agent.cli -config "$CONFIG" )
}

LOAD_LINE="Please load the design from the file $CASE.v located in the directory testcase/$CASE/."

case "$MODE" in
  ask)
    bold "Interactive: $CASE  ($(grep -c '' "$WORK/testcase/$CASE/$CASE.v") lines of Verilog)"
    dim  "Type one request per line, then Ctrl-D. These all work:"
    dim  "  Please count all the gates in this design and report the total count broken down by gate type."
    dim  "  What is the maximum combinational logic depth in the design?"
    dim  "  Which primary input has the highest fanout in this design?"
    dim  "  Find all pairs of back-to-back inverters and collapse them into direct wire connections."
    dim  "  Confirm that the design is still functionally equivalent to the original."
    dim  "  Please write the current design to the output file ${CASE}_out.v."
    dim  ""
    dim  "This public build has no rule-based router: every request above is answered by"
    dim  "the LLM. Set OPENAI_API_KEY or ANTHROPIC_API_KEY first, or every line will come"
    dim  "back as an 'Internal error ... API_KEY is not set' response."
    echo
    { printf 'This is the beginning of a new testcase. The case name is %s.\n%s\n' "$CASE" "$LOAD_LINE"; cat; } | run_agent
    ;;
  file)
    [ -f "$REQ_FILE" ] || die "no such request file: $REQ_FILE"
    bold "Requests from $REQ_FILE, against $CASE"
    echo
    # If the file doesn't open with the framing line, supply it plus the load,
    # so a file of bare questions just works.
    if head -1 "$REQ_FILE" | grep -qi 'beginning of a new testcase'; then
      run_agent < "$REQ_FILE"
    else
      { printf 'This is the beginning of a new testcase. The case name is %s.\n%s\n' "$CASE" "$LOAD_LINE"; cat "$REQ_FILE"; } | run_agent
    fi
    ;;
  case)
    bold "Replaying $CASE  ($(( $(grep -c '' "$CASE_ROOT/$CASE/prompt.txt") - 2 )) requests over $(grep -c '' "$CASE_ROOT/$CASE/$CASE.v") lines of Verilog)"
    dim  "requests: $CASE_ROOT/$CASE/prompt.txt"
    echo
    START=$(date +%s)
    run_agent < "$CASE_ROOT/$CASE/prompt.txt"
    ELAPSED=$(( $(date +%s) - START ))
    echo
    dim "-- $ELAPSED seconds wall clock, all requests answered --"
    ;;
esac

# --- what came out ----------------------------------------------------------
echo
bold "Files written to demo/runs/$CASE/"
( cd "$WORK" && find . -type f | sed 's|^\./||' | sort | while read -r f; do
    case "$f" in
      testcase/*/"$CASE".v|testcase/*/prompt.txt) continue ;;   # the copied inputs
    esac
    printf '  %-44s %s bytes\n' "$f" "$(wc -c < "$f" | tr -d ' ')"
  done )

if [ "$VERIFY" = "1" ]; then
  OUT_V="$WORK/testcase/$CASE/${CASE}_out.v"
  if [ ! -f "$OUT_V" ]; then
    dim "no ${CASE}_out.v was written, so there is nothing to verify."
  else
    echo
    bold "Equivalence check: ${CASE}_out.v vs the original, via ABC cec"
    dim  "(Only meaningful for a run whose requests are all equivalence-preserving."
    dim  " scripts/run_corpus.py is the harness that tracks that properly.)"
    PYTHONPATH="$REPO_ROOT" "$PY" - "$CASE_ROOT/$CASE/$CASE.v" "$OUT_V" <<'PYEOF'
import sys
from netlist_agent.abc_bridge import verify_equivalence
from netlist_agent.parser import parse_verilog

before, after = (parse_verilog(p) for p in sys.argv[1:3])
result = verify_equivalence(before, after)
verdict = "  equivalent" if result.equivalent else "  NOT equivalent"
print(verdict)
for line in (result.detail or "").strip().splitlines()[:6]:
    print("   ", line)
PYEOF
  fi
fi
