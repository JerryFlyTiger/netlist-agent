# demo

A way to drive the thing by hand instead of reading about it.

```sh
./demo/run_demo.sh --ask                    # type your own requests at a prompt
./demo/run_demo.sh --ask -netlist my.v       # ...against your own gate-level Verilog
```

The corpus-based tour below (`./demo/run_demo.sh` with no `-netlist`, `--list`,
replaying a named testcase, `--verify`) needs the contest's released testcases,
which this public repo does not ship -- see *Bring your own testcases* in the
top-level README. `--ask -netlist` does not need them: point it at any
gate-level Verilog file using only built-in primitive gates.

Nothing here wraps or simplifies the engine. `run_demo.sh` invokes the same
entry point the contest's evaluation environment would —
`python -m netlist_agent.cli -config <file>`, one request per line on stdin,
`#RESPONSE n` / `#END n` on stdout — and the only thing it adds is a scratch
working directory (`demo/runs/<case>/`, wiped at the start of each run) so
output netlists and logs don't land in `Alpha_Testcase/`. This public repo does
not include the `Alpha_Testcase/` corpus itself -- `--list`, replaying a
testcase by name, `--verify`, and bare `--ask`/`--file` (no `-netlist`) all need
it (put the contest's released testcases under `Alpha_Testcase/testcase/testNN/`
to use them). `--ask -netlist <path>` and `--file req.txt -netlist <path>` do not
need it.

## What to run

Works with no corpus:

| command | what it does |
|---|---|
| `./demo/run_demo.sh --ask -netlist my.v` | interactive against your own gate-level Verilog — type requests one per line, Ctrl-D to finish |
| `./demo/run_demo.sh --file req.txt -netlist my.v` | your own request file against your own netlist (the testcase framing and load lines are supplied for you if the file doesn't open with them) |

Needs the contest's released testcases under `Alpha_Testcase/testcase/testNN/`
(see *Bring your own testcases* in the top-level README):

| command | what it does |
|---|---|
| `./demo/run_demo.sh` | replays **test38**: 18 requests over a 4,700-line netlist — gate census, fanout capping with buffer insertion, inverter-pair collapsing, dead-logic removal, signal renaming, constant propagation, and an equivalence confirmation at the end |
| `./demo/run_demo.sh test30` | shorter, heavier: remap the whole design to AND/NOT only, sweep floating nodes, merge duplicate gates, then **minimize depth via ABC** (23 → 19 on this design) |
| `./demo/run_demo.sh --list` | every released testcase, with request count and design size |
| `./demo/run_demo.sh --ask [case]` | interactive against a named testcase instead of your own file |
| `./demo/run_demo.sh --file req.txt [case]` | your own request file against a named testcase |
| `./demo/run_demo.sh --verify [case]` | after the run, prove the written netlist is still equivalent to the original with ABC `cec` |

`--verify` is opt-in because it is only meaningful for a run whose requests are
all equivalence-preserving. One transform deliberately changes function.

## Requirements

- The project venv (`python3 -m venv .venv && .venv/bin/pip install -e '.[test]'`).
  The script falls back to `python3` if there's no venv, and says so if `yaml`
  can't be imported.
- ABC, for equivalence checks and the ABC-backed optimizations:
  `./scripts/setup_abc.sh`. Analysis-only requests work without it; the script
  warns up front rather than failing one request at a time.

## An API key IS needed

This public repo does not include the private rule-based router (see the top-level
README for why), so every request -- however it's phrased -- goes to the LLM.
Export a real `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` before running `--ask` or
replaying a testcase; without one, every line comes back as:

```
Internal error while handling this request (ConfigError: environment
variable 'OPENAI_API_KEY' is not set ...); skipping to the next line.
```

`demo/config.yaml` has the shape the problem statement's Section 6.2 specifies,
with `api_key: env:OPENAI_API_KEY` — resolved lazily, only when a request
actually reaches the LLM, but in this build that's every request.
