# netlist-agent

A natural-language front end over a gate-level Verilog netlist engine. Requests
arrive one per line on stdin; each gets a `#RESPONSE n` / `#END n` pair on
stdout and in `<case>.log`.

Ask it what drives a net, whether a path exists that avoids a node, or what a
registered output actually computes, and it answers from the netlist. Tell it to
cap fanout at 4, remap a cone to NAND, or optimize for depth, and it rewrites the
netlist — then proves with an equivalence checker that it didn't change the
function before accepting the result.

## What is and isn't in this repository

This is the open part of a personal project exploring ICCAD 2026 Problem A
(Cadence). Three things are deliberately not here:

- **The rule-based router.** In the full version, a router of ~120 regex
  patterns over 96 handlers recognizes a request and calls the engine directly —
  deterministic, no model in the loop, no latency. It is the part that took the
  most work and it stays private. This repository ships a stub in its place, so
  every request goes to the LLM path described below.
- **The contest problem statement, the Q&A, and the released testcases.** Those
  are Cadence's documents and Cadence's benchmark data. Redistributing them is
  not mine to do — the problem is Problem A of the 2026 CAD Contest at ICCAD,
  and the organizers publish all of it themselves. Tests that need the testcases
  skip cleanly when they are absent; see *Bring your own testcases*.
- **The measurement experiments** built on top of both.

What remains is the engine itself and the LLM path over it — which is a complete,
runnable system, not a husk. The engine is the part that actually computes
answers, and it is all here.

## The design idea worth stealing

The model is there to decide *which* operation a sentence means. It never
computes the answer.

Deriving a Boolean function, proving two nets equivalent, enumerating paths — a
model handed generic inspection tools and asked to do that gets it wrong. So the
LLM drives the engine through 65 typed tool schemas (`netlist_agent/llm/`), and
every claim about preserved function is checked by ABC's combinational
equivalence checker before the result is accepted. The one transform that
deliberately *changes* function says so.

That split is why the engine is worth publishing on its own: it is the half that
has to be right.

## Running it

```sh
python -m netlist_agent.cli -config config.yaml < requests.txt
```

The config file names the LLM provider and model. An `api_key` of `env:VARNAME`
reads the key from the environment rather than storing it in the file. Because
the rule-based router is not in this repository, a working LLM config is
required here — there is no offline path.

```sh
./demo/run_demo.sh --ask        # type your own requests at a prompt
```

## What the engine can do

Read and write gate-level Verilog; report gate and port inventories; walk
fanin/fanout cones, paths, depth, and fanout; answer whether a path exists,
avoids a node, or must pass through one; compare signals for equivalence,
symmetry, or constancy; find a pair of existing signals whose AND/NAND/OR/NOR/
XOR/XNOR matches a target net; prove that an output is asserted only under a
stated condition, and produce a counterexample when it isn't; derive what a
registered output actually computes; and find which flip-flops have an enable or
hold structure in their D logic, by proving that forcing one net makes D collapse
back to the flop's own Q.

On the transform side: rename, remove dangling logic, deduplicate, propagate
constants, collapse inverter chains, remap a cone or the whole design to a gate
basis, cap fanout, balance depth across a set of sinks with buffers, rewrite
gates matched by name pattern, reconnect one gate pin from one net to another,
and optimize for depth or for gate count under a depth bound — the last two
backed by ABC, with every result equivalence-checked before it is accepted.

A structural bound stated by one request keeps being enforced against every later
one, so "cap fanout at 4" is not undone by a remap three requests later. Six of
eight transforms break such a bound if nothing re-enforces it.

## Results

Two released testcase sets exist for this problem: an earlier one of 40 designs
carrying 459 request lines, and a later one of 91 designs carrying 680. Both were
replayed end to end through the real request loop — the same `stdin` → `#RESPONSE n`
path an evaluation run takes, with only the LLM swapped for a recorder so the run
costs no quota.

| | earlier set | later set |
|---|---|---|
| designs / request lines | 40 / 459 | 91 / 680 |
| crashed | 0 | 0 |
| requests left unanswered | 0 | 0 |
| written netlists that re-parse | all | all |
| outputs equivalent to their input | all | all |
| requests the rule router did not recognize | **0** | **8** (1.2%) |

The later set matters more than the first, because 51 of its 91 designs had never
been seen when the router's patterns were written, and the request wording is the
organizers' rather than ours. Recognizing 672 of 680 lines is therefore a
held-out number, not a number about the corpus the patterns were fitted to.

The eight misses are phrasing, not missing capability. Each one has a minimal pair
that routes: insert one adjective, name the subject differently, or use a synonym
for the verb, and the same handler answers it. Nothing about those eight requests
is beyond the engine.

The organizers also published a Q&A after this work was largely done. Eleven of its
entries settle semantics rather than clarify logistics — what counts as inside a
cone, what a register-to-register path is, when a structure is an enable. Checking
the implementation against all eleven: **eight already agreed**, **two disagreed and
were fixed**, and one names a runtime budget for which there is no mechanism here at
all, which is recorded rather than papered over. One of the eleven is externally
checkable: asked which of two ways to count a register-to-register path is correct —
one of them arrives at 107 for a particular released design, the other at 36 — the
organizers ruled for the first. The engine arrives at 107. It is the only figure in
this project whose expected value was published by someone else before we looked at
it, which makes it the one number here that cannot be a case of the implementation
grading its own homework.

**What is deliberately not claimed: any accuracy rate, and any rank.** The
organizers were asked twice for reference answers and declined both times, so no
ground truth exists outside their own grader. Everything above says the system does
not fall over, answers every line, and preserves function — necessary conditions,
not sufficient ones. A number for "how often is the answer right" would have to be
graded by the thing being graded, which is worth less than no number.

The corpus figures cannot be reproduced from this repository: the corpora are the
organizers' and are not redistributed here, and the harness that replays them lives
in the private half. What *can* be run here is the test suite below.

## Bring your own testcases

Some tests exercise the engine against real gate-level designs rather than
synthetic ones. Those designs are not in this repository, because they are the
contest organizers' benchmark data and not mine to redistribute.

They are published by the organizers themselves: look for Problem A on the
official site of the 2026 CAD Contest at ICCAD. There have been two releases,
and each has its own directory here:

```
Alpha_Testcase/testcase/testNN/testNN.v     # the earlier release
Beta_Testcase/testcase/testNN/testNN.v      # the later one
```

Each case directory also holds the release's own `prompt.txt`. Drop either or
both in and the corpus-backed tests for that release activate. Without them they
skip with a message naming the directory they wanted — they do not fail, and
they do not silently disappear.

Any gate-level Verilog netlist using only built-in primitive gates works with the
engine; the layout above is just what the corpus-backed tests look for.

## ABC dependency

This project uses ABC (Berkeley Logic Synthesis and Verification System) for
netlist reading, logic optimization, and combinational equivalence checking.

ABC has no release tags — its behavior tracks upstream commits directly — so the
exact commit this project builds against is pinned in [`abc.lock`](abc.lock),
not left to "whatever `git clone` gives you today".

**The ABC binary is never committed.** It's a ~30 MB platform-specific build
artifact, and git history is forever. Instead:

- `scripts/setup_abc.sh` builds the pinned commit into a shared location outside
  this repo (`~/opt/eda/abc/<commit>/<platform>/bin/abc` by default), so several
  projects on one machine reuse a single build.
- ABC's Makefile links libreadline unconditionally — no configure step, no
  fallback — so on a machine without the development headers the build dies
  at `mainUtils.c:32: fatal error: readline/readline.h`. `setup_abc.sh`
  compiles a one-line probe for the header and passes
  `ABC_USE_NO_READLINE=1` when it is missing, so the build works on a bare
  machine with no extra packages. Nothing here needs readline: the engine
  drives ABC as `abc -c "..."`, and readline only serves a human typing at
  ABC's own `abc>` prompt.
- `scripts/find_abc.sh` resolves which binary to use, in order: `$ABC_BIN` →
  `vendor/<platform>/abc` → the shared install → `abc` on `$PATH`. It fails
  loudly if none exist rather than silently doing something else.

```sh
./scripts/setup_abc.sh      # builds (or reuses) the pinned ABC commit
./scripts/find_abc.sh       # prints the resolved binary path
```

## Tests

[![CI](https://github.com/JerryFlyTiger/netlist-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/JerryFlyTiger/netlist-agent/actions/workflows/ci.yml)

`pytest` — **614 passing, 69 skipped** on a clean checkout with no testcases
present. The skips are not hidden: `addopts = ["-rs"]` makes pytest print the
reason for every one, and every reason names what is missing: an
"Alpha_Testcase corpus not present" or "Beta_Testcase corpus not present" for
the two benchmark releases, and "requires the private rule-based router" for
the part that stays private. Nothing fails, and nothing silently vanishes.

Drop both releases in and the corpus-backed tests activate, taking the suite to
**1060 passing, 13 skipped** — measured, not estimated. With only the earlier
release it is 1057 and 16.

CI builds ABC from scratch at the pinned commit on a clean Ubuntu runner and
runs the suite there — the actual proof that the dependency is reproducible on a
machine that has never seen this project, not just a claim in a README.

## Layout

```
netlist_agent/
  cli.py                stdin/stdout request loop, config, LLM fallback wiring
  router.py             stub — the real router is not in this repository
  ir.py parser.py writer.py    netlist representation, Verilog in and out
  graph.py analysis.py  connectivity, cones, paths, depth, fanout
  transform.py          equivalence-preserving rewrites and sweeps
  constraints.py        structural bounds that outlive the request that set them
  abc_bridge.py         ABC: equivalence checking, constancy, counterexamples
  abc_synth.py          ABC: depth and gate-count optimization
  signal_pair_search.py property_check.py    the two heavier analyses
  llm/                  65 tool schemas and the provider client
demo/run_demo.sh        drive it by hand
abc.lock                pinned ABC commit
```

## License

Apache-2.0. See [LICENSE](LICENSE).

The contest problem statement, Q&A, and testcases are not covered by this
license and are not distributed here — they belong to Cadence Design Systems.
