"""Cross-validate our own parser's gate counts against a third-party
Verilog front end (`iverilog`) for every design in a corpus root.

Why iverilog and not ABC: ABC's `read_verilog` cannot parse the corpus's
named-connection instantiation style (`dff g15(.RN(...), ...)`) -- the very
first `dff` instance in the corpus makes it choke. `abc_bridge.py` works
around that by first running the netlist through *our own*
`extract_combinational_view` and *our own* writer to produce a file ABC can
read -- but that means ABC never sees anything except our parser's output,
so it cannot be used to validate the parser itself; that would be circular.
`iverilog` has its own independent parser and elaborator and can read the
corpus's original .v files directly, so it is used here instead.

Method (see also tests/test_gate_count_crosscheck.py):

  1. Elaborate `<design>.v` together with a stub `dff` module (the corpus
     depends on `dff` as an external primitive, with ports RN/SN/CK/D/Q) via
     `iverilog -s top -o out.vvp -tvvp <design>.v dff_stub.v`.
  2. Count occurrences of `.functor <TYPE>` in the compiled `out.vvp` for
     TYPE in {AND, OR, NAND, NOR, NOT, BUF, XOR, XNOR} -- these are iverilog's
     own gate primitives, one per line, and correspond 1:1 with our
     `GateType` values (minus DFF, counted separately below).
  3. Count occurrences of `.scope module, "<inst>" "dff"` for the DFF count.

Two pitfalls, discovered empirically and baked into the code below:

  - `BUFT` and `BUFZ` must be excluded from the BUF count: they are buffers
    iverilog inserts itself for net/port fanout, not gates present in the
    source. Evidence: test58 (Beta) has zero source `buf` gates, yet iverilog
    reports BUFT 8 + BUFZ 6 there; test01 (Alpha) has 8 source `buf` gates,
    and iverilog reports BUF 8 (matching) *plus* its own BUFZ/BUFT on top --
    i.e. BUF and BUFZ/BUFT are counted separately by iverilog, and only BUF
    corresponds to a real source gate.
  - `-s top` (explicit root module) must always be passed. Without it, the
    uninstantiated `dff` stub module becomes its own elaboration root, so a
    design with zero real DFF instances (e.g. test01) gets counted as having
    one. With `-s top`, test01 correctly comes back DFF=0 and test58 DFF=8.

Usage:  .venv/bin/python scripts/crosscheck_gate_counts.py [--root Alpha_Testcase]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from netlist_agent.analysis import gate_count_by_type
from netlist_agent.ir import GateType
from netlist_agent.parser import parse_verilog

DFF_STUB = """\
module dff(RN, SN, CK, D, Q);
  input RN, SN, CK, D; output Q; reg Q;
  always @(posedge CK or negedge RN) if (!RN) Q <= 1'b0; else Q <= D;
endmodule
"""

# iverilog .functor TYPE token -> our GateType. BUFT/BUFZ are deliberately
# left out: they are iverilog-synthesized fanout buffers, not source gates
# (see module docstring).
FUNCTOR_TO_GATE_TYPE = {
    "AND": GateType.AND,
    "OR": GateType.OR,
    "NAND": GateType.NAND,
    "NOR": GateType.NOR,
    "NOT": GateType.NOT,
    "BUF": GateType.BUF,
    "XOR": GateType.XOR,
    "XNOR": GateType.XNOR,
}

FUNCTOR_RE = re.compile(r"\.functor ([A-Z]+)")
DFF_SCOPE_RE = re.compile(r'\.scope module, "[^"]*" "dff"')


def iverilog_counts(design_path: str, workdir: str) -> dict[GateType, int]:
    """Elaborate `design_path` with iverilog and return its gate counts,
    keyed by our own GateType (DFF included)."""
    stub_path = os.path.join(workdir, "dff_stub.v")
    with open(stub_path, "w") as f:
        f.write(DFF_STUB)
    vvp_path = os.path.join(workdir, "out.vvp")

    subprocess.run(
        ["iverilog", "-s", "top", "-o", vvp_path, "-tvvp", design_path, stub_path],
        check=True,
        capture_output=True,
        text=True,
    )

    with open(vvp_path) as f:
        vvp_text = f.read()

    counts: dict[GateType, int] = {gt: 0 for gt in GateType}
    for token in FUNCTOR_RE.findall(vvp_text):
        gate_type = FUNCTOR_TO_GATE_TYPE.get(token)
        if gate_type is not None:
            counts[gate_type] += 1
    counts[GateType.DFF] = len(DFF_SCOPE_RE.findall(vvp_text))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.path.join(REPO_ROOT, "Alpha_Testcase"),
        help="corpus root holding testcase/<name>/<name>.v (default: Alpha_Testcase)",
    )
    args = ap.parse_args()

    if shutil.which("iverilog") is None:
        print("iverilog not found on PATH -- install it (e.g. `brew install icarus-verilog`) "
              "to run this cross-check.", file=sys.stderr)
        return 1

    testcase_dir = os.path.join(args.root, "testcase")
    names = sorted(d for d in os.listdir(testcase_dir) if d.startswith("test"))

    mismatches = []
    for name in names:
        design_path = os.path.join(testcase_dir, name, f"{name}.v")
        if not os.path.isfile(design_path):
            continue
        ours = gate_count_by_type(parse_verilog(design_path))
        with tempfile.TemporaryDirectory(dir="/tmp") as workdir:
            theirs = iverilog_counts(design_path, workdir)

        for gate_type in GateType:
            if ours[gate_type] != theirs[gate_type]:
                mismatches.append((name, gate_type, ours[gate_type], theirs[gate_type]))
        print(f"{name}: ours={ {gt.value: ours[gt] for gt in GateType} } "
              f"iverilog={ {gt.value: theirs[gt] for gt in GateType} }")

    print("\n=== SUMMARY ===")
    print(f"designs checked: {len(names)}")
    if mismatches:
        print(f"MISMATCHES: {len(mismatches)}")
        for name, gate_type, ours_n, theirs_n in mismatches:
            print(f"  {name} {gate_type.value}: ours={ours_n} iverilog={theirs_n}")
        return 1
    print("no mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
