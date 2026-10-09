"""Synthetic/unit-level tests for the ABC bridge (netlist_agent/abc_bridge.py).

These exercise the module against small hand-built designs so each check is
cheap and the expected answer is known by inspection -- see
tests/test_abc_bridge_real_files.py for integration coverage against the real
40-testcase corpus.
"""

from __future__ import annotations

import copy
import os

import pytest

from netlist_agent.abc_bridge import (
    are_equivalent,
    check_symmetry,
    extract_combinational_view,
    is_constant,
    verify_equivalence,
)
from netlist_agent.ir import Const, Direction, GateType, NetBit
from netlist_agent.parser import parse_verilog
from netlist_agent.transform import collapse_double_inverters, remap_to_basis


def _write(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content)
    return str(path)


# ----------------------------------------------------------------------
# verify_equivalence
# ----------------------------------------------------------------------


def test_verify_equivalence_genuinely_equivalent_after_basis_remap(tmp_path) -> None:
    src = """
    module top(a, b, c, y);
      input a, b, c;
      output y;
      wire n1;
      and g0(n1, a, b);
      or g1(y, n1, c);
    endmodule
    """
    path = _write(tmp_path, "eq_basis.v", src)
    original = parse_verilog(path)
    transformed = parse_verilog(path)
    remap_to_basis(transformed, "nand_not")

    disallowed = {g.gate_type for g in transformed.gates} - {GateType.NAND, GateType.NOT, GateType.BUF, GateType.DFF}
    assert not disallowed

    result = verify_equivalence(original, transformed)
    assert result.equivalent, result.detail


def test_verify_equivalence_genuinely_equivalent_after_double_inverter_collapse(tmp_path) -> None:
    src = """
    module top(a, y);
      input a;
      output y;
      wire n1, n2;
      not g0(n1, a);
      not g1(n2, n1);
      buf g2(y, n2);
    endmodule
    """
    path = _write(tmp_path, "eq_dinv.v", src)
    original = parse_verilog(path)
    transformed = parse_verilog(path)
    collapse_double_inverters(transformed)

    result = verify_equivalence(original, transformed)
    assert result.equivalent, result.detail


def test_verify_equivalence_genuinely_not_equivalent(tmp_path) -> None:
    src = """
    module top(a, b, y);
      input a, b;
      output y;
      and g0(y, a, b);
    endmodule
    """
    path = _write(tmp_path, "neq.v", src)
    original = parse_verilog(path)
    mutated = parse_verilog(path)
    # Flip AND -> OR: a genuine functional change (e.g. a=1,b=0 differs).
    mutated.gates[0].gate_type = GateType.OR

    result = verify_equivalence(original, mutated)
    assert not result.equivalent
    assert "NOT EQUIVALENT" in result.detail


# ----------------------------------------------------------------------
# is_constant
# ----------------------------------------------------------------------


def test_is_constant_always_zero(tmp_path) -> None:
    src = """
    module top(a, y);
      input a;
      output y;
      wire n1;
      and g0(n1, a, 1'b0);
      buf g1(y, n1);
    endmodule
    """
    path = _write(tmp_path, "const_zero.v", src)
    design = parse_verilog(path)
    assert is_constant(design, NetBit("n1", None)) == Const.ZERO


def test_is_constant_always_one(tmp_path) -> None:
    src = """
    module top(a, y);
      input a;
      output y;
      wire n1;
      or g0(n1, a, 1'b1);
      buf g1(y, n1);
    endmodule
    """
    path = _write(tmp_path, "const_one.v", src)
    design = parse_verilog(path)
    assert is_constant(design, NetBit("n1", None)) == Const.ONE


def test_is_constant_neither(tmp_path) -> None:
    src = """
    module top(a, b, y);
      input a, b;
      output y;
      and g0(y, a, b);
    endmodule
    """
    path = _write(tmp_path, "not_const.v", src)
    design = parse_verilog(path)
    assert is_constant(design, NetBit("y", None)) is None


# ----------------------------------------------------------------------
# check_symmetry
# ----------------------------------------------------------------------


def test_check_symmetry_and_is_symmetric(tmp_path) -> None:
    src = """
    module top(a, b, y);
      input a, b;
      output y;
      and g0(y, a, b);
    endmodule
    """
    path = _write(tmp_path, "sym_and.v", src)
    design = parse_verilog(path)
    assert check_symmetry(design, NetBit("y", None), NetBit("a", None), NetBit("b", None)) is True


def test_check_symmetry_and_not_is_not_symmetric(tmp_path) -> None:
    src = """
    module top(a, b, y);
      input a, b;
      output y;
      wire nb;
      not g0(nb, b);
      and g1(y, a, nb);
    endmodule
    """
    path = _write(tmp_path, "sym_and_not.v", src)
    design = parse_verilog(path)
    assert check_symmetry(design, NetBit("y", None), NetBit("a", None), NetBit("b", None)) is False


def test_check_symmetry_vacuous_case_is_symmetric(tmp_path) -> None:
    src = """
    module top(a, b, c, y);
      input a, b, c;
      output y;
      buf g0(y, c);
    endmodule
    """
    path = _write(tmp_path, "sym_vacuous.v", src)
    design = parse_verilog(path)
    # Neither `a` nor `b` appears in y's fanin cone at all -- vacuously symmetric
    # (note: probing a *used* input against an *unused* one, e.g. (a, b) with y
    # depending only on a, is correctly NOT vacuous -- swapping would actually
    # change behavior. The vacuous case needs BOTH probed inputs absent.)
    assert check_symmetry(design, NetBit("y", None), NetBit("a", None), NetBit("b", None)) is True


# ----------------------------------------------------------------------
# are_equivalent
# ----------------------------------------------------------------------


def test_are_equivalent_same_function_different_gates(tmp_path) -> None:
    src = """
    module top(a, b, y1, y2);
      input a, b;
      output y1, y2;
      wire n1, n2;
      and g0(n1, a, b);
      and g1(n2, a, b);
      buf g2(y1, n1);
      buf g3(y2, n2);
    endmodule
    """
    path = _write(tmp_path, "eq_same_fn.v", src)
    design = parse_verilog(path)
    assert are_equivalent(design, NetBit("n1", None), NetBit("n2", None)) is True


def test_are_equivalent_identical_net_is_trivially_equivalent(tmp_path) -> None:
    src = """
    module top(a, b, y);
      input a, b;
      output y;
      wire n1;
      and g0(n1, a, b);
      buf g1(y, n1);
    endmodule
    """
    path = _write(tmp_path, "eq_same_net.v", src)
    design = parse_verilog(path)
    assert are_equivalent(design, NetBit("n1", None), NetBit("n1", None)) is True


def test_are_equivalent_different_functions_is_false(tmp_path) -> None:
    src = """
    module top(a, b, y1, y2);
      input a, b;
      output y1, y2;
      wire n1, n2;
      and g0(n1, a, b);
      or g1(n2, a, b);
      buf g2(y1, n1);
      buf g3(y2, n2);
    endmodule
    """
    path = _write(tmp_path, "eq_diff_fn.v", src)
    design = parse_verilog(path)
    assert are_equivalent(design, NetBit("n1", None), NetBit("n2", None)) is False


# ----------------------------------------------------------------------
# extract_combinational_view with an actual DFF
# ----------------------------------------------------------------------


def test_extract_combinational_view_dff_boundary(tmp_path) -> None:
    src = """
    module top(clk, rst, d_in, q_out);
      input clk, rst, d_in;
      output q_out;
      wire n_q, n_next;
      and g0(n_next, d_in, rst);
      dff g1(.RN(rst), .SN(1'b1), .CK(clk), .D(n_next), .Q(n_q));
      buf g2(q_out, n_q);
    endmodule
    """
    path = _write(tmp_path, "dff_boundary.v", src)
    design = parse_verilog(path)

    free_pi = extract_combinational_view(design, "free_pi")
    assert all(g.gate_type != GateType.DFF for g in free_pi.gates)
    assert {"g0", "g2"} <= {g.inst_name for g in free_pi.gates}
    # The Q net keeps its name and is re-driven by a BUF from the
    # per-instance PI `__dff_Q__g1`; it is not a port itself.
    assert not any(p.name == "n_q" for p in free_pi.ports)
    q_pi = next(p for p in free_pi.ports if p.name == "__dff_Q__g1")
    assert q_pi.direction == Direction.INPUT
    assert free_pi.signals["__dff_Q__g1"].direction == Direction.INPUT
    q_bufs = [g for g in free_pi.gates if g.gate_type == GateType.BUF and g.pins.get("O") == NetBit("n_q", None)]
    assert len(q_bufs) == 1
    assert q_bufs[0].pins["I0"] == NetBit("__dff_Q__g1", None)
    # The D-pin boundary is exposed via a canonical per-instance tap, keyed
    # on the DFF instance name (stable across transforms), not the net name.
    assert not any(p.name == "n_next" for p in free_pi.ports)
    d_port = next(p for p in free_pi.ports if p.name == "__dff_D__g1")
    assert d_port.direction == Direction.OUTPUT
    tap = next(
        g for g in free_pi.gates if g.gate_type == GateType.BUF and g.pins.get("O") == NetBit("__dff_D__g1", None)
    )
    assert tap.pins["I0"] == NetBit("n_next", None)

    const_zero = extract_combinational_view(design, "const_zero")
    assert all(g.gate_type != GateType.DFF for g in const_zero.gates)
    assert not any(p.name == "n_q" for p in const_zero.ports)
    assert const_zero.signals["n_q"].direction == Direction.INTERNAL
    tie_gates = [
        g
        for g in const_zero.gates
        if g.gate_type == GateType.BUF and g.pins.get("O") == NetBit("n_q", None) and g.pins.get("I0") == Const.ZERO
    ]
    assert len(tie_gates) == 1
    d_port_cz = next(p for p in const_zero.ports if p.name == "__dff_D__g1")
    assert d_port_cz.direction == Direction.OUTPUT


def test_extract_combinational_view_q_wired_straight_to_existing_po(tmp_path) -> None:
    """A DFF's Q net literally IS an already-declared primary output. It
    stays an OUTPUT (with real truth: driven by a BUF from the
    `__dff_Q__<inst>` PI); it is not turned into an INPUT."""
    src = """
    module top(clk, rst, d_in, q_out);
      input clk, rst, d_in;
      output q_out;
      dff g0(.RN(rst), .SN(1'b1), .CK(clk), .D(d_in), .Q(q_out));
    endmodule
    """
    path = _write(tmp_path, "dff_q_is_po.v", src)
    design = parse_verilog(path)

    free_pi = extract_combinational_view(design, "free_pi")
    ports_named_q_out = [p for p in free_pi.ports if p.name == "q_out"]
    assert len(ports_named_q_out) == 1
    assert ports_named_q_out[0].direction == Direction.OUTPUT
    assert free_pi.signals["q_out"].direction == Direction.OUTPUT
    drv = free_pi.net_driver[NetBit("q_out", None)]
    assert drv.gate_type == GateType.BUF
    assert drv.pins["I0"] == NetBit("__dff_Q__g0", None)
    assert next(p for p in free_pi.ports if p.name == "__dff_Q__g0").direction == Direction.INPUT

    result = verify_equivalence(design, copy.deepcopy(design))
    assert result.equivalent, result.detail


def test_extract_combinational_view_direct_dff_to_dff_chain(tmp_path) -> None:
    """Edge case (b): one DFF's Q net is literally another DFF's D net, with
    zero combinational gates between them. The D-side promotion must be
    skipped as a no-op (the value is already observable via the Q-side INPUT
    port), not turned into a second, conflicting OUTPUT port entry."""
    src = """
    module top(clk, rst, q1, q2);
      input clk, rst;
      output q1, q2;
      dff g0(.RN(rst), .SN(1'b1), .CK(clk), .D(q2), .Q(q1));
      dff g1(.RN(rst), .SN(1'b1), .CK(clk), .D(q1), .Q(q2));
    endmodule
    """
    path = _write(tmp_path, "dff_chain.v", src)
    design = parse_verilog(path)

    free_pi = extract_combinational_view(design, "free_pi")
    # Both DFFs dropped; the only gates are the two canonical D-pin taps and
    # the two Q BUFs re-driving q1/q2 from their per-instance PIs.
    assert all(g.gate_type == GateType.BUF for g in free_pi.gates)
    taps = {g.pins["O"]: g.pins["I0"] for g in free_pi.gates}
    assert taps == {
        NetBit("__dff_D__g0", None): NetBit("q2", None),
        NetBit("__dff_D__g1", None): NetBit("q1", None),
        NetBit("q1", None): NetBit("__dff_Q__g0", None),
        NetBit("q2", None): NetBit("__dff_Q__g1", None),
    }
    for name in ("q1", "q2"):
        matching_ports = [p for p in free_pi.ports if p.name == name]
        assert len(matching_ports) == 1
        assert matching_ports[0].direction == Direction.OUTPUT
        assert free_pi.signals[name].direction == Direction.OUTPUT


def test_extract_combinational_view_dff_q_shares_bus_with_combinational_bit(tmp_path) -> None:
    """Regression test for a real bug found in test39.v: a DFF's Q pin can be
    a bit-select of a wider bus whose OTHER bits are independently driven by
    ordinary combinational gates. Whole-Signal Direction promotion can't turn
    the whole bus into a primary input (bit 0 would become simultaneously a
    PI bit and gate-driven) -- the fix leaves the bus alone and re-drives the
    DFF's bit through a BUF from its own per-instance PI."""
    src = """
    module top(clk, rst, a, b, y_gate, y_dff);
      input clk, rst, a, b;
      output y_gate, y_dff;
      wire [1:0] shared;
      and g0(shared[0], a, b);
      dff g1(.RN(rst), .SN(1'b1), .CK(clk), .D(a), .Q(shared[1]));
      buf g2(y_gate, shared[0]);
      buf g3(y_dff, shared[1]);
    endmodule
    """
    path = _write(tmp_path, "dff_q_shares_bus.v", src)
    design = parse_verilog(path)

    free_pi = extract_combinational_view(design, "free_pi")
    assert all(g.gate_type != GateType.DFF for g in free_pi.gates)

    # The bus is never promoted or split: it stays INTERNAL and no consumer
    # is rewired.
    assert free_pi.signals["shared"].direction == Direction.INTERNAL
    assert not any(p.name == "shared" for p in free_pi.ports)

    # bit 0 (g0's output) is completely untouched.
    g0 = next(g for g in free_pi.gates if g.inst_name == "g0")
    assert g0.pins["O"] == NetBit("shared", 0)
    g2 = next(g for g in free_pi.gates if g.inst_name == "g2")
    assert g2.pins["I0"] == NetBit("shared", 0)

    # g3 still reads shared[1]; that bit is now driven by a BUF from the
    # per-instance PI.
    g3 = next(g for g in free_pi.gates if g.inst_name == "g3")
    assert g3.pins["I0"] == NetBit("shared", 1)
    drv = free_pi.net_driver[NetBit("shared", 1)]
    assert drv.gate_type == GateType.BUF
    assert drv.pins["I0"] == NetBit("__dff_Q__g1", None)
    assert free_pi.signals["__dff_Q__g1"].direction == Direction.INPUT

    # Equivalence checking across this exact boundary shape must still work
    # end to end (this is what actually matters -- the split is plumbing).
    unchanged = parse_verilog(path)
    result = verify_equivalence(design, unchanged)
    assert result.equivalent, result.detail


def test_verify_equivalence_survives_d_pin_rewire(tmp_path) -> None:
    """Regression test for a real bug surfaced by the corpus run (test30/
    test36/test39): a transform that changes WHICH net feeds a DFF's D pin
    (double-inverter collapse here; buffer insertion in the corpus) used to
    make verify_equivalence error out with "PO name sets differ", because
    the D boundary was keyed on the net name. With canonical per-instance
    taps it must compare cleanly and report equivalent."""
    src = """
    module top(clk, rst, a, q_out);
      input clk, rst, a;
      output q_out;
      wire n1, n2;
      not g0(n1, a);
      not g1(n2, n1);
      dff g2(.RN(rst), .SN(1'b1), .CK(clk), .D(n2), .Q(q_out));
    endmodule
    """
    path = _write(tmp_path, "d_rewire.v", src)
    original = parse_verilog(path)
    transformed = parse_verilog(path)
    collapsed = collapse_double_inverters(transformed)
    assert collapsed >= 1
    # Confirm the premise: the collapse really did rewire g2.D away from n2.
    dff = next(g for g in transformed.gates if g.inst_name == "g2")
    assert dff.pins["D"] == NetBit("a", None)

    result = verify_equivalence(original, transformed)
    assert result.equivalent, result.detail


# ----------------------------------------------------------------------
# `_run_abc`: nonzero exit code (an ABC crash, e.g. an internal assert) must
# raise ABCBridgeError with the exit code and a stderr excerpt, rather than
# silently returning empty/partial stdout (batch 6b, 2026-09-30).
# ----------------------------------------------------------------------


def test_run_abc_nonzero_exit_raises_with_stderr(monkeypatch) -> None:
    import subprocess as subprocess_module

    import pytest

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _run_abc

    class _FakeResult:
        returncode = -6
        stdout = ""
        stderr = "abc: some_file.c:123: some_func: Assertion `x == y' failed.\nAborted"

    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(subprocess_module, "run", lambda *a, **k: _FakeResult())

    from netlist_agent.abc_bridge import ABCInconclusiveError

    with pytest.raises(ABCBridgeError, match="Assertion") as exc:
        _run_abc("some script", timeout=5.0)
    # A crash is not "no verdict": it must stay loud.
    assert not isinstance(exc.value, ABCInconclusiveError)


# ----------------------------------------------------------------------
# Batch 14: "ABC gave no verdict" is its own exception family
# ----------------------------------------------------------------------


def test_parse_cec_output_verdicts_and_failures() -> None:
    import pytest

    from netlist_agent.abc_bridge import (
        ABCBridgeError,
        ABCInconclusiveError,
        CecUndecidedError,
        _parse_cec_output,
    )

    assert _parse_cec_output("Networks are equivalent after structural hashing.  Time = 0.00 sec").equivalent is True
    assert _parse_cec_output("Networks are NOT EQUIVALENT.  Time = 0.00 sec\nINPUT: ...").equivalent is False

    with pytest.raises(ABCBridgeError, match="could not compare") as miter:
        _parse_cec_output("Miter computation has failed.")
    assert not isinstance(miter.value, ABCInconclusiveError)

    with pytest.raises(CecUndecidedError, match="SAT solver timed out") as und:
        _parse_cec_output("Networks are undecided (SAT solver timed out).")
    assert type(und.value) is CecUndecidedError
    assert isinstance(und.value, ABCInconclusiveError) and isinstance(und.value, ABCBridgeError)

    with pytest.raises(CecUndecidedError):
        _parse_cec_output("Networks are UNDECIDED. Time = 1.00 sec")

    with pytest.raises(ABCBridgeError, match="unrecognized") as unk:
        _parse_cec_output("some totally unexpected banner")
    assert type(unk.value) is ABCBridgeError
    assert not isinstance(unk.value, ABCInconclusiveError)


def test_run_abc_timeout_raises_abc_timeout_error(monkeypatch) -> None:
    import subprocess as subprocess_module

    import pytest

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCInconclusiveError, ABCTimeoutError, _run_abc

    def _boom(*a, **k):
        raise subprocess_module.TimeoutExpired(cmd="abc", timeout=5.0)

    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(subprocess_module, "run", _boom)

    with pytest.raises(ABCTimeoutError, match="timed out after 5.0s") as exc:
        _run_abc("some script", timeout=5.0)
    assert isinstance(exc.value, ABCInconclusiveError)

    # 14b: the message names the ABC command but not the temp-file paths.
    with pytest.raises(ABCTimeoutError, match="timed out after 5.0s") as exc2:
        _run_abc('cec "/var/folders/zz/a.blif" "/var/folders/zz/b.blif"', timeout=5.0)
    assert "'cec'" in str(exc2.value)
    assert "/var/folders" not in str(exc2.value)
    assert ".blif" not in str(exc2.value)


def test_run_abc_nonzero_exit_message_has_command_code_stderr_but_no_paths(monkeypatch) -> None:
    import subprocess as subprocess_module
    import types

    import pytest

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, ABCInconclusiveError, _run_abc

    def _fake_run(*a, **k):
        return types.SimpleNamespace(returncode=134, stdout="", stderr="Assertion failed: boom")

    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(subprocess_module, "run", _fake_run)

    with pytest.raises(ABCBridgeError) as exc:
        _run_abc('cec "/var/folders/zz/a.blif" "/var/folders/zz/b.blif"', timeout=5.0)
    msg = str(exc.value)
    assert not isinstance(exc.value, ABCInconclusiveError)
    assert "code 134" in msg and "'cec'" in msg and "Assertion failed: boom" in msg
    assert "/var/folders" not in msg and ".blif" not in msg


def test_parse_cec_output_order_decided_verdicts_beat_undecided_word() -> None:
    import pytest

    from netlist_agent.abc_bridge import ABCBridgeError, ABCInconclusiveError, _parse_cec_output

    # A definite verdict wins even if the word "undecided" also appears.
    res = _parse_cec_output("Networks are NOT EQUIVALENT.  (1 output undecided)")
    assert res.equivalent is False
    res = _parse_cec_output("Networks are equivalent.  (0 outputs undecided)")
    assert res.equivalent is True
    # A failed miter stays a loud, non-inconclusive error even with "undecided" in the text.
    with pytest.raises(ABCBridgeError) as miter:
        _parse_cec_output("Miter computation has failed. Result is undecided.")
    assert not isinstance(miter.value, ABCInconclusiveError)


def test_resolve_abc_timeout_is_not_inconclusive(monkeypatch) -> None:
    import subprocess as subprocess_module

    import pytest

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, ABCInconclusiveError, _resolve_abc

    def _boom(*a, **k):
        raise subprocess_module.TimeoutExpired(cmd="bash", timeout=1.0)

    # Reset the global cache through monkeypatch so it is restored afterwards.
    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    monkeypatch.setattr(subprocess_module, "run", _boom)

    with pytest.raises(ABCBridgeError, match="timed out resolving") as exc:
        _resolve_abc()
    # Not finding the tool is not "ABC gave no verdict".
    assert not isinstance(exc.value, ABCInconclusiveError)


# ----------------------------------------------------------------------
# Batch 15 (F3): ABC's temp-dir paths must not reach user-facing text
# ----------------------------------------------------------------------

_ECHO = '======== ABC command line "cec "{d}/a.blif" "{d}/b.blif""'


def test_scrub_abc_output_unit() -> None:
    from netlist_agent.abc_bridge import _scrub_abc_output

    d = "/var/folders/xx/T/abc_bridge_q1"
    # Echo line removed, first line or not.
    assert _scrub_abc_output(_ECHO.format(d=d) + "\nNetworks are NOT EQUIVALENT.\n", d) == "Networks are NOT EQUIVALENT.\n"
    assert _scrub_abc_output("banner\n" + _ECHO.format(d=d) + "\nInput pattern: 01\n", d) == "banner\nInput pattern: 01\n"
    # Error line keeps the file name, loses the directory.
    assert _scrub_abc_output(f'Cannot open input file "{d}/x.blif". \n', d) == 'Cannot open input file "x.blif". \n'
    # Bare directory becomes <tmp>.
    assert _scrub_abc_output(f"dir is {d}\n", d) == "dir is <tmp>\n"
    # Lines without the directory are untouched.
    text = "Networks are NOT EQUIVALENT.  Time = 0.00 sec\nInput pattern: 101\n"
    assert _scrub_abc_output(text, d) == text


def _patch_abc(monkeypatch, make_result):
    """Patch ABC out; `make_result(tmpdir)` builds the fake CompletedProcess
    from the real temp dir recovered out of the script string."""
    import re
    import subprocess as subprocess_module
    import types

    from netlist_agent import abc_bridge as abc_bridge_module

    def _fake_run(argv, **k):
        tmpdir = re.search(r'"(.*?)/a\.blif"', argv[2]).group(1)
        rc, out, err = make_result(tmpdir)
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)

    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(subprocess_module, "run", _fake_run)


def _tiny_designs(tmp_path):
    path = _write(tmp_path, "t.v", "module top(a, b, y);\n input a, b;\n output y;\n and g0(y, a, b);\nendmodule\n")
    return parse_verilog(path), parse_verilog(path)


def test_run_cec_detail_has_no_tmpdir_or_echo_line(monkeypatch, tmp_path) -> None:
    from netlist_agent.abc_bridge import _run_cec

    def _res(d):
        return 0, _ECHO.format(d=d) + "\nNetworks are NOT EQUIVALENT.  Time = 0.00 sec\nInput pattern: 10\n", ""

    _patch_abc(monkeypatch, _res)
    a, b = _tiny_designs(tmp_path)
    res = _run_cec(a, b, 5.0)
    assert not res.equivalent
    assert "NOT EQUIVALENT" in res.detail and "Input pattern: 10" in res.detail
    assert "abc_bridge_" not in res.detail and "ABC command line" not in res.detail


@pytest.mark.parametrize(
    ("body", "expect_in"),
    [
        ("Miter computation has failed.", "Miter computation has failed"),
        ("Networks are undecided (SAT solver timed out).", "undecided (SAT solver"),
        ("some totally unexpected banner", "some totally unexpected banner"),
        ('Cannot open input file "{d}/nope.blif". ', 'Cannot open input file "nope.blif"'),
    ],
)
def test_run_cec_exception_messages_have_no_tmpdir(monkeypatch, tmp_path, body, expect_in) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    def _res(d):
        return 0, _ECHO.format(d=d) + "\n" + body.format(d=d) + "\n", ""

    _patch_abc(monkeypatch, _res)
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCBridgeError) as exc:
        _run_cec(a, b, 5.0)
    msg = str(exc.value)
    assert expect_in in msg
    assert "abc_bridge_" not in msg and "ABC command line" not in msg and "/a.blif" not in msg


def test_run_cec_nonzero_exit_stderr_is_scrubbed(monkeypatch, tmp_path) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    _patch_abc(monkeypatch, lambda d: (1, "", f'Cannot open input file "{d}/a.blif". '))
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCBridgeError) as exc:
        _run_cec(a, b, 5.0)
    msg = str(exc.value)
    assert "code 1" in msg and "Cannot open input file" in msg
    assert "abc_bridge_" not in msg


def test_run_cec_long_stderr_is_scrubbed_before_it_is_truncated(monkeypatch, tmp_path) -> None:
    # The message keeps only the last 500 chars of stderr. Truncating first
    # would cut the temp dir in half, and the half left over no longer
    # matches `tmpdir`, so it would survive the scrub (long macOS temp dirs).
    # With a short temp dir the cut falls before the path instead; then the
    # positive `"a.blif"` assertion is the one that fails.
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    _patch_abc(monkeypatch, lambda d: (1, "", f'Cannot open input file "{d}/a.blif". ' + "x" * 460))
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCBridgeError) as exc:
        _run_cec(a, b, 5.0)
    msg = str(exc.value)
    assert 'Cannot open input file "a.blif"' in msg
    assert "abc_bridge_" not in msg and "/a.blif" not in msg


def test_real_abc_not_equivalent_detail_has_no_tmpdir_and_keeps_counterexample(tmp_path) -> None:
    original, mutated = _tiny_designs(tmp_path)
    mutated.gates[0].gate_type = GateType.OR
    res = verify_equivalence(original, mutated)
    assert not res.equivalent
    assert "NOT EQUIVALENT" in res.detail
    assert "abc_bridge_" not in res.detail and "ABC command line" not in res.detail
    # AND vs OR differ exactly when one input is 1; either is a valid witness.
    from netlist_agent.property_check import parse_counterexample

    assert parse_counterexample(res.detail) in ({"a": 1, "b": 0}, {"a": 0, "b": 1})


# ----------------------------------------------------------------------
# Batch 16a: OSErrors become loud ABCBridgeError (never "inconclusive"),
# with no absolute path in the message; empty stderr gives no dangling colon.
# ----------------------------------------------------------------------

_INJECTED_PATH = "/var/folders/zz/leak_dir/abc_bridge_xyz/a.blif"


def _injected_os_error() -> OSError:
    return OSError(28, "No space left on device", _INJECTED_PATH)


def _assert_loud_and_pathless(exc_info) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, ABCInconclusiveError

    assert type(exc_info.value) is ABCBridgeError
    assert not isinstance(exc_info.value, ABCInconclusiveError)
    msg = str(exc_info.value)
    assert "No space left on device" in msg
    assert "/var/folders" not in msg and "leak_dir" not in msg and "abc_bridge_xyz" not in msg
    assert isinstance(exc_info.value.__cause__, OSError)


def test_run_cec_tempdir_creation_oserror_is_bridge_error(monkeypatch, tmp_path) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    a, b = _tiny_designs(tmp_path)

    def _boom(*args, **kwargs):
        raise _injected_os_error()

    monkeypatch.setattr(abc_bridge_module.tempfile, "TemporaryDirectory", _boom)
    with pytest.raises(ABCBridgeError) as exc:
        _run_cec(a, b, 5.0)
    _assert_loud_and_pathless(exc)


@pytest.mark.parametrize("failing_call", [1, 2])
def test_run_cec_write_blif_oserror_is_bridge_error(monkeypatch, tmp_path, failing_call) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    a, b = _tiny_designs(tmp_path)
    real_write = abc_bridge_module.write_blif
    calls = {"n": 0}

    def _fake_write(design, path):
        calls["n"] += 1
        if calls["n"] == failing_call:
            raise _injected_os_error()
        return real_write(design, path)

    monkeypatch.setattr(abc_bridge_module, "write_blif", _fake_write)
    with pytest.raises(ABCBridgeError) as exc:
        _run_cec(a, b, 5.0)
    assert calls["n"] == failing_call
    _assert_loud_and_pathless(exc)


def test_run_abc_exec_oserror_is_bridge_error(monkeypatch) -> None:
    import subprocess as subprocess_module

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _run_abc

    def _boom(*a, **k):
        raise OSError(8, "Exec format error", _INJECTED_PATH)

    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(subprocess_module, "run", _boom)
    with pytest.raises(ABCBridgeError) as exc:
        _run_abc("cec x y", timeout=5.0)
    msg = str(exc.value)
    assert type(exc.value) is ABCBridgeError
    assert "Exec format error" in msg and "a.blif" in msg
    assert "/var/folders" not in msg and "leak_dir" not in msg


def test_os_error_detail_without_strerror_or_filename() -> None:
    from netlist_agent.abc_bridge import _os_error_detail

    assert _os_error_detail(OSError()) == "OSError"
    assert _os_error_detail(PermissionError()) == "PermissionError"
    assert _os_error_detail(OSError(13, "Permission denied")) == "Permission denied"


def test_run_abc_nonzero_exit_empty_stderr_has_no_dangling_colon(monkeypatch) -> None:
    import subprocess as subprocess_module
    import types

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _run_abc

    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(
        subprocess_module, "run", lambda *a, **k: types.SimpleNamespace(returncode=-11, stdout="", stderr="  \n")
    )
    with pytest.raises(ABCBridgeError) as exc:
        _run_abc("cec x y", timeout=5.0)
    msg = str(exc.value)
    assert "code -11" in msg
    assert not msg.rstrip().endswith(":")


# ----------------------------------------------------------------------
# Batch 16 round-1 fixes: F1 (strerror may carry a path), F2 (temp-dir
# cleanup errors), F3 (locator exec failure), F9 note, gaps 2 and 3.
# ----------------------------------------------------------------------


def test_os_error_detail_ignores_instance_strerror_with_path() -> None:
    # `tempfile` puts the searched directories into `strerror` itself.
    from netlist_agent.abc_bridge import _os_error_detail

    exc = FileNotFoundError(2, "No usable temporary directory found in ['/var/folders/zz/T', '/tmp']")
    detail = _os_error_detail(exc)
    assert "/" not in detail and "var" not in detail
    assert detail == os.strerror(2)


def test_run_cec_tempdir_search_failure_message_has_no_path(monkeypatch, tmp_path) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    a, b = _tiny_designs(tmp_path)

    def _boom(*args, **kwargs):
        raise FileNotFoundError(2, "No usable temporary directory found in ['/var/folders/zz/T']")

    monkeypatch.setattr(abc_bridge_module.tempfile, "TemporaryDirectory", _boom)
    with pytest.raises(ABCBridgeError) as exc:
        _run_cec(a, b, 5.0)
    assert "/" not in str(exc.value)


@pytest.fixture
def _rmdir_fails_for_abc_dirs(monkeypatch):
    """Make `os.rmdir` fail for this module's temp dirs, as a cleanup error
    in `TemporaryDirectory.__exit__` would; remove the leftovers afterwards."""
    import shutil

    real_rmdir = os.rmdir
    left = []

    def _rmdir(path, *a, **k):
        if os.path.basename(str(path)).startswith(("abc_bridge_", "abc_synth_")):
            left.append(str(path))
            raise OSError(16, "Device or resource busy", str(path))
        return real_rmdir(path, *a, **k)

    monkeypatch.setattr(os, "rmdir", _rmdir)
    yield left
    monkeypatch.undo()
    for p in left:
        shutil.rmtree(p, ignore_errors=True)


def test_run_cec_cleanup_error_does_not_mask_bridge_error(monkeypatch, tmp_path, _rmdir_fails_for_abc_dirs) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    _patch_abc(monkeypatch, lambda d: (1, "", "boom"))
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCBridgeError) as exc:
        _run_cec(a, b, 5.0)
    assert "boom" in str(exc.value)
    assert _rmdir_fails_for_abc_dirs, "the cleanup failure was never exercised"


def test_run_cec_cleanup_error_does_not_break_success(monkeypatch, tmp_path, _rmdir_fails_for_abc_dirs) -> None:
    from netlist_agent.abc_bridge import _run_cec

    _patch_abc(monkeypatch, lambda d: (0, "Networks are equivalent.  Time = 0.00 sec\n", ""))
    a, b = _tiny_designs(tmp_path)
    res = _run_cec(a, b, 5.0)
    assert res.equivalent
    assert _rmdir_fails_for_abc_dirs, "the cleanup failure was never exercised"


def test_resolve_abc_locator_oserror_is_bridge_error(monkeypatch) -> None:
    import subprocess as subprocess_module

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, ABCInconclusiveError, _resolve_abc

    injected = OSError(2, "No such file or directory", "/usr/bin/bash")

    def _boom(*a, **k):
        raise injected

    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    monkeypatch.setattr(subprocess_module, "run", _boom)
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert type(exc.value) is ABCBridgeError and not isinstance(exc.value, ABCInconclusiveError)
    assert exc.value.__cause__ is injected
    assert "/" not in str(exc.value)


def test_run_abc_exec_oserror_cause_is_the_injected_error(monkeypatch) -> None:
    import subprocess as subprocess_module

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _run_abc

    injected = OSError(8, "Exec format error", _INJECTED_PATH)

    def _boom(*a, **k):
        raise injected

    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(subprocess_module, "run", _boom)
    with pytest.raises(ABCBridgeError) as exc:
        _run_abc("cec x y", timeout=5.0)
    assert exc.value.__cause__ is injected


def test_run_abc_nonzero_exit_scrubs_before_truncating(monkeypatch, tmp_path) -> None:
    # stderr is the temp path, "/in.blif", then 487 chars: 495 + len(tmpdir)
    # in all, so a 500-char tail cut BEFORE the scrub starts inside the temp
    # dir (for any tmpdir longer than 5 chars) and leaves half a path behind.
    import subprocess as subprocess_module
    import types

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _run_abc

    d = str(tmp_path)
    err = f"{d}/in.blif" + "y" * 487
    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(
        subprocess_module, "run", lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr=err)
    )
    with pytest.raises(ABCBridgeError) as exc:
        _run_abc("cec x y", timeout=5.0, tmpdir=d)
    msg = str(exc.value)
    assert "in.blif" in msg and "/" not in msg
