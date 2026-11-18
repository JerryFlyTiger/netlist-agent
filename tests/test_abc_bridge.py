"""Synthetic/unit-level tests for the ABC bridge (netlist_agent/abc_bridge.py).

These exercise the module against small hand-built designs so each check is
cheap and the expected answer is known by inspection -- see
tests/test_abc_bridge_real_files.py for integration coverage against the real
40-testcase corpus.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import threading
import time

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


def test_write_blif_model_name_is_fixed(tmp_path) -> None:
    from netlist_agent.abc_bridge import write_blif

    long_name = "m" * 600
    src = f"module {long_name}(a, b, y);\n input a, b;\n output y;\n and g0(y, a, b);\nendmodule\n"
    design = parse_verilog(_write(tmp_path, "long.v", src))
    out = tmp_path / "o.blif"
    write_blif(design, str(out))
    model_lines = [ln for ln in out.read_text().splitlines() if ln.startswith(".model")]
    assert model_lines == [".model top"]
    assert design.module_name == long_name  # the user-visible name is untouched


def test_verify_equivalence_survives_600_byte_module_name(tmp_path) -> None:
    # ABC used to SIGTRAP on a .model name of >= 497 bytes.
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    try:
        _resolve_abc()
    except ABCBridgeError:
        pytest.skip("ABC binary not available")
    src = f"module {'m' * 600}(a, b, y);\n input a, b;\n output y;\n and g0(y, a, b);\nendmodule\n"
    path = _write(tmp_path, "long.v", src)
    result = verify_equivalence(parse_verilog(path), parse_verilog(path))
    assert result.equivalent, result.detail


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

    injected = subprocess_module.TimeoutExpired(cmd="bash", timeout=1.0)

    def _boom(*a, **k):
        raise injected

    # Reset the global cache through monkeypatch so it is restored afterwards.
    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    monkeypatch.setattr(subprocess_module, "run", _boom)

    with pytest.raises(ABCBridgeError, match="timed out resolving") as exc:
        _resolve_abc()
    # Not finding the tool is not "ABC gave no verdict".
    assert not isinstance(exc.value, ABCInconclusiveError)
    assert abc_bridge_module.REPO_ROOT not in str(exc.value)
    assert "scripts/find_abc.sh" in str(exc.value)
    assert f"after {abc_bridge_module._RESOLVE_TIMEOUT}s" in str(exc.value)
    assert exc.value.__cause__ is injected


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
    """Patch ABC out; `make_result(tmpdir)` builds the `(returncode, stdout, stderr)` that both race processes
    (P and L) finish with at once, the real temp dir recovered out of the script string. Which of the two reaches
    the queue first is not fixed; callers only use outcomes that do not depend on it."""
    import re

    from netlist_agent import abc_bridge as abc_bridge_module

    def _fake_start(abc_path, script):
        tmpdir = re.search(r'"(.*?)/a\.blif"', script).group(1)
        return _FakeProc("X", make_result(tmpdir))

    monkeypatch.setattr(abc_bridge_module, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(abc_bridge_module, "_start_abc", _fake_start)


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


# ----------------------------------------------------------------------
# Batch 17 (F10): the locator's failure text is one clean line, no paths
# ----------------------------------------------------------------------
def _fake_locator(monkeypatch, rc, stdout, stderr) -> None:
    import subprocess as subprocess_module

    from netlist_agent import abc_bridge as abc_bridge_module

    def _run(args, *a, **k):
        return subprocess_module.CompletedProcess(args, rc, stdout, stderr)

    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    monkeypatch.setattr(subprocess_module, "run", _run)


def _not_found_stderr() -> str:
    from netlist_agent import abc_bridge as abc_bridge_module

    root = abc_bridge_module.REPO_ROOT
    home = os.path.expanduser("~")
    return (
        "error: could not find an ABC binary for platform 'darwin-arm64'.\n\n"
        "Checked, in order:\n"
        "  $ABC_BIN (unset)\n"
        f"  {root}/vendor/darwin-arm64/abc\n"
        f"  {home}/abc-install/current/darwin-arm64/bin/abc\n"
        "  abc on $PATH\n\n"
        f"Run scripts/setup_abc.sh ... ({root}/scripts/find_abc.sh)\n"
    )


def _assert_clean_not_found(msg: str) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module

    assert abc_bridge_module.REPO_ROOT not in msg
    assert os.path.basename(abc_bridge_module.REPO_ROOT) not in msg
    assert os.path.expanduser("~") not in msg
    assert "\n" not in msg
    assert "could not find an ABC binary for platform" in msg
    assert "setup_abc.sh" in msg
    assert "exited with code 1" in msg
    assert "Checked, in order" not in msg
    assert "$ABC_BIN" in msg
    assert ".;" not in msg


def test_resolve_abc_not_found_message_is_one_line_without_paths(monkeypatch) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 1, "", _not_found_stderr())
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert type(exc.value) is ABCBridgeError
    _assert_clean_not_found(str(exc.value))


@pytest.mark.parametrize("stderr", ["", "  \n \n", "error:\n", "  error:   \n"])
def test_resolve_abc_empty_stderr_has_no_dangling_colon(monkeypatch, stderr) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 3, "", stderr)
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert ": ;" not in msg and not msg.endswith(":") and not msg.endswith(": ")
    assert "no message" in msg and "exited with code 3" in msg
    assert "\n" not in msg


def test_resolve_abc_bash_own_error_keeps_relative_script_name(monkeypatch) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(
        monkeypatch, 127, "", f"{abc_bridge_module.FIND_ABC_SCRIPT}: line 22: uname: command not found\n"
    )
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert "): scripts/find_abc.sh: line 22: uname: command not found;" in msg
    assert abc_bridge_module.REPO_ROOT not in msg
    assert os.path.basename(abc_bridge_module.REPO_ROOT) not in msg


def test_resolve_abc_abc_bin_diagnosis_keeps_value_with_home_shortened(monkeypatch) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    home = os.path.expanduser("~")
    _fake_locator(
        monkeypatch, 1, "", f"error: $ABC_BIN is set to '{home}/x/abc' but it is not an executable file\n"
    )
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert "'~/x/abc'" in msg
    assert home not in msg
    assert "error: $ABC_BIN" not in msg and "$ABC_BIN is set to" in msg


def test_resolve_abc_home_slash_does_not_mangle_paths(monkeypatch) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    monkeypatch.setenv("HOME", "/")
    stderr = "error: $ABC_BIN is set to '/usr/bin/x' but '/usr/bin/' and /tmp/ are not it, nor /\n"
    _fake_locator(monkeypatch, 1, "", stderr)
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert "'/usr/bin/x' but '/usr/bin/' and /tmp/ are not it, nor /;" in str(exc.value)


def test_resolve_abc_blank_stdout_is_an_error_and_not_cached(monkeypatch) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 0, "\n", "")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert type(exc.value) is ABCBridgeError
    assert "printed no path" in str(exc.value)
    assert abc_bridge_module._abc_path is None


def test_resolve_abc_real_script_not_found_message(monkeypatch, tmp_path) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    if (
        os.path.exists("/usr/bin/abc")
        or os.path.exists("/bin/abc")
        or os.path.exists(os.path.join(abc_bridge_module.REPO_ROOT, "vendor"))
    ):
        pytest.skip("an ABC binary is reachable; the not-found path cannot be exercised")
    monkeypatch.delenv("ABC_BIN", raising=False)
    monkeypatch.setenv("ABC_INSTALL_ROOT", str(tmp_path / "nonexistent"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(abc_bridge_module, "_abc_path", None)
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    _assert_clean_not_found(str(exc.value))


def test_resolve_abc_detail_is_first_non_blank_stderr_line(monkeypatch) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 1, "", "\n  \nerror: first problem\nsecond line\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert "): first problem;" in msg and "second line" not in msg


def test_resolve_abc_success_is_cached(monkeypatch) -> None:
    import subprocess as subprocess_module

    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import _resolve_abc

    _fake_locator(monkeypatch, 0, "/some/abc\n", "")
    assert _resolve_abc() == "/some/abc"

    def _no_more(*a, **k):
        raise AssertionError("locator re-run despite cached path")

    monkeypatch.setattr(subprocess_module, "run", _no_more)
    assert _resolve_abc() == "/some/abc"


def test_resolve_abc_bare_repo_root_becomes_placeholder(monkeypatch) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 1, "", f"error: searched {abc_bridge_module.REPO_ROOT} and {abc_bridge_module.REPO_ROOT}/vendor/x\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert "searched <repo> and vendor/x;" in msg
    assert abc_bridge_module.REPO_ROOT not in msg


def test_resolve_abc_detail_prefers_error_line_over_earlier_warning(monkeypatch) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(
        monkeypatch, 1, "", "bash: warning: setlocale: LC_ALL: cannot change locale\n  error: the real problem.\n"
    )
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert "): the real problem;" in msg and "setlocale" not in msg


def test_resolve_abc_detail_strips_indent_and_extra_spaces_after_error(monkeypatch) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 1, "", "   error:   foo  \n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert "): foo;" in str(exc.value)

    _fake_locator(monkeypatch, 1, "", "   plain line  \nerror:\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    # an `error:` line that is empty after stripping does not hide real text
    assert "): plain line;" in str(exc.value)

    _fake_locator(monkeypatch, 1, "", "warn\nerror:\nerror: real\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert "): real;" in str(exc.value)


@pytest.mark.parametrize(
    "stderr, expected",
    [("first\nsecond\n", "first"), ("error: a\nerror: b\n", "a"), ("error: foo .\n", "foo")],
)
def test_resolve_abc_detail_takes_first_line_and_trims_before_semicolon(monkeypatch, stderr, expected) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 1, "", stderr)
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert f"): {expected};" in str(exc.value)


def test_resolve_abc_detail_strips_only_one_trailing_dot(monkeypatch) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 1, "", "error: gone...\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert "): gone..;" in str(exc.value)


def test_resolve_abc_detail_strips_only_one_error_prefix(monkeypatch) -> None:
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 1, "", "error: error: x\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert "): error: x;" in str(exc.value)


def test_resolve_abc_path_replacement_respects_path_boundaries(monkeypatch) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    monkeypatch.setenv("HOME", "/Users/a")
    root = abc_bridge_module.REPO_ROOT
    _fake_locator(
        monkeypatch,
        1,
        "",
        f"error: x /Users/ab/y {root}2/z '/Users/a' {root}: /Users/a /Users/a.b/c {root}.bak\n",
    )
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert "/Users/ab/y" in msg
    assert f"{root}2/z" in msg
    assert "/Users/a.b/c" in msg and f"{root}.bak" in msg
    assert " '~' <repo>: ~ /Users/a.b/c" in msg


def test_resolve_abc_path_replacement_ignores_trailing_punctuation_and_quotes(monkeypatch) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    monkeypatch.setenv("HOME", "/Users/a")
    root = abc_bridge_module.REPO_ROOT
    stderr = f'error: {root}. ({root}) {root}, x "/Users/a" ({"/Users/a"}) /Users/a.\n'
    _fake_locator(monkeypatch, 1, "", stderr)
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert "): <repo>. (<repo>) <repo>, x \"~\" (~) ~;" in str(exc.value)


def test_resolve_abc_path_replacement_never_leaks_when_glued_to_other_text(monkeypatch) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    monkeypatch.setenv("HOME", "/Users/a")
    root = abc_bridge_module.REPO_ROOT
    stderr = f"error: -L{root}/lib x.{root}/sub x-{root}/sub -I/Users/a/inc 路徑/Users/a/x /mnt{root}/y -L{root} x.{root} y-{root} x./Users/a/z x-/Users/a end\n"
    _fake_locator(monkeypatch, 1, "", stderr)
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert root not in msg
    assert "/Users/a" not in msg
    assert "-Llib" in msg and "-I~/inc" in msg and "路徑~/x" in msg
    assert "-L<repo> x.<repo> y-<repo> x.~/z x-~ end;" in msg


@pytest.mark.parametrize("tail, expected", [("/Users/a", "~"), ("{R}", "<repo>")])
def test_resolve_abc_path_replacement_at_end_of_line(monkeypatch, tail, expected) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    monkeypatch.setenv("HOME", "/Users/a")
    tail = tail.replace("{R}", abc_bridge_module.REPO_ROOT)
    _fake_locator(monkeypatch, 1, "", f"error: no abc under {tail}\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert f"): no abc under {expected};" in str(exc.value)


def test_resolve_abc_repo_root_is_replaced_before_home(monkeypatch) -> None:
    # Pins the order without relying on where the real repo or $HOME live: both
    # repo steps (`<root>/` and bare `<root>`) must run before home is shortened,
    # or the line keeps `~/proj/...` / `~/proj`.
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    monkeypatch.setattr(abc_bridge_module, "REPO_ROOT", "/home/u/proj")
    monkeypatch.setenv("HOME", "/home/u")
    _fake_locator(
        monkeypatch,
        127,
        "",
        "/home/u/proj/scripts/find_abc.sh: line 22: uname: command not found; searched /home/u/proj\n",
    )
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert "): scripts/find_abc.sh: line 22: uname: command not found; searched <repo>;" in str(exc.value)


def test_resolve_abc_path_replacement_keeps_names_with_dash_suffix(monkeypatch) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    monkeypatch.setenv("HOME", "/Users/a")
    root = abc_bridge_module.REPO_ROOT
    _fake_locator(monkeypatch, 1, "", f"error: {root}-x /Users/a-b\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert f"{root}-x" in msg and "/Users/a-b" in msg


@pytest.mark.parametrize(
    "stderr, expected",
    [
        ("error:\nplain line\n", "plain line"),
        ("error: {R}/\nerror: real\n", "real"),
    ],
)
def test_resolve_abc_detail_is_chosen_after_cleaning(monkeypatch, stderr, expected) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 1, "", stderr.replace("{R}", abc_bridge_module.REPO_ROOT))
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert f"): {expected};" in str(exc.value)


def test_resolve_abc_detail_strips_dot_left_by_path_removal(monkeypatch) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    _fake_locator(monkeypatch, 1, "", f"error: cannot read config. {abc_bridge_module.REPO_ROOT}/\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    msg = str(exc.value)
    assert ".;" not in msg and "config;" in msg


def test_resolve_abc_path_replacement_escapes_regex_metacharacters(monkeypatch) -> None:
    from netlist_agent import abc_bridge as abc_bridge_module
    from netlist_agent.abc_bridge import ABCBridgeError, _resolve_abc

    monkeypatch.setattr(abc_bridge_module, "REPO_ROOT", "/tmp/a.b+c")
    _fake_locator(monkeypatch, 1, "", "error: /tmp/aXb+c/x /tmp/a.b+c/y /tmp/a.b+c done /tmp/aXbbc end\n")
    with pytest.raises(ABCBridgeError) as exc:
        _resolve_abc()
    assert "): /tmp/aXb+c/x y <repo> done /tmp/aXbbc end;" in str(exc.value)


# ----------------------------------------------------------------------
# Batch 36: `_run_cec` races the fast prover P (miter; &fraig -x; iprove) against the legacy `cec` L
# ----------------------------------------------------------------------

_CEC_EQ = "Networks are equivalent.  Time = 0.00 sec\n"
_CEC_NEQ = "Networks are NOT EQUIVALENT.  Time = 0.00 sec\nInput pattern: 10\n"
_P_UNSAT = "UNSATISFIABLE   Time = 0.1 sec\n"
_P_SAT = "SATISFIABLE (output = 0)  Time = 0.00 sec\n"
_NEVER = None  # outcome of a process that only ends when it is killed


class _FakeProc:
    """Stands in for a `subprocess.Popen`. `outcome` is `(returncode, stdout, stderr)` or `_NEVER`. `communicate()`
    blocks on `gate` (when given) and, for `_NEVER`, on `kill()`. `delay` (seconds) makes a finishing process wait
    that long, killable -- only for the deadline-relative test."""

    _pid = 1000

    def __init__(self, key, outcome, gate=None, delay=0.0):
        _FakeProc._pid += 1
        self.pid = _FakeProc._pid
        self.key, self.outcome, self.gate, self.delay = key, outcome, gate, delay
        self.returncode = None
        self.killed = False
        self.waited = False
        self._kill_ev = threading.Event()
        self._exited = threading.Event()

    def communicate(self):
        if self.outcome is _NEVER:
            self._kill_ev.wait(30)
            self.returncode = -9
            self._exited.set()
            return "", ""
        if self.gate is not None:
            self.gate()
        if self.delay and self._kill_ev.wait(self.delay):
            self.returncode = -9
            self._exited.set()
            return "", ""
        if isinstance(self.outcome, BaseException):
            self.returncode = 0
            self._exited.set()
            raise self.outcome  # a read failure (e.g. UnicodeDecodeError)
        self.returncode, out, err = self.outcome
        self._exited.set()
        return out, err

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        if self.returncode is None:
            self.returncode = -9
        self._kill_ev.set()

    def wait(self, timeout=None):
        self.waited = True
        self._exited.wait(10)
        return self.returncode


def _as_outcome(x):
    return x if x is _NEVER or isinstance(x, (tuple, BaseException)) else (0, x, "")


class _Race:
    """Replaces `_resolve_abc`, `_start_abc` and `_race_reader` inside abc_bridge. `p` / `l` are the outcomes of
    the two processes (a str = rc 0 with that stdout, a tuple = (rc, stdout, stderr), `_NEVER`). `order` is who
    reaches the queue first when both finish: the second finisher's `communicate()` waits until the first one's
    `_race_reader` has returned (so already put on the queue). No sleeps decide the order."""

    def __init__(self, monkeypatch, p, l, order=("P", "L"), p_delay=0.0):
        from netlist_agent import abc_bridge as m

        self.scripts: dict[str, str] = {}
        self.procs: dict[str, _FakeProc] = {}
        self.start_calls = 0
        self._done = {"P": threading.Event(), "L": threading.Event()}
        outcomes = {"P": _as_outcome(p), "L": _as_outcome(l)}
        delays = {"P": p_delay, "L": 0.0}
        first = order[0]
        orig_reader = m._race_reader

        def _gate_for(key):
            if key == first or outcomes[first] is _NEVER:
                return None
            return lambda: self._done[first].wait(10)

        def _start(abc_path, script):
            self.start_calls += 1
            key = "P" if script.startswith("miter") else "L"
            self.scripts[key] = script
            proc = _FakeProc(key, outcomes[key], gate=_gate_for(key), delay=delays[key])
            self.procs[key] = proc
            return proc

        def _reader(key, proc, q):
            orig_reader(key, proc, q)
            self._done[key].set()

        monkeypatch.setattr(m, "_resolve_abc", lambda: "/fake/abc")
        monkeypatch.setattr(m, "_start_abc", _start)
        monkeypatch.setattr(m, "_race_reader", _reader)


def _timed(fn):
    """`(fn(), wall seconds)`."""
    start = time.monotonic()
    result = fn()
    return result, time.monotonic() - start


def test_run_cec_p_unsat_first_is_equivalent_and_kills_legacy(monkeypatch, tmp_path) -> None:
    """kills: P's UNSAT not taken as a verdict; the detail built from anything but the matched verdict line; the
    legacy process left running (not killed / not waited for); the call waiting for L."""
    from netlist_agent.abc_bridge import _run_cec

    race = _Race(monkeypatch, "Networks are equivalent after sweeping.\n" + _P_UNSAT, _NEVER)
    a, b = _tiny_designs(tmp_path)
    res, wall = _timed(lambda: _run_cec(a, b, 100.0))
    assert res.equivalent
    assert res.detail == "Networks are equivalent (proved by SAT sweeping: UNSATISFIABLE   Time = 0.1 sec)"
    assert race.start_calls == 2
    assert 'miter "' in race.scripts["P"] and "&fraig -x" in race.scripts["P"]
    assert race.scripts["L"].startswith('cec "')
    assert race.procs["L"].killed and race.procs["L"].waited and race.procs["P"].waited
    assert wall < 5


def test_run_cec_legacy_equivalent_first_does_not_wait_for_p(monkeypatch, tmp_path) -> None:
    """kills: the call waiting for P (or giving P priority) before taking L's verdict; P not killed; L's detail
    replaced by P-style text."""
    from netlist_agent.abc_bridge import _run_cec

    race = _Race(monkeypatch, _NEVER, _CEC_EQ)
    a, b = _tiny_designs(tmp_path)
    res, wall = _timed(lambda: _run_cec(a, b, 100.0))
    assert res.equivalent and res.detail == _CEC_EQ.strip()
    assert race.procs["P"].killed and race.procs["P"].waited
    assert wall < 5


def test_run_cec_legacy_counterexample_first_does_not_wait_for_p(monkeypatch, tmp_path) -> None:
    """kills: a NOT EQUIVALENT answer from L held back until P finishes; the counterexample text altered."""
    from netlist_agent.abc_bridge import _run_cec

    race = _Race(monkeypatch, _NEVER, _CEC_NEQ)
    a, b = _tiny_designs(tmp_path)
    res, wall = _timed(lambda: _run_cec(a, b, 100.0))
    assert not res.equivalent and res.detail == _CEC_NEQ.strip()
    assert "Input pattern:" in res.detail
    assert race.procs["P"].killed and race.procs["P"].waited
    assert wall < 5


def test_run_cec_p_sat_then_legacy_equivalent_is_a_loud_error(monkeypatch, tmp_path) -> None:
    """kills: the engine-disagreement check removed (SAT from P silently overridden by L's equivalent), or `p_sat`
    only recorded when P is still running."""
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    _Race(monkeypatch, _P_SAT, _CEC_EQ, order=("P", "L"))
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCBridgeError, match="disagree"):
        _run_cec(a, b, 100.0)


def test_run_cec_p_sat_then_legacy_not_equivalent_returns_legacy_verbatim(monkeypatch, tmp_path) -> None:
    """kills: P's SAT taken as the answer (it has no counterexample text); L's result not passed through."""
    from netlist_agent.abc_bridge import _run_cec

    _Race(monkeypatch, _P_SAT, _CEC_NEQ, order=("P", "L"))
    a, b = _tiny_designs(tmp_path)
    res = _run_cec(a, b, 100.0)
    assert not res.equivalent and res.detail == _CEC_NEQ.strip()


# Real ABC words the warning `Warning: Constant-0 drivers added to N non-driven nets in network "top":` and then
# prints each such net's bare name on its own line. The production path never reaches it (`write_blif` ties every
# undriven net explicitly); these fixtures are defense in depth.
_WARN = 'Warning: Constant-0 drivers added to 1 non-driven nets in network "top":\n'
_FORGED = _WARN + "UNSATISFIABLE\nSATISFIABLE (output = 0)  Time = 0.00 sec\n"


def _non_unsat_p_outcomes():
    return [
        pytest.param("UNDECIDED\n", id="undecided"),
        pytest.param("something ABC never prints\n", id="unrecognized"),
        pytest.param("", id="empty"),
        pytest.param((134, "", "Assertion failed"), id="crash"),
        pytest.param((1, _P_UNSAT, ""), id="unsat-text-but-nonzero-exit"),
        pytest.param("the answer is not UNSATISFIABLE  Time = 0.00 sec\n", id="unsat-mid-line"),
        pytest.param("UNSATISFIABLE\nTime = 0.00 sec\n", id="unsat-spans-lines"),
        pytest.param(_WARN + "SATISFIABLE\nUNDECIDED      Time = 0.00 sec\n", id="bare-sat-net-name"),
        pytest.param("note: SATISFIABLE (output = 0) was not printed here\n", id="sat-mid-line"),
        pytest.param("SATISFIABLE\n(output = 0) Time = 0.00 sec\n", id="sat-spans-lines"),
    ]


@pytest.mark.parametrize("p", _non_unsat_p_outcomes())
def test_run_cec_non_unsat_p_first_then_legacy_equivalent_is_not_a_disagreement(monkeypatch, tmp_path, p) -> None:
    """kills: any non-UNSAT P outcome taken as a verdict; the SAT regex loosened (bare word / mid-line / spanning
    lines / `^` or `re.M` dropped) so that it raises "disagree" against L's equivalent; the UNSAT regex loosened
    (`^`, `[ \\t]+`, rc check dropped) so that P decides equivalence."""
    from netlist_agent.abc_bridge import _run_cec

    _Race(monkeypatch, p, _CEC_EQ, order=("P", "L"))
    a, b = _tiny_designs(tmp_path)
    res = _run_cec(a, b, 100.0)
    assert res.equivalent and res.detail == _CEC_EQ.strip()


@pytest.mark.parametrize("p", _non_unsat_p_outcomes())
def test_run_cec_non_unsat_p_first_then_legacy_not_equivalent_returns_legacy(monkeypatch, tmp_path, p) -> None:
    """kills: a non-UNSAT P outcome (incl. the forged/bare-name UNSAT lines) turning the result into equivalent."""
    from netlist_agent.abc_bridge import _run_cec

    _Race(monkeypatch, p, _CEC_NEQ, order=("P", "L"))
    a, b = _tiny_designs(tmp_path)
    res = _run_cec(a, b, 100.0)
    assert not res.equivalent and res.detail == _CEC_NEQ.strip()


def test_run_cec_forged_unsat_line_is_not_a_verdict(monkeypatch, tmp_path) -> None:
    """ABC prints an undriven net's name on its own line; a net called UNSATISFIABLE must not prove equivalence.
    kills: the `Time` suffix dropped from the UNSAT regex (bare `^UNSATISFIABLE` would take line 2)."""
    from netlist_agent.abc_bridge import _run_cec

    _Race(monkeypatch, _FORGED, _CEC_NEQ, order=("P", "L"))
    a, b = _tiny_designs(tmp_path)
    res = _run_cec(a, b, 100.0)
    assert not res.equivalent and res.detail == _CEC_NEQ.strip()


def test_run_cec_sat_line_after_a_warning_still_contradicts_legacy(monkeypatch, tmp_path) -> None:
    """The SAT verdict line is not the first line when ABC printed a warning; the contradiction check must still
    see it. kills: `re.M` dropped from the SAT regex (the guard goes silently dead)."""
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    _Race(monkeypatch, _WARN + "x\n" + _P_SAT, _CEC_EQ, order=("P", "L"))
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCBridgeError, match="disagree"):
        _run_cec(a, b, 100.0)


def test_run_cec_legacy_crash_then_p_unsat_is_equivalent(monkeypatch, tmp_path) -> None:
    """kills: L's failure raised at once while P is still running (L's exception must be held until P is done)."""
    from netlist_agent.abc_bridge import _run_cec

    _Race(monkeypatch, _P_UNSAT, (134, "", "boom"), order=("L", "P"))
    a, b = _tiny_designs(tmp_path)
    res = _run_cec(a, b, 100.0)
    assert res.equivalent and "proved by SAT sweeping" in res.detail


@pytest.mark.parametrize("p", ["UNDECIDED\n", (1, "", "p crashed"), "junk\n"], ids=["undecided", "crash", "junk"])
def test_run_cec_legacy_crash_then_p_without_verdict_raises_legacy_error(monkeypatch, tmp_path, p) -> None:
    """kills: P's non-verdict end swallowed (hanging until the deadline / timeout error instead); L's exception
    replaced by P's; the P branch's `if l_exc is not None: raise l_exc` dropped (the call would wait for the
    deadline -> the wall check)."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import ABCBridgeError, ABCTimeoutError, _run_cec

    _Race(monkeypatch, p, (134, "", "L boom"), order=("L", "P"))
    monkeypatch.setattr(m, "_real_wait", lambda t: 3.0)
    a, b = _tiny_designs(tmp_path)
    start = time.monotonic()
    with pytest.raises(ABCBridgeError, match=r"ABC exited with code 134 running 'cec'.*L boom") as exc:
        _run_cec(a, b, 100.0)
    assert not isinstance(exc.value, ABCTimeoutError)
    assert time.monotonic() - start < 2.0


def test_run_cec_legacy_crash_then_deadline_raises_legacy_error_not_timeout(monkeypatch, tmp_path) -> None:
    """L crashed, P never ends, the deadline comes: L's error, not a timeout message.
    kills: the `queue.Empty` branch raising the timeout without looking at L's held exception."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import ABCBridgeError, ABCTimeoutError, _run_cec

    race = _Race(monkeypatch, _NEVER, (134, "", "L boom"))
    monkeypatch.setattr(m, "_real_wait", lambda t: 0.3)
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCBridgeError, match=r"code 134 running 'cec'") as exc:
        _run_cec(a, b, 100.0)
    assert not isinstance(exc.value, ABCTimeoutError)
    assert race.procs["P"].killed and race.procs["P"].waited


def test_run_cec_legacy_undecided_then_p_unsat_is_equivalent(monkeypatch, tmp_path) -> None:
    """kills: `CecUndecidedError` from `_parse_cec_output` raised at once (it is an ABCBridgeError subclass that
    must be held like any other L failure)."""
    from netlist_agent.abc_bridge import _run_cec

    _Race(monkeypatch, _P_UNSAT, "Networks are undecided (SAT solver timed out).\n", order=("L", "P"))
    a, b = _tiny_designs(tmp_path)
    assert _run_cec(a, b, 100.0).equivalent


def test_run_cec_legacy_undecided_then_p_without_verdict_raises_undecided(monkeypatch, tmp_path) -> None:
    """kills: L's `CecUndecidedError` turned into a different exception type; the P branch's
    `if l_exc is not None: raise l_exc` dropped (the call would wait for the deadline -> the wall check)."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import CecUndecidedError, _run_cec

    _Race(monkeypatch, "UNDECIDED\n", "Networks are undecided (SAT solver timed out).\n", order=("L", "P"))
    monkeypatch.setattr(m, "_real_wait", lambda t: 3.0)
    a, b = _tiny_designs(tmp_path)
    start = time.monotonic()
    with pytest.raises(CecUndecidedError):
        _run_cec(a, b, 100.0)
    assert time.monotonic() - start < 2.0


def test_run_cec_p_without_verdict_then_legacy_crash_raises_at_once(monkeypatch, tmp_path) -> None:
    """P finished first with no verdict, L fails afterwards: L's exception at once.
    kills: the `p_done` check dropped (the call would wait for a deadline that never comes -> the wall check)."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import ABCBridgeError, _run_cec

    _Race(monkeypatch, "UNDECIDED\n", (134, "", "L boom"), order=("P", "L"))
    monkeypatch.setattr(m, "_real_wait", lambda t: 3.0)
    a, b = _tiny_designs(tmp_path)
    start = time.monotonic()
    with pytest.raises(ABCBridgeError, match="code 134"):
        _run_cec(a, b, 100.0)
    assert time.monotonic() - start < 2.0


def test_run_cec_p_read_failure_does_not_affect_legacy_verdict(monkeypatch, tmp_path) -> None:
    """P's output is unreadable (`UnicodeDecodeError` from `communicate()`): P counts as "no verdict"; L's
    verdict stands. kills: P's read error treated as a verdict, or raised instead of ignored; `_race_reader`
    without the `except` (P's thread dies before `_Race` releases L's gate, which then opens only on its
    10s fallback -> the wall check)."""
    from netlist_agent.abc_bridge import _run_cec

    err = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    _Race(monkeypatch, err, _CEC_NEQ, order=("P", "L"))
    a, b = _tiny_designs(tmp_path)
    start = time.monotonic()
    res = _run_cec(a, b, 100.0)
    assert time.monotonic() - start < 5.0
    assert not res.equivalent and res.detail == _CEC_NEQ.strip()


def test_run_cec_legacy_read_failure_with_p_without_verdict_raises_at_once(monkeypatch, tmp_path) -> None:
    """L's output is unreadable, P ended without a verdict: the read error is raised at once, without a path.
    kills: `_race_reader` without the `except` (L's thread dies, the main loop waits out the budget and reports
    a timeout -> the wall and type checks); the L read error not treated as an L failure (e.g. swallowed)."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import ABCBridgeError, ABCTimeoutError, _run_cec

    err = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    _Race(monkeypatch, "UNDECIDED\n", err, order=("P", "L"))
    monkeypatch.setattr(m, "_real_wait", lambda t: 3.0)
    a, b = _tiny_designs(tmp_path)
    start = time.monotonic()
    with pytest.raises(ABCBridgeError) as exc:
        _run_cec(a, b, 100.0)
    assert time.monotonic() - start < 2.0
    assert not isinstance(exc.value, ABCTimeoutError)
    assert str(exc.value) == "could not read ABC output running 'cec': UnicodeDecodeError"


def test_run_cec_legacy_read_failure_then_p_unsat_is_equivalent(monkeypatch, tmp_path) -> None:
    """L's read error is held like any L failure; P's UNSAT then still wins.
    kills: the L read error raised at once while P is still running; `_race_reader` without the `except`
    (L's thread dies before `_Race` releases P's gate, which then opens only on its 10s fallback -> the wall
    check)."""
    from netlist_agent.abc_bridge import _run_cec

    err = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    _Race(monkeypatch, _P_UNSAT, err, order=("L", "P"))
    a, b = _tiny_designs(tmp_path)
    start = time.monotonic()
    assert _run_cec(a, b, 100.0).equivalent
    assert time.monotonic() - start < 5.0


def test_run_cec_zero_remaining_wait_is_a_timeout_not_a_crash(monkeypatch, tmp_path) -> None:
    """The wait budget is already gone (`_real_wait` -> 0): `q.get` gets a 0 timeout and the result is
    `ABCTimeoutError`. kills: the `max(0.0, ...)` clamp dropped (a negative timeout makes `Queue.get` raise
    `ValueError`; with a tiny positive budget the elapsed time can already exceed it)."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import ABCTimeoutError, _run_cec

    _Race(monkeypatch, _NEVER, _NEVER)
    monkeypatch.setattr(m, "_real_wait", lambda t: 0.0)
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCTimeoutError):
        _run_cec(a, b, 100.0)


def test_run_cec_neither_finishes_times_out_with_the_nominal_wording(monkeypatch, tmp_path) -> None:
    """No request scope: the timeout message quotes `timeout`, verbatim.
    kills: the nominal / un-clamped wording changed or given the clamped suffix."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import ABCTimeoutError, _run_cec

    race = _Race(monkeypatch, _NEVER, _NEVER)
    monkeypatch.setattr(m, "_real_wait", lambda t: 0.2)
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCTimeoutError) as exc:
        _run_cec(a, b, 600.0)
    assert str(exc.value) == "ABC invocation timed out after 600.0s running 'cec'"
    assert all(p.killed and p.waited for p in race.procs.values())


def test_run_cec_neither_finishes_clamped_timeout_quotes_the_time_left(monkeypatch, tmp_path) -> None:
    """Inside a request scope the budget is clamped to the time left (30s of 100s): the message quotes 30.0s with
    the budget suffix, not the nominal 100.
    kills: the un-clamped branch used for the clamped message; the `t < timeout` test; `t` not the clamped value."""
    from netlist_agent import abc_bridge as m
    from netlist_agent import deadline
    from netlist_agent.abc_bridge import ABCTimeoutError, _run_cec

    _Race(monkeypatch, _NEVER, _NEVER)
    monkeypatch.setattr(m, "_real_wait", lambda t: 0.2)
    monkeypatch.setattr(deadline, "_monotonic", lambda: 0.0)
    a, b = _tiny_designs(tmp_path)
    with deadline.request_scope(30.0):
        with pytest.raises(ABCTimeoutError) as exc:
            _run_cec(a, b, 100.0)
    assert str(exc.value) == (
        "ABC invocation timed out after 30.0s running 'cec' (the time left in this request's budget)"
    )


def test_run_cec_not_started_when_the_request_has_no_time_left(monkeypatch, tmp_path) -> None:
    """kills: processes started although `spawn_timeout` said no; the not-started wording changed."""
    from netlist_agent import deadline
    from netlist_agent.abc_bridge import ABCTimeoutError, _run_cec

    race = _Race(monkeypatch, _P_UNSAT, _CEC_EQ)
    now = {"t": 0.0}
    monkeypatch.setattr(deadline, "_monotonic", lambda: now["t"])
    a, b = _tiny_designs(tmp_path)
    with deadline.request_scope(10.0):
        now["t"] = 100.0  # the budget is spent before `_run_cec` is reached
        with pytest.raises(ABCTimeoutError) as exc:
            _run_cec(a, b, 100.0)
    assert race.start_calls == 0
    assert str(exc.value).startswith("ABC was not started for 'cec': this request had less than ")
    assert "/" not in str(exc.value)


def test_run_cec_p_gets_the_whole_budget_not_a_quarter(monkeypatch, tmp_path) -> None:
    """L never ends; P's UNSAT arrives after 0.6 of the (real) wait budget. A P deadline of a quarter of the
    budget would give up on P first. The race's real wait is shrunk to 1.0s through `_real_wait`; P's fake process
    holds its answer for 0.6s (killable). The call must return the P verdict, between 0.6 and 1.0s.
    kills: a P-specific deadline of 0.25 x t (K8): it expires at 0.25s, before P answers, and the call raises the
    timeout instead; L's wait budget not being the whole one."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import _run_cec

    race = _Race(monkeypatch, _P_UNSAT, _NEVER, p_delay=0.6)
    monkeypatch.setattr(m, "_real_wait", lambda t: 1.0)
    a, b = _tiny_designs(tmp_path)
    res, wall = _timed(lambda: _run_cec(a, b, 100.0))
    assert res.equivalent and "proved by SAT sweeping" in res.detail
    assert wall >= 0.5
    assert race.procs["L"].killed


@pytest.mark.parametrize("fail_on", [0, 1], ids=["p-start-fails", "legacy-start-fails"])
def test_run_cec_start_oserror_is_a_bridge_error_and_cleans_up(monkeypatch, tmp_path, fail_on) -> None:
    """kills: the OSError escaping raw or becoming an inconclusive error; the message carrying a path; a process
    started before the failing one left running (not killed / not waited for)."""
    import errno

    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import ABCBridgeError, ABCInconclusiveError, _run_cec

    started: list[_FakeProc] = []

    def _start(abc_path, script):
        if len(started) == fail_on:
            raise OSError(errno.ENOENT, "No such file or directory", "/some/where/abc_bridge_x/abc")
        proc = _FakeProc("P", _NEVER)
        started.append(proc)
        return proc

    monkeypatch.setattr(m, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(m, "_start_abc", _start)
    a, b = _tiny_designs(tmp_path)
    with pytest.raises(ABCBridgeError) as exc:
        _run_cec(a, b, 100.0)
    assert not isinstance(exc.value, ABCInconclusiveError)
    assert str(exc.value) == "could not run ABC: No such file or directory (abc)"
    assert len(started) == fail_on
    assert all(p.killed and p.waited for p in started)


def test_run_cec_every_wait_goes_through_real_wait(monkeypatch, tmp_path) -> None:
    """kills: a wait that bypasses `_real_wait` (sim.py scales only that); the argument not being the budget
    actually granted (`t`, clamped inside a request scope)."""
    from netlist_agent import abc_bridge as m
    from netlist_agent import deadline
    from netlist_agent.abc_bridge import _run_cec

    seen: list[float] = []

    def _rw(t):
        seen.append(t)
        return t

    _Race(monkeypatch, _P_UNSAT, _NEVER)
    monkeypatch.setattr(m, "_real_wait", _rw)
    a, b = _tiny_designs(tmp_path)
    _run_cec(a, b, 100.0)
    assert seen == [100.0]

    seen.clear()
    _Race(monkeypatch, _P_UNSAT, _NEVER)
    monkeypatch.setattr(deadline, "_monotonic", lambda: 0.0)
    with deadline.request_scope(30.0):
        _run_cec(a, b, 100.0)
    assert seen == [30.0]


def test_run_cec_real_wait_scales_the_deadline(monkeypatch, tmp_path) -> None:
    """kills: `wait_budget` computed from `t` instead of `_real_wait(t)`: with a huge `t` and a tiny real wait the
    call must time out quickly."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import ABCTimeoutError, _run_cec

    _Race(monkeypatch, _NEVER, _NEVER)
    monkeypatch.setattr(m, "_real_wait", lambda t: t / 1000.0)
    a, b = _tiny_designs(tmp_path)
    start = time.monotonic()
    with pytest.raises(ABCTimeoutError):
        _run_cec(a, b, 200.0)
    assert time.monotonic() - start < 5



def test_verify_equivalence_real_abc_fast_path_equivalent(tmp_path) -> None:
    """Real ABC. kills: the race wired to the wrong scripts (an ABC error would surface or the verdict flip).
    Which engine wins on a tiny design is not fixed (P's UNSAT text or L's `Networks are equivalent after ...`),
    so only the verdict and its common prefix are asserted."""
    path = _write(tmp_path, "fp_eq.v", "module top(a, b, y);\n input a, b;\n output y;\n and g0(y, a, b);\nendmodule\n")
    original = parse_verilog(path)
    other = parse_verilog(path)
    result = verify_equivalence(original, other)
    assert result.equivalent, result.detail
    assert result.detail.startswith("Networks are equivalent")


def test_verify_equivalence_real_abc_not_equivalent_keeps_input_pattern(tmp_path) -> None:
    """Real ABC. kills: a SAT result from the fast path being returned instead of the legacy `cec` text (which
    carries `Input pattern:`)."""
    path = _write(tmp_path, "fp_neq.v", "module top(a, b, y);\n input a, b;\n output y;\n and g0(y, a, b);\nendmodule\n")
    original = parse_verilog(path)
    mutated = parse_verilog(path)
    mutated.gates[0].gate_type = GateType.OR
    result = verify_equivalence(original, mutated)
    assert not result.equivalent
    assert "Input pattern:" in result.detail


def test_fast_cec_script_real_abc_verdict_lines_match_the_regexes(tmp_path) -> None:
    """Real ABC, P's script run alone (no race). kills: `; iprove` dropped from `_FAST_CEC_SCRIPT` (no verdict
    line -> the UNSAT assertion); the UNSAT or SAT regex no longer matching real ABC's text; the script wired so
    the not-equivalent pair also prints UNSAT."""
    from netlist_agent.abc_bridge import _FAST_CEC_SAT_RE, _FAST_CEC_SCRIPT, _FAST_CEC_UNSAT_RE, _run_abc, write_blif

    path = _write(tmp_path, "fs.v", "module top(a, b, y);\n input a, b;\n output y;\n and g0(y, a, b);\nendmodule\n")
    same_a, same_b = parse_verilog(path), parse_verilog(path)
    diff_b = parse_verilog(path)
    diff_b.gates[0].gate_type = GateType.OR
    pa, pb, pc = (os.path.join(str(tmp_path), n) for n in ("fa.blif", "fb.blif", "fc.blif"))
    write_blif(same_a, pa)
    write_blif(same_b, pb)
    write_blif(diff_b, pc)
    eq_out = _run_abc(_FAST_CEC_SCRIPT.format(a=pa, b=pb), timeout=60.0)
    assert _FAST_CEC_UNSAT_RE.search(eq_out), eq_out
    assert not _FAST_CEC_SAT_RE.search(eq_out), eq_out
    neq_out = _run_abc(_FAST_CEC_SCRIPT.format(a=pa, b=pc), timeout=60.0)
    assert _FAST_CEC_SAT_RE.search(neq_out), neq_out
    assert not _FAST_CEC_UNSAT_RE.search(neq_out), neq_out


# ----------------------------------------------------------------------
# Batch 36-R2: `_run_cec` leaves no process behind (REAL child processes)
# ----------------------------------------------------------------------

_SLEEP30 = [sys.executable, "-c", "import time; time.sleep(30)"]
_PRINT_UNSAT = [sys.executable, "-c", "print('UNSATISFIABLE   Time = 0.1 sec')"]


@pytest.fixture
def real_race(monkeypatch):
    """`_run_cec` with `_start_abc` replaced by real `subprocess.Popen`s. `cmds` maps "P"/"L" to the argv to
    start (default: sleep 30s). Every started Popen is recorded in `started`; whatever the test did, the
    fixture kills and reaps them afterwards so a failing assertion cannot leave a sleeper behind."""
    from netlist_agent import abc_bridge as m

    cmds = {"P": _SLEEP30, "L": _SLEEP30}
    started: list = []

    def _start(abc_path, script):
        key = "P" if script.startswith("miter") else "L"
        proc = subprocess.Popen(cmds[key], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        started.append(proc)
        return proc

    monkeypatch.setattr(m, "_resolve_abc", lambda: "/fake/abc")
    monkeypatch.setattr(m, "_start_abc", _start)
    try:
        yield cmds, started
    finally:
        for proc in started:
            if proc.poll() is None:
                proc.kill()
        for proc in started:
            proc.wait()
            for f in (proc.stdout, proc.stderr):
                if f is not None:
                    f.close()


def _assert_reaped(started) -> None:
    """Every recorded process has exited AND been waited for (a zombie still answers `os.kill(pid, 0)`)."""
    assert len(started) == 2
    for proc in started:
        assert proc.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.kill(proc.pid, 0)


def test_run_cec_real_processes_reaped_after_normal_return(real_race, tmp_path) -> None:
    """P prints UNSAT and exits at once; L is a real 30s sleeper. The call returns the verdict without waiting
    for L, and L is killed and reaped.
    kills: the `finally` `kill()` of a still-running process removed (L would sleep on; `proc.wait()` and the
    reader's `communicate()` then block ~30s -> the wall check). Not claimed: the `wait()` loop removed (the
    reader's `communicate()` reaps the child by itself)."""
    from netlist_agent.abc_bridge import _run_cec

    cmds, started = real_race
    cmds["P"] = _PRINT_UNSAT
    a, b = _tiny_designs(tmp_path)
    result, wall = _timed(lambda: _run_cec(a, b, 100.0))
    assert result.equivalent and "SAT sweeping" in result.detail
    assert wall < 10.0
    _assert_reaped(started)


def test_run_cec_real_processes_reaped_after_deadline(real_race, monkeypatch, tmp_path) -> None:
    """Both processes sleep; the 0.3s wait budget runs out -> `ABCTimeoutError`, and both sleepers are killed
    and reaped.
    kills: the `finally` `kill()` removed (both sleepers would run ~30s -> the wall check). Not claimed: the
    `wait()` loop removed (the reader's `communicate()` reaps the child by itself)."""
    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import ABCTimeoutError, _run_cec

    _, started = real_race
    monkeypatch.setattr(m, "_real_wait", lambda t: 0.3)
    a, b = _tiny_designs(tmp_path)
    start = time.monotonic()
    with pytest.raises(ABCTimeoutError):
        _run_cec(a, b, 100.0)
    assert time.monotonic() - start < 10.0
    _assert_reaped(started)


def test_run_cec_real_processes_reaped_after_keyboard_interrupt(real_race, monkeypatch, tmp_path) -> None:
    """The main thread gets `KeyboardInterrupt` inside the wait (`queue.Queue.get`) while both processes are
    alive: it propagates, and both are killed and reaped first.
    kills: the whole cleanup moved out of `finally` (e.g. only on the normal path or only `except Exception`:
    `KeyboardInterrupt` propagates at once with both sleepers alive -> `_assert_reaped`); the `finally`
    `kill()` removed with the `wait()` loop kept (the wait blocks ~30s on the sleepers -> the wall check).
    Not claimed: the `wait()` loop removed (the reader's `communicate()` reaps the child by itself)."""
    import queue as queue_mod
    import types

    from netlist_agent import abc_bridge as m
    from netlist_agent.abc_bridge import _run_cec

    class _InterruptedQueue(queue_mod.Queue):
        def get(self, *args, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(m, "queue", types.SimpleNamespace(Queue=_InterruptedQueue, Empty=queue_mod.Empty))
    _, started = real_race
    a, b = _tiny_designs(tmp_path)
    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run_cec(a, b, 100.0)
    assert time.monotonic() - start < 10.0
    _assert_reaped(started)
