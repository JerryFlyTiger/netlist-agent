"""Synthetic/unit-level tests for the ABC bridge (netlist_agent/abc_bridge.py).

These exercise the module against small hand-built designs so each check is
cheap and the expected answer is known by inspection -- see
tests/test_abc_bridge_real_files.py for integration coverage against the real
40-testcase corpus.
"""

from __future__ import annotations

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
    q_port = next(p for p in free_pi.ports if p.name == "n_q")
    assert q_port.direction == Direction.INPUT
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
    """Edge case (a) of the promotion-order rule: a DFF's Q net literally IS
    an already-declared primary output. free_pi promotion must still win
    (INPUT), overwriting the pre-existing OUTPUT port entry rather than
    leaving two conflicting Port entries for the same name."""
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
    assert ports_named_q_out[0].direction == Direction.INPUT
    assert free_pi.signals["q_out"].direction == Direction.INPUT


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
    # Both DFFs dropped; the only gates are the two canonical D-pin taps.
    assert all(g.gate_type == GateType.BUF for g in free_pi.gates)
    taps = {g.pins["O"]: g.pins["I0"] for g in free_pi.gates}
    assert taps == {
        NetBit("__dff_D__g0", None): NetBit("q2", None),
        NetBit("__dff_D__g1", None): NetBit("q1", None),
    }
    for name in ("q1", "q2"):
        matching_ports = [p for p in free_pi.ports if p.name == name]
        assert len(matching_ports) == 1
        assert matching_ports[0].direction == Direction.INPUT
        assert free_pi.signals[name].direction == Direction.INPUT


def test_extract_combinational_view_dff_q_shares_bus_with_combinational_bit(tmp_path) -> None:
    """Regression test for a real bug found in test39.v: a DFF's Q pin can be
    a bit-select of a wider bus whose OTHER bits are independently driven by
    ordinary combinational gates. Whole-Signal Direction promotion can't turn
    the whole bus into a primary input (bit 0 would become simultaneously a
    PI bit and gate-driven) -- the fix splits just the DFF's own bit off into
    its own fresh single-bit input, leaving the rest of the bus untouched."""
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

    # The bus as a whole is never promoted: bit 0 is still ordinarily
    # gate-driven, so Signal-granularity promotion of "shared" would be invalid.
    assert free_pi.signals["shared"].direction == Direction.INTERNAL
    assert not any(p.name == "shared" for p in free_pi.ports)

    # bit 0 (g0's output) is completely untouched.
    g0 = next(g for g in free_pi.gates if g.inst_name == "g0")
    assert g0.pins["O"] == NetBit("shared", 0)
    g2 = next(g for g in free_pi.gates if g.inst_name == "g2")
    assert g2.pins["I0"] == NetBit("shared", 0)

    # bit 1 (the DFF's Q) was split into its own fresh input; nothing in the
    # extracted design references NetBit("shared", 1) anymore.
    assert not any(v == NetBit("shared", 1) for g in free_pi.gates for v in g.pins.values())
    g3 = next(g for g in free_pi.gates if g.inst_name == "g3")
    split_net = g3.pins["I0"]
    assert isinstance(split_net, NetBit) and split_net.bit is None and split_net != NetBit("shared", 1)
    split_port = next(p for p in free_pi.ports if p.name == split_net.name)
    assert split_port.direction == Direction.INPUT
    assert free_pi.signals[split_net.name].direction == Direction.INPUT

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
