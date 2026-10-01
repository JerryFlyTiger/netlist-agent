"""Unit tests for the ABC BLIF writer (`netlist_agent.abc_bridge.write_blif`),
added in batch 6c to fix ABC `read_verilog`'s crash on >=128-bit buses (see
abc_bridge.py module docstring, finding 4). Each detail called out there gets
its own test here, isolated from the whole-corpus integration coverage in
tests/test_abc_bridge_real_files.py.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from netlist_agent.abc_bridge import (
    ABCBridgeError,
    _ONE_INPUT_BLIF_COVERS,
    _TWO_INPUT_BLIF_COVERS,
    _resolve_abc,
    verify_equivalence,
    write_blif,
)
from netlist_agent.ir import (
    Design,
    Direction,
    Gate,
    GateType,
    NetBit,
    POSITIONAL_PIN_ORDER,
    Port,
    Signal,
)
from netlist_agent.parser import parse_verilog
from netlist_agent.writer import write_verilog


def _write(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def _cec(path_a: str, path_b: str, timeout: float = 60.0) -> str:
    abc = _resolve_abc()
    result = subprocess.run([abc, "-c", f'cec "{path_a}" "{path_b}"'], capture_output=True, text=True, timeout=timeout)
    return result.stdout


# ----------------------------------------------------------------------
# Gate-type cover encoding: every primitive, checked against write_verilog's
# own reference semantics via ABC cec (not just by inspection).
# ----------------------------------------------------------------------


def test_write_blif_gate_covers_match_write_verilog_semantics(tmp_path) -> None:
    src = """
    module top(a, b, y_and, y_or, y_nand, y_nor, y_xor, y_xnor, y_not, y_buf);
      input a, b;
      output y_and, y_or, y_nand, y_nor, y_xor, y_xnor, y_not, y_buf;
      and g_and1(y_and, a, b);
      or g_or1(y_or, a, b);
      nand g_nand1(y_nand, a, b);
      nor g_nor1(y_nor, a, b);
      xor g_xor1(y_xor, a, b);
      xnor g_xnor1(y_xnor, a, b);
      not g_not1(y_not, a);
      buf g_buf1(y_buf, a);
    endmodule
    """
    path = _write(tmp_path, "gates.v", src)
    design = parse_verilog(path)
    v_path = str(tmp_path / "gates_ref.v")
    blif_path = str(tmp_path / "gates_out.blif")
    write_verilog(design, v_path)
    write_blif(design, blif_path)
    stdout = _cec(v_path, blif_path)
    assert "Networks are equivalent" in stdout, stdout


def test_write_blif_covers_exist_for_every_non_dff_gate_type() -> None:
    # Documents/guards the current IR's structural limit: every non-DFF
    # primitive takes at most 2 inputs (POSITIONAL_PIN_ORDER only ever
    # defines "I0"/"I1"), so a "gate with 3+ inputs" the spec asked to check
    # for cannot be represented at all today -- this assertion is what would
    # break the day that stops being true, forcing write_blif to grow an
    # n-input cover generator before such a gate could ever reach it.
    for gate_type, pin_order in POSITIONAL_PIN_ORDER.items():
        assert len(pin_order) <= 3, f"{gate_type} has more than 2 inputs -- write_blif's covers assume <= 2"
    non_dff = {gt for gt in GateType if gt != GateType.DFF}
    covered = set(_ONE_INPUT_BLIF_COVERS) | set(_TWO_INPUT_BLIF_COVERS)
    assert covered == non_dff


def test_write_blif_rejects_dff_instance(tmp_path) -> None:
    design = Design(module_name="top")
    design.signals["a"] = Signal(name="a", msb=None, lsb=None, direction=Direction.INPUT)
    design.signals["q"] = Signal(name="q", msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports = [Port(name="a", direction=Direction.INPUT), Port(name="q", direction=Direction.OUTPUT)]
    design.gates.append(
        Gate(inst_name="dff0", gate_type=GateType.DFF, pins={"D": NetBit("a", None), "Q": NetBit("q", None)})
    )
    with pytest.raises(ABCBridgeError, match="dff"):
        write_blif(design, str(tmp_path / "bad.blif"))


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------


def test_write_blif_const_zero_and_one_pins(tmp_path) -> None:
    src = """
    module top(a, y0, y1);
      input a;
      output y0, y1;
      and g_and2(y0, a, 1'b0);
      or g_or2(y1, a, 1'b1);
    endmodule
    """
    path = _write(tmp_path, "const.v", src)
    design = parse_verilog(path)
    v_path = str(tmp_path / "const_ref.v")
    blif_path = str(tmp_path / "const_out.blif")
    write_verilog(design, v_path)
    write_blif(design, blif_path)
    stdout = _cec(v_path, blif_path)
    assert "Networks are equivalent" in stdout, stdout


def test_write_blif_const_name_does_not_collide_with_existing_net(tmp_path) -> None:
    # A real net is already named "c0" (write_blif's default constant-0 net
    # name) -- the const allocator must pick a different name instead of
    # silently aliasing onto it.
    src = """
    module top(a, c0, y0, y1);
      input a, c0;
      output y0, y1;
      and g_and3(y0, a, 1'b0);
      or g_or3(y1, c0, a);
    endmodule
    """
    path = _write(tmp_path, "const_collide.v", src)
    design = parse_verilog(path)
    v_path = str(tmp_path / "const_collide_ref.v")
    blif_path = str(tmp_path / "const_collide_out.blif")
    write_verilog(design, v_path)
    write_blif(design, blif_path)
    stdout = _cec(v_path, blif_path)
    assert "Networks are equivalent" in stdout, stdout


# ----------------------------------------------------------------------
# Undriven nets
# ----------------------------------------------------------------------


def test_write_blif_undriven_net_reads_back_as_constant_zero(tmp_path) -> None:
    # `w` is read by two gates but driven by neither a PI nor any gate's O --
    # both write_verilog (via ABC's read_verilog auto-tie) and write_blif
    # (via this module's own explicit undriven-net tie, see docstring) must
    # treat it identically as constant 0.
    src = """
    module top(a, y, z);
      input a;
      output y, z;
      wire w;
      and g_and4(y, a, w);
      buf g_buf2(z, w);
    endmodule
    """
    path = _write(tmp_path, "undriven.v", src)
    design = parse_verilog(path)
    v_path = str(tmp_path / "undriven_ref.v")
    blif_path = str(tmp_path / "undriven_out.blif")
    write_verilog(design, v_path)
    write_blif(design, blif_path)
    stdout = _cec(v_path, blif_path)
    assert "Networks are equivalent" in stdout, stdout

    # White-box: ABC's read_blif would auto-tie `w` to 0 on its own, so the
    # cec checks above pass even with write_blif's explicit tie deleted.
    # Pin the tie itself: a `.names w` line with no cover row (= constant 0).
    with open(blif_path) as f:
        blif_lines = f.read().splitlines()
    assert ".names w" in blif_lines, blif_lines
    assert blif_lines[blif_lines.index(".names w") + 1].startswith("."), blif_lines

    # And directly confirm the "constant 0" part, not just "write_blif
    # matches write_verilog's own (also-implicit) behavior": z should always
    # read 0 regardless of a.
    const_zero_path = str(tmp_path / "const_zero_ref.v")
    with open(const_zero_path, "w") as f:
        f.write("module top(a, y, z);\n  input a;\n  output y, z;\n  and g0(y, a, 1'b0);\n  buf g1(z, 1'b0);\nendmodule\n")
    stdout2 = _cec(const_zero_path, blif_path)
    assert "Networks are equivalent" in stdout2, stdout2


def test_write_blif_undriven_output_reads_back_as_constant_zero(tmp_path) -> None:
    # A primary output never referenced by any gate at all (not even as
    # another gate's input) -- must also come back constant 0.
    design = Design(module_name="top")
    design.signals["a"] = Signal(name="a", msb=None, lsb=None, direction=Direction.INPUT)
    design.signals["y"] = Signal(name="y", msb=None, lsb=None, direction=Direction.OUTPUT)
    design.signals["z"] = Signal(name="z", msb=None, lsb=None, direction=Direction.OUTPUT)
    design.ports = [
        Port(name="a", direction=Direction.INPUT),
        Port(name="y", direction=Direction.OUTPUT),
        Port(name="z", direction=Direction.OUTPUT),
    ]
    design.add_gate(Gate(inst_name="g0", gate_type=GateType.BUF, pins={"O": NetBit("y", None), "I0": NetBit("a", None)}))
    blif_path = str(tmp_path / "undriven_po.blif")
    write_blif(design, blif_path)

    ref_path = str(tmp_path / "undriven_po_ref.v")
    with open(ref_path, "w") as f:
        f.write("module top(a, y, z);\n  input a;\n  output y, z;\n  buf g0(y, a);\n  buf g1(z, 1'b0);\nendmodule\n")
    stdout = _cec(ref_path, blif_path)
    assert "Networks are equivalent" in stdout, stdout


# ----------------------------------------------------------------------
# PI-to-PO / PO-to-PO passthrough (always via an explicit BUF gate in this
# IR -- a Signal has exactly one Direction, so a net cannot literally be
# both an input port and an output port at once; tested anyway, defensively,
# per spec item 32).
# ----------------------------------------------------------------------


def test_write_blif_pi_directly_buffered_to_po(tmp_path) -> None:
    src = """
    module top(a, y);
      input a;
      output y;
      buf g_buf3(y, a);
    endmodule
    """
    path = _write(tmp_path, "pi_po.v", src)
    design = parse_verilog(path)
    v_path = str(tmp_path / "pi_po_ref.v")
    blif_path = str(tmp_path / "pi_po_out.blif")
    write_verilog(design, v_path)
    write_blif(design, blif_path)
    stdout = _cec(v_path, blif_path)
    assert "Networks are equivalent" in stdout, stdout


def test_write_blif_po_directly_buffered_from_another_po(tmp_path) -> None:
    src = """
    module top(a, y, z);
      input a;
      output y, z;
      buf g_buf4(y, a);
      buf g_buf5(z, y);
    endmodule
    """
    path = _write(tmp_path, "po_po.v", src)
    design = parse_verilog(path)
    v_path = str(tmp_path / "po_po_ref.v")
    blif_path = str(tmp_path / "po_po_out.blif")
    write_verilog(design, v_path)
    write_blif(design, blif_path)
    stdout = _cec(v_path, blif_path)
    assert "Networks are equivalent" in stdout, stdout


# ----------------------------------------------------------------------
# Wide buses (>= 128 bits): the whole point of this batch.
# ----------------------------------------------------------------------


def _wide_bus_source(width: int) -> str:
    lines = [
        f"module top(a, b, y);",
        f"  input [{width - 1}:0] a;",
        f"  input b;",
        f"  output [{width - 1}:0] y;",
    ]
    for i in range(width):
        lines.append(f"  and g{i}(y[{i}], a[{i}], b);")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def test_wide_bus_verify_equivalence_no_longer_crashes(tmp_path) -> None:
    # width 130 (msb index 129) is past the nMsb<128 assertion boundary that
    # motivated this whole batch (abc_bridge.py module docstring, finding 4).
    src = _wide_bus_source(130)
    path = _write(tmp_path, "wide.v", src)
    design_a = parse_verilog(path)
    design_b = parse_verilog(path)
    result = verify_equivalence(design_a, design_b)
    assert result.equivalent, result.detail


def test_wide_bus_verify_equivalence_detects_a_real_difference(tmp_path) -> None:
    src = _wide_bus_source(130)
    path = _write(tmp_path, "wide.v", src)
    design_a = parse_verilog(path)
    design_b = parse_verilog(path)
    # Flip one bit's gate type so the two designs are genuinely NOT
    # equivalent -- confirms write_blif's per-gate covers are actually wired
    # into the miter, not just "doesn't crash".
    for g in design_b.gates:
        if g.pins.get("O") == NetBit("y", 64):
            g.gate_type = GateType.NAND
            break
    else:
        raise AssertionError("expected to find the y[64] gate")
    result = verify_equivalence(design_a, design_b)
    assert not result.equivalent


def test_wide_bus_optimize_depth_runs_without_crashing(tmp_path) -> None:
    from netlist_agent.abc_synth import optimize_depth

    src = _wide_bus_source(130)
    path = _write(tmp_path, "wide_opt.v", src)
    design = parse_verilog(path)
    result = optimize_depth(design)
    # `note` is filled on every branch, including "ABC failed"; `failure` is
    # None only when the BLIF -> ABC -> read-back pipeline actually ran.
    assert result.failure is None, result.failure


# ----------------------------------------------------------------------
# Long lines (many PIs on one .inputs line) -- test051-scale token count.
# ----------------------------------------------------------------------


def test_write_blif_handles_a_very_long_inputs_line(tmp_path) -> None:
    width = 2000
    lines = [f"module top(a, y);", f"  input [{width - 1}:0] a;", f"  output [{width - 1}:0] y;"]
    for i in range(width):
        lines.append(f"  buf g{i}(y[{i}], a[{i}]);")
    lines.append("endmodule")
    path = _write(tmp_path, "longline.v", "\n".join(lines) + "\n")
    design = parse_verilog(path)
    blif_path = str(tmp_path / "longline.blif")
    write_blif(design, blif_path)
    with open(blif_path) as f:
        text = f.read()
    # Every bit, in full -- not just the last one, so a partial expansion
    # (some bits dropped) is caught too. Order is irrelevant: cec and the
    # read-back resolver both match by name.
    inputs = text.splitlines()[1].split()
    assert inputs[0] == ".inputs"
    assert sorted(inputs[1:]) == sorted(f"a[{i}]" for i in range(width))

    abc = _resolve_abc()
    out_path = str(tmp_path / "longline_roundtrip.blif")
    result = subprocess.run(
        [abc, "-c", f'read_blif "{blif_path}"; strash; write_blif "{out_path}"'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(out_path)
