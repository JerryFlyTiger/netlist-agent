"""Regression tests for the DFF-Q boundary the verification side extracts
(batch 6d redesign, B').

`extract_combinational_view("free_pi")` gives every DFF one primary input
`__dff_Q__<inst>` (keyed by DFF *instance*, symmetric with the D-side
`__dff_D__<inst>` taps) and re-drives the ORIGINAL Q net through a BUF from
it. No bus is split or promoted and no consumer is rewired, so every original
net keeps its name and meaning in the view.

Fixtures in tests/fixtures/dff_q_split/ are synthetic netlists written to
reproduce the minimal case in experiments/final_release_100_gap_2026-09-28/test062_min/;
no line is taken from the contest netlist.
"""

from __future__ import annotations

import os

import pytest

import netlist_agent.abc_bridge as abc_bridge
import netlist_agent.abc_synth as abc_synth
from netlist_agent.abc_bridge import (
    ABCBridgeError,
    DEFAULT_ABC_TIMEOUT,
    DEFAULT_VERIFY_TIMEOUT,
    EquivResult,
    _is_declared_bit,
    are_equivalent,
    check_symmetry,
    extract_combinational_view,
    verify_equivalence,
)
from netlist_agent.abc_synth import (
    optimize_cone_depth,
    optimize_cone_gate_count,
    optimize_depth,
    optimize_gate_count,
)
from netlist_agent.ir import Const, Direction, GateType, NetBit, Signal
from netlist_agent.netref import netbit_token
from netlist_agent.parser import parse_verilog
from netlist_agent import signal_pair_search
from netlist_agent.signal_pair_search import find_pair_for_op

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "dff_q_split")
BUS_A = os.path.join(FIXTURES, "bus_comb_sibling.v")
BUS_B = os.path.join(FIXTURES, "bus_sibling_renamed_scalar.v")
BUS_C = os.path.join(FIXTURES, "bus_sibling_cone_target_y.v")


def _parse(tmp_path, src: str, name: str = "x.v"):
    path = tmp_path / name
    path.write_text(src)
    return parse_verilog(str(path))


def _two_dff_bus(r1_q: str, r2_q: str, order: str = "r1r2", c1: str | None = None, c2: str | None = None) -> str:
    """3-bit bus w: w[0] comb-driven, two DFF Qs. The output is asymmetric in
    the two consumers' nets. `c1`/`c2` are the nets the consumers g1/g2 read
    (default: r1_q/r2_q, i.e. a pure relabelling of the Q nets)."""
    c1 = c1 or r1_q
    c2 = c2 or r2_q
    r1 = f"  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q({r1_q}));\n"
    r2 = f"  dff r2(.RN(1'b1), .SN(1'b1), .CK(clk), .D(b), .Q({r2_q}));\n"
    dffs = r1 + r2 if order == "r1r2" else r2 + r1
    return (
        "module top(a, b, c, clk, y);\n"
        "  input a, b, c, clk;\n"
        "  output y;\n"
        "  wire [2:0] w;\n"
        "  wire t1, t2, t3;\n"
        "  and g0(w[0], a, b);\n"
        + dffs
        + f"  and g1(t1, {c1}, c);\n"
        f"  not g2(t2, {c2});\n"
        "  or g3(t3, t1, t2);\n"
        "  xor g4(y, t3, w[0]);\n"
        "endmodule\n"
    )


def _pi_names(comb) -> set[str]:
    return {p.name for p in comb.ports if comb.signals[p.name].direction == Direction.INPUT}


def _and_chain(first: str, stages: int = 7) -> str:
    """`stages`-deep AND chain from `first` to y, using inputs b..h."""
    ins = "bcdefgh"[:stages]
    lines = []
    prev = first
    for i, x in enumerate(ins):
        out = "y" if i == stages - 1 else f"t{i + 1}"
        lines.append(f"  and h{i}({out}, {prev}, {x});")
        prev = out
    return "\n".join(lines) + "\n"


F1_SRC = (
    "module top(a, b, clk, y);\n"
    "  input a, b, clk;\n"
    "  output y;\n"
    "  wire [1:0] w;\n"
    "  wire q2;\n"
    "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[1]));\n"
    "  dff r2(.RN(1'b1), .SN(1'b1), .CK(clk), .D({d2}), .Q(q2));\n"
    "  and g1(y, q2, b);\n"
    "endmodule\n"
)

CHAIN_OPT3 = (
    "module top(a, b, c, d, e, f, g, h, clk, y);\n"
    "  input a, b, c, d, e, f, g, h, clk;\n"
    "  output y;\n"
    "  wire [1:0] w;\n"
    "  wire q2, t1, t2, t3, t4, t5, t6;\n"
    "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[1]));\n"
    "  dff r2(.RN(1'b1), .SN(1'b1), .CK(clk), .D(w[1]), .Q(q2));\n"
    "  and g1(t1, q2, b);\n"
    "  and g2(t2, t1, c);\n"
    "  and g3(t3, t2, d);\n"
    "  and g4(t4, t3, e);\n"
    "  and g5(t5, t4, f);\n"
    "  and g6(t6, t5, g);\n"
    "  and g7(y, t6, h);\n"
    "endmodule\n"
)


# ----------------------------------------------------------------------
# T1: verify_equivalence on the test062 minimal reproduction
# ----------------------------------------------------------------------


def test_bus_fixtures_extract_same_pi_set() -> None:
    a = extract_combinational_view(parse_verilog(BUS_A), "free_pi")
    b = extract_combinational_view(parse_verilog(BUS_B), "free_pi")
    assert _pi_names(a) == _pi_names(b)


def test_verify_equivalence_bus_fixtures_equivalent() -> None:
    eq = verify_equivalence(parse_verilog(BUS_A), parse_verilog(BUS_B))
    assert eq.equivalent, eq.detail


def test_verify_equivalence_bus_fixtures_negative_control(tmp_path) -> None:
    src = open(BUS_B).read().replace("or g3(y, t, c);", "and g3(y, t, c);")
    assert src != open(BUS_B).read()
    eq = verify_equivalence(parse_verilog(BUS_A), _parse(tmp_path, src, "b_broken.v"))
    assert not eq.equivalent


# ----------------------------------------------------------------------
# T2: the four optimizers on the fixtures
# ----------------------------------------------------------------------


def _assert_clean_result(res, original) -> None:
    assert res.failure is None, res.failure
    orig_dffs = {g.inst_name: g.pins["Q"] for g in original.gates if g.gate_type == GateType.DFF}
    new_dffs = {g.inst_name: g.pins["Q"] for g in res.design.gates if g.gate_type == GateType.DFF}
    assert new_dffs == orig_dffs
    for g in res.design.gates:
        for v in g.pins.values():
            if isinstance(v, NetBit):
                assert v.name in res.design.signals, (g.inst_name, v)
    assert not any(n.startswith("__dff_") for n in res.design.signals)


@pytest.mark.parametrize("fn", [optimize_depth, optimize_gate_count])
def test_whole_design_optimizers_bus_fixture(fn) -> None:
    d = parse_verilog(BUS_A)
    _assert_clean_result(fn(d), d)


@pytest.mark.parametrize("fn", [optimize_cone_depth, optimize_cone_gate_count])
def test_cone_optimizers_bus_fixture(fn) -> None:
    d = parse_verilog(BUS_C)
    _assert_clean_result(fn(d, NetBit("y", None)), d)


# ----------------------------------------------------------------------
# T3: extraction structure
# ----------------------------------------------------------------------


def test_extraction_structure(tmp_path) -> None:
    design = parse_verilog(BUS_A)
    src: dict[str, NetBit] = {}
    comb = extract_combinational_view(design, "free_pi", src)
    pis = _pi_names(comb)
    assert pis == {"a", "b", "c", "clk", "__dff_Q__r1"}
    assert pis == _pi_names(extract_combinational_view(parse_verilog(BUS_A), "free_pi"))
    assert not any("[" in n or "]" in n for n in pis)
    assert comb.signals["w"].direction == Direction.INTERNAL
    drv = comb.net_driver[NetBit("w", 1)]
    assert drv.gate_type == GateType.BUF
    assert drv.pins["I0"] == NetBit("__dff_Q__r1", None)
    g2 = next(g for g in comb.gates if g.inst_name == "g2")
    assert g2.pins["I1"] == NetBit("w", 1) or g2.pins["I0"] == NetBit("w", 1)
    assert src == {"__dff_Q__r1": NetBit("w", 1)}


# ----------------------------------------------------------------------
# T4: F1 -- a D pin reading a Q net must keep the Q value
# ----------------------------------------------------------------------


def test_f1_d_pin_reading_q_net_not_equivalent_to_constant(tmp_path) -> None:
    a = _parse(tmp_path, F1_SRC.format(d2="w[1]"), "f1a.v")
    b = _parse(tmp_path, F1_SRC.format(d2="1'b0"), "f1b.v")
    assert not verify_equivalence(a, b).equivalent


def test_f1_optimize_depth_keeps_q_value_on_d_pin(tmp_path) -> None:
    design = _parse(tmp_path, CHAIN_OPT3)
    res = optimize_depth(design)
    assert res.failure is None, res.failure
    r2 = next(g for g in res.design.gates if g.inst_name == "r2")
    d = r2.pins["D"]
    assert isinstance(d, NetBit)
    # Follow BUFs back: must reach w[1], never a constant.
    seen = 0
    while d != NetBit("w", 1):
        drv = res.design.net_driver.get(d)
        assert drv is not None and drv.gate_type == GateType.BUF and seen < 20
        assert drv.pins["I0"] != Const.ZERO and drv.pins["I0"] != Const.ONE
        d = drv.pins["I0"]
        seen += 1
    for g in res.design.gates:
        if g.gate_type == GateType.DFF:
            assert g.pins["D"] not in (Const.ZERO, Const.ONE)


# ----------------------------------------------------------------------
# T5: F3 -- query entry points take original net names
# ----------------------------------------------------------------------

F3_SYM = (
    "module top(a, clk, y);\n"
    "  input a, clk;\n"
    "  output y;\n"
    "  wire [2:0] w;\n"
    "  wire nw2;\n"
    "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[1]));\n"
    "  dff r2(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[2]));\n"
    "  not g1(nw2, w[2]);\n"
    "  {y_gate}\n"
    "endmodule\n"
)


def test_f3_query_entry_points_on_q_bits(tmp_path) -> None:
    w1, w2 = NetBit("w", 1), NetBit("w", 2)
    asym = _parse(tmp_path, F3_SYM.format(y_gate="and g2(y, w[1], nw2);"), "asym.v")
    assert are_equivalent(asym, w1, w2) is False
    assert check_symmetry(asym, NetBit("y", None), w1, w2) is False
    sym = _parse(tmp_path, F3_SYM.format(y_gate="and g2(y, w[1], w[2]);"), "sym.v")
    assert check_symmetry(sym, NetBit("y", None), w1, w2) is True


def test_f3_pair_search_on_q_bit(tmp_path) -> None:
    src = (
        "module top(a, clk, y, z);\n"
        "  input a, clk;\n"
        "  output y, z;\n"
        "  wire [1:0] w;\n"
        "  wire t;\n"
        "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[1]));\n"
        "  buf g1(t, w[1]);\n"
        "  not g2(y, t);\n"
        "  and g3(z, t, a);\n"
        "endmodule\n"
    )
    design = _parse(tmp_path, src)
    res = find_pair_for_op(design, NetBit("w", 1), "AND")
    assert res.found, res.explanation
    assert res.pair is not None and not any("__dff_" in n for n in res.pair)


def test_internal_net_symmetry_is_reader_swap(tmp_path) -> None:
    src = (
        "module top(a, b, c, y);\n"
        "  input a, b, c;\n"
        "  output y;\n"
        "  wire a1, b1, nb1;\n"
        "  and g0(a1, a, c);\n"
        "  and g1(b1, b, c);\n"
        "  not g2(nb1, b1);\n"
        "  and g3(y, a1, nb1);\n"
        "endmodule\n"
    )
    design = _parse(tmp_path, src)
    assert check_symmetry(design, NetBit("y", None), NetBit("a1", None), NetBit("b1", None)) is False
    # primary inputs, same asymmetry
    assert check_symmetry(design, NetBit("y", None), NetBit("a", None), NetBit("b", None)) is False


def test_internal_net_symmetry_with_unread_duplicate_gate(tmp_path) -> None:
    # b1 duplicates a1 functionally but has no reader; swapping y's reader
    # from a1 to b1 leaves y unchanged, so the pair is symmetric.
    src = (
        "module top(a, c, d, y);\n"
        "  input a, c, d;\n"
        "  output y;\n"
        "  wire a1, b1;\n"
        "  and g0(a1, a, c);\n"
        "  and g1(b1, a, c);\n"
        "  or g2(y, a1, d);\n"
        "endmodule\n"
    )
    design = _parse(tmp_path, src)
    assert check_symmetry(design, NetBit("y", None), NetBit("a1", None), NetBit("b1", None)) is True


# ----------------------------------------------------------------------
# T6: F4 -- Q-pin identity, and DFF order
# ----------------------------------------------------------------------


def test_f4_true_q_pin_swap_is_not_equivalent(tmp_path) -> None:
    x = _parse(tmp_path, _two_dff_bus("w[1]", "w[2]"), "x.v")
    # r1/r2 exchange their .Q() nets; consumers g1/g2 keep reading w[1]/w[2].
    y = _parse(tmp_path, _two_dff_bus("w[2]", "w[1]", c1="w[1]", c2="w[2]"), "y.v")
    assert not verify_equivalence(x, y).equivalent


def test_f4_label_swap_is_equivalent(tmp_path) -> None:
    x = _parse(tmp_path, _two_dff_bus("w[1]", "w[2]"), "x.v")
    y = _parse(tmp_path, _two_dff_bus("w[2]", "w[1]"), "y.v")
    eq = verify_equivalence(x, y)
    assert eq.equivalent, eq.detail


def test_dff_order_swap_is_equivalent(tmp_path) -> None:
    x = _parse(tmp_path, _two_dff_bus("w[1]", "w[2]", "r1r2"), "x.v")
    y = _parse(tmp_path, _two_dff_bus("w[1]", "w[2]", "r2r1"), "y.v")
    assert [g.inst_name for g in x.gates if g.gate_type == GateType.DFF] != [
        g.inst_name for g in y.gates if g.gate_type == GateType.DFF
    ]
    eq = verify_equivalence(x, y)
    assert eq.equivalent, eq.detail


# ----------------------------------------------------------------------
# T7: F5 -- a PI token that maps to nothing fails loudly
# ----------------------------------------------------------------------


def test_f5_unmapped_q_token_fails_loudly(monkeypatch) -> None:
    real = abc_synth.extract_combinational_view

    def stripped(design, mode, pqs=None):
        comb = real(design, mode, pqs)
        if pqs is not None:
            for k in [k for k in pqs if k.startswith("__dff_Q__")]:
                del pqs[k]
        return comb

    monkeypatch.setattr(abc_synth, "extract_combinational_view", stripped)
    c = parse_verilog(BUS_C)
    r = optimize_cone_depth(c, NetBit("y", None))
    assert r.failure is not None and "maps to no primary input" in r.failure
    assert r.design is c and r.changed is False
    a = parse_verilog(BUS_A)
    r = optimize_depth(a)
    assert r.failure is not None and "maps to no primary input" in r.failure
    assert r.design is a and r.changed is False


# ----------------------------------------------------------------------
# T8: F6 -- floating PO bit next to a Q PO bit
# ----------------------------------------------------------------------


def test_f6_floating_po_bit_stays_floating(tmp_path) -> None:
    src = (
        "module top(a, b, c, d, e, f, g, h, clk, q, y);\n"
        "  input a, b, c, d, e, f, g, h, clk;\n"
        "  output [1:0] q;\n"
        "  output y;\n"
        "  wire t1, t2, t3, t4, t5, t6;\n"
        "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(q[1]));\n"
        "  and g1(t1, q[1], b);\n"
        "  and g2(t2, t1, c);\n"
        "  and g3(t3, t2, d);\n"
        "  and g4(t4, t3, e);\n"
        "  and g5(t5, t4, f);\n"
        "  and g6(t6, t5, g);\n"
        "  and g7(y, t6, h);\n"
        "endmodule\n"
    )
    design = _parse(tmp_path, src)
    res = optimize_depth(design)
    assert res.failure is None, res.failure
    assert len(res.design.gates) <= 10
    assert NetBit("q", 0) not in res.design.net_driver
    r1 = next(g for g in res.design.gates if g.inst_name == "r1")
    assert r1.pins["Q"] == NetBit("q", 1)


# ----------------------------------------------------------------------
# T11: malformed Q wiring raises
# ----------------------------------------------------------------------

_MALFORMED_HEAD = "module top(a, b, clk, y);\n  input a, b, clk;\n  output y;\n  wire [1:0] w;\n  wire t;\n"


def test_malformed_two_dffs_same_q(tmp_path) -> None:
    src = (
        _MALFORMED_HEAD
        + "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[1]));\n"
        + "  dff r2(.RN(1'b1), .SN(1'b1), .CK(clk), .D(b), .Q(w[1]));\n"
        + "  not g1(y, w[1]);\nendmodule\n"
    )
    with pytest.raises(ABCBridgeError, match="both drive"):
        extract_combinational_view(_parse(tmp_path, src), "free_pi")


def test_malformed_q_also_gate_driven(tmp_path) -> None:
    src = (
        _MALFORMED_HEAD
        + "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[1]));\n"
        + "  and g0(w[1], a, b);\n  not g1(y, w[1]);\nendmodule\n"
    )
    with pytest.raises(ABCBridgeError, match="also driven by a combinational gate"):
        extract_combinational_view(_parse(tmp_path, src), "free_pi")


def test_malformed_q_bit_outside_declared_range(tmp_path) -> None:
    src = (
        _MALFORMED_HEAD
        + "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[5]));\n"
        + "  not g1(y, a);\nendmodule\n"
    )
    with pytest.raises(ABCBridgeError, match="not a declared bit"):
        extract_combinational_view(_parse(tmp_path, src), "free_pi")


def test_malformed_q_on_input_port(tmp_path) -> None:
    src = (
        _MALFORMED_HEAD
        + "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(b), .Q(a));\n"
        + "  not g1(y, a);\nendmodule\n"
    )
    with pytest.raises(ABCBridgeError, match="primary input"):
        extract_combinational_view(_parse(tmp_path, src), "free_pi")


def test_malformed_pi_name_collision(tmp_path) -> None:
    src = (
        _MALFORMED_HEAD.replace("wire t;", "wire t;\n  wire __dff_Q__r1;")
        + "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[1]));\n"
        + "  not g1(y, w[1]);\nendmodule\n"
    )
    with pytest.raises(ABCBridgeError, match="collides"):
        extract_combinational_view(_parse(tmp_path, src), "free_pi")


# ----------------------------------------------------------------------
# T13: F2 -- no quadratic Signal.bits() use, and _is_declared_bit agrees
# ----------------------------------------------------------------------


def test_extraction_does_not_call_signal_bits(tmp_path, monkeypatch) -> None:
    n = 256
    lines = [
        f"module top(a, clk, y);\n  input a, clk;\n  output y;\n  wire [{n}:0] w;\n",
        f"  and gc(w[{n}], a, a);\n",
    ]
    lines += [f"  dff r{i}(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[{i}]));\n" for i in range(n)]
    lines.append(f"  not gy(y, w[{n}]);\nendmodule\n")
    design = _parse(tmp_path, "".join(lines))
    calls = []
    real = Signal.bits

    def counting(self):
        calls.append(self.name)
        return real(self)

    monkeypatch.setattr(Signal, "bits", counting)
    extract_combinational_view(design, "free_pi", {})
    assert calls == []


@pytest.mark.parametrize(
    "msb, lsb, nb",
    [
        (None, None, NetBit("s", None)),
        (None, None, NetBit("s", 0)),
        (7, 0, NetBit("s", 0)),
        (7, 0, NetBit("s", 7)),
        (7, 0, NetBit("s", 8)),
        (7, 0, NetBit("s", -1)),
        (7, 0, NetBit("s", None)),
        (0, 7, NetBit("s", 0)),
        (0, 7, NetBit("s", 7)),
        (0, 7, NetBit("s", 8)),
        (5, 3, NetBit("s", 2)),
    ],
)
def test_is_declared_bit_agrees_with_bits(msb, lsb, nb) -> None:
    sig = Signal("s", msb, lsb, Direction.INTERNAL)
    assert _is_declared_bit(sig, nb) == (nb in sig.bits())


# ----------------------------------------------------------------------
# T14: F8 -- timeouts
# ----------------------------------------------------------------------


def test_verify_equivalence_default_timeout_is_verify_timeout(monkeypatch) -> None:
    seen: list[float] = []

    def fake_cec(a, b, timeout):
        seen.append(timeout)
        return EquivResult(True, "ok")

    monkeypatch.setattr(abc_bridge, "_run_cec", fake_cec)
    d = parse_verilog(BUS_A)
    assert verify_equivalence(d, d).equivalent
    assert seen == [DEFAULT_VERIFY_TIMEOUT]


@pytest.mark.parametrize(
    "fn, path, target",
    [
        (optimize_depth, BUS_A, None),
        (optimize_gate_count, BUS_A, None),
        (optimize_cone_depth, BUS_C, NetBit("y", None)),
        (optimize_cone_gate_count, BUS_C, NetBit("y", None)),
    ],
)
def test_optimizers_split_synthesis_and_verify_timeouts(monkeypatch, fn, path, target) -> None:
    verify_seen: list[float] = []
    synth_seen: list[float] = []
    real_v = abc_synth.verify_equivalence
    real_s = abc_synth._run_abc_synthesis

    def spy_v(a, b, timeout=None, **kw):
        verify_seen.append(timeout)
        return real_v(a, b, timeout=timeout, **kw)

    def spy_s(view, basis, timeout, *args, **kw):
        synth_seen.append(timeout)
        return real_s(view, basis, timeout, *args, **kw)

    monkeypatch.setattr(abc_synth, "verify_equivalence", spy_v)
    monkeypatch.setattr(abc_synth, "_run_abc_synthesis", spy_s)
    args = (parse_verilog(path),) if target is None else (parse_verilog(path), target)
    fn(*args, timeout=7, verify_timeout=777.0)
    assert synth_seen and set(synth_seen) == {7}
    assert verify_seen and set(verify_seen) == {777.0}


def test_default_timeout_constants() -> None:
    assert DEFAULT_VERIFY_TIMEOUT == 600.0
    assert DEFAULT_VERIFY_TIMEOUT > DEFAULT_ABC_TIMEOUT


@pytest.mark.parametrize(
    "fn, path, target",
    [
        (optimize_depth, BUS_A, None),
        (optimize_gate_count, BUS_A, None),
        (optimize_cone_depth, BUS_C, NetBit("y", None)),
        (optimize_cone_gate_count, BUS_C, NetBit("y", None)),
    ],
)
def test_optimizers_default_timeouts(monkeypatch, fn, path, target) -> None:
    verify_seen: list[float] = []
    synth_seen: list[float] = []
    real_v = abc_synth.verify_equivalence
    real_s = abc_synth._run_abc_synthesis

    def spy_v(a, b, timeout=None, **kw):
        verify_seen.append(timeout)
        return real_v(a, b, timeout=timeout, **kw)

    def spy_s(view, basis, timeout, *args, **kw):
        synth_seen.append(timeout)
        return real_s(view, basis, timeout, *args, **kw)

    monkeypatch.setattr(abc_synth, "verify_equivalence", spy_v)
    monkeypatch.setattr(abc_synth, "_run_abc_synthesis", spy_s)
    args = (parse_verilog(path),) if target is None else (parse_verilog(path), target)
    fn(*args)
    assert synth_seen and set(synth_seen) == {DEFAULT_ABC_TIMEOUT}
    assert verify_seen and set(verify_seen) == {DEFAULT_VERIFY_TIMEOUT}


# ----------------------------------------------------------------------
# S2: consumers and user-visible text
# ----------------------------------------------------------------------

F9_SRC = (
    "module top(a, b, clk, y, z);\n"
    "  input a, b, clk;\n"
    "  output y, z;\n"
    "  wire [1:0] w;\n"
    "  wire t;\n"
    "  dff r0(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(w[0]));\n"
    "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(b), .Q(w[1]));\n"
    "  buf g1(t, w[1]);\n"
    "  not g2(y, w[0]);\n"
    "  not g3(z, t);\n"
    "endmodule\n"
)


def test_t9_pair_search_all_q_bus_names_original_bits(tmp_path) -> None:
    design = _parse(tmp_path, F9_SRC)
    res = find_pair_for_op(design, NetBit("t", None), "AND")
    assert res.found, res.explanation
    assert res.pair == ("w[1]", "w[1]")


def test_t10b_pair_search_does_not_count_synthetic_dff_pis(tmp_path) -> None:
    design = _parse(tmp_path, F9_SRC)
    res = find_pair_for_op(design, NetBit("t", None), "AND")
    # `find_pair_for_op` only exposes counts, so rebuild the candidate set with
    # the same rule: every signature-tracked net-bit of the "free_pi" view
    # except the target and the synthetic boundary nets.
    comb = extract_combinational_view(design, "free_pi")
    sig = signal_pair_search._simulate_signatures(comb, 64, 0)
    names = {
        netbit_token(nb)
        for nb in sig
        if nb != NetBit("t", None) and not nb.name.startswith(signal_pair_search._SYNTHETIC_PREFIXES)
    }
    # `t` is the target (excluded); `clk` is a PI and so a candidate.
    assert names == {"a", "b", "clk", "w[0]", "w[1]", "y", "z"}
    assert not any(n.startswith("__dff_Q__") for n in names)
    # the two synthetic __dff_Q__ PIs are in the view but must not be counted
    assert any(nb.name.startswith("__dff_Q__") for nb in sig)
    assert res.stats["nets_scanned"] == len(names) == 7, res.stats


def test_t10_property_counterexample_uses_design_net_names(tmp_path) -> None:
    from netlist_agent.property_check import check_asserted_only_when

    src = (
        "module top(a, c, clk, y);\n"
        "  input a, c, clk;\n"
        "  output y;\n"
        "  wire [1:0] w;\n"
        "  and g0(w[0], a, c);\n"
        "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(w[0]), .Q(w[1]));\n"
        "  and g1(y, w[1], a);\n"
        "endmodule\n"
    )
    design = _parse(tmp_path, src)
    res = check_asserted_only_when(design, "y", "a is 0")
    assert res.holds is False
    assert res.assignment is not None
    assert set(res.assignment) == {"a", "w[1]"}, res.assignment
    assert res.caveat is not None


def test_t15_legend_on_dff_counterexample(tmp_path) -> None:
    from netlist_agent.property_check import parse_counterexample

    x = _parse(tmp_path, _two_dff_bus("w[1]", "w[2]"), "x.v")
    y = _parse(tmp_path, _two_dff_bus("w[2]", "w[1]", c1="w[1]", c2="w[2]"), "y.v")
    res = verify_equivalence(x, y)
    assert not res.equivalent
    last = res.detail.splitlines()[-1]
    assert "flip-flop" in last and "Input pattern:" not in last
    assert "__dff_Q__" in res.detail
    bare = res.detail.rsplit("\n", 1)[0]
    assert parse_counterexample(res.detail) == parse_counterexample(bare)
    # signals= branch carries the legend too
    res2 = verify_equivalence(x, y, signals=["y"])
    assert not res2.equivalent and "flip-flop" in res2.detail.splitlines()[-1]
