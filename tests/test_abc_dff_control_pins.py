"""Regression tests for batch 7 (F-A): the equivalence boundary must see a
DFF's RN/SN/CK pins, not only Q and D.

`verify_equivalence` passes `tap_control_pins=True` to
`extract_combinational_view`, which adds one `__dff_<PIN>__<inst>` PO per
connected control pin. Every other caller keeps the default (off).
(On the verify path, a pin that is the same constant / same PI on both sides
is not tapped; see the skip-rule tests below.)

All designs are built from source strings written to `tmp_path`; no
fixture files are added.
"""

from __future__ import annotations

import pytest

import netlist_agent.abc_bridge as abc_bridge
from netlist_agent.abc_bridge import (
    ABCBridgeError,
    extract_combinational_view,
    verify_equivalence,
)
from netlist_agent import abc_synth
from netlist_agent.graph import NetlistGraph
from netlist_agent.ir import Direction, GateType, NetBit
from netlist_agent.parser import parse_verilog

BASE = (
    "module top(a, b, c, clk, y);\n"
    "  input a, b, c, clk;\n"
    "  output y;\n"
    "  wire bb, q;\n"
    "  buf gb(bb, a);\n"
    "  dff r1(.RN({rn}), .SN({sn}), .CK({ck}), .D(c), .Q(q));\n"
    "  and g1(y, q, b);\n"
    "endmodule\n"
)


def _parse(tmp_path, src: str, name: str):
    path = tmp_path / name
    path.write_text(src)
    return parse_verilog(str(path))


def _design(tmp_path, name: str, rn="1'b1", sn="1'b1", ck="clk"):
    return _parse(tmp_path, BASE.format(rn=rn, sn=sn, ck=ck), name)


@pytest.mark.parametrize(
    "pin,new",
    [("sn", "b"), ("ck", "b"), ("rn", "b"), ("sn", "a"), ("ck", "c")],
)
def test_control_pin_rewire_is_not_equivalent(tmp_path, pin, new) -> None:
    # Defaults: RN=SN=1'b1, CK=clk. Rewire one pin to another PI or a constant.
    a = _design(tmp_path, "a.v")
    b = _design(tmp_path, "b.v", **{pin: new})
    res = verify_equivalence(a, b)
    assert not res.equivalent


def test_rn_const_one_to_pi_is_not_equivalent(tmp_path) -> None:
    a = _design(tmp_path, "a.v", rn="1'b1")
    b = _design(tmp_path, "b.v", rn="a")
    assert not verify_equivalence(a, b).equivalent


def test_equivalent_rewire_through_buffer_is_equivalent(tmp_path) -> None:
    # bb = BUF(a): SN from a to bb is the same function.
    a = _design(tmp_path, "a.v", sn="a")
    b = _design(tmp_path, "b.v", sn="bb")
    assert verify_equivalence(a, b).equivalent


def test_identical_designs_still_equivalent(tmp_path) -> None:
    a = _design(tmp_path, "a.v", sn="a")
    b = _design(tmp_path, "b.v", sn="a")
    assert verify_equivalence(a, b).equivalent


@pytest.mark.parametrize("pin,new", [("sn", "b"), ("ck", "b"), ("rn", "b")])
def test_detail_carries_legend_for_control_pin_counterexample(tmp_path, pin, new) -> None:
    a = _design(tmp_path, "a.v")
    b = _design(tmp_path, "b.v", **{pin: new})
    res = verify_equivalence(a, b)
    assert not res.equivalent
    assert "reset, set and clock pins" in res.detail.splitlines()[-1]
    assert "Input pattern:" not in res.detail.splitlines()[-1]


@pytest.mark.parametrize("pin", ["RN", "SN", "CK"])
def test_tap_name_collision_raises(tmp_path, pin) -> None:
    src = (
        "module top(a, b, clk, y);\n"
        "  input a, b, clk;\n"
        "  output y;\n"
        f"  wire __dff_{pin}__r1;\n"
        "  wire q;\n"
        "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(q));\n"
        "  and g1(y, q, b);\n"
        "endmodule\n"
    )
    d = _parse(tmp_path, src, "col.v")
    with pytest.raises(ABCBridgeError):
        extract_combinational_view(d, "free_pi", tap_control_pins=True)
    # default view is unaffected by the colliding name
    extract_combinational_view(d, "free_pi")


def test_default_view_has_no_control_pin_ports(tmp_path) -> None:
    d = _design(tmp_path, "a.v")
    for mode in ("free_pi", "const_zero"):
        view = extract_combinational_view(d, mode)
        names = {p.name for p in view.ports}
        assert not any(n.startswith(("__dff_RN__", "__dff_SN__", "__dff_CK__")) for n in names), names
    tapped = extract_combinational_view(d, "free_pi", tap_control_pins=True)
    outs = {p.name for p in tapped.ports if p.direction == Direction.OUTPUT}
    assert {"__dff_RN__r1", "__dff_SN__r1", "__dff_CK__r1", "__dff_D__r1"} <= outs


def test_unconnected_control_pin_is_skipped(tmp_path) -> None:
    src = (
        "module top(a, b, clk, y);\n"
        "  input a, b, clk;\n"
        "  output y;\n"
        "  wire q;\n"
        "  dff r1(.RN(), .SN(1'b1), .CK(clk), .D(a), .Q(q));\n"
        "  and g1(y, q, b);\n"
        "endmodule\n"
    )
    d = _parse(tmp_path, src, "unc.v")
    view = extract_combinational_view(d, "free_pi", tap_control_pins=True)
    names = {p.name for p in view.ports}
    assert "__dff_RN__r1" not in names
    assert "__dff_SN__r1" in names


def test_legend_line_when_only_a_control_pin_differs(tmp_path) -> None:
    # No .D() and no .Q(): the only __dff_*__ token that can appear in the
    # detail is __dff_SN__r1, so the legend must be triggered by the control
    # pin tokens alone.
    tmpl = (
        "module top(a, b, clk, y);\n"
        "  input a, b, clk;\n"
        "  output y;\n"
        "  dff r1(.SN({sn}), .CK(clk));\n"
        "  and g1(y, a, b);\n"
        "endmodule\n"
    )
    da = _parse(tmp_path, tmpl.format(sn="a"), "a.v")
    db = _parse(tmp_path, tmpl.format(sn="b"), "b.v")
    res = verify_equivalence(da, db)
    assert not res.equivalent
    assert res.detail.splitlines()[-1].startswith("(__dff_Q__<name>")
    assert "reset, set and clock pins" in res.detail.splitlines()[-1]


def test_swapping_control_pins_between_two_dffs_is_not_equivalent(tmp_path) -> None:
    tmpl = (
        "module top(a, b, c, clk, y);\n"
        "  input a, b, c, clk;\n"
        "  output y;\n"
        "  wire q1, q2;\n"
        "  dff r1(.RN(1'b1), .SN({s1}), .CK(clk), .D(c), .Q(q1));\n"
        "  dff r2(.RN(1'b1), .SN({s2}), .CK(clk), .D(c), .Q(q2));\n"
        "  and g1(y, q1, q2);\n"
        "endmodule\n"
    )
    da = _parse(tmp_path, tmpl.format(s1="a", s2="b"), "a.v")
    db = _parse(tmp_path, tmpl.format(s1="b", s2="a"), "b.v")
    assert not verify_equivalence(da, db).equivalent
    assert verify_equivalence(da, da).equivalent


@pytest.mark.parametrize("pin", ["RN", "SN", "CK"])
def test_control_pin_connected_on_one_side_only_raises(tmp_path, pin) -> None:
    tmpl = (
        "module top(a, b, clk, y);\n"
        "  input a, b, clk;\n"
        "  output y;\n"
        "  wire q;\n"
        "  dff r1(.RN({rn}), .SN({sn}), .CK({ck}), .D(a), .Q(q));\n"
        "  and g1(y, q, b);\n"
        "endmodule\n"
    )
    full = dict(rn="a", sn="a", ck="clk")
    cut = dict(full, **{pin.lower(): ""})
    da = _parse(tmp_path, tmpl.format(**full), "a.v")
    db = _parse(tmp_path, tmpl.format(**cut), "b.v")
    with pytest.raises(ABCBridgeError, match=f"__dff_{pin}__"):
        verify_equivalence(da, db)
    with pytest.raises(ABCBridgeError, match=f"__dff_{pin}__"):
        verify_equivalence(db, da)


def test_extra_pin_in_dff_pin_order_is_tapped(tmp_path, monkeypatch) -> None:
    # The tap is derived from DFF_PIN_ORDER, not a hard-coded RN/SN/CK list.
    monkeypatch.setattr(abc_bridge, "DFF_PIN_ORDER", ["RN", "SN", "CK", "EN", "D", "Q"])
    src = (
        "module top(a, b, clk, y);\n"
        "  input a, b, clk;\n"
        "  output y;\n"
        "  wire q;\n"
        "  dff r1(.RN(1'b1), .SN(1'b1), .CK(clk), .EN(a), .D(b), .Q(q));\n"
        "  and g1(y, q, b);\n"
        "endmodule\n"
    )
    d = _parse(tmp_path, src, "en.v")
    view = extract_combinational_view(d, "free_pi", tap_control_pins=True)
    assert "__dff_EN__r1" in {p.name for p in view.ports}


# ----------------------------------------------------------------------
# Skip rule: identical constant / identical PI control pins are not tapped.
# ----------------------------------------------------------------------

SKIP_TMPL = (
    "module top(a, b, c, clk, y);\n"
    "  input a, b, c, clk;\n"
    "  output y;\n"
    "  wire q, w, n;\n"
    "  and gn(n, a, b);\n"
    "  buf gw(w, n);\n"
    "  dff r1(.RN({rn}), .SN({sn}), .CK(clk), .D(c), .Q(q));\n"
    "  and g1(y, q, b);\n"
    "endmodule\n"
)


def _skip_pair(tmp_path, a_kw, b_kw):
    da = _parse(tmp_path, SKIP_TMPL.format(**a_kw), "a.v")
    db = _parse(tmp_path, SKIP_TMPL.format(**b_kw), "b.v")
    return da, db


def _tap_ports(design):
    return {p.name for p in design.ports if p.name.startswith(("__dff_RN__", "__dff_SN__", "__dff_CK__"))}


def _views(da, db):
    skip = abc_bridge._identical_control_pins(da, db)
    kw = dict(tap_control_pins=True, skip_control_taps=skip)
    return (
        extract_combinational_view(da, "free_pi", **kw),
        extract_combinational_view(db, "free_pi", **kw),
    )


def test_same_const_and_same_pi_control_pins_are_not_tapped(tmp_path) -> None:
    da, db = _skip_pair(tmp_path, dict(rn="1'b1", sn="a"), dict(rn="1'b1", sn="a"))
    va, vb = _views(da, db)
    assert _tap_ports(va) == set() == _tap_ports(vb)  # RN const, SN PI, CK = clk PI
    assert verify_equivalence(da, db).equivalent


def test_pi_to_other_pi_is_still_caught_after_skip(tmp_path) -> None:
    da, db = _skip_pair(tmp_path, dict(rn="1'b1", sn="a"), dict(rn="1'b1", sn="b"))
    va, vb = _views(da, db)
    assert _tap_ports(va) == {"__dff_SN__r1"} == _tap_ports(vb)
    assert not verify_equivalence(da, db).equivalent


def test_gate_driven_same_name_net_is_still_tapped(tmp_path) -> None:
    # w is driven by a gate, so even with the same name on both sides it is tapped.
    da, db = _skip_pair(tmp_path, dict(rn="1'b1", sn="w"), dict(rn="1'b1", sn="w"))
    va, vb = _views(da, db)
    assert _tap_ports(va) == {"__dff_SN__r1"} == _tap_ports(vb)


def test_gate_driven_control_pin_whose_logic_changed_is_caught(tmp_path) -> None:
    da = _parse(tmp_path, SKIP_TMPL.format(rn="1'b1", sn="w"), "a.v")
    db = _parse(tmp_path, SKIP_TMPL.format(rn="1'b1", sn="w").replace("and gn(n, a, b)", "or gn(n, a, b)"), "b.v")
    assert not verify_equivalence(da, db).equivalent


def test_const_on_one_side_pi_on_other_is_still_caught(tmp_path) -> None:
    da, db = _skip_pair(tmp_path, dict(rn="1'b1", sn="1'b1"), dict(rn="1'b1", sn="a"))
    va, vb = _views(da, db)
    assert _tap_ports(va) == {"__dff_SN__r1"} == _tap_ports(vb)
    assert not verify_equivalence(da, db).equivalent


# ----------------------------------------------------------------------
# The only consumer of the control-pin tap: abc_synth's optimisation must
# refuse a result that lost a function only the SN pin depended on.
# ----------------------------------------------------------------------

ONLY_CONSUMER_SRC = (
    "module top(a, b, c, p1, p2, p3, clk, y);\n"
    "  input a, b, c, p1, p2, p3, clk;\n"
    "  output y;\n"
    "  wire q, t1, t2, t3, s1, s;\n"
    "  and g1(t1, a, b);\n"
    "  and g2(t2, t1, c);\n"
    "  or g3(t3, t1, t2);\n"  # absorption: t3 == t1, so there is something to reduce
    "  and k1(s1, p1, p2);\n"
    "  and k2(s, s1, p3);\n"  # SN-only chain
    "  dff d1(.RN(1'b1), .SN(s), .CK(clk), .D(t3), .Q(q));\n"
    "  and h1(y, q, b);\n"
    "endmodule\n"
)


def _remove_dangling_without_sn(design):
    """Liveness list that forgets SN, so the SN-only chain is swept."""
    graph = NetlistGraph(design)
    live = set()
    for po in graph.po_bits:
        live |= graph.backward_reachable_gates(po)
    for dff in graph.dff_gates:
        for pin in ("D", "CK", "RN"):
            v = dff.pins.get(pin)
            if isinstance(v, NetBit):
                live |= graph.backward_reachable_gates(v)
    dangling = [g for g in design.gates if g.gate_type != GateType.DFF and g.inst_name not in live]
    for g in dangling:
        design.remove_gate(g)
    return len(dangling)


def test_optimize_refuses_a_result_that_lost_the_sn_driver(tmp_path, monkeypatch) -> None:
    design = _parse(tmp_path, ONLY_CONSUMER_SRC, "only.v")
    gates_before = list(design.gates)
    monkeypatch.setattr(abc_synth, "remove_dangling_gates", _remove_dangling_without_sn)
    res = abc_synth.optimize_gate_count(design)
    assert res.failure is not None
    assert "not equivalent" in res.failure.lower()
    assert not res.changed
    assert list(design.gates) == gates_before  # the caller's design is untouched


# ----------------------------------------------------------------------
# Skip rule: pinned down (spy, renames, multi-DFF, buses, duplicate names).
# ----------------------------------------------------------------------


def test_verify_equivalence_applies_skip_to_the_cec_views(tmp_path, monkeypatch) -> None:
    da, db = _skip_pair(tmp_path, dict(rn="1'b1", sn="a"), dict(rn="1'b1", sn="b"))
    seen = []

    def spy(a, b, timeout):
        seen.append(({p.name for p in a.ports if p.direction == Direction.OUTPUT},
                     {p.name for p in b.ports if p.direction == Direction.OUTPUT}))
        return abc_bridge.EquivResult(True, "")

    monkeypatch.setattr(abc_bridge, "_run_cec", spy)
    verify_equivalence(da, db)
    (po_a, po_b), = seen
    assert po_a == po_b
    assert "__dff_RN__r1" not in po_a and "__dff_CK__r1" not in po_a  # skipped
    assert "__dff_SN__r1" in po_a  # changed pin is still tapped


@pytest.mark.parametrize("swap", [False, True])
def test_dff_present_under_different_names_raises(tmp_path, swap) -> None:
    tmpl = (
        "module top(a, b, c, clk, y);\n"
        "  input a, b, c, clk;\n"
        "  output y;\n"
        "  wire q;\n"
        "  dff {n}(.RN(1'b1), .SN(a), .CK(clk), .D(c), .Q(q));\n"
        "  and g1(y, q, b);\n"
        "endmodule\n"
    )
    d1 = _parse(tmp_path, tmpl.format(n="r1"), "a.v")
    d2 = _parse(tmp_path, tmpl.format(n="r2"), "b.v")
    x, y = (d2, d1) if swap else (d1, d2)
    with pytest.raises(ABCBridgeError):
        verify_equivalence(x, y)


def test_skipped_dff_does_not_hide_another_dffs_changed_pin(tmp_path) -> None:
    tmpl = (
        "module top(a, b, c, clk, y);\n"
        "  input a, b, c, clk;\n"
        "  output y;\n"
        "  wire q1, q2;\n"
        "  dff r1(.RN(1'b1), .SN(a), .CK(clk), .D(c), .Q(q1));\n"
        "  dff r2(.RN(1'b1), .SN({s2}), .CK(clk), .D(c), .Q(q2));\n"
        "  and g1(y, q1, q2);\n"
        "endmodule\n"
    )
    da = _parse(tmp_path, tmpl.format(s2="a"), "a.v")
    db = _parse(tmp_path, tmpl.format(s2="b"), "b.v")
    assert abc_bridge._identical_control_pins(da, db) >= {("r1", "SN")}
    assert not verify_equivalence(da, db).equivalent


def test_bus_bit_change_on_a_control_pin_is_not_equivalent(tmp_path) -> None:
    tmpl = (
        "module top(a, c, clk, y);\n"
        "  input [1:0] a;\n"
        "  input c, clk;\n"
        "  output y;\n"
        "  wire q;\n"
        "  dff r1(.RN(1'b1), .SN(a[{i}]), .CK(clk), .D(c), .Q(q));\n"
        "  and g1(y, q, c);\n"
        "endmodule\n"
    )
    da = _parse(tmp_path, tmpl.format(i=0), "a.v")
    db = _parse(tmp_path, tmpl.format(i=1), "b.v")
    assert not verify_equivalence(da, db).equivalent
    assert verify_equivalence(da, da).equivalent


def test_duplicate_dff_instance_names_are_not_skipped_and_raise(tmp_path) -> None:
    # (inst, pin) is not a unique key when an instance name repeats; skipping
    # would swallow the tap-name collision and return a wrong EQ.
    tmpl = (
        "module top(a, b, clk, y);\n"
        "  input a, b, clk;\n"
        "  output y;\n"
        "  dff r1(.SN({s1}));\n"
        "  dff r1(.SN({s2}));\n"
        "  and g1(y, a, b);\n"
        "endmodule\n"
    )
    da = _parse(tmp_path, tmpl.format(s1="a", s2="b"), "a.v")
    db = _parse(tmp_path, tmpl.format(s1="a", s2="a"), "b.v")
    assert abc_bridge._identical_control_pins(da, db) == frozenset()
    with pytest.raises(ABCBridgeError):
        verify_equivalence(da, db)
    with pytest.raises(ABCBridgeError):
        verify_equivalence(db, da)


@pytest.mark.parametrize("swap", [False, True])
def test_duplicate_dff_instance_name_on_one_side_only_raises(tmp_path, swap) -> None:
    # The duplicate check must look at BOTH designs: here only one side
    # repeats `r1`, so checking a single side lets the skip swallow the
    # collision in one direction and return a wrong EQ.
    single = (
        "module top(a, b, clk, y);\n"
        "  input a, b, clk;\n"
        "  output y;\n"
        "  dff r1(.SN(a));\n"
        "  and g1(y, a, b);\n"
        "endmodule\n"
    )
    double = single.replace("  dff r1(.SN(a));\n", "  dff r1(.SN(a));\n  dff r1(.SN(a));\n")
    d1 = _parse(tmp_path, single, "a.v")
    d2 = _parse(tmp_path, double, "b.v")
    x, y = (d2, d1) if swap else (d1, d2)
    assert abc_bridge._identical_control_pins(x, y) == frozenset()
    with pytest.raises(ABCBridgeError):
        verify_equivalence(x, y)


@pytest.mark.parametrize("swap", [False, True])
def test_dff_with_only_control_pins_missing_on_one_side_raises(tmp_path, swap) -> None:
    # No D/Q, so the Q-PI / D-tap name-set checks cannot catch the missing
    # instance; its SN/CK taps must survive the skip so the PO names differ.
    # swap=False: the "instance on one side only is never skipped" guard is
    # what keeps them.  swap=True: the skip only iterates design_a's DFFs, so
    # a B-only instance is never a candidate; this direction guards against a
    # future refactor making the skip symmetric and swallowing it.
    with_dff = (
        "module top(a, b, clk, y);\n"
        "  input a, b, clk;\n"
        "  output y;\n"
        "  dff r1(.SN(a), .CK(clk));\n"
        "  and g1(y, a, b);\n"
        "endmodule\n"
    )
    without = with_dff.replace("  dff r1(.SN(a), .CK(clk));\n", "")
    d1 = _parse(tmp_path, with_dff, "a.v")
    d2 = _parse(tmp_path, without, "b.v")
    x, y = (d2, d1) if swap else (d1, d2)
    # Checked without ABC, so a missing/broken ABC binary (which also raises
    # ABCBridgeError) cannot mask a skip regression below.
    assert abc_bridge._identical_control_pins(x, y) == frozenset()
    with pytest.raises(ABCBridgeError):
        verify_equivalence(x, y)
