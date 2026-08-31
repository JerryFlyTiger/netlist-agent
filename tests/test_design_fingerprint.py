"""`design_fingerprint` has one job: never say "unchanged" about a design
that changed.

It exists because `router.handle_request` uses it to decide whether an LLM
fallback turn silently edited the netlist, and a false "unchanged" there is
the exact failure the whole mechanism was added to stop -- five of sixty
held-out requests in `experiments/heldout_fallback_2026-08-28/` modified a
netlist without saying so, and two of those told the user they had failed.

These tests exist because a mutation run found they were missing. Knifing the
port loop, the signal loop, and the pin digest out of `design_fingerprint` --
three of the four things it claims to cover -- left the entire suite green.
The disclosure tests only ever exercised the gate list, because every fixture
that mutates a design adds or removes gates. A rename or a rewire does not,
and those are precisely the edits the count and the histogram already miss.
"""
from __future__ import annotations

import copy

from netlist_agent.ir import Design, Direction, Gate, GateType, NetBit, Port, Signal, design_fingerprint


def _design() -> Design:
    """Two gates, one net between them, plus a primary input no gate reads.

    The unread port is deliberate: `check_floating_signals` exists because
    such nets occur, and `rename_signal` is a tool the model can call on one.
    A gates-only digest cannot see that edit at all.
    """
    return Design(
        module_name="top",
        ports=[
            Port("a", Direction.INPUT),
            Port("y", Direction.OUTPUT),
            Port("unread", Direction.INPUT),
        ],
        signals={
            "a": Signal("a", None, None, Direction.INPUT),
            "y": Signal("y", None, None, Direction.OUTPUT),
            "unread": Signal("unread", None, None, Direction.INPUT),
            "mid": Signal("mid", None, None, Direction.INTERNAL),
        },
        gates=[
            Gate("g0", GateType.NOT, {"A": NetBit("a", None), "Y": NetBit("mid", None)}),
            Gate("g1", GateType.BUF, {"A": NetBit("mid", None), "Y": NetBit("y", None)}),
        ],
    )


def test_identical_designs_agree() -> None:
    assert design_fingerprint(_design()) == design_fingerprint(_design())


def test_none_design_has_no_fingerprint() -> None:
    # handle_request relies on this: "nothing was loaded" must not compare
    # equal to "a design is loaded", nor raise.
    assert design_fingerprint(None) is None


def test_gate_insertion_order_does_not_change_it() -> None:
    """`remove_gate` is swap-and-pop, so the list order is not stable across
    edits that cancel out. Order is not content, and must not read as one."""
    shuffled = _design()
    shuffled.gates.reverse()
    assert design_fingerprint(shuffled) == design_fingerprint(_design())


def test_renaming_a_net_no_gate_references_is_detected() -> None:
    """The realistic edit: `Design.rename_signal` moves the port and the
    signal together, and no gate mentions this one, so the gate list is
    identical before and after."""
    edited = _design()
    edited.ports[2].name = "RENAMED"
    edited.signals["RENAMED"] = edited.signals.pop("unread")
    edited.signals["RENAMED"].name = "RENAMED"
    assert len(edited.gates) == len(_design().gates)  # the gate list is untouched
    assert design_fingerprint(edited) != design_fingerprint(_design())


def test_the_port_list_is_digested_independently_of_the_signal_table() -> None:
    """Isolates the port record on purpose, by editing `ports` and nothing
    else.

    The test above does not do this job, and a mutation run is how that was
    found: deleting the port loop from `design_fingerprint` left it green,
    because it renames the signal too and the signal loop caught the change
    on its own. The test asserted the right thing for the wrong reason.

    No tool today changes a port without its signal -- `rename_signal`
    updates both -- so this input is deliberately inconsistent, and the port
    record is defence in depth rather than a path any current caller takes.
    That is a reason to pin it, not to drop it: the next tool that touches
    `ports` alone will be covered, and a silent netlist edit is the one thing
    this digest exists to refuse to miss.
    """
    edited = _design()
    edited.ports[2].direction = Direction.OUTPUT
    assert {n: (s.msb, s.lsb) for n, s in edited.signals.items()} == {
        n: (s.msb, s.lsb) for n, s in _design().signals.items()
    }
    assert design_fingerprint(edited) != design_fingerprint(_design())


def test_changing_a_signal_width_is_detected() -> None:
    edited = _design()
    edited.signals["unread"] = Signal("unread", 7, 0, Direction.INPUT)
    assert design_fingerprint(edited) != design_fingerprint(_design())


def test_rewiring_a_pin_between_identical_gates_is_detected() -> None:
    """Same gates, same names, same types, same count, same histogram -- only
    the wiring moved. This is the case the gate count and the gate-type
    histogram both call "unchanged"."""
    edited = _design()
    edited.gates[1].pins["A"] = NetBit("a", None)  # g1 now reads `a`, not `mid`
    assert [g.inst_name for g in edited.gates] == [g.inst_name for g in _design().gates]
    assert design_fingerprint(edited) != design_fingerprint(_design())


def test_renaming_a_gate_is_detected() -> None:
    edited = _design()
    edited.gates[0].inst_name = "g99"
    assert design_fingerprint(edited) != design_fingerprint(_design())


def test_changing_a_gate_type_is_detected() -> None:
    edited = _design()
    edited.gates[1].gate_type = GateType.NOT
    assert design_fingerprint(edited) != design_fingerprint(_design())


def test_renaming_the_module_is_detected() -> None:
    edited = _design()
    edited.module_name = "renamed_top"
    assert design_fingerprint(edited) != design_fingerprint(_design())


def test_deep_copy_is_not_aliased_to_its_source() -> None:
    """The measurement harness snapshots a design with `copy.deepcopy` and
    compares fingerprints across a mutation; if the copy tracked the original
    the comparison would be of a design with itself."""
    original = _design()
    snapshot = copy.deepcopy(original)
    original.gates[0].inst_name = "g99"
    assert design_fingerprint(snapshot) != design_fingerprint(original)
