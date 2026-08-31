"""Shared net-reference parsing helpers.

Used by both router.py's regex handlers (which capture net tokens out of
free-text requests) and the LLM tool registry (netlist_agent/llm/tools_schema.py,
which receives net references as plain strings in tool-call arguments) --
both surfaces need the exact same "name" / "name[bit]" bit-select parsing, so
it lives here once rather than being duplicated.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from netlist_agent.ir import NetBit

if TYPE_CHECKING:
    from netlist_agent.ir import Design, Signal

_NET_RE = re.compile(r"^(\w+)(?:\[(\d+)\])?$")


class NetRefError(ValueError):
    """A user-given net reference cannot be resolved to a concrete net-bit
    (or set of net-bits) on this design: no such signal, a bit-select on a
    scalar signal, a missing bit-select on a vector signal (`resolve_bit`
    only -- ambiguous which bit is meant), or a bit-select outside the
    signal's declared `[msb:lsb]` range. Always a single-line, actionable
    message -- callers surface `str(exc)` directly to the user/model rather
    than letting a traceback through."""


def parse_net(token: str) -> NetBit:
    """Parse a net reference like "n6" or "n6[3]" into a NetBit.

    Strips surrounding whitespace and trailing sentence punctuation (router.py's
    regex captures sometimes include a trailing '.', ',', etc.; an LLM tool
    call argument should not, but stripping is harmless either way).
    """
    token = token.strip().rstrip(".,;:?")
    m = _NET_RE.match(token)
    if not m:
        raise ValueError(f"cannot parse net reference: {token!r}")
    name, bit = m.groups()
    return NetBit(name, int(bit) if bit is not None else None)


def signal_name_only(token: str) -> str:
    """Strip an optional trailing bit-select, leaving just the signal name."""
    return token.split("[")[0].strip()


def netbit_token(nb: NetBit) -> str:
    """Render a NetBit back to its "name" / "name[bit]" token form."""
    return nb.name if nb.bit is None else f"{nb.name}[{nb.bit}]"


def netbit_sort_key(nb: NetBit) -> tuple[str, int]:
    """Sort key for NetBit that orders by (name, bit) NUMERICALLY, not by
    lexicographic string comparison of the rendered "name[bit]" token --
    string sort puts "n17[10]" before "n17[2]" (F6: on a design with 35+
    bits of the same signal this reads as scrambled). `bit is None` (a
    scalar net, no bit-select) sorts before any bit-selected entry of the
    same name.
    """
    return (nb.name, -1 if nb.bit is None else nb.bit)


# ----------------------------------------------------------------------
# Width-aware resolution (validates against the design's declared signal
# widths, unlike `parse_net`/`signal_name_only` above, which are pure syntax
# and know nothing about any particular Design). See the module-level
# problem statement this was introduced to fix: `parse_net` alone lets a
# dropped/mismatched/out-of-range bit-select silently resolve to a NetBit
# key that simply isn't in `design.net_driver`/`net_fanout`, so every
# lookup quietly returns "0"/"none" instead of failing loudly.
# ----------------------------------------------------------------------


def _checked_signal(design: "Design", nb: NetBit) -> "Signal":
    """The validation `resolve_bit` and `resolve_bits` share: the signal must
    be declared, and a bit-select (if given) must be legal for it.

    This lives in one place deliberately. When the two resolvers each carried
    their own copy of these checks, the copies were already drifting: only
    `resolve_bit`'s had a test for the nonzero-`lsb` boundary, so a mutation
    to `resolve_bits`' range check alone passed the whole suite.

    What is NOT shared is the bare-name case, which is exactly where the two
    differ: `resolve_bit` rejects a bare vector name as ambiguous, while
    `resolve_bits` expands it to every bit.
    """
    sig = design.signals.get(nb.name)
    if sig is None:
        raise NetRefError(f"no such signal {nb.name!r}")
    if sig.msb is None:
        if nb.bit is not None:
            raise NetRefError(f"{nb.name} is a scalar signal; a bit-select {nb.name}[{nb.bit}] is not valid")
        return sig
    if nb.bit is not None:
        # min/max, not (lsb, msb): the declared range may be ascending.
        lsb = sig.lsb if sig.lsb is not None else sig.msb
        lo, hi = min(sig.msb, lsb), max(sig.msb, lsb)
        if not (lo <= nb.bit <= hi):
            raise NetRefError(f"{nb.name}[{nb.bit}] is out of the declared range [{sig.msb}:{sig.lsb}]")
    return sig


def resolve_bit(design: "Design", token: str) -> NetBit:
    """Resolve `token` to EXACTLY ONE net-bit on `design`, for callers whose
    operation is semantically about a single net-bit (fanout, depth, path,
    is_constant, symmetry, signal equivalence, Boolean function, balance
    depth, etc.). Raises `NetRefError` if:

      * no signal named `token`'s name is declared in `design`;
      * `token` bit-selects a scalar (1-bit, `msb is None`) signal
        (e.g. `clk[0]`);
      * `token` is a bare name with no bit-select on a vector (multi-bit)
        signal -- ambiguous which bit is meant, so this is rejected rather
        than silently guessing bit 0 or "the whole bus";
      * `token`'s bit-select falls outside the signal's declared
        `[msb:lsb]` range (inclusive of both endpoints -- `lsb` is NOT
        assumed to be 0, a signal may be declared e.g. `[31:1]`).
    """
    nb = parse_net(token)
    sig = _checked_signal(design, nb)
    if sig.msb is None:
        return NetBit(nb.name, None)
    if nb.bit is None:
        raise NetRefError(
            f"{nb.name} is a {sig.width}-bit signal (declared [{sig.msb}:{sig.lsb}]) -- a bit-select is "
            f"required, e.g. {nb.name}[{sig.lsb}]"
        )
    return nb


def resolve_bits(design: "Design", token: str) -> list[NetBit]:
    """Resolve `token` to the list of net-bits it refers to, for callers
    whose operation is semantically about a whole signal (e.g.
    `_h_max_fanout_of_signal`, `_h_cut_signal`, `_h_gates_connected_to_signal`,
    `_h_dffs_on_clock`). A bit-selected token (`n8[3]`) resolves to that one
    bit only; a bare signal name resolves to every bit of that signal
    (`Signal.bits()`, MSB-first) -- unlike `resolve_bit`, a bare vector name
    is NOT an error here. Raises `NetRefError` under the same no-such-signal/
    scalar-bit-select/out-of-range conditions as `resolve_bit`.
    """
    nb = parse_net(token)
    sig = _checked_signal(design, nb)
    if sig.msb is None:
        return [NetBit(nb.name, None)]
    if nb.bit is None:
        return sig.bits()
    return [nb]
