from __future__ import annotations

import re

from netlist_agent.ir import (
    Const,
    Design,
    Direction,
    Gate,
    GateType,
    NetBit,
    Pin,
    POSITIONAL_PIN_ORDER,
    Port,
    Signal,
)

COMMENT_RE = re.compile(r"//[^\n]*")
MODULE_RE = re.compile(r"^module\s+(\w+)\s*\((.*)\)\s*$", re.DOTALL)
DECL_RE = re.compile(r"^(input|output|inout|wire)\s*(?:\[(\d+):(\d+)\])?\s*(.*)$", re.DOTALL)
INSTANCE_RE = re.compile(r"^(\w+)\s+(\w+)\s*\((.*)\)\s*$", re.DOTALL)
NAMED_PIN_RE = re.compile(r"^\.(\w+)\s*\((.*)\)$", re.DOTALL)
BITSEL_RE = re.compile(r"^(\w+)(?:\[(\d+)\])?$")
CONST_RE = re.compile(r"^1'b([01])$")

GATE_KEYWORDS = {gt.value for gt in POSITIONAL_PIN_ORDER} | {"dff"}
DIRECTION_KEYWORDS = {"input": Direction.INPUT, "output": Direction.OUTPUT, "inout": Direction.INPUT}


class NetlistParseError(ValueError):
    """Raised when the *input netlist itself* is malformed -- unrecognized
    statement, redeclared signal, undeclared port, and the like -- as
    opposed to a bug in this module. Deliberately a `ValueError` subclass so
    every existing `except ValueError` call site keeps working unchanged;
    callers that want to distinguish "bad input" from "bug in the parser"
    can catch this specifically (see router.py's `_h_load`)."""


def _parse_pin_value(text: str) -> Pin | None:
    text = text.strip()
    if not text:
        return None
    m = CONST_RE.match(text)
    if m:
        return Const.ZERO if m.group(1) == "0" else Const.ONE
    m = BITSEL_RE.match(text)
    if m:
        bit = m.group(2)
        return NetBit(m.group(1), int(bit) if bit is not None else None)
    raise NetlistParseError(f"unrecognized pin connection: {text!r}")


def _register_signal(design: Design, name: str, msb: int | None, lsb: int | None, direction: Direction) -> None:
    existing = design.signals.get(name)
    if existing is None:
        design.signals[name] = Signal(name=name, msb=msb, lsb=lsb, direction=direction)
        return
    # Ports are sometimes redeclared later as `wire` with identical name+width;
    # that's a no-op merge, not a new/duplicate signal.
    if existing.msb != msb or existing.lsb != lsb:
        raise NetlistParseError(f"signal {name!r} redeclared with a different width")
    if direction == Direction.INTERNAL:
        return
    existing.direction = direction


def _parse_decl(design: Design, keyword: str, msb_s: str | None, lsb_s: str | None, names_part: str) -> None:
    direction = DIRECTION_KEYWORDS.get(keyword, Direction.INTERNAL)
    msb = int(msb_s) if msb_s is not None else None
    lsb = int(lsb_s) if lsb_s is not None else None
    for name in names_part.split(","):
        name = name.strip()
        if name:
            _register_signal(design, name, msb, lsb, direction)


def _parse_instance(design: Design, gate_type_s: str, inst_name: str, body: str) -> None:
    gate_type = GateType(gate_type_s)
    gate = Gate(inst_name=inst_name, gate_type=gate_type)
    items = [item.strip() for item in body.split(",")]
    if gate_type == GateType.DFF:
        # dff connections are named (.RN/.SN/.CK/.D/.Q), unlike every other
        # primitive; source order of the named connections is not guaranteed.
        for item in items:
            if not item:
                continue
            m = NAMED_PIN_RE.match(item)
            if not m:
                raise NetlistParseError(f"expected named dff connection, got {item!r}")
            pin_name, value = m.group(1), m.group(2)
            gate.pins[pin_name] = _parse_pin_value(value)
    else:
        pin_order = POSITIONAL_PIN_ORDER[gate_type]
        for pin_name, item in zip(pin_order, items):
            gate.pins[pin_name] = _parse_pin_value(item) if item else None
    design.gates.append(gate)


_MODULE_NAME_RE = re.compile(r"^module\s+(\w+)")


def _multiple_modules_error(first_name: str, stmt: str) -> NetlistParseError:
    """The one rejection for a second `module` in the same file.

    Hierarchy has no downstream support (gate counts, cones and equivalence
    checks all assume one flat module), so a second module is refused rather
    than merged or silently dropped. The second name is best-effort: if the
    header is too malformed for `_MODULE_NAME_RE` the message just omits it.
    No file path here -- see the note on "no module declaration found" below.
    """
    m = _MODULE_NAME_RE.match(stmt)
    second = f" and {m.group(1)!r}" if m else ""
    return NetlistParseError(
        f"multiple modules in one file are not supported (found {first_name!r}{second})"
    )


def _after_endmodule_error(stmt: str) -> NetlistParseError:
    shown = stmt if len(stmt) <= 80 else stmt[:77] + "..."
    return NetlistParseError(f"unexpected content after endmodule: {shown!r}")


def parse_verilog(path: str) -> Design:
    with open(path, "r") as f:
        text = f.read()
    text = COMMENT_RE.sub("", text)

    design: Design | None = None
    port_order: list[str] = []
    # `endmodule` has no `;`, so after the `;` split it arrives glued to
    # whatever follows it (the next `module` header, a stray gate, ...) as one
    # segment. The old `first_word == "endmodule": continue` swallowed that
    # whole segment, so a second module's header was never processed and its
    # declarations and gates landed silently in the first module. Track
    # whether we have seen it so anything after it is rejected loudly.
    seen_endmodule = False

    for raw_stmt in text.split(";"):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        first_word = stmt.split(None, 1)[0]

        if seen_endmodule:
            # Any non-empty statement after the module closed (this also
            # covers `endmodule;` followed by a fresh `module b(...)`).
            if first_word == "module":
                raise _multiple_modules_error(design.module_name if design else "?", stmt)
            raise _after_endmodule_error(stmt)

        if first_word == "module":
            if design is not None:
                # A second header with no `endmodule` in between: the old code
                # replaced `design` here and dropped the first module.
                raise _multiple_modules_error(design.module_name, stmt)
            m = MODULE_RE.match(stmt)
            if not m:
                raise NetlistParseError(f"malformed module header: {stmt!r}")
            module_name, ports_text = m.group(1), m.group(2)
            design = Design(module_name=module_name)
            port_order = [p.strip() for p in ports_text.split(",") if p.strip()]
            continue

        if design is None:
            raise NetlistParseError("statement encountered before module header")

        if first_word in DIRECTION_KEYWORDS or first_word == "wire":
            m = DECL_RE.match(stmt)
            if not m:
                # measured unreachable by construction: DECL_RE's tail
                # (`(?:\[(\d+):(\d+)\])?\s*(.*)$`) is all-optional/`.*`, so
                # the match is guaranteed once `first_word` is one of the
                # four DIRECTION_KEYWORDS/"wire" values that got us into
                # this branch (exhaustively checked over the 4 keywords x
                # every 0-2 char continuation). Left in place so a future
                # tightening of DECL_RE fails loud instead of silently
                # accepting garbage -- see tests/test_load_write_errors.py.
                raise NetlistParseError(f"malformed declaration: {stmt!r}")
            keyword, msb_s, lsb_s, names_part = m.groups()
            _parse_decl(design, keyword, msb_s, lsb_s, names_part)
            continue

        if first_word in GATE_KEYWORDS:
            m = INSTANCE_RE.match(stmt)
            if not m:
                raise NetlistParseError(f"malformed instance: {stmt!r}")
            gate_type_s, inst_name, body = m.groups()
            _parse_instance(design, gate_type_s, inst_name, body)
            continue

        if first_word == "endmodule":
            seen_endmodule = True
            rest = stmt[len("endmodule"):].strip()
            if rest:
                # Glued remainder, e.g. `endmodule\nmodule other (...)`.
                if rest.split(None, 1)[0] == "module":
                    raise _multiple_modules_error(design.module_name, rest)
                raise _after_endmodule_error(rest)
            continue

        raise NetlistParseError(f"unrecognized statement: {stmt!r}")

    if design is None:
        # No `path` here -- the caller (router.py's _h_load) already knows
        # the path and prefixes it onto the message; repeating it here would
        # just duplicate it in the user-facing string.
        raise NetlistParseError("no module declaration found")
    if not seen_endmodule:
        # Checked after the no-module case on purpose. A file cut off at a
        # statement boundary would otherwise load with gates silently missing.
        raise NetlistParseError(f"missing endmodule for module {design.module_name!r}")

    ports = []
    for name in port_order:
        signal = design.signals.get(name)
        if signal is None:
            raise NetlistParseError(f"port {name!r} appears in the module header but is never declared")
        ports.append(Port(name=name, direction=signal.direction))
    design.ports = ports
    design.build_indices()
    return design
