"""SPICE-subset netlist parser.

Grammar (one statement per line):

    <title line>          -- always the first line, ignored no matter what
    * a comment line
    Rname n+ n- value      ; inline comments after ';' are also stripped
    Vname n+ n- value
    Iname n+ n- value
    .op
    .end

Deliberate SPICE-compatibility details this parser gets right, because
they're the classic gotchas that separate "reads a text file" from "reads a
netlist":

  - The FIRST line of the file is always a title line and is discarded, even
    if it looks like a perfectly valid element -- real SPICE decks always
    have one, and forgetting this silently eats your first component.
  - A line starting with '+' is a continuation of the previous logical line.
  - Engineering suffixes distinguish "meg" (1e6) from "m" (1e-3); trailing
    unit letters ("10kOhm", "5V", "100mA") are ignored once the recognized
    prefix is consumed, matching how real SPICE tolerates unit annotations.
  - Node names are case-sensitive arbitrary strings; "0" and "gnd"
    (case-insensitive) both mean ground -- that aliasing is handled down in
    Circuit, not here.
  - Directives this phase doesn't implement yet (.dc/.ac/.tran) are parsed
    and kept, not rejected -- the CLI decides what to do with them, so a
    netlist written for a later phase doesn't fail to even *parse*.
"""

import re
from dataclasses import dataclass, field

from .elements import CurrentSource, Resistor, VoltageSource
from .matrix import Circuit

_SI_SUFFIXES = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "g": 1e9,
    "t": 1e12,
}

_VALUE_RE = re.compile(r"^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)([a-zA-Z]*)$")

_ELEMENT_CLASSES = {"R": Resistor, "V": VoltageSource, "I": CurrentSource}

_KNOWN_DIRECTIVES = {".op", ".end", ".dc", ".ac", ".tran", ".print"}


class NetlistError(ValueError):
    def __init__(self, message, line_no=None, line_text=None):
        self.line_no = line_no
        self.line_text = line_text
        if line_no is not None:
            located = f"line {line_no}: {message}"
            if line_text:
                located += f" ({line_text.strip()!r})"
            message = located
        super().__init__(message)


def parse_value(token, *, line_no=None, line_text=None):
    """Parse a SPICE-style numeric literal with an optional engineering suffix.

    "1k" -> 1000.0, "4.7u" -> 4.7e-6, "1meg" -> 1e6 (note: NOT the same as
    "1m" = 1e-3 -- this is the single most common netlist typo). Trailing
    unit letters after a recognized prefix ("10kOhm") are ignored.
    """
    match = _VALUE_RE.match(token)
    if not match:
        raise NetlistError(f"invalid numeric value {token!r}", line_no, line_text)
    num_str, suffix = match.groups()
    num = float(num_str)
    suffix_lower = suffix.lower()
    if suffix_lower.startswith("meg"):
        mult = 1e6
    elif suffix_lower[:1] in _SI_SUFFIXES:
        mult = _SI_SUFFIXES[suffix_lower[0]]
    else:
        mult = 1.0
    return num * mult


@dataclass
class Directive:
    name: str  # lowercased, includes the leading '.', e.g. ".dc"
    args: list[str] = field(default_factory=list)
    line_no: int = 0


@dataclass
class ParsedNetlist:
    circuit: Circuit
    directives: list[Directive]
    title: str


def _join_continuations(raw_lines):
    """Merge lines starting with '+' into the previous logical line."""
    logical = []
    for physical_no, raw in enumerate(raw_lines, start=1):
        if raw.strip().startswith("+") and logical:
            prev_no, prev_text = logical[-1]
            logical[-1] = (prev_no, prev_text + " " + raw.strip()[1:].strip())
        else:
            logical.append((physical_no, raw))
    return logical


def parse_source(text: str) -> ParsedNetlist:
    logical = _join_continuations(text.splitlines())
    if not logical:
        raise NetlistError("empty netlist (missing title line)")

    title = logical[0][1].strip()
    body = logical[1:]

    circuit = Circuit()
    directives = []

    for line_no, raw in body:
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("*"):
            continue

        if line.startswith("."):
            parts = line.split()
            name = parts[0].lower()
            if name == ".end":
                break
            if name not in _KNOWN_DIRECTIVES:
                raise NetlistError(f"unknown directive {parts[0]!r}", line_no, raw)
            directives.append(Directive(name=name, args=parts[1:], line_no=line_no))
            continue

        parts = line.split()
        if len(parts) < 4:
            raise NetlistError(
                f"expected 'name n+ n- value', got {len(parts)} field(s)", line_no, raw
            )
        name, n1, n2, value_tok = parts[0], parts[1], parts[2], parts[3]
        prefix = name[0].upper()
        cls = _ELEMENT_CLASSES.get(prefix)
        if cls is None:
            raise NetlistError(
                f"unknown element prefix {prefix!r} in {name!r} "
                "(supported in this phase: R, V, I)",
                line_no,
                raw,
            )
        value = parse_value(value_tok, line_no=line_no, line_text=raw)
        circuit.add(cls(name, n1, n2, value))

    return ParsedNetlist(circuit=circuit, directives=directives, title=title)


def parse_file(path) -> ParsedNetlist:
    with open(path) as f:
        return parse_source(f.read())
