"""Command-line entry point: ``python -m mna circuit.cir``.

Phase 2 only computes the DC operating point (that's all the parsed element
set -- R/V/I -- can produce anyway). If a netlist requests an analysis that
isn't implemented yet, this prints a clear note pointing at the phase that
adds it and falls back to the operating point, rather than crashing or
silently ignoring the directive.
"""

import argparse
import sys

from .matrix import CircuitError
from .netlist import NetlistError, parse_file

_PHASE_FOR_DIRECTIVE = {".dc": 3, ".ac": 4, ".tran": 5}


def _node_sort_key(name: str):
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def _format_value(value: float) -> str:
    return f"{value:.6g}"


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mna",
        description="Simple SPICE-style DC operating-point solver",
    )
    parser.add_argument("netlist", help="path to a .cir netlist file")
    args = parser.parse_args(argv)

    try:
        parsed = parse_file(args.netlist)
    except NetlistError as exc:
        print(f"netlist error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for directive in parsed.directives:
        phase = _PHASE_FOR_DIRECTIVE.get(directive.name)
        if phase is not None:
            print(
                f"note: {directive.name} requested (line {directive.line_no}) but "
                f"that analysis isn't implemented until Phase {phase}; "
                "running the DC operating point instead.",
                file=sys.stderr,
            )

    print(f"* {parsed.title}")

    try:
        voltages, currents = parsed.circuit.solve()
    except CircuitError as exc:
        print(f"circuit error: {exc}", file=sys.stderr)
        return 1

    print(f"{'node':<10}{'voltage (V)':>15}")
    for name in sorted(voltages, key=_node_sort_key):
        print(f"{name:<10}{_format_value(voltages[name]):>15}")

    if currents:
        print()
        print(f"{'source':<10}{'current (A)':>15}")
        for name in sorted(currents):
            print(f"{name:<10}{_format_value(currents[name]):>15}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
