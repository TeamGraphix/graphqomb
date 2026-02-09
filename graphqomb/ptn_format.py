"""Pattern text format (.ptn) module.

This module provides:

- `dump`: Write a pattern to a .ptn file or string.
- `load`: Read a pattern from a .ptn file or string.
- `dumps`: Serialize a pattern to a .ptn format string.
- `loads`: Deserialize a pattern from a .ptn format string.
"""

from __future__ import annotations

import math
import operator
import re
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from graphqomb.command import TICK, Command, E, M, N, X, Z
from graphqomb.common import Plane, PlannerMeasBasis

if TYPE_CHECKING:
    from graphqomb.pattern import Pattern
    from graphqomb.pauli_frame import PauliFrame

PTN_VERSION = 1


def _format_angle(angle: float) -> str:
    """Format angle for output, using pi fractions where appropriate.

    Parameters
    ----------
    angle : `float`
        The angle in radians.

    Returns
    -------
    `str`
        Formatted angle string.
    """
    # Check for common pi fractions
    if math.isclose(angle, 0.0, abs_tol=1e-10):
        return "0"
    if math.isclose(angle, math.pi, rel_tol=1e-10):
        return "pi"
    if math.isclose(angle, -math.pi, rel_tol=1e-10):
        return "-pi"
    if math.isclose(angle, math.pi / 2, rel_tol=1e-10):
        return "pi/2"
    if math.isclose(angle, -math.pi / 2, rel_tol=1e-10):
        return "-pi/2"
    if math.isclose(angle, math.pi / 4, rel_tol=1e-10):
        return "pi/4"
    if math.isclose(angle, -math.pi / 4, rel_tol=1e-10):
        return "-pi/4"
    if math.isclose(angle, 3 * math.pi / 2, rel_tol=1e-10):
        return "3pi/2"
    if math.isclose(angle, 3 * math.pi / 4, rel_tol=1e-10):
        return "3pi/4"
    return f"{angle}"


def _parse_angle(s: str) -> float:
    """Parse angle string to float.

    Parameters
    ----------
    s : `str`
        Angle string (e.g., "0", "pi", "pi/2", "3pi/4", "1.5707963").

    Returns
    -------
    `float`
        The angle in radians.
    """
    s = s.strip()
    if s == "0":
        return 0.0
    if s == "pi":
        return math.pi
    if s == "-pi":
        return -math.pi
    if s == "pi/2":
        return math.pi / 2
    if s == "-pi/2":
        return -math.pi / 2
    if s == "pi/4":
        return math.pi / 4
    if s == "-pi/4":
        return -math.pi / 4
    if s == "3pi/2":
        return 3 * math.pi / 2
    if s == "3pi/4":
        return 3 * math.pi / 4

    # Try to parse as a general pi expression (e.g., "2pi/3")
    pi_match = re.match(r"^(-?\d*)pi(?:/(\d+))?$", s)
    if pi_match:
        numerator = pi_match.group(1)
        denominator = pi_match.group(2)
        num = int(numerator) if numerator and numerator != "-" else (1 if numerator != "-" else -1)
        denom = int(denominator) if denominator else 1
        return num * math.pi / denom

    return float(s)


def _format_coord(coord: tuple[float, ...]) -> str:
    """Format coordinate tuple for output.

    Parameters
    ----------
    coord : `tuple`[`float`, ...]
        Coordinate tuple (2D or 3D).

    Returns
    -------
    `str`
        Space-separated coordinate string.
    """
    return " ".join(str(c) for c in coord)


def _parse_coord(parts: list[str]) -> tuple[float, ...]:
    """Parse coordinate from string parts.

    Parameters
    ----------
    parts : `list`[`str`]
        List of coordinate value strings.

    Returns
    -------
    `tuple`[`float`, ...]
        Coordinate tuple.
    """
    return tuple(float(p) for p in parts)


def _write_header(
    out: StringIO,
    pattern: Pattern,
) -> None:
    """Write header section to output.

    Parameters
    ----------
    out : `StringIO`
        Output stream.
    pattern : `Pattern`
        The pattern to write.
    """
    out.write(f"# GraphQOMB Pattern Format v{PTN_VERSION}\n")
    out.write("\n")
    out.write("#======== HEADER ========\n")
    out.write(f".version {PTN_VERSION}\n")

    # Input nodes
    if pattern.input_node_indices:
        input_parts = [
            f"{node}:{qidx}" for node, qidx in sorted(pattern.input_node_indices.items(), key=operator.itemgetter(1))
        ]
        out.write(f".input {' '.join(input_parts)}\n")

    # Output nodes
    if pattern.output_node_indices:
        output_parts = [
            f"{node}:{qidx}" for node, qidx in sorted(pattern.output_node_indices.items(), key=operator.itemgetter(1))
        ]
        out.write(f".output {' '.join(output_parts)}\n")

    # Input coordinates
    for node, coord in sorted(pattern.input_coordinates.items()):
        out.write(f".coord {node} {_format_coord(coord)}\n")


def _write_quantum_section(
    out: StringIO,
    pattern: Pattern,
) -> None:
    """Write quantum instructions section to output.

    Parameters
    ----------
    out : `StringIO`
        Output stream.
    pattern : `Pattern`
        The pattern to write.
    """
    out.write("\n")
    out.write("#======== QUANTUM ========\n")

    # Group commands by timeslice
    timeslice = 0
    current_slice_commands: list[Command] = []

    def write_slice(slice_num: int, commands: list[Command]) -> None:
        if commands or slice_num == 0:
            out.write(f"[{slice_num}]\n")
            for cmd in commands:
                _write_command(out, cmd)

    for cmd in pattern.commands:
        if isinstance(cmd, TICK):
            write_slice(timeslice, current_slice_commands)
            current_slice_commands = []
            timeslice += 1
        else:
            current_slice_commands.append(cmd)

    # Write remaining commands in last slice
    if current_slice_commands or timeslice == 0:
        write_slice(timeslice, current_slice_commands)


def _write_command(out: StringIO, cmd: Command) -> None:
    """Write a single command to output.

    Parameters
    ----------
    out : `StringIO`
        Output stream.
    cmd : `Command`
        The command to write.
    """
    if isinstance(cmd, N):
        if cmd.coordinate is not None:
            out.write(f"N {cmd.node} {_format_coord(cmd.coordinate)}\n")
        else:
            out.write(f"N {cmd.node}\n")
    elif isinstance(cmd, E):
        out.write(f"E {cmd.nodes[0]} {cmd.nodes[1]}\n")
    elif isinstance(cmd, M):
        plane_name = cmd.meas_basis.plane.name
        angle_str = _format_angle(cmd.meas_basis.angle)
        out.write(f"M {cmd.node} {plane_name} {angle_str}\n")
    elif isinstance(cmd, X):
        out.write(f"X {cmd.node}\n")
    elif isinstance(cmd, Z):
        out.write(f"Z {cmd.node}\n")
    elif isinstance(cmd, TICK):
        pass  # TICK is handled by timeslice grouping


def _write_classical_section(
    out: StringIO,
    pauli_frame: PauliFrame,
) -> None:
    """Write classical frame section to output.

    Parameters
    ----------
    out : `StringIO`
        Output stream.
    pauli_frame : `PauliFrame`
        The Pauli frame to write.
    """
    out.write("\n")
    out.write("#======== CLASSICAL ========\n")

    # Write xflow
    for source, targets in sorted(pauli_frame.xflow.items()):
        if targets:
            targets_str = " ".join(str(t) for t in sorted(targets))
            out.write(f".xflow {source} -> {targets_str}\n")

    # Write zflow
    for source, targets in sorted(pauli_frame.zflow.items()):
        if targets:
            targets_str = " ".join(str(t) for t in sorted(targets))
            out.write(f".zflow {source} -> {targets_str}\n")

    # Write parity check groups (detectors)
    for group in pauli_frame.parity_check_group:
        if group:
            group_str = " ".join(str(n) for n in sorted(group))
            out.write(f".detector {group_str}\n")


def dumps(pattern: Pattern) -> str:
    """Serialize a pattern to a .ptn format string.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to serialize.

    Returns
    -------
    `str`
        The .ptn format string.
    """
    out = StringIO()
    _write_header(out, pattern)
    _write_quantum_section(out, pattern)
    _write_classical_section(out, pattern.pauli_frame)
    return out.getvalue()


def dump(pattern: Pattern, file: Path | str) -> None:
    """Write a pattern to a .ptn file.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to write.
    file : `Path` | `str`
        The file path to write to.
    """
    path = Path(file)
    path.write_text(dumps(pattern), encoding="utf-8")


def _parse_node_qubit_pairs(parts: list[str]) -> dict[int, int]:
    """Parse node:qubit pairs from string parts.

    Parameters
    ----------
    parts : `list`[`str`]
        List of "node:qubit" strings.

    Returns
    -------
    `dict`[`int`, `int`]
        Mapping from node to qubit index.
    """
    result: dict[int, int] = {}
    for part in parts:
        node_str, qidx_str = part.split(":")
        result[int(node_str)] = int(qidx_str)
    return result


def _parse_flow(line: str) -> tuple[int, set[int]]:
    """Parse a flow line (xflow or zflow).

    Parameters
    ----------
    line : `str`
        The flow line content after ".xflow" or ".zflow".

    Returns
    -------
    `tuple`[`int`, `set`[`int`]]
        Source node and set of target nodes.
    """
    parts = line.split("->")
    source = int(parts[0].strip())
    targets = {int(t) for t in parts[1].strip().split()}
    return source, targets


class PatternData:
    """Container for parsed pattern data from .ptn format.

    Attributes
    ----------
    input_node_indices : `dict`[`int`, `int`]
        Mapping from node to qubit index for input nodes.
    output_node_indices : `dict`[`int`, `int`]
        Mapping from node to qubit index for output nodes.
    input_coordinates : `dict`[`int`, `tuple`[`float`, ...]]
        Coordinates for input nodes.
    commands : `list`[`Command`]
        List of quantum commands.
    xflow : `dict`[`int`, `set`[`int`]]
        X correction flow mapping.
    zflow : `dict`[`int`, `set`[`int`]]
        Z correction flow mapping.
    parity_check_groups : `list`[`set`[`int`]]
        Parity check groups for error detection.
    """

    def __init__(self) -> None:
        self.input_node_indices: dict[int, int] = {}
        self.output_node_indices: dict[int, int] = {}
        self.input_coordinates: dict[int, tuple[float, ...]] = {}
        self.commands: list[Command] = []
        self.xflow: dict[int, set[int]] = {}
        self.zflow: dict[int, set[int]] = {}
        self.parity_check_groups: list[set[int]] = []


def loads(s: str) -> PatternData:
    """Deserialize a .ptn format string to pattern components.

    Parameters
    ----------
    s : `str`
        The .ptn format string.

    Returns
    -------
    `PatternData`
        Container with pattern components.

    Raises
    ------
    ValueError
        If the format is invalid or unsupported version.
    """
    result = PatternData()

    current_timeslice = -1
    version_found = False

    for line_num, line in enumerate(s.splitlines(), 1):
        # Remove inline comments
        if "#" in line:
            line = line[: line.index("#")]
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Parse directives
        if line.startswith("."):
            parts = line.split(maxsplit=1)
            directive = parts[0]
            content = parts[1] if len(parts) > 1 else ""

            if directive == ".version":
                version = int(content)
                if version != PTN_VERSION:
                    msg = f"Unsupported .ptn version: {version} (expected {PTN_VERSION})"
                    raise ValueError(msg)
                version_found = True

            elif directive == ".input":
                result.input_node_indices = _parse_node_qubit_pairs(content.split())

            elif directive == ".output":
                result.output_node_indices = _parse_node_qubit_pairs(content.split())

            elif directive == ".coord":
                coord_parts = content.split()
                node = int(coord_parts[0])
                coord = _parse_coord(coord_parts[1:])
                result.input_coordinates[node] = coord

            elif directive == ".xflow":
                source, targets = _parse_flow(content)
                result.xflow[source] = targets

            elif directive == ".zflow":
                source, targets = _parse_flow(content)
                result.zflow[source] = targets

            elif directive == ".detector":
                nodes = {int(n) for n in content.split()}
                result.parity_check_groups.append(nodes)

            elif directive == ".observable":
                # Observable parsing - store for future use
                pass

            continue

        # Parse timeslice header
        if line.startswith("[") and line.endswith("]"):
            slice_num = int(line[1:-1])
            # Add TICK commands for timeslice transitions
            while current_timeslice < slice_num - 1:
                result.commands.append(TICK())
                current_timeslice += 1
            if current_timeslice < slice_num:
                if current_timeslice >= 0:
                    result.commands.append(TICK())
                current_timeslice = slice_num
            continue

        # Parse commands
        parts = line.split()
        cmd_type = parts[0]

        if cmd_type == "N":
            node = int(parts[1])
            n_coord: tuple[float, ...] | None = _parse_coord(parts[2:]) if len(parts) > 2 else None
            result.commands.append(N(node=node, coordinate=n_coord))

        elif cmd_type == "E":
            node1 = int(parts[1])
            node2 = int(parts[2])
            result.commands.append(E(nodes=(node1, node2)))

        elif cmd_type == "M":
            node = int(parts[1])
            plane = Plane[parts[2]]
            angle = _parse_angle(parts[3])
            meas_basis = PlannerMeasBasis(plane, angle)
            result.commands.append(M(node=node, meas_basis=meas_basis))

        elif cmd_type == "X":
            node = int(parts[1])
            result.commands.append(X(node=node))

        elif cmd_type == "Z":
            node = int(parts[1])
            result.commands.append(Z(node=node))

        else:
            msg = f"Unknown command at line {line_num}: {cmd_type}"
            raise ValueError(msg)

    if not version_found:
        msg = "Missing .version directive"
        raise ValueError(msg)

    return result


def load(file: Path | str) -> PatternData:
    """Read pattern components from a .ptn file.

    Parameters
    ----------
    file : `Path` | `str`
        The file path to read from.

    Returns
    -------
    `PatternData`
        Container with pattern components.
        See `loads` for details.
    """
    path = Path(file)
    return loads(path.read_text(encoding="utf-8"))
