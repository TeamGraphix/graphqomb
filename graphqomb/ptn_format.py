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
from graphqomb.common import (
    Axis,
    AxisMeasBasis,
    MeasBasis,
    Plane,
    PlannerMeasBasis,
    Sign,
    determine_pauli_axis,
    is_close_angle,
)

if TYPE_CHECKING:
    from graphqomb.pattern import Pattern
    from graphqomb.pauli_frame import PauliFrame

PTN_VERSION = 1

# Angle formatting/parsing lookup tables
_ANGLE_TO_STR: dict[float, str] = {
    0.0: "0",
    math.pi: "pi",
    -math.pi: "-pi",
    math.pi / 2: "pi/2",
    -math.pi / 2: "-pi/2",
    math.pi / 4: "pi/4",
    -math.pi / 4: "-pi/4",
    3 * math.pi / 2: "3pi/2",
    3 * math.pi / 4: "3pi/4",
}

_STR_TO_ANGLE: dict[str, float] = {
    "0": 0.0,
    "pi": math.pi,
    "-pi": -math.pi,
    "pi/2": math.pi / 2,
    "-pi/2": -math.pi / 2,
    "pi/4": math.pi / 4,
    "-pi/4": -math.pi / 4,
    "3pi/2": 3 * math.pi / 2,
    "3pi/4": 3 * math.pi / 4,
}

_PI_PATTERN = re.compile(r"^(-?\d*)pi(?:/(\d+))?$")


def _format_angle(angle: float) -> str:
    r"""Format angle for output, using pi fractions where appropriate.

    Parameters
    ----------
    angle : `float`
        The angle in radians.

    Returns
    -------
    `str`
        Formatted angle string.
    """
    for ref_angle, label in _ANGLE_TO_STR.items():
        tol = 1e-10 if ref_angle == 0.0 else None
        if tol is not None:
            if math.isclose(angle, ref_angle, abs_tol=tol):
                return label
        elif math.isclose(angle, ref_angle, rel_tol=1e-10):
            return label
    return f"{angle}"


def _parse_angle(s: str) -> float:
    r"""Parse angle string to float.

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
    if s in _STR_TO_ANGLE:
        return _STR_TO_ANGLE[s]

    pi_match = _PI_PATTERN.match(s)
    if pi_match:
        numerator = pi_match.group(1)
        denominator = pi_match.group(2)
        num = int(numerator) if numerator and numerator != "-" else (1 if numerator != "-" else -1)
        denom = int(denominator) if denominator else 1
        return num * math.pi / denom

    return float(s)


def _format_coord(coord: tuple[float, ...]) -> str:
    r"""Format coordinate tuple for output.

    Parameters
    ----------
    coord : `tuple`\[`float`, ...\]
        Coordinate tuple (2D or 3D).

    Returns
    -------
    `str`
        Space-separated coordinate string.
    """
    return " ".join(str(c) for c in coord)


def _parse_coord(parts: list[str]) -> tuple[float, ...]:
    r"""Parse coordinate from string parts.

    Parameters
    ----------
    parts : `list`\[`str`\]
        List of coordinate value strings.

    Returns
    -------
    `tuple`\[`float`, ...\]
        Coordinate tuple.
    """
    return tuple(float(p) for p in parts)


# ============================================================
# Serialization (dumps/dump)
# ============================================================


def _write_header(out: StringIO, pattern: Pattern) -> None:
    """Write header section to output."""
    out.write(f"# GraphQOMB Pattern Format v{PTN_VERSION}\n")
    out.write("\n")
    out.write("#======== HEADER ========\n")
    out.write(f".version {PTN_VERSION}\n")

    if pattern.input_node_indices:
        input_parts = [
            f"{node}:{qidx}" for node, qidx in sorted(pattern.input_node_indices.items(), key=operator.itemgetter(1))
        ]
        out.write(f".input {' '.join(input_parts)}\n")

    if pattern.output_node_indices:
        output_parts = [
            f"{node}:{qidx}" for node, qidx in sorted(pattern.output_node_indices.items(), key=operator.itemgetter(1))
        ]
        out.write(f".output {' '.join(output_parts)}\n")

    out.writelines(
        f".coord {node} {_format_coord(coord)}\n" for node, coord in sorted(pattern.input_coordinates.items())
    )


def _write_command(out: StringIO, cmd: Command) -> None:
    """Write a single command to output."""
    if isinstance(cmd, N):
        if cmd.coordinate is not None:
            out.write(f"N {cmd.node} {_format_coord(cmd.coordinate)}\n")
        else:
            out.write(f"N {cmd.node}\n")
    elif isinstance(cmd, E):
        out.write(f"E {cmd.nodes[0]} {cmd.nodes[1]}\n")
    elif isinstance(cmd, M):
        _write_measurement(out, cmd)
    elif isinstance(cmd, X):
        out.write(f"X {cmd.node}\n")
    elif isinstance(cmd, Z):
        out.write(f"Z {cmd.node}\n")


def _write_measurement(out: StringIO, cmd: M) -> None:
    """Write measurement command with appropriate format."""
    pauli_axis = determine_pauli_axis(cmd.meas_basis)
    if pauli_axis is not None:
        angle = cmd.meas_basis.angle
        if pauli_axis == Axis.Y:
            sign = "+" if is_close_angle(angle, math.pi / 2) else "-"
        else:
            sign = "+" if is_close_angle(angle, 0.0) else "-"
        out.write(f"M {cmd.node} {pauli_axis.name} {sign}\n")
    else:
        plane_name = cmd.meas_basis.plane.name
        angle_str = _format_angle(cmd.meas_basis.angle)
        out.write(f"M {cmd.node} {plane_name} {angle_str}\n")


def _write_quantum_section(out: StringIO, pattern: Pattern) -> None:
    """Write quantum instructions section to output."""
    out.write("\n")
    out.write("#======== QUANTUM ========\n")

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

    if current_slice_commands or timeslice == 0:
        write_slice(timeslice, current_slice_commands)


def _write_classical_section(out: StringIO, pauli_frame: PauliFrame) -> None:
    """Write classical frame section to output."""
    out.write("\n")
    out.write("#======== CLASSICAL ========\n")

    for source, targets in sorted(pauli_frame.xflow.items()):
        if targets:
            targets_str = " ".join(str(t) for t in sorted(targets))
            out.write(f".xflow {source} -> {targets_str}\n")

    for source, targets in sorted(pauli_frame.zflow.items()):
        if targets:
            targets_str = " ".join(str(t) for t in sorted(targets))
            out.write(f".zflow {source} -> {targets_str}\n")

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


# ============================================================
# Deserialization (loads/load)
# ============================================================


def _parse_node_qubit_pairs(parts: list[str]) -> dict[int, int]:
    r"""Parse node:qubit pairs from string parts.

    Parameters
    ----------
    parts : `list`\[`str`\]
        List of "node:qubit" strings.

    Returns
    -------
    `dict`\[`int`, `int`\]
        Mapping from node to qubit index.
    """
    result: dict[int, int] = {}
    for part in parts:
        node_str, qidx_str = part.split(":")
        result[int(node_str)] = int(qidx_str)
    return result


def _parse_flow(line: str) -> tuple[int, set[int]]:
    r"""Parse a flow line (xflow or zflow).

    Parameters
    ----------
    line : `str`
        The flow line content after ".xflow" or ".zflow".

    Returns
    -------
    `tuple`\[`int`, `set`\[`int`\]\]
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


class _Parser:
    """Internal parser state for loads()."""

    def __init__(self) -> None:
        self.result = PatternData()
        self.current_timeslice = -1
        self.version_found = False

    def parse(self, s: str) -> PatternData:
        r"""Parse the input string and return PatternData.

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
        for line_num, raw_line in enumerate(s.splitlines(), 1):
            self._parse_line(line_num, raw_line)

        if not self.version_found:
            msg = "Missing .version directive"
            raise ValueError(msg)

        return self.result

    def _parse_line(self, line_num: int, raw_line: str) -> None:
        """Parse a single line."""
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            return

        if line.startswith("."):
            self._parse_directive(line)
        elif line.startswith("[") and line.endswith("]"):
            self._parse_timeslice(line)
        else:
            self._parse_command(line_num, line)

    def _parse_directive(self, line: str) -> None:
        """Parse a directive line (starts with '.')."""
        parts = line.split(maxsplit=1)
        directive = parts[0]
        content = parts[1] if len(parts) > 1 else ""

        if directive == ".version":
            self._handle_version(content)
        elif directive == ".input":
            self.result.input_node_indices = _parse_node_qubit_pairs(content.split())
        elif directive == ".output":
            self.result.output_node_indices = _parse_node_qubit_pairs(content.split())
        elif directive == ".coord":
            self._handle_coord(content)
        elif directive == ".xflow":
            source, targets = _parse_flow(content)
            self.result.xflow[source] = targets
        elif directive == ".zflow":
            source, targets = _parse_flow(content)
            self.result.zflow[source] = targets
        elif directive == ".detector":
            nodes = {int(n) for n in content.split()}
            self.result.parity_check_groups.append(nodes)

    def _handle_version(self, content: str) -> None:
        r"""Handle .version directive.

        Raises
        ------
        ValueError
            If the version is unsupported.
        """
        version = int(content)
        if version != PTN_VERSION:
            msg = f"Unsupported .ptn version: {version} (expected {PTN_VERSION})"
            raise ValueError(msg)
        self.version_found = True

    def _handle_coord(self, content: str) -> None:
        """Handle .coord directive."""
        coord_parts = content.split()
        node = int(coord_parts[0])
        coord = _parse_coord(coord_parts[1:])
        self.result.input_coordinates[node] = coord

    def _parse_timeslice(self, line: str) -> None:
        """Parse timeslice marker [n]."""
        slice_num = int(line[1:-1])
        while self.current_timeslice < slice_num - 1:
            self.result.commands.append(TICK())
            self.current_timeslice += 1
        if self.current_timeslice < slice_num:
            if self.current_timeslice >= 0:
                self.result.commands.append(TICK())
            self.current_timeslice = slice_num

    def _parse_command(self, line_num: int, line: str) -> None:
        r"""Parse a command line.

        Raises
        ------
        ValueError
            If the command type is unknown.
        """
        parts = line.split()
        cmd_type = parts[0]

        if cmd_type == "N":
            self._parse_n_command(parts)
        elif cmd_type == "E":
            self._parse_e_command(parts)
        elif cmd_type == "M":
            self._parse_m_command(parts)
        elif cmd_type == "X":
            self.result.commands.append(X(node=int(parts[1])))
        elif cmd_type == "Z":
            self.result.commands.append(Z(node=int(parts[1])))
        else:
            msg = f"Unknown command at line {line_num}: {cmd_type}"
            raise ValueError(msg)

    def _parse_n_command(self, parts: list[str]) -> None:
        """Parse N (node) command."""
        node = int(parts[1])
        coord: tuple[float, ...] | None = _parse_coord(parts[2:]) if len(parts) > 2 else None  # noqa: PLR2004
        self.result.commands.append(N(node=node, coordinate=coord))

    def _parse_e_command(self, parts: list[str]) -> None:
        """Parse E (entangle) command."""
        node1 = int(parts[1])
        node2 = int(parts[2])
        self.result.commands.append(E(nodes=(node1, node2)))

    def _parse_m_command(self, parts: list[str]) -> None:
        """Parse M (measure) command."""
        node = int(parts[1])
        basis_spec = parts[2]
        meas_basis: MeasBasis

        if basis_spec in {"X", "Y", "Z"}:
            sign_str = parts[3]
            sign = Sign.PLUS if sign_str == "+" else Sign.MINUS
            axis = Axis[basis_spec]
            meas_basis = AxisMeasBasis(axis, sign)
        else:
            plane = Plane[basis_spec]
            angle = _parse_angle(parts[3])
            meas_basis = PlannerMeasBasis(plane, angle)

        self.result.commands.append(M(node=node, meas_basis=meas_basis))


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

    See Also
    --------
    _Parser.parse : Internal parser that may raise ValueError for invalid input.
    """
    return _Parser().parse(s)


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
