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
from graphqomb.pattern import (
    Pattern,
    _ensure_measurement_consistency,
    _ensure_no_operations_on_measured_qubits,
    _ensure_no_unprepared_qubit_operations,
)
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
_INT_PATTERN = re.compile(r"^-?\d+$")
_FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


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


def _parse_float(token: str, *, line_num: int | None = None, context: str = "float") -> float:
    """Parse a floating-point token with strict validation.

    Parameters
    ----------
    token : `str`
        Token to parse.
    line_num : `int` | `None`, optional
        Source line number for error reporting.
    context : `str`, optional
        Context string used in error messages.

    Returns
    -------
    `float`
        Parsed floating-point value.

    Raises
    ------
    ValueError
        If the token is not a valid decimal floating-point literal.
    """
    if not _FLOAT_PATTERN.fullmatch(token):
        where = f" at line {line_num}" if line_num is not None else ""
        msg = f"Invalid float for {context}{where}: {token!r}"
        raise ValueError(msg)
    return float(token)


def _parse_angle(s: str, *, line_num: int | None = None, context: str = "angle") -> float:
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

    return _parse_float(s, line_num=line_num, context=context)


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


def _parse_coord(parts: list[str], *, line_num: int | None = None, context: str = "coordinate") -> tuple[float, ...]:
    r"""Parse coordinate from string parts.

    Parameters
    ----------
    parts : `list`\[`str`\]
        List of coordinate value strings.
    line_num : `int` | `None`, optional
        Source line number for error reporting.
    context : `str`, optional
        Context string used in error messages.

    Returns
    -------
    `tuple`\[`float`, ...\]
        Coordinate tuple.

    Raises
    ------
    ValueError
        If coordinate dimensions are not 2D/3D or contain non-numeric values.
    """
    if len(parts) not in {2, 3}:
        where = f" at line {line_num}" if line_num is not None else ""
        msg = f"Invalid {context}{where}: expected 2D or 3D coordinates, got {len(parts)} values."
        raise ValueError(msg)
    return tuple(_parse_float(p, line_num=line_num, context=context) for p in parts)


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

    for logical_idx, seed_nodes in sorted(pauli_frame.logical_observables.items()):
        if seed_nodes:
            nodes_str = " ".join(str(n) for n in sorted(seed_nodes))
            out.write(f".observable {logical_idx} {nodes_str}\n")
        else:
            out.write(f".observable {logical_idx}\n")


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


def _parse_int(token: str, *, line_num: int, context: str) -> int:
    """Parse an integer token with contextualized error messages.

    Parameters
    ----------
    token : `str`
        Token to parse.
    line_num : `int`
        Source line number for error reporting.
    context : `str`
        Context string used in error messages.

    Returns
    -------
    `int`
        Parsed integer value.

    Raises
    ------
    ValueError
        If the token cannot be parsed as an integer.
    """
    if not _INT_PATTERN.fullmatch(token):
        msg = f"Invalid integer for {context} at line {line_num}: {token!r}"
        raise ValueError(msg)
    return int(token)


def _parse_node_qubit_pairs_strict(parts: list[str], *, line_num: int, directive: str) -> dict[int, int]:
    r"""Parse node:qubit pairs with strict duplicate validation.

    Parameters
    ----------
    parts : `list`\[`str`\]
        Tokenized entries of ``node:qindex`` pairs.
    line_num : `int`
        Source line number for error reporting.
    directive : `str`
        Directive label (e.g., ``.input`` or ``.output``).

    Returns
    -------
    `dict`\[`int`, `int`\]
        Mapping from node ID to logical qubit index.

    Raises
    ------
    ValueError
        If entries are malformed or contain duplicate nodes/qindices.
    """
    result: dict[int, int] = {}
    seen_qindices: set[int] = set()
    for part in parts:
        node_str, sep, qidx_str = part.partition(":")
        if sep != ":":
            msg = f"Invalid {directive} entry at line {line_num}: expected 'node:qindex', got {part!r}"
            raise ValueError(msg)
        node = _parse_int(node_str, line_num=line_num, context=f"{directive} node")
        qidx = _parse_int(qidx_str, line_num=line_num, context=f"{directive} qindex")
        if node in result:
            msg = f"Duplicate node in {directive} at line {line_num}: {node}"
            raise ValueError(msg)
        if qidx in seen_qindices:
            msg = f"Duplicate logical qubit index in {directive} at line {line_num}: {qidx}"
            raise ValueError(msg)
        result[node] = qidx
        seen_qindices.add(qidx)
    return result


def _parse_flow_strict(line: str, *, line_num: int, directive: str) -> tuple[int, set[int]]:
    r"""Parse a flow line (xflow or zflow) with strict syntax checks.

    Parameters
    ----------
    line : `str`
        Raw flow directive content.
    line_num : `int`
        Source line number for error reporting.
    directive : `str`
        Directive label (``.xflow`` or ``.zflow``).

    Returns
    -------
    `tuple`\[`int`, `set`\[`int`\]\]
        Source node and target-node set.

    Raises
    ------
    ValueError
        If syntax is invalid or source/targets cannot be parsed.
    """
    parts = line.split("->")
    if len(parts) != 2:  # noqa: PLR2004
        msg = f"Invalid {directive} syntax at line {line_num}: expected 'source -> target ...'"
        raise ValueError(msg)

    source = _parse_int(parts[0].strip(), line_num=line_num, context=f"{directive} source")
    target_tokens = parts[1].strip().split()
    if not target_tokens:
        msg = f"Invalid {directive} at line {line_num}: no targets specified."
        raise ValueError(msg)

    targets = {_parse_int(token, line_num=line_num, context=f"{directive} target") for token in target_tokens}
    return source, targets


def _parse_observable_strict(content: str, *, line_num: int) -> tuple[int, set[int]]:
    r"""Parse an observable directive with strict syntax checks.

    Parameters
    ----------
    content : `str`
        Raw ``.observable`` directive content.
    line_num : `int`
        Source line number for error reporting.

    Returns
    -------
    `tuple`\[`int`, `set`\[`int`\]\]
        Logical index and seed-node set.

    Raises
    ------
    ValueError
        If syntax is invalid or values cannot be parsed.
    """
    tokens = content.split()
    if not tokens:
        msg = f"Invalid .observable directive at line {line_num}: expected '.observable <index> [nodes ...]'"
        raise ValueError(msg)
    logical_idx = _parse_int(tokens[0], line_num=line_num, context=".observable index")
    nodes = {_parse_int(token, line_num=line_num, context=".observable node") for token in tokens[1:]}
    return logical_idx, nodes


def _validate_pattern_for_import(pattern: Pattern) -> None:
    """Run importer-time validations except DAG/schedule causality checks.

    Parameters
    ----------
    pattern : `Pattern`
        Reconstructed pattern from .ptn content.
    """
    _ensure_no_operations_on_measured_qubits(pattern)
    _ensure_no_unprepared_qubit_operations(pattern)
    _ensure_measurement_consistency(pattern)


class _ParsedPattern:
    """Container for parsed pattern components before semantic validation."""

    def __init__(self) -> None:
        self.input_node_indices: dict[int, int] = {}
        self.output_node_indices: dict[int, int] = {}
        self.input_coordinates: dict[int, tuple[float, ...]] = {}
        self.commands: list[Command] = []
        self.xflow: dict[int, set[int]] = {}
        self.zflow: dict[int, set[int]] = {}
        self.parity_check_groups: list[set[int]] = []
        self.logical_observables: dict[int, set[int]] = {}


class _Parser:
    """Internal parser state for loads()."""

    def __init__(self) -> None:
        self.result = _ParsedPattern()
        self.current_timeslice = -1
        self.version_found = False
        self._input_defined = False
        self._output_defined = False

    def parse(self, s: str) -> Pattern:
        r"""Parse the input string and return a validated pattern.

        Parameters
        ----------
        s : `str`
            The .ptn format string.

        Returns
        -------
        `Pattern`
            Reconstructed measurement pattern.

        Raises
        ------
        ValueError
            If the format is invalid or semantically inconsistent.
        """
        for line_num, raw_line in enumerate(s.splitlines(), 1):
            self._parse_line(line_num, raw_line)

        if not self.version_found:
            msg = "Missing .version directive"
            raise ValueError(msg)
        return self._build_pattern()

    def _parse_line(self, line_num: int, raw_line: str) -> None:
        """Parse a single line."""
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            return

        if line.startswith("."):
            self._parse_directive(line_num, line)
        elif line.startswith("[") and line.endswith("]"):
            self._parse_timeslice(line_num, line)
        else:
            self._parse_command(line_num, line)

    def _parse_directive(self, line_num: int, line: str) -> None:  # noqa: C901, PLR0912
        """Parse a directive line (starts with '.').

        Raises
        ------
        ValueError
            If the directive is unknown or malformed.
        """
        parts = line.split(maxsplit=1)
        directive = parts[0]
        content = parts[1] if len(parts) > 1 else ""

        if directive == ".version":
            self._handle_version(line_num, content)
        elif directive == ".input":
            self._handle_input(line_num, content)
        elif directive == ".output":
            self._handle_output(line_num, content)
        elif directive == ".coord":
            self._handle_coord(line_num, content)
        elif directive == ".xflow":
            source, targets = _parse_flow_strict(content, line_num=line_num, directive=directive)
            if source in self.result.xflow:
                msg = f"Duplicate {directive} source at line {line_num}: {source}"
                raise ValueError(msg)
            self.result.xflow[source] = targets
        elif directive == ".zflow":
            source, targets = _parse_flow_strict(content, line_num=line_num, directive=directive)
            if source in self.result.zflow:
                msg = f"Duplicate {directive} source at line {line_num}: {source}"
                raise ValueError(msg)
            self.result.zflow[source] = targets
        elif directive == ".detector":
            nodes = {
                _parse_int(token, line_num=line_num, context=".detector node") for token in content.split() if token
            }
            if not nodes:
                msg = f"Empty .detector directive at line {line_num}"
                raise ValueError(msg)
            self.result.parity_check_groups.append(nodes)
        elif directive == ".observable":
            logical_idx, seed_nodes = _parse_observable_strict(content, line_num=line_num)
            if logical_idx in self.result.logical_observables:
                msg = f"Duplicate .observable index at line {line_num}: {logical_idx}"
                raise ValueError(msg)
            self.result.logical_observables[logical_idx] = seed_nodes
        else:
            msg = f"Unknown directive at line {line_num}: {directive}"
            raise ValueError(msg)

    def _handle_version(self, line_num: int, content: str) -> None:
        r"""Handle .version directive.

        Raises
        ------
        ValueError
            If version is duplicated, malformed, or unsupported.
        """
        if self.version_found:
            msg = f"Duplicate .version directive at line {line_num}"
            raise ValueError(msg)
        version = _parse_int(content.strip(), line_num=line_num, context=".version")
        if version != PTN_VERSION:
            msg = f"Unsupported .ptn version: {version} (expected {PTN_VERSION})"
            raise ValueError(msg)
        self.version_found = True

    def _handle_input(self, line_num: int, content: str) -> None:
        """Handle .input directive.

        Raises
        ------
        ValueError
            If .input is duplicated or malformed.
        """
        if self._input_defined:
            msg = f"Duplicate .input directive at line {line_num}"
            raise ValueError(msg)
        self._input_defined = True
        self.result.input_node_indices = _parse_node_qubit_pairs_strict(
            content.split(),
            line_num=line_num,
            directive=".input",
        )

    def _handle_output(self, line_num: int, content: str) -> None:
        """Handle .output directive.

        Raises
        ------
        ValueError
            If .output is duplicated or malformed.
        """
        if self._output_defined:
            msg = f"Duplicate .output directive at line {line_num}"
            raise ValueError(msg)
        self._output_defined = True
        self.result.output_node_indices = _parse_node_qubit_pairs_strict(
            content.split(),
            line_num=line_num,
            directive=".output",
        )

    def _handle_coord(self, line_num: int, content: str) -> None:
        """Handle .coord directive.

        Raises
        ------
        ValueError
            If coordinate syntax is invalid or duplicated.
        """
        coord_parts = content.split()
        if len(coord_parts) < 3:  # noqa: PLR2004
            msg = f"Invalid .coord directive at line {line_num}: expected '.coord <node> <x> <y> [z]'"
            raise ValueError(msg)
        node = _parse_int(coord_parts[0], line_num=line_num, context=".coord node")
        if node in self.result.input_coordinates:
            msg = f"Duplicate .coord directive for node {node} at line {line_num}"
            raise ValueError(msg)
        coord = _parse_coord(coord_parts[1:], line_num=line_num, context=".coord")
        self.result.input_coordinates[node] = coord

    def _parse_timeslice(self, line_num: int, line: str) -> None:
        """Parse timeslice marker [n].

        Raises
        ------
        ValueError
            If the marker is malformed or timeslices go backwards.
        """
        slice_num = _parse_int(line[1:-1], line_num=line_num, context="timeslice index")
        if slice_num < self.current_timeslice:
            msg = f"Timeslice index moved backwards at line {line_num}: [{slice_num}]"
            raise ValueError(msg)
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
            If the command is malformed or appears before the first timeslice.
        """
        if self.current_timeslice < 0:
            msg = f"Command appeared before first timeslice at line {line_num}: {line}"
            raise ValueError(msg)

        parts = line.split()
        cmd_type = parts[0]

        if cmd_type == "N":
            self._parse_n_command(line_num, parts)
        elif cmd_type == "E":
            self._parse_e_command(line_num, parts)
        elif cmd_type == "M":
            self._parse_m_command(line_num, parts)
        elif cmd_type == "X":
            if len(parts) != 2:  # noqa: PLR2004
                msg = f"Invalid X command at line {line_num}: expected 'X <node>'"
                raise ValueError(msg)
            node = _parse_int(parts[1], line_num=line_num, context="X node")
            self.result.commands.append(X(node=node))
        elif cmd_type == "Z":
            if len(parts) != 2:  # noqa: PLR2004
                msg = f"Invalid Z command at line {line_num}: expected 'Z <node>'"
                raise ValueError(msg)
            node = _parse_int(parts[1], line_num=line_num, context="Z node")
            self.result.commands.append(Z(node=node))
        else:
            msg = f"Unknown command at line {line_num}: {cmd_type}"
            raise ValueError(msg)

    def _parse_n_command(self, line_num: int, parts: list[str]) -> None:
        """Parse N (node) command.

        Raises
        ------
        ValueError
            If the command arguments are malformed.
        """
        if len(parts) not in {2, 4, 5}:
            msg = f"Invalid N command at line {line_num}: expected 'N <node>' or 'N <node> <x> <y> [z]'"
            raise ValueError(msg)
        node = _parse_int(parts[1], line_num=line_num, context="N node")
        coord: tuple[float, ...] | None = None
        if len(parts) > 2:  # noqa: PLR2004
            coord = _parse_coord(parts[2:], line_num=line_num, context="N command coordinate")
        self.result.commands.append(N(node=node, coordinate=coord))

    def _parse_e_command(self, line_num: int, parts: list[str]) -> None:
        """Parse E (entangle) command.

        Raises
        ------
        ValueError
            If the command arguments are malformed.
        """
        if len(parts) != 3:  # noqa: PLR2004
            msg = f"Invalid E command at line {line_num}: expected 'E <node1> <node2>'"
            raise ValueError(msg)
        node1 = _parse_int(parts[1], line_num=line_num, context="E node1")
        node2 = _parse_int(parts[2], line_num=line_num, context="E node2")
        self.result.commands.append(E(nodes=(node1, node2)))

    def _parse_m_command(self, line_num: int, parts: list[str]) -> None:
        """Parse M (measure) command.

        Raises
        ------
        ValueError
            If the command arguments are malformed.
        """
        if len(parts) != 4:  # noqa: PLR2004
            msg = f"Invalid M command at line {line_num}: expected 'M <node> <plane|axis> <angle|+/->'"
            raise ValueError(msg)
        node = _parse_int(parts[1], line_num=line_num, context="M node")
        basis_spec = parts[2]
        meas_basis: MeasBasis

        if basis_spec in {"X", "Y", "Z"}:
            sign_str = parts[3]
            if sign_str not in {"+", "-"}:
                msg = f"Invalid Pauli sign at line {line_num}: {sign_str!r}"
                raise ValueError(msg)
            sign = Sign.PLUS if sign_str == "+" else Sign.MINUS
            axis = Axis[basis_spec]
            meas_basis = AxisMeasBasis(axis, sign)
        else:
            if basis_spec not in Plane.__members__:
                msg = f"Unknown measurement plane at line {line_num}: {basis_spec!r}"
                raise ValueError(msg)
            plane = Plane[basis_spec]
            angle = _parse_angle(parts[3], line_num=line_num, context="measurement angle")
            meas_basis = PlannerMeasBasis(plane, angle)

        self.result.commands.append(M(node=node, meas_basis=meas_basis))

    def _build_pattern(self) -> Pattern:  # noqa: C901, PLR0912, PLR0915, PLR0914
        """Build a validated Pattern object from parsed components.

        Returns
        -------
        `Pattern`
            Reconstructed and validated pattern.

        Raises
        ------
        ValueError
            If parsed components are semantically inconsistent.
        """
        input_nodes = self.result.input_node_indices
        output_nodes = self.result.output_node_indices

        unknown_coord_nodes = set(self.result.input_coordinates) - set(input_nodes)
        if unknown_coord_nodes:
            unknown = sorted(unknown_coord_nodes)
            msg = f".coord nodes must be declared in .input. Unknown nodes: {unknown}"
            raise ValueError(msg)

        all_nodes = set(input_nodes) | set(output_nodes)
        prepared_nodes = set(input_nodes)
        measured_nodes: set[int] = set()
        meas_bases: dict[int, MeasBasis] = {}

        for cmd in self.result.commands:
            if isinstance(cmd, N):
                if cmd.node in prepared_nodes:
                    msg = f"Node {cmd.node} is prepared more than once."
                    raise ValueError(msg)
                if cmd.node in measured_nodes:
                    msg = f"Node {cmd.node} is prepared after measurement."
                    raise ValueError(msg)
                prepared_nodes.add(cmd.node)
                all_nodes.add(cmd.node)
            elif isinstance(cmd, E):
                for node in cmd.nodes:
                    if node in measured_nodes:
                        msg = f"Entanglement operation targets measured node {node}."
                        raise ValueError(msg)
                    if node not in prepared_nodes:
                        msg = f"Entanglement operation targets unprepared node {node}."
                        raise ValueError(msg)
            elif isinstance(cmd, M):
                if cmd.node in measured_nodes:
                    msg = f"Node {cmd.node} is measured more than once."
                    raise ValueError(msg)
                if cmd.node not in prepared_nodes:
                    msg = f"Measurement operation targets unprepared node {cmd.node}."
                    raise ValueError(msg)
                measured_nodes.add(cmd.node)
                meas_bases[cmd.node] = cmd.meas_basis
            elif isinstance(cmd, (X, Z)):
                if cmd.node in measured_nodes:
                    msg = f"Correction operation targets measured node {cmd.node}."
                    raise ValueError(msg)
                if cmd.node not in prepared_nodes:
                    msg = f"Correction operation targets unprepared node {cmd.node}."
                    raise ValueError(msg)

        missing_output_nodes = set(output_nodes) - prepared_nodes
        if missing_output_nodes:
            missing = sorted(missing_output_nodes)
            msg = f"Output nodes must be prepared. Missing nodes: {missing}"
            raise ValueError(msg)

        for flow_name, flow in (("xflow", self.result.xflow), ("zflow", self.result.zflow)):
            for source, targets in flow.items():
                if source not in all_nodes:
                    msg = f"{flow_name} source node is unknown: {source}"
                    raise ValueError(msg)
                if source not in measured_nodes:
                    msg = f"{flow_name} source node must be measured: {source}"
                    raise ValueError(msg)
                unknown_targets = sorted(set(targets) - all_nodes)
                if unknown_targets:
                    msg = f"{flow_name} has unknown target nodes from source {source}: {unknown_targets}"
                    raise ValueError(msg)

        for group in self.result.parity_check_groups:
            unknown_nodes = sorted(group - measured_nodes)
            if unknown_nodes:
                msg = f".detector group contains non-measured nodes: {unknown_nodes}"
                raise ValueError(msg)
            non_pauli_nodes = sorted(node for node in group if determine_pauli_axis(meas_bases[node]) is None)
            if non_pauli_nodes:
                msg = f".detector group contains non-Pauli measurements: {non_pauli_nodes}"
                raise ValueError(msg)

        for logical_idx, seed_nodes in self.result.logical_observables.items():
            unknown_nodes = sorted(seed_nodes - measured_nodes)
            if unknown_nodes:
                msg = f".observable {logical_idx} contains non-measured nodes: {unknown_nodes}"
                raise ValueError(msg)
            non_pauli_nodes = sorted(node for node in seed_nodes if determine_pauli_axis(meas_bases[node]) is None)
            if non_pauli_nodes:
                msg = f".observable {logical_idx} contains non-Pauli measurements: {non_pauli_nodes}"
                raise ValueError(msg)

        pauli_frame = PauliFrame.from_nodes(
            nodes=all_nodes,
            meas_bases=meas_bases,
            xflow=self.result.xflow,
            zflow=self.result.zflow,
            parity_check_group=self.result.parity_check_groups,
            logical_observables=self.result.logical_observables,
        )

        pattern = Pattern(
            input_node_indices=input_nodes,
            output_node_indices=output_nodes,
            commands=tuple(self.result.commands),
            pauli_frame=pauli_frame,
            input_coordinates=self.result.input_coordinates,
        )
        _validate_pattern_for_import(pattern)
        return pattern


def loads(s: str) -> Pattern:
    """Deserialize a .ptn format string to a Pattern object.

    Parameters
    ----------
    s : `str`
        The .ptn format string.

    Returns
    -------
    `Pattern`
        Reconstructed pattern.
    """
    return _Parser().parse(s)


def load(file: Path | str) -> Pattern:
    """Read a Pattern object from a .ptn file.

    Parameters
    ----------
    file : `Path` | `str`
        The file path to read from.

    Returns
    -------
    `Pattern`
        Reconstructed pattern.
    """
    path = Path(file)
    return loads(path.read_text(encoding="utf-8"))
