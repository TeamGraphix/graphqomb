"""Pattern module.

This module provides:

- `Pattern`: Pattern class
- `is_runnable`: Check if the pattern is runnable
- `print_pattern`: Print a pattern
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING

from graphix_zx.command import Clifford, Command, D, E, M, N, X, Z

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Callable


@dataclasses.dataclass(frozen=True)
class Pattern(Sequence[Command]):
    r"""Pattern class.

    Attributes
    ----------
    input_node_indices : `dict`\[`int`, `int`\]
        The map of input nodes to their logical qubit indices
    output_node_indices : `dict`\[`int`, `int`\]
        The map of output nodes to their logical qubit indices
    commands : `tuple`\[`Command`, ...\]
        Commands of the pattern
    """

    input_node_indices: dict[int, int]
    output_node_indices: dict[int, int]
    commands: tuple[Command, ...]

    def __len__(self) -> int:
        return len(self.commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    @typing.overload
    def __getitem__(self, index: int) -> Command: ...
    @typing.overload
    def __getitem__(self, index: slice) -> tuple[Command, ...]: ...
    def __getitem__(self, index: int | slice) -> Command | tuple[Command, ...]:
        return self.commands[index]

    @functools.cached_property
    def max_space(self) -> int:
        """Maximum number of qubits prepared at any point in the pattern.

        Returns
        -------
        `int`
            Maximum number of qubits prepared at any point in the pattern
        """
        return max(self.space)

    @functools.cached_property
    def space(self) -> list[int]:
        r"""List of qubits prepared at each point in the pattern.

        Returns
        -------
        `list`\[`int`\]
            List of qubits prepared at each point in the pattern
        """
        nodes = len(self.input_node_indices)
        space_list = [nodes]
        for cmd in self.commands:
            if isinstance(cmd, N):
                nodes += 1
                space_list.append(nodes)
            elif isinstance(cmd, M):
                nodes -= 1
                space_list.append(nodes)
        return space_list


def is_runnable(pattern: Pattern) -> None:
    """Check if the pattern is runnable.

    Parameters
    ----------
    pattern : `Pattern`
        Pattern to check
    """
    _ensure_no_unmeasured_output_dependencies(pattern)
    _ensure_no_operations_on_measured_qubits(pattern)
    _ensure_no_unprepared_qubit_operations(pattern)
    _ensure_measurement_consistency(pattern)


def _ensure_no_unmeasured_output_dependencies(pattern: Pattern) -> None:
    """Ensure that no command depends on an output that has not yet been measured.

    Parameters
    ----------
    pattern : `Pattern`
        The sequence of commands to validate.

    Raises
    ------
    ValueError
        If any command depends on an output that has not yet been measured.
    """
    measured: set[int] = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            measured.add(cmd.node)
        if isinstance(cmd, D) and any(c not in measured for c in cmd.input_cbits):
            msg = f"D command depends on an output that hasn't been measured yet: {cmd}"
            raise ValueError(msg)


def _ensure_no_operations_on_measured_qubits(pattern: Pattern) -> None:
    """Ensure that no command operates on a qubit that has already been measured.

    Parameters
    ----------
    pattern : `Pattern`
        The sequence of commands to validate.

    Raises
    ------
    ValueError
        If any command attempts to act on a qubit that has already been measured.
    TypeError
        If an unknown command type is encountered.
    """
    measured = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            if cmd.node in measured:
                msg = f"A measurement is repeated on the same qubit: {cmd}"
                raise ValueError(msg)
            measured.add(cmd.node)
        elif isinstance(cmd, E):
            if not set(cmd.nodes).isdisjoint(measured):
                msg = f"Entanglement operation targets a measured qubit: {cmd}"
                raise ValueError(msg)
        elif isinstance(cmd, (N, X, Z, Clifford)):
            if cmd.node in measured:
                msg = f"Operation on a measured qubit: {cmd}"
                raise ValueError(msg)
        else:
            msg = f"Unknown command kind: {type(cmd)}"
            raise TypeError(msg)


def _ensure_no_unprepared_qubit_operations(pattern: Pattern) -> None:
    """Ensure that no command operates on a qubit that hasn't been prepared yet, except for input qubits.

    Parameters
    ----------
    pattern : `Pattern`
        The sequence of commands to validate.

    Raises
    ------
    ValueError
        If any command targets a qubit that hasn't been prepared.
    """
    prepared = set(pattern.input_node_indices)
    for cmd in pattern:
        if isinstance(cmd, N):
            prepared.add(cmd.node)
        elif isinstance(cmd, E):
            if cmd.nodes[0] not in prepared or cmd.nodes[1] not in prepared:
                msg = f"Entanglement operation targets a qubit that hasn't been prepared yet: {cmd}"
                raise ValueError(msg)
        elif isinstance(cmd, (M, X, Z, Clifford)) and cmd.node not in prepared:
            msg = f"Operation on a qubit that hasn't been prepared yet: {cmd}"
            raise ValueError(msg)


def _ensure_measurement_consistency(pattern: Pattern) -> None:
    """Ensure that measurements are applied exactly to all non-output qubits and that no output qubit is ever measured.

    Parameters
    ----------
    pattern : `Pattern`
        The sequence of commands to validate.

    Raises
    ------
    ValueError
        If a measurement targets an output qubit, or if some non-output qubits are never measured.
    """
    output_nodes = set(pattern.output_node_indices)
    non_output_nodes = {cmd.node for cmd in pattern if isinstance(cmd, N)} | set(
        pattern.input_node_indices
    ) - output_nodes
    measured = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            if cmd.node in output_nodes:
                msg = f"The command measures an output qubit: {cmd}"
                raise ValueError(msg)
            measured.add(cmd.node)
    if measured != non_output_nodes:
        missing = non_output_nodes - measured
        msg = f"Missing measurements on qubit(s): {sorted(missing)}"
        raise ValueError(msg)


def print_pattern(
    pattern: Pattern,
    *,
    file: typing.TextIO | None = None,
    lim: int = 40,
    cmd_filter: Callable[[Command], Command | None] | None = None,
) -> None:
    r"""Print a pattern.

    Parameters
    ----------
    pattern : `Pattern`
        Pattern to print
    file : `typing.TextIO`, optional
        File to print to, by default None (prints to stdout)
    lim : `int`, optional
        Maximum number of commands to print, by default 40
    cmd_filter : `typing.Callable`\[\[`Command`\], `Command` | `None`\] | `None`, optional
        Command filter, by default None
    """
    if cmd_filter is None:

        def cmd_filter(cmd: Command) -> Command | None:
            return cmd if isinstance(cmd, (N, E, M, X, Z, D, Clifford)) else None

    nmax = min(lim, len(pattern))
    print_count = 0
    for i, cmd in enumerate(pattern):
        cmd_filtered = cmd_filter(cmd)
        if cmd_filtered is None:
            continue
        print(cmd_filtered, file=file)
        print_count += 1

        if print_count >= nmax:
            print(
                f"{len(pattern) - i - 1} more commands truncated. Change lim argument of print_pattern() to show more",
                file=file,
            )
            break
