"""Pattern module.

This module provides:

- `ImmutablePattern`: Immutable pattern class
- `MutablePattern`: Mutable pattern class
- `is_runnable`: Check if the pattern is runnable
- `check_rule0`: Check if no command depends on an output not yet measured
- `check_rule1`: Check if no command acts on a qubit already measured
- `check_rule2`: Check if no command acts on a qubit not yet prepared, unless it is an input qubit
- `check_rule3`: Check if a qubit is measured if and only if it is not an output
- `print_command`: Print a command
- `print_pattern`: Print a pattern
"""

from __future__ import annotations

import dataclasses
import functools
from types import MappingProxyType
from typing import TYPE_CHECKING

from graphix_zx.command import Clifford, Command, D, E, M, N, X, Z

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from collections.abc import Set as AbstractSet


@dataclasses.dataclass(frozen=True)
class ImmutablePattern:
    r"""Immutable pattern class.

    Attributes
    ----------
    input_node_indices : `collections.abc.Mapping`\[`int`, `int`\]
        The map of input nodes to their logical qubit indices
    output_node_indices : `collections.abc.Mapping`\[`int`, `int`\]
        The map of output nodes to their logical qubit indices
    commands : `collections.abc.Sequence`\[`Command`\]
        Commands of the pattern
    """

    input_node_indices: Mapping[int, int]
    output_node_indices: Mapping[int, int]
    commands: Sequence[Command]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_node_indices", MappingProxyType(self.input_node_indices))
        object.__setattr__(self, "output_node_indices", MappingProxyType(self.output_node_indices))
        object.__setattr__(self, "commands", tuple(self.input_node_indices))

    def __len__(self) -> int:
        return len(self.commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    @functools.cached_property
    def max_space(self) -> int:
        """Maximum number of qubits prepared at any point in the pattern.

        Returns
        -------
        `int`
            Maximum number of qubits prepared at any point in the pattern
        """
        return max(self.space_list)

    @functools.cached_property
    def space_list(self) -> list[int]:
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


class MutablePattern:
    r"""Mutable pattern class.

    Attributes
    ----------
    input_node_indices : `dict`\[`int`, `int`\]
        The map of input nodes to their logical qubit indices
    output_node_indices : `dict`\[`int`, `int`\]
        The map of output nodes to their logical qubit indices
    """

    def __init__(
        self,
        input_node_indices: Mapping[int, int] | None = None,
        output_node_indices: Mapping[int, int] | None = None,
    ) -> None:
        if input_node_indices is None:
            input_node_indices = {}
        if output_node_indices is None:
            output_node_indices = {}
        self.input_node_indices: dict[int, int] = dict(input_node_indices)  # input node indices
        self.output_node_indices: dict[int, int] = dict(output_node_indices)  # output node indices

        self.__commands: list[Command] = []

        self._already_used_nodes: set[int] = set(self.input_node_indices)  # nodes already used

    def __len__(self) -> int:
        return len(self.__commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.__commands)

    def add(self, cmd: Command) -> None:
        """Add a command to the pattern.

        Parameters
        ----------
        cmd : `Command`
            Command to add to the pattern

        Raises
        ------
        ValueError
            If the node has already been used
        """
        if isinstance(cmd, N):
            if cmd.node in self._already_used_nodes:
                msg = f"The node {cmd.node} has already been used"
                raise ValueError(msg)
            self._already_used_nodes.add(cmd.node)
        self.__commands.append(cmd)

    def extend(self, cmds: Iterable[Command]) -> None:
        r"""Extend the pattern with a list of commands.

        Parameters
        ----------
        cmds : `collections.abc.Iterable`\[`Command`\]
            List of commands to add to the pattern
        """
        for cmd in cmds:
            self.add(cmd)

    @property
    def commands(self) -> list[Command]:
        r"""Commands of the pattern.

        Returns
        -------
        `list`\[`Command`\]
            List of commands of the pattern
        """
        return self.__commands

    @property
    def max_space(self) -> int:
        """Maximum number of qubits prepared at any point in the pattern.

        Returns
        -------
        `int`
            Maximum number of qubits prepared at any point in the pattern
        """
        return max(self.space_list)

    @property
    def space_list(self) -> list[int]:
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

    def freeze(self) -> ImmutablePattern:
        """Immutarize the pattern.

        Returns
        -------
        `ImmutablePattern`
            Immutable pattern
        """
        return ImmutablePattern(
            input_node_indices=self.input_node_indices,
            output_node_indices=self.output_node_indices,
            commands=self.commands,
        )


def is_runnable(pattern: MutablePattern | ImmutablePattern) -> bool:
    """Check if the pattern is runnable.

    Parameters
    ----------
    pattern : `MutablePattern` | `ImmutablePattern`
        Pattern to check

    Returns
    -------
    `bool`
        True if the pattern is runnable
    """
    check_rule0(pattern)
    check_rule1(pattern)
    check_rule2(pattern)
    check_rule3(pattern)
    return True


def check_rule0(pattern: MutablePattern | ImmutablePattern) -> None:
    """Check if no command depends on an output not yet measured.

    Parameters
    ----------
    pattern : `MutablePattern` | `ImmutablePattern`
        Pattern to check

    Raises
    ------
    ValueError
        If the command depends on an output not yet measured
    """
    measured: set[int] = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            measured.add(cmd.node)
        if isinstance(cmd, D) and len(set(cmd.input_cbits) & measured) > 0:
            msg = "The above command depends on an unmeasured qubit(s)"
            print_command(cmd)
            raise ValueError(msg)


def check_rule1(pattern: MutablePattern | ImmutablePattern) -> None:
    """Check if no command acts on a qubit already measured.

    Parameters
    ----------
    pattern : `MutablePattern` | `ImmutablePattern`
        Pattern to check

    Raises
    ------
    ValueError
        If the command acts on a qubit already measured
    TypeError
        If the command kind is unknown
    """
    measured = set()
    verror_msg = "The above command acts on a qubit already measured"
    for cmd in pattern:
        if isinstance(cmd, M):
            if cmd.node in measured:
                print_command(cmd)
                raise ValueError(verror_msg)
            measured.add(cmd.node)
        elif isinstance(cmd, E):
            if len(set(cmd.nodes) & measured) > 0:
                print_command(cmd)
                raise ValueError(verror_msg)
        elif isinstance(cmd, (N, X, Z, Clifford)):
            if cmd.node in measured:
                print_command(cmd)
                raise ValueError(verror_msg)
        else:
            msg = f"Unknown command kind: {type(cmd)}"
            raise TypeError(msg)


def check_rule2(pattern: MutablePattern | ImmutablePattern) -> None:
    """Check if no command acts on a qubit not yet prepared, unless it is an input qubit.

    Parameters
    ----------
    pattern : `MutablePattern` | `ImmutablePattern`
        Pattern to check

    Raises
    ------
    ValueError
        If the command acts on a qubit not yet prepared
    """
    prepared = set(pattern.input_node_indices)
    verror_msg = "The above command acts on a qubit not yet prepared"
    for cmd in pattern:
        if isinstance(cmd, N):
            prepared.add(cmd.node)
        elif isinstance(cmd, E):
            if cmd.nodes[0] not in prepared or cmd.nodes[1] not in prepared:
                print_command(cmd)
                raise ValueError(verror_msg)
        elif isinstance(cmd, (M, X, Z, Clifford)) and cmd.node not in prepared:
            print_command(cmd)
            raise ValueError(verror_msg)


def check_rule3(pattern: MutablePattern | ImmutablePattern) -> None:
    """Check if a qubit is measured if and only if it is not an output.

    Parameters
    ----------
    pattern : `MutablePattern` | `ImmutablePattern`
        Pattern to check

    Raises
    ------
    ValueError
        1. If the command measures an output qubit
        2. If not all the non-output qubits are measured
    """
    output_nodes = set(pattern.output_node_indices)
    # if not all(not (isinstance(cmd, M) and cmd.node in output_nodes) for cmd in pattern)
    non_output_nodes = {cmd.node for cmd in pattern if isinstance(cmd, N)} | set(
        pattern.input_node_indices
    ) - output_nodes
    measured = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            if cmd.node in output_nodes:
                msg = "The above command measures an output qubit"
                print_command(cmd)
                raise ValueError(msg)
            measured.add(cmd.node)
    if measured != non_output_nodes:
        msg = "Not all the non-output qubits are measured"
        raise ValueError(msg)


def print_command(cmd: Command) -> None:
    """Print a command.

    Parameters
    ----------
    cmd : `Command`
        Command to print
    """
    if isinstance(cmd, N):
        print(f"N, node = {cmd.node}")  # noqa: T201
    elif isinstance(cmd, E):
        print(f"E, nodes = {cmd.nodes}")  # noqa: T201
    elif isinstance(cmd, M):
        print(  # noqa: T201
            f"M, node = {cmd.node}",
            f"plane = {cmd.meas_basis.plane}",
            f"angle = {cmd.meas_basis.angle}",
            f"s-domain = {cmd.s_cbit}",
            f"t-domain = {cmd.t_cbit}",
        )
    elif isinstance(cmd, X):
        print(f"X, node = {cmd.node}, domain = {cmd.cbit}")  # noqa: T201
    elif isinstance(cmd, Z):
        print(f"Z, node = {cmd.node}, domain = {cmd.cbit}")  # noqa: T201
    elif isinstance(cmd, D):
        print(f"D, input_cbits = {cmd.input_cbits}, output_cbits = {cmd.output_cbits}, backend = {cmd.decoder}")  # noqa: T201
    elif isinstance(cmd, Clifford):
        print(f"Clifford, node = {cmd.node}")  # noqa: T201
        cmd.local_clifford.print_angles()
    else:
        print(f"Unknown command: {cmd}")  # noqa: T201


def print_pattern(
    pattern: MutablePattern | ImmutablePattern, lim: int = 40, cmd_filter: AbstractSet[type[Command]] | None = None
) -> None:
    r"""Print a pattern.

    Parameters
    ----------
    pattern : `MutablePattern` | `ImmutablePattern`
        Pattern to print
    lim : `int`, optional
        Maximum number of commands to print, by default 40
    cmd_filter : `collections.abc.Set`\[`type`\[`Command`\]\] | None, optional
        Command filter, by default None
    """
    if cmd_filter is None:
        cmd_filter = {
            N,
            E,
            M,
            X,
            Z,
            D,
            Clifford,
        }
    nmax = min(lim, len(pattern))
    print_count = 0
    for i, cmd in enumerate(pattern):
        if type(cmd) in cmd_filter:
            print_command(cmd)
            print_count += 1

        if print_count > nmax:
            print(  # noqa: T201
                f"{len(pattern) - i - 1} more commands truncated. Change lim argument of print_pattern() to show more"
            )
            break
