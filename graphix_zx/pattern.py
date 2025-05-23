"""Pattern module.

This module provides:

- `ImmutablePattern`: Immutable pattern class
- `MutablePattern`: Mutable pattern class
- `is_runnable`: Check if the pattern is runnable
- `check_rule0`: Check if no command depends on an output not yet measured
- `check_rule1`: Check if no command acts on a qubit already measured
- `check_rule2`: Check if no command acts on a qubit not yet prepared, unless it is an input qubit
- `check_rule3`: Check if a qubit is measured if and only if it is not an output
- `print_pattern`: Print a pattern
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

from graphix_zx.command import Clifford, Command, D, E, M, N, X, Z

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from collections.abc import Set as AbstractSet


@dataclasses.dataclass(frozen=True)
class ImmutablePattern(Sequence[Command]):
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
        object.__setattr__(self, "input_node_indices", MappingProxyType(dict(self.input_node_indices)))
        object.__setattr__(self, "output_node_indices", MappingProxyType(dict(self.output_node_indices)))
        object.__setattr__(self, "commands", tuple(self.commands))

    def __len__(self) -> int:
        return len(self.commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    @typing.overload
    def __getitem__(self, index: int) -> Command: ...
    @typing.overload
    def __getitem__(self, index: slice) -> tuple[Command, ...]: ...
    def __getitem__(self, index: int | slice) -> Command | tuple[Command, ...]:
        if isinstance(index, slice):
            return tuple(self.commands[index])
        if isinstance(index, int):
            return self.commands[index]
        msg = f"Index must be int or slice, not {type(index)}"
        raise TypeError(msg)

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


class MutablePattern(Sequence[Command]):
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

    @typing.overload
    def __getitem__(self, index: int) -> Command: ...

    @typing.overload
    def __getitem__(self, index: slice) -> tuple[Command, ...]: ...

    def __getitem__(self, index: int | slice) -> Command | tuple[Command, ...]:
        if isinstance(index, slice):
            return tuple(self.__commands[index])
        if isinstance(index, int):
            return self.__commands[index]
        msg = f"Index must be int or slice, not {type(index)}"
        raise TypeError(msg)

    @property
    def commands(self) -> tuple[Command, ...]:
        r"""List of commands in the pattern.

        Returns
        -------
        `tuple`\[`Command`, ...\]
            List of commands in the pattern
        """
        return tuple(self.__commands)

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

    def _add(self, cmd: Command) -> None:
        if isinstance(cmd, N):
            if cmd.node in self._already_used_nodes:
                msg = f"The node {cmd.node} has already been used"
                raise ValueError(msg)
            self._already_used_nodes.add(cmd.node)
        self.__commands.append(cmd)

    def _invalidate_cache(self, name: str) -> None:
        self.__dict__.pop(name, None)

    def add(self, cmd: Command) -> None:
        """Add a command to the pattern.

        Parameters
        ----------
        cmd : `Command`
            Command to add to the pattern
        """
        self._add(cmd)
        self._invalidate_cache("max_space")
        self._invalidate_cache("space_list")

    def extend(self, cmds: Sequence[Command]) -> None:
        r"""Extend the pattern with a list of commands.

        Parameters
        ----------
        cmds : `collections.abc.Sequence`\[`Command`\]
            Commands to add to the pattern
        """
        for cmd in cmds:
            self._add(cmd)
        self._invalidate_cache("max_space")
        self._invalidate_cache("space_list")

    def freeze(self) -> ImmutablePattern:
        """Freeze the pattern.

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
        if isinstance(cmd, D) and any(c in measured for c in cmd.input_cbits):
            msg = f"The command depends on an output not yet measured: {cmd}"
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
    verror_msg = "The command acts on a qubit already measured: "
    for cmd in pattern:
        if isinstance(cmd, M):
            if cmd.node in measured:
                msg = verror_msg + f"{cmd}"
                raise ValueError(msg)
            measured.add(cmd.node)
        elif isinstance(cmd, E):
            if len(set(cmd.nodes) & measured) > 0:
                msg = verror_msg + f"{cmd}"
                raise ValueError(msg)
        elif isinstance(cmd, (N, X, Z, Clifford)):
            if cmd.node in measured:
                msg = verror_msg + f"{cmd}"
                raise ValueError(msg)
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
    verror_msg = "The command acts on a qubit not yet prepared: "
    for cmd in pattern:
        if isinstance(cmd, N):
            prepared.add(cmd.node)
        elif isinstance(cmd, E):
            if cmd.nodes[0] not in prepared or cmd.nodes[1] not in prepared:
                msg = verror_msg + f"{cmd}"
                raise ValueError(msg)
        elif isinstance(cmd, (M, X, Z, Clifford)) and cmd.node not in prepared:
            msg = verror_msg + f"{cmd}"
            raise ValueError(msg)


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
                msg = f"The command measures an output qubit: {cmd}"
                raise ValueError(msg)
            measured.add(cmd.node)
    if measured != non_output_nodes:
        msg = "Not all the non-output qubits are measured"
        raise ValueError(msg)


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
            print(cmd)  # noqa: T201
            print_count += 1

        if print_count >= nmax:
            print(  # noqa: T201
                f"{len(pattern) - i - 1} more commands truncated. Change lim argument of print_pattern() to show more"
            )
            break
