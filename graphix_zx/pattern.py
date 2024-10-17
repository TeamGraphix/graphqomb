"""Pattern module.

This module provides:
- ImmutablePattern: Immutable pattern class
- BasePattern: Base pattern class
- MutablePattern: Mutable pattern class
- NodeAlreadyPreparedError: Raised when a node is prepared more than once
- is_standardized: Check if the pattern is standardized
- is_runnable: Check if the pattern is runnable
- print_command: Print a command
- print_pattern: Print a pattern
"""

from __future__ import annotations

import dataclasses
import typing
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

from graphix_zx.command import C, Command, E, M, N, X, Z

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from collections.abc import Set as AbstractSet

    from graphix_zx.common import Plane


@dataclasses.dataclass(frozen=True)
class ImmutablePattern:
    """Immutable pattern class.

    Attributes
    ----------
    input_nodes : set[int]
        Input nodes of the pattern
    output_nodes : set[int]
        Output nodes of the pattern
    q_indices : dict[int, int]
        Logical qubit indices map of the pattern
    commands : list[Command]
        Commands of the pattern
    runnable : bool
        True if the pattern is runnable
    deterministic : bool
        True if the pattern is deterministic
    """

    input_nodes: set[int]
    output_nodes: set[int]
    q_indices: dict[int, int]
    commands: list[Command]
    runnable: bool = False
    deterministic: bool = False

    def __len__(self) -> int:
        return len(self.commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    @cached_property
    def nodes(self) -> set[int]:
        """Nodes of the pattern.

        Returns
        -------
        set[int]
            Set of nodes of the pattern
        """
        nodes = set(self.input_nodes)
        for cmd in self.commands:
            if isinstance(cmd, N):
                nodes |= {cmd.node}
        return nodes

    @cached_property
    def max_space(self) -> int:
        """Maximum number of qubits prepared at any point in the pattern.

        Returns
        -------
        int
            Maximum number of qubits prepared at any point in the pattern
        """
        nodes = len(self.input_nodes)
        max_nodes = nodes
        for cmd in self.commands:
            if isinstance(cmd, N):
                nodes += 1
            elif isinstance(cmd, M):
                nodes -= 1
            max_nodes = max(nodes, max_nodes)
        return max_nodes

    @cached_property
    def space_list(self) -> list[int]:
        """List of qubits prepared at each point in the pattern.

        Returns
        -------
        list[int]
            List of qubits prepared at each point in the pattern
        """
        nodes = len(self.input_nodes)
        space_list = [nodes]
        for cmd in self.commands:
            if isinstance(cmd, N):
                nodes += 1
                space_list.append(nodes)
            elif isinstance(cmd, M):
                nodes -= 1
                space_list.append(nodes)
        return space_list

    # TODO: will be removed
    def is_runnable(self) -> bool:
        """Return True if the pattern is runnable.

        Returns
        -------
        bool
            True if the pattern is runnable
        """
        return self.runnable

    def is_deterministic(self) -> bool:
        """Return True if the pattern is deterministic.

        Returns
        -------
        bool
            True if the pattern is deterministic
        """
        return self.deterministic


class BasePattern(ABC):
    """Base pattern class."""

    def __len__(self) -> int:
        return len(self.commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    @typing.overload
    def __getitem__(self, index: int) -> Command: ...

    @typing.overload
    def __getitem__(self, index: slice) -> list[Command]: ...

    def __getitem__(self, index: int | slice) -> Command | list[Command]:
        commands = self.commands
        if isinstance(index, (int, slice)):
            return commands[index]
        msg = f"Index type not supported: {type(index)}"
        raise TypeError(msg)

    @property
    @abstractmethod
    def input_nodes(self) -> set[int]:
        """Input nodes of the pattern.

        Returns
        -------
        set[int]
            Set of input nodes of the pattern
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_nodes(self) -> set[int]:
        """Output nodes of the pattern.

        Returns
        -------
        set[int]
            Set of output nodes of the pattern
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def nodes(self) -> set[int]:
        """Nodes of the pattern.

        Returns
        -------
        set[int]
            Set of nodes of the pattern
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def q_indices(self) -> dict[int, int]:
        """Logical qubit indices map of the pattern.

        Returns
        -------
        dict[int, int]
            Map of logical qubit indices
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def commands(self) -> list[Command]:
        """Commands of the pattern.

        Returns
        -------
        list[Command]
            List of commands of the pattern
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def max_space(self) -> int:
        """Maximum number of qubits prepared at any point in the pattern.

        Returns
        -------
        int
            Maximum number of qubits prepared at any point in the pattern
        """
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def space_list(self) -> list[int]:
        """List of qubits prepared at each point in the pattern.

        Returns
        -------
        list[int]
            List of qubits prepared at each point in the pattern
        """
        raise NotImplementedError

    @abstractmethod
    def is_runnable(self) -> bool:
        """Return True if the pattern is runnable.

        Returns
        -------
        bool
            True if the pattern is runnable
        """
        raise NotImplementedError

    @abstractmethod
    def is_deterministic(self) -> bool:
        """Return True if the pattern is deterministic.

        Returns
        -------
        bool
            True if the pattern is deterministic
        """
        raise NotImplementedError


class MutablePattern(BasePattern):
    """Mutable pattern class.

    Attributes
    ----------
    __input_nodes : set[int]
        Input nodes of the pattern
    __output_nodes : set[int]
        Output nodes of the pattern
    __q_indices : dict[int, int]
        Logical qubit indices map of the pattern
    __num_nodes : int
        Total number of nodes in the graph state
    __commands : list[Command]
        Commands of the pattern
    __runnable : bool
        True if the pattern is runnable
    __deterministic : bool
        True if the pattern is deterministic
    """

    def __init__(
        self,
        input_nodes: AbstractSet[int] | None = None,
        q_indices: Mapping[int, int] | None = None,
    ) -> None:
        if input_nodes is None:
            input_nodes = set()
        self.__input_nodes: set[int] = set(input_nodes)  # input nodes (list() makes our own copy of the list)
        self.__num_nodes: int = len(input_nodes)  # total number of nodes in the graph state

        self.__commands: list[Command] = []
        # output nodes are initially input nodes, since none are measured yet
        self.__output_nodes: set[int] = set(self.__input_nodes)

        if q_indices is None:
            q_indices = {}
            for i, input_node in enumerate(input_nodes):
                q_indices[input_node] = i

        self.__q_indices: dict[int, int] = dict(q_indices)  # qubit index. used for simulation

        self.__runnable: bool = False
        self.__deterministic: bool = False

    def __len__(self) -> int:
        return len(self.__commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.__commands)

    @typing.overload
    def __getitem__(self, index: int) -> Command: ...

    @typing.overload
    def __getitem__(self, index: slice) -> list[Command]: ...

    def __getitem__(self, index: int | slice) -> Command | list[Command]:
        commands = self.__commands
        if isinstance(index, (int, slice)):
            return commands[index]
        msg = f"Index type not supported: {type(index)}"
        raise TypeError(msg)

    def __add(self, cmd: Command) -> None:
        if isinstance(cmd, N):
            if cmd.node in self.__output_nodes:
                raise NodeAlreadyPreparedError(cmd.node)
            self.__num_nodes += 1
            self.__output_nodes |= {cmd.node}
            self.__q_indices[cmd.node] = cmd.q_index
        elif isinstance(cmd, M):
            self.__output_nodes -= {cmd.node}
        self.__commands.append(cmd)

    def add(self, cmd: Command) -> None:
        """Add a command to the pattern.

        note: This method does not guarantee the pattern to be runnable or deterministic.

        Parameters
        ----------
        cmd : Command
            Command to add to the pattern
        """
        self.__add(cmd)

        # runnablility and determinism are not guaranteed after adding a command
        self.__runnable = False
        self.__deterministic = False

    def extend(self, cmds: Iterable[Command]) -> None:
        """Extend the pattern with a list of commands.

        note: This method does not guarantee the pattern to be runnable or deterministic.

        Parameters
        ----------
        cmds : Iterable[Command]
            List of commands to add to the pattern
        """
        for cmd in cmds:
            self.__add(cmd)
        self.__runnable = False
        self.__deterministic = False

    def clear(self) -> None:
        """Clear the pattern.

        note: This method does not guarantee the pattern to be runnable or deterministic.
        """
        self.__num_nodes = len(self.__input_nodes)
        self.__commands = []
        self.__output_nodes = set(self.__input_nodes)

        self.__runnable = False
        self.__deterministic = False

    def replace(self, cmds: Iterable[Command], input_nodes: AbstractSet[int] | None = None) -> None:
        """Replace the pattern with a list of commands.

        Parameters
        ----------
        cmds : Iterable[Command]
            List of commands to replace the pattern
        input_nodes : AbstractSet[int] | None, optional
            input nodes, by default None
        """
        if input_nodes is not None:
            self.__input_nodes = set(input_nodes)
        self.clear()
        self.extend(cmds)

    def append_pattern(self, pattern: MutablePattern | ImmutablePattern) -> MutablePattern:
        """Append a pattern to the current pattern.

        Parameters
        ----------
        pattern : MutablePattern | ImmutablePattern
            Pattern to append to the current pattern

        Returns
        -------
        MutablePattern
            Combined pattern

        Raises
        ------
        ValueError
            If duplicated nodes without border of two patterns
        """
        common_nodes = self.nodes & pattern.nodes
        border_nodes = self.output_nodes & pattern.input_nodes

        if common_nodes != border_nodes:
            msg = f"Detect duplicated nodes without border of two patterns. duplicated nodes: {common_nodes}"
            raise ValueError(msg)
        new_input_nodes = self.input_nodes | (pattern.input_nodes - self.output_nodes)
        new_input_q_indices = {}
        for node in new_input_nodes:
            if node in self.input_nodes:
                new_input_q_indices[node] = self.q_indices[node]
            else:
                new_input_q_indices[node] = pattern.q_indices[node]

        new_pattern = MutablePattern(input_nodes=new_input_nodes, q_indices=new_input_q_indices)
        for cmd in self.commands:
            new_pattern.add(cmd)

        for cmd in pattern.commands:
            new_pattern.add(cmd)

        if self.is_runnable() and pattern.is_runnable():
            new_pattern.mark_runnable()

        if self.is_deterministic() and pattern.is_deterministic():
            new_pattern.mark_deterministic()

        return new_pattern

    @property
    def input_nodes(self) -> set[int]:
        """Input nodes of the pattern.

        Returns
        -------
        set[int]
            Set of input nodes of the pattern
        """
        return set(self.__input_nodes)

    @property
    def output_nodes(self) -> set[int]:
        """Output nodes of the pattern.

        Returns
        -------
        set[int]
            Set of output nodes of the pattern
        """
        return set(self.__output_nodes)

    @property
    def q_indices(self) -> dict[int, int]:
        """Logical qubit indices map of the pattern.

        Returns
        -------
        dict[int, int]
            Map of logical qubit indices
        """
        return dict(self.__q_indices)

    @property
    def num_nodes(self) -> int:
        """Total number of nodes in the graph state.

        Returns
        -------
        int
            Total number of nodes in the graph state
        """
        return self.__num_nodes

    @cached_property
    def nodes(self) -> set[int]:
        """Nodes of the pattern.

        Returns
        -------
        set[int]
            Set of nodes of the pattern
        """
        nodes = set(self.__input_nodes)
        for cmd in self.commands:
            if isinstance(cmd, N):
                nodes |= {cmd.node}
        return nodes

    @property
    def commands(self) -> list[Command]:
        """Commands of the pattern.

        Returns
        -------
        list[Command]
            List of commands of the pattern
        """
        return self.__commands

    @cached_property
    def max_space(self) -> int:
        """Maximum number of qubits prepared at any point in the pattern.

        Returns
        -------
        int
            Maximum number of qubits prepared at any point in the pattern
        """
        nodes = len(self.input_nodes)
        max_nodes = nodes
        for cmd in self.commands:
            if isinstance(cmd, N):
                nodes += 1
            elif isinstance(cmd, M):
                nodes -= 1
            max_nodes = max(nodes, max_nodes)
        return max_nodes

    @cached_property
    def space_list(self) -> list[int]:
        """List of qubits prepared at each point in the pattern.

        Returns
        -------
        list[int]
            List of qubits prepared at each point in the pattern
        """
        nodes = len(self.input_nodes)
        space_list = [nodes]
        for cmd in self.commands:
            if isinstance(cmd, N):
                nodes += 1
                space_list.append(nodes)
            elif isinstance(cmd, M):
                nodes -= 1
                space_list.append(nodes)
        return space_list

    @property
    def meas_planes(self) -> dict[int, Plane]:
        """Measurement planes of the pattern.

        Returns
        -------
        dict[int, Plane]
            Measurement planes of each node in the pattern
        """
        meas_plane = {}
        for cmd in self.commands:
            if isinstance(cmd, M):
                mplane = cmd.plane
                meas_plane[cmd.node] = mplane
        return meas_plane

    @property
    def meas_angles(self) -> dict[int, float]:
        """Measurement angles of the pattern.

        Returns
        -------
        dict[int, float]
            Measurement angles of each node in the pattern
        """
        angles = {}
        for cmd in self.commands:
            if isinstance(cmd, M):
                angles[cmd.node] = cmd.angle
        return angles

    def is_runnable(self) -> bool:
        """Return True if the pattern is runnable.

        Returns
        -------
        bool
            True if the pattern is runnable
        """
        return self.__runnable

    def is_deterministic(self) -> bool:
        """Return True if the pattern is deterministic.

        Returns
        -------
        bool
            True if the pattern is deterministic
        """
        return self.__deterministic

    def mark_runnable(self) -> None:
        """Mark the pattern as runnable. Called where the pattern is guaranteed to be runnable."""
        self.__runnable = True

    def mark_deterministic(self) -> None:
        """Mark the pattern as deterministic. Called where flow preservation is guaranteed."""
        self.__deterministic = True

    def freeze(self) -> ImmutablePattern:
        """Immutarize the pattern.

        Returns
        -------
        ImmutablePattern
            Immutable pattern
        """
        return ImmutablePattern(
            input_nodes=self.input_nodes,
            output_nodes=self.output_nodes,
            q_indices=self.q_indices,
            commands=self.commands,
            runnable=self.is_runnable(),
            deterministic=self.is_deterministic(),
        )

    def standardize(self) -> None:
        """Standardize the pattern.

        Raises
        ------
        NotImplementedError
            The method is not implemented
        """
        raise NotImplementedError

    def shift_signals(self) -> None:
        """Shift signals in the pattern.

        Raises
        ------
        NotImplementedError
            The method is not implemented
        """
        raise NotImplementedError

    def pauli_simplification(self) -> None:
        """Simplify the pattern using Pauli simplification.

        Raises
        ------
        NotImplementedError
            The method is not implemented
        """
        raise NotImplementedError


class NodeAlreadyPreparedError(Exception):
    """Raised when a node is prepared more than once."""

    def __init__(self, node: int) -> None:
        self.__node: int = node

    @property
    def node(self) -> int:
        """Node index of the command.

        Returns
        -------
        int
            Node index of the command
        """
        return self.__node

    def __str__(self) -> str:
        return f"Node already prepared: {self.__node}"


def is_standardized(pattern: BasePattern | ImmutablePattern) -> bool:
    """Check if the pattern is standardized.

    Parameters
    ----------
    pattern : BasePattern | ImmutablePattern
        Pattern to check

    Returns
    -------
    bool
        True if the pattern is standardized

    Raises
    ------
    ValueError
        If the command is unknown
    """
    standardized = True
    standardized_order = [
        N,
        E,
        M,
        X,
        Z,
        C,
    ]
    current_cmd_type: type[Command] = N
    for cmd in pattern:
        if isinstance(cmd, current_cmd_type):
            continue
        if type(cmd) not in standardized_order:
            msg = f"Unknown command: {cmd}"
            raise ValueError(msg)
        if standardized_order.index(type(cmd)) < standardized_order.index(current_cmd_type):
            standardized = False
            break
        current_cmd_type = type(cmd)
    return standardized


def is_runnable(pattern: BasePattern | ImmutablePattern) -> bool:
    """Check if the pattern is runnable.

    Parameters
    ----------
    pattern : BasePattern | ImmutablePattern
        Pattern to check

    Returns
    -------
    bool
        True if the pattern is runnable
    """
    runnable = True
    if not _check_rule0(pattern):
        runnable = False
    if not _check_rule1(pattern):
        runnable = False
    if not _check_rule2(pattern):
        runnable = False
    if not _check_rule3(pattern):
        runnable = False
    return runnable


def _check_rule0(pattern: BasePattern | ImmutablePattern) -> bool:
    """Check if no command depends on an output not yet measured.

    Parameters
    ----------
    pattern : BasePattern | ImmutablePattern
        Pattern to check

    Returns
    -------
    bool
        True if no command depends on an output not yet measured
    """
    measured: set[int] = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            if len(cmd.s_domain & measured) > 0 or len(cmd.t_domain & measured) > 0:
                return False
            measured.add(cmd.node)
        elif isinstance(cmd, (X, Z)):
            if len(cmd.domain & measured) > 0:
                return False

    return True


def _check_rule1(pattern: BasePattern | ImmutablePattern) -> bool:
    """Check if no command acts on a qubit already measured.

    Parameters
    ----------
    pattern : BasePattern | ImmutablePattern
        Pattern to check

    Returns
    -------
    bool
        True if no command acts on a qubit already measured

    Raises
    ------
    TypeError
        If the command kind is unknown
    """
    measured = set()
    for cmd in pattern:
        if isinstance(cmd, M):
            if cmd.node in measured:
                return False
            measured.add(cmd.node)
        elif isinstance(cmd, E):
            if len(set(cmd.nodes) & measured) > 0:
                return False
        elif isinstance(cmd, (N, X, Z, C)):
            if cmd.node in measured:
                return False
        else:
            msg = f"Unknown command kind: {cmd.kind}"
            raise TypeError(msg)
    return True


def _check_rule2(pattern: BasePattern | ImmutablePattern) -> bool:
    """Check if no command acts on a qubit not yet prepared, unless it is an input qubit.

    Parameters
    ----------
    pattern : BasePattern | ImmutablePattern
        Pattern to check

    Returns
    -------
    bool
        True if no command acts on a qubit not yet prepared, unless it is an input qubit
    """
    prepared = set(pattern.input_nodes)
    for cmd in pattern:
        if isinstance(cmd, N):
            prepared.add(cmd.node)
        elif isinstance(cmd, E):
            if cmd.nodes[0] not in prepared or cmd.nodes[1] not in prepared:
                return False
        elif isinstance(cmd, (M, X, Z, C)) and cmd.node not in prepared:
            return False
    return True


def _check_rule3(pattern: BasePattern | ImmutablePattern) -> bool:
    """Check if a qubit is measured if and only if it is not an output.

    Parameters
    ----------
    pattern : BasePattern | ImmutablePattern
        Pattern to check

    Returns
    -------
    bool
        True if a qubit is measured if and only if it is not an output
    """
    output_nodes = pattern.output_nodes
    return all(not (isinstance(cmd, M) and cmd.node in output_nodes) for cmd in pattern)


# # NOTE: generally, difficult to prove that a pattern is deterministic
# def is_deterministic(pattern: BasePattern) -> bool:
#     raise NotImplementedError


def print_command(cmd: Command) -> None:
    """Print a command.

    Parameters
    ----------
    cmd : Command
        Command to print
    """
    if isinstance(cmd, N):
        print(f"N, node = {cmd.node}")  # noqa: T201
    elif isinstance(cmd, E):
        print(f"E, nodes = {cmd.nodes}")  # noqa: T201
    elif isinstance(cmd, M):
        print(  # noqa: T201
            f"M, node = {cmd.node}",
            f"plane = {cmd.plane}",
            f"angle = {cmd.angle}",
            f"s-domain = {cmd.s_domain}",
            f"t_domain = {cmd.t_domain}",
        )
    elif isinstance(cmd, X):
        print(f"X, node = {cmd.node}, domain = {cmd.domain}")  # noqa: T201
    elif isinstance(cmd, Z):
        print(f"Z, node = {cmd.node}, domain = {cmd.domain}")  # noqa: T201
    elif isinstance(cmd, C):
        print(f"C, node = {cmd.node}")  # noqa: T201
        cmd.local_clifford.print_angles()
    else:
        print(f"Unknown command: {cmd}")  # noqa: T201


def print_pattern(
    pattern: BasePattern | ImmutablePattern, lim: int = 40, cmd_filter: list[type[Command]] | None = None
) -> None:
    """Print a pattern.

    Parameters
    ----------
    pattern : BasePattern | ImmutablePattern
        Pattern to print
    lim : int, optional
        Maximum number of commands to print, by default 40
    cmd_filter : list[type[Command]] | None, optional
        Command filter, by default None
    """
    if cmd_filter is None:
        cmd_filter = [
            N,
            E,
            M,
            X,
            Z,
            C,
        ]
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
