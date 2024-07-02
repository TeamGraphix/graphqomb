"""Data validator command classes."""

from __future__ import annotations

import copy
import enum
from typing import Literal

import numpy as np
from pydantic import BaseModel

Node = int
Plane = Literal["XY", "YZ", "XZ"]


class CommandKind(str, enum.Enum):
    N = "N"
    M = "M"
    E = "E"
    C = "C"
    X = "X"
    Z = "Z"


class Command(BaseModel):
    """
    Base command class.
    """

    kind: CommandKind = None


class N(Command):
    """
    Preparation command.
    """

    kind: CommandKind = CommandKind.N
    node: Node


class M(Command):
    """
    Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop.
    """

    kind: CommandKind = CommandKind.M
    node: Node
    plane: Plane = "XY"
    angle: float = 0.0
    s_domain: list[Node] = []
    t_domain: list[Node] = []


class E(Command):
    """
    Entanglement command.
    """

    kind: CommandKind = CommandKind.E
    nodes: tuple[Node, Node]


class C(Command):
    """
    Clifford command.
    """

    kind: CommandKind = CommandKind.C
    node: Node
    cliff_index: int


class Correction(Command):
    """
    Correction command.
    Either X or Z.
    """

    node: Node
    domain: set[Node] = []


class X(Correction):
    """
    X correction command.
    """

    kind: CommandKind = CommandKind.X


class Z(Correction):
    """
    Z correction command.
    """

    kind: CommandKind = CommandKind.Z


class NodeAlreadyPreparedError(Exception):
    def __init__(self, node: int):
        self.__node = node

    @property
    def node(self):
        return self.__node

    @property
    def __str__(self) -> str:
        return f"Node already prepared: {self.__node}"


class Pattern:
    """
    MBQC pattern class

    Pattern holds a sequence of commands to operate the MBQC (Pattern.seq),
    and provide modification strategies to improve the structure and simulation
    efficiency of the pattern accoring to measurement calculus.

    ref: V. Danos, E. Kashefi and P. Panangaden. J. ACM 54.2 8 (2007)

    Attributes
    ----------
    list(self) :
        list of commands.

        .. line-block::
            each command is a list [type, nodes, attr] which will be applied in the order of list indices.
            type: one of {'N', 'M', 'E', 'X', 'Z', 'S', 'C'}
            nodes: int for {'N', 'M', 'X', 'Z', 'S', 'C'} commands, tuple (i, j) for {'E'} command
            attr for N: none
            attr for M: meas_plane, angle, s_domain, t_domain
            attr for X: signal_domain
            attr for Z: signal_domain
            attr for S: signal_domain
            attr for C: clifford_index, as defined in :py:mod:`graphix.clifford`
    Nnode : int
        total number of nodes in the resource state
    """

    def __init__(self, input_nodes: list[int] | None = None):
        """
        :param input_nodes:  optional, list of input qubits
        """
        if input_nodes is None:
            input_nodes = []
        self.__input_nodes = list(input_nodes)  # input nodes (list() makes our own copy of the list)
        self.__Nnode = len(input_nodes)  # total number of nodes in the graph state

        self.__seq: list[Command] = []
        # output nodes are initially input nodes, since none are measured yet
        self.__output_nodes = list()

        self.extend([N(node=node) for node in input_nodes])

    def add(self, cmd: Command):
        """add command to the end of the pattern.
        an MBQC command is specified by a list of [type, node, attr], where

            type : 'N', 'M', 'E', 'X', 'Z', 'S' or 'C'
            nodes : int for 'N', 'M', 'X', 'Z', 'S', 'C' commands
            nodes : tuple (i, j) for 'E' command
            attr for N (node preparation):
                none
            attr for E (entanglement):
                none
            attr for M (measurement):
                meas_plane : 'XY','YZ' or 'XZ'
                angle : float, in radian / pi
                s_domain : list
                t_domain : list
            attr for X:
                signal_domain : list
            attr for Z:
                signal_domain : list
            attr for S:
                signal_domain : list
            attr for C:
                clifford_index : int

        Parameters
        ----------
        cmd : list
            MBQC command.
        """
        if cmd.kind == CommandKind.N:
            if cmd.node in self.__output_nodes:
                raise NodeAlreadyPreparedError(cmd.node)
            self.__Nnode += 1
            self.__output_nodes.append(cmd.node)
        elif cmd.kind == CommandKind.M:
            self.__output_nodes.remove(cmd.node)
        self.__seq.append(cmd)

    def extend(self, cmds: list[Command]):
        """Add a list of commands.

        :param cmds: list of commands
        """
        for cmd in cmds:
            self.add(cmd)

    def clear(self):
        """Clear the sequence of pattern commands."""
        self.__Nnode = len(self.__input_nodes)
        self.__seq = []
        self.__output_nodes = list(self.__input_nodes)

    def replace(self, cmds: list[Command], input_nodes=None):
        """Replace pattern with a given sequence of pattern commands.

        :param cmds: list of commands

        :param input_nodes:  optional, list of input qubits
        (by default, keep the same input nodes as before)
        """
        if input_nodes is not None:
            self.__input_nodes = list(input_nodes)
        self.clear()
        self.extend(cmds)

    @property
    def input_nodes(self):
        """list of input nodes"""
        return list(self.__input_nodes)  # copy for preventing modification

    @property
    def output_nodes(self):
        """list of all nodes that are either `input_nodes` or prepared with
        `N` commands and that have not been measured with an `M` command
        """
        return list(self.__output_nodes)  # copy for preventing modification

    def __len__(self):
        """length of command sequence"""
        return len(self.__seq)

    def __iter__(self):
        """iterate over commands"""
        return iter(self.__seq)

    def __getitem__(self, index):
        return self.__seq[index]

    def __add__(self, pattern):
        if self.__output_nodes != pattern.__input_nodes:
            raise ValueError("Output nodes of the first pattern must be the input nodes of the second pattern")
        new_pattern = copy.deepcopy(self)
        for cmd in pattern:
            new_pattern.add(cmd)

        return new_pattern

    def calc_max_space(self):
        """The maximum number of nodes that must be present in the graph (graph space) during the execution of the pattern.
        For statevector simulation, this is equivalent to the maximum memory
        needed for classical simulation.

        Returns
        -------
        n_nodes : int
            max number of nodes present in the graph during pattern execution.
        """
        nodes = len(self.input_nodes)
        max_nodes = nodes
        for cmd in self.__seq:
            if cmd.kind == N:
                nodes += 1
            elif cmd.kind == M:
                nodes -= 1
            if nodes > max_nodes:
                max_nodes = nodes
        return max_nodes

    def get_space_list(self):
        """Returns the list of the number of nodes present in the graph (space)
        during each step of execution of the pattern (for N and M commands).

        Returns
        -------
        N_list : list
            time evolution of 'space' at each 'N' and 'M' commands of pattern.
        """
        nodes = 0
        space_list = []
        for cmd in self.__seq:
            if cmd.kind == N:
                nodes += 1
                space_list.append(nodes)
            elif cmd.kind == M:
                nodes -= 1
                space_list.append(nodes)
        return space_list

    def print_pattern(self, lim=40, cmd_filter=None):
        """print the pattern sequence (Pattern.seq).

        Parameters
        ----------
        lim: int, optional
            maximum number of commands to show
        cmd_filter : list of str, optional
            show only specified commands, e.g. ['M', 'X', 'Z']
        """
        if len(self.__seq) < lim:
            nmax = len(self.__seq)
        else:
            nmax = lim
        if cmd_filter is None:
            cmd_filter = ["N", "E", "M", "X", "Z", "C"]
        count = 0
        i = -1
        while count < nmax:
            i = i + 1
            if i == len(self.__seq):
                break
            if self.__seq[i].kind == CommandKind.N and ("N" in cmd_filter):
                count += 1
                print(f"N, node = {self.__seq[i].node}")
            elif self.__seq[i].kind == CommandKind.E and ("E" in cmd_filter):
                count += 1
                print(f"E, nodes = {self.__seq[i].node}")
            elif self.__seq[i].kind == CommandKind.M and ("M" in cmd_filter):
                count += 1
                print(
                    f"M, node = {self.__seq[i].node}, "
                    + "plane = {self.__seq[i].plane}, angle(pi) = {self.__seq[i].angle}, "
                    + f"s-domain = {self.__seq[i].s_domain}, t_domain = {self.__seq[i].t_domain}"
                )
            elif self.__seq[i].kind == CommandKind.X and ("X" in cmd_filter):
                count += 1
                # remove duplicates
                _domain = np.array(self.__seq[i].domain)
                uind = np.unique(_domain)
                unique_domain = []
                for ind in uind:
                    if np.mod(np.count_nonzero(_domain == ind), 2) == 1:
                        unique_domain.append(ind)
                print(f"X byproduct, node = {self.__seq[i].node}, domain = {unique_domain}")
            elif self.__seq[i].kind == CommandKind.Z and ("Z" in cmd_filter):
                count += 1
                # remove duplicates
                _domain = np.array(self.__seq[i].domain)
                uind = np.unique(_domain)
                unique_domain = []
                for ind in uind:
                    if np.mod(np.count_nonzero(_domain == ind), 2) == 1:
                        unique_domain.append(ind)
                print(f"Z byproduct, node = {self.__seq[i].node}, domain = {unique_domain}")
            elif self.__seq[i].kind == CommandKind.C and ("C" in cmd_filter):
                count += 1
                print(f"Clifford, node = {self.__seq[i].node}, Clifford index = {self.__seq[i].cliff_index}")

        if len(self.__seq) > i + 1:
            print(f"{len(self.__seq)-lim} more commands truncated. Change lim argument of print_pattern() to show more")

    def get_meas_plane(self):
        """get measurement plane from the pattern.

        Returns
        -------
        meas_plane: dict of str
            list of str representing measurement plane for each node.
        """
        meas_plane = dict()
        for cmd in self.__seq:
            if cmd.kind == CommandKind.M:
                mplane = cmd.plane
                meas_plane[cmd.node] = mplane
        return meas_plane

    def get_meas_angles(self):
        """Get measurement angles of the pattern.

        Returns
        -------
        angles : dict
            measurement angles of the each node.
        """
        angles = {}
        for cmd in self.__seq:
            if cmd.kind == CommandKind.M:
                angles[cmd.node] = cmd.angle
        return angles

    def simulate(self, backend):
        raise NotImplementedError

    def execute(self, backend):
        raise NotImplementedError

    def to_qasm3(self):
        raise NotImplementedError

    def to_text(self):
        raise NotImplementedError


def is_standardized(pattern: Pattern) -> bool:
    """Check if the pattern is standardized.

    Args:
        pattern (Pattern): Pattern to check.

    Returns:
        bool: is the pattern standardized.
    """
    standardized = True
    standardized_order = [
        CommandKind.N,
        CommandKind.E,
        CommandKind.M,
        CommandKind.X,
        CommandKind.Z,
        CommandKind.C,
    ]
    current_cmd_kind = CommandKind.N
    for cmd in pattern:
        if cmd.kind == current_cmd_kind:
            continue
        if cmd.kind not in standardized_order:
            raise ValueError(f"Unknown command kind: {cmd.kind}")
        if standardized_order.index(cmd.kind) < standardized_order.index(current_cmd_kind):
            standardized = False
            break
        current_cmd_kind = cmd.kind
    return standardized
