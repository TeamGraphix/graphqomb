from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np
import dataclasses

from graphix_zx.command import Command, CommandKind


class NodeAlreadyPreparedError(Exception):
    def __init__(self, node: int):
        self.__node = node

    @property
    def node(self):
        return self.__node

    @property
    def __str__(self) -> str:
        return f"Node already prepared: {self.__node}"


class BasePattern(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def get_input_nodes(self):
        raise NotImplementedError

    @abstractmethod
    def get_output_nodes(self):
        raise NotImplementedError

    @abstractmethod
    def get_commands(self):
        raise NotImplementedError

    @abstractmethod
    def calc_max_space(self):
        raise NotImplementedError

    @abstractmethod
    def is_runnable(self):
        raise NotImplementedError

    @abstractmethod
    def is_deterministic(self):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ImmutablePattern(BasePattern):
    input_nodes: list[int]
    output_nodes: list[int]
    seq: list[Command]
    runnable: bool = False
    deterministic: bool = False

    def get_input_nodes(self):
        return self.input_nodes

    def get_output_nodes(self):
        return self.output_nodes

    def get_commands(self):
        return self.seq

    def calc_max_space(self):
        nodes = len(self.input_nodes)
        max_nodes = nodes
        for cmd in self.seq:
            if cmd.kind == CommandKind.N:
                nodes += 1
            elif cmd.kind == CommandKind.M:
                nodes -= 1
            if nodes > max_nodes:
                max_nodes = nodes
        return max_nodes

    def is_runnable(self):
        return self.runnable

    def is_deterministic(self):
        return self.deterministic


class MutablePattern(BasePattern):
    def __init__(self, input_nodes: list[int] | None = None):
        if input_nodes is None:
            input_nodes = []
        self.__input_nodes: list[int] = list(input_nodes)  # input nodes (list() makes our own copy of the list)
        self.__Nnode: int = len(input_nodes)  # total number of nodes in the graph state

        self.__seq: list[Command] = []
        # output nodes are initially input nodes, since none are measured yet
        self.__output_nodes: list[int] = list(self.__input_nodes)

        self.__runnable: bool = False
        self.__deterministic: bool = False

    def add(self, cmd: Command):
        if cmd.kind == CommandKind.N:
            if cmd.node in self.__output_nodes:
                raise NodeAlreadyPreparedError(cmd.node)
            self.__Nnode += 1
            self.__output_nodes.append(cmd.node)
        elif cmd.kind == CommandKind.M:
            self.__output_nodes.remove(cmd.node)
        self.__seq.append(cmd)

    def extend(self, cmds: list[Command]):
        for cmd in cmds:
            self.add(cmd)

    def clear(self):
        self.__Nnode = len(self.__input_nodes)
        self.__seq = []
        self.__output_nodes = list(self.__input_nodes)

    def replace(self, cmds: list[Command], input_nodes=None):
        if input_nodes is not None:
            self.__input_nodes = list(input_nodes)
        self.clear()
        self.extend(cmds)

    @property
    def input_nodes(self):
        return list(self.__input_nodes)  # copy for preventing modification

    @property
    def output_nodes(self):
        return list(self.__output_nodes)  # copy for preventing modification

    def get_commands(self):
        return self.__seq

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
        nodes = len(self.input_nodes)
        max_nodes = nodes
        for cmd in self.__seq:
            if cmd.kind == CommandKind.N:
                nodes += 1
            elif cmd.kind == CommandKind.M:
                nodes -= 1
            if nodes > max_nodes:
                max_nodes = nodes
        return max_nodes

    def get_space_list(self):
        nodes = len(self.input_nodes)
        space_list = [nodes]
        for cmd in self.__seq:
            if cmd.kind == CommandKind.N:
                nodes += 1
                space_list.append(nodes)
            elif cmd.kind == CommandKind.M:
                nodes -= 1
                space_list.append(nodes)
        return space_list

    def get_meas_planes(self):
        meas_plane = dict()
        for cmd in self.__seq:
            if cmd.kind == CommandKind.M:
                mplane = cmd.plane
                meas_plane[cmd.node] = mplane
        return meas_plane

    def get_meas_angles(self):
        angles = {}
        for cmd in self.__seq:
            if cmd.kind == CommandKind.M:
                angles[cmd.node] = cmd.angle
        return angles

    def is_runnable(self):
        return self.__runnable

    def is_deterministic(self):
        return self.__deterministic

    def freeze(self) -> ImmutablePattern:
        return ImmutablePattern(
            input_nodes=self.__input_nodes,
            output_nodes=self.__output_nodes,
            seq=self.__seq,
            runnable=self.__runnable,
            deterministic=self.__deterministic,
        )

    def standardize(self):
        raise NotImplementedError

    def shift_signals(self):
        raise NotImplementedError

    def pauli_simplification(self):
        raise NotImplementedError


def is_standardized(pattern: BasePattern) -> bool:
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


def print_pattern(pattern: BasePattern, lim: int = 40, cmd_filter: list[CommandKind] | None = None):
    if len(pattern) < lim:
        nmax = len(pattern)
    else:
        nmax = lim
    if cmd_filter is None:
        cmd_filter = [
            CommandKind.N,
            CommandKind.E,
            CommandKind.M,
            CommandKind.X,
            CommandKind.Z,
            CommandKind.C,
        ]
    count = 0
    i = -1
    while count < nmax:
        i = i + 1
        if i == len(pattern):
            break
        if pattern[i].kind == CommandKind.N and (CommandKind.N in cmd_filter):
            count += 1
            print(f"N, node = {pattern[i].node}")
        elif pattern[i].kind == CommandKind.E and (CommandKind.E in cmd_filter):
            count += 1
            print(f"E, nodes = {pattern[i].nodes}")
        elif pattern[i].kind == CommandKind.M and (CommandKind.M in cmd_filter):
            count += 1
            print(
                f"M, node = {pattern[i].node}, "
                + f"plane = {pattern[i].plane}, angle(pi) = {pattern[i].angle}, "
                + f"s-domain = {pattern[i].s_domain}, t_domain = {pattern[i].t_domain}"
            )
        elif pattern[i].kind == CommandKind.X and (CommandKind.X in cmd_filter):
            count += 1
            # remove duplicates
            _domain = np.array(pattern[i].domain)
            uind = np.unique(_domain)
            unique_domain = []
            for ind in uind:
                if np.mod(np.count_nonzero(_domain == ind), 2) == 1:
                    unique_domain.append(ind)
            print(f"X byproduct, node = {pattern[i].node}, domain = {unique_domain}")
        elif pattern[i].kind == CommandKind.Z and (CommandKind.Z in cmd_filter):
            count += 1
            # remove duplicates
            _domain = np.array(pattern[i].domain)
            uind = np.unique(_domain)
            unique_domain = []
            for ind in uind:
                if np.mod(np.count_nonzero(_domain == ind), 2) == 1:
                    unique_domain.append(ind)
            print(f"Z byproduct, node = {pattern[i].node}, domain = {unique_domain}")
        elif pattern[i].kind == CommandKind.C and (CommandKind.C in cmd_filter):
            count += 1
            print(f"Clifford, node = {pattern[i].node}, Clifford index = {pattern[i].cliff_index}")
        else:
            print(f"Command {pattern[i].kind} not recognized")

    if len(pattern) > i + 1:
        print(f"{len(pattern)-lim} more commands truncated. Change lim argument of print_pattern() to show more")
