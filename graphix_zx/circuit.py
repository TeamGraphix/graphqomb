from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from graphix_zx.common import Plane
from graphix_zx.graphstate import GraphState, BaseGraphState


class GateKind(Enum):
    """Enum class for gate kind"""

    J = auto()
    CZ = auto()
    PhaseGadget = auto()


@dataclass(frozen=True)
class Gate:
    kind: GateKind

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass(frozen=True)
class J(Gate):
    kind: GateKind = GateKind.J
    qubit: int | None = None
    angle: float = 0

    def get_matrix(self) -> NDArray:
        return np.array([[1, np.exp(-1j * self.angle)], [1, -np.exp(-1j * self.angle)]]) / np.sqrt(2)


@dataclass(frozen=True)
class CZ(Gate):
    kind: GateKind = GateKind.CZ
    qubits: tuple[int, int] | None = None

    def get_matrix(self) -> NDArray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


@dataclass(frozen=True)
class PhaseGadget(Gate):
    kind: GateKind = GateKind.PhaseGadget
    qubits: list[int] | None = None
    angle: float = 0

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


class BaseCircuit(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def num_qubits(self):
        raise NotImplementedError

    @abstractmethod
    def get_instructions(self):
        raise NotImplementedError

    @abstractmethod
    def j(self, qubit: int, angle: float):
        raise NotImplementedError

    @abstractmethod
    def cz(self, qubit1: int, qubit2: int):
        raise NotImplementedError

    @abstractmethod
    def phase_gadget(self, qubits: list[int], angle: float):
        raise NotImplementedError


class MBQCCircuit(BaseCircuit):
    def __init__(self, qubits: int):
        self.__num_qubits: int = qubits
        self.__gate_instructions: list[Gate] = []

    @property
    def input_nodes(self):
        return self.__input_nodes

    @property
    def num_qubits(self):
        return self.__num_qubits

    def get_instructions(self):
        return self.__gate_instructions

    def j(self, qubit: int, angle: float):
        self.__gate_instructions.append(J(qubit=qubit, angle=angle))

    def cz(self, qubit1: int, qubit2: int):
        self.__gate_instructions.append(CZ(qubits=(qubit1, qubit2)))

    def phase_gadget(self, qubits: list[int], angle: float):
        self.__gate_instructions.append(PhaseGadget(qubits=qubits, angle=angle))


def circuit2graph(circuit: BaseCircuit) -> tuple[BaseGraphState, dict[int, set[int]]]:
    graph = GraphState()
    flow = dict()

    front_nodes = []  # list index  corresponds to qubit index
    num_nodes = 0

    # input nodes
    for _ in range(circuit.num_qubits):
        graph.add_physical_node(num_nodes, is_input=True)
        front_nodes.append(num_nodes)
        num_nodes += 1

    for instruction in circuit.get_instructions():
        if isinstance(instruction, J):
            graph.add_physical_node(num_nodes)
            graph.add_physical_edge(front_nodes[instruction.qubit], num_nodes)
            graph.set_meas_plane(front_nodes[instruction.qubit], Plane.XY)
            graph.set_meas_angle(front_nodes[instruction.qubit], instruction.angle)

            flow[front_nodes[instruction.qubit]] = {num_nodes}
            front_nodes[instruction.qubit] = num_nodes

            num_nodes += 1

        elif isinstance(instruction, CZ):
            graph.add_physical_edge(front_nodes[instruction.qubits[0]], front_nodes[instruction.qubits[1]])
        elif isinstance(instruction, PhaseGadget):
            raise NotImplementedError
        else:
            raise ValueError("Invalid instruction")

    for node in front_nodes:
        graph.set_output(node)

    return graph, flow
