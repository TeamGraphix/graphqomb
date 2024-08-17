from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import Plane
from graphix_zx.graphstate import GraphState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from graphix_zx.flow import FlowLike


class GateKind(Enum):
    """Enum class for gate kind"""

    J = auto()
    CZ = auto()
    PhaseGadget = auto()


class Gate(ABC):
    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass(frozen=True)
class J(Gate):
    qubit: int
    angle: float
    kind: GateKind = GateKind.J

    def get_matrix(self) -> NDArray:
        return np.array([[1, np.exp(1j * self.angle)], [1, -np.exp(1j * self.angle)]]) / np.sqrt(2)


@dataclass(frozen=True)
class CZ(Gate):
    qubits: tuple[int, int]
    kind: GateKind = GateKind.CZ

    def get_matrix(self) -> NDArray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


@dataclass(frozen=True)
class PhaseGadget(Gate):
    qubits: list[int]
    angle: float
    kind: GateKind = GateKind.PhaseGadget

    def get_matrix(self) -> NDArray:
        def count_ones_in_binary(array: NDArray) -> NDArray:
            count_ones = np.vectorize(lambda x: bin(x).count("1"))
            return count_ones(array)

        index_array = np.arange(2 ** len(self.qubits))
        z_sign = (-1) ** count_ones_in_binary(index_array)
        return np.diag(np.exp(-1j * self.angle / 2 * z_sign))


class MacroGate(ABC):
    @abstractmethod
    def get_unit_gates(self) -> list[Gate]:
        raise NotImplementedError

    @abstractmethod
    def get_matrix(self) -> NDArray:
        raise NotImplementedError


class BaseCircuit(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_instructions(self) -> list[Gate]:  # TODO: should be unit gate class
        raise NotImplementedError


class MBQCCircuit(BaseCircuit):
    def __init__(self, qubits: int) -> None:
        self.__num_qubits: int = qubits
        self.__gate_instructions: list[Gate] = []

    @property
    def num_qubits(self) -> int:
        return self.__num_qubits

    def get_instructions(self) -> list[Gate]:  # TODO: should be unit gate class
        return self.__gate_instructions

    def j(self, qubit: int, angle: float) -> None:
        self.__gate_instructions.append(J(qubit=qubit, angle=angle))

    def cz(self, qubit1: int, qubit2: int) -> None:
        self.__gate_instructions.append(CZ(qubits=(qubit1, qubit2)))

    def phase_gadget(self, qubits: list[int], angle: float) -> None:
        self.__gate_instructions.append(PhaseGadget(qubits=qubits, angle=angle))


class MacroCircuit(BaseCircuit):
    def __init__(self, qubits: int) -> None:
        self.__num_qubits: int = qubits
        self.__macro_gate_instructions: list[MacroGate] = []

    @property
    def num_qubits(self) -> int:
        return self.__num_qubits

    def get_instructions(self) -> list[Gate]:  # TODO: should be unit gate class
        gate_instructions = []
        for macro_gate in self.__macro_gate_instructions:
            gate_instructions.extend(macro_gate.get_unit_gates())
        return gate_instructions

    def apply_macro_gate(self, gate: MacroGate) -> None:
        self.__macro_gate_instructions.append(gate)


def circuit2graph(circuit: BaseCircuit) -> tuple[GraphState, FlowLike]:
    graph = GraphState()
    gflow = {}

    front_nodes = []  # list index  corresponds to qubit index
    num_nodes = 0

    # input nodes
    for i in range(circuit.num_qubits):
        graph.add_physical_node(num_nodes, is_input=True)
        graph.set_q_index(num_nodes, i)
        front_nodes.append(num_nodes)
        num_nodes += 1

    for instruction in circuit.get_instructions():
        if isinstance(instruction, J):
            graph.add_physical_node(num_nodes)
            graph.add_physical_edge(front_nodes[instruction.qubit], num_nodes)
            graph.set_meas_plane(front_nodes[instruction.qubit], Plane.XY)
            graph.set_meas_angle(front_nodes[instruction.qubit], -instruction.angle)
            graph.set_q_index(num_nodes, instruction.qubit)

            gflow[front_nodes[instruction.qubit]] = {num_nodes}
            front_nodes[instruction.qubit] = num_nodes

            num_nodes += 1

        elif isinstance(instruction, CZ):
            graph.add_physical_edge(front_nodes[instruction.qubits[0]], front_nodes[instruction.qubits[1]])
        elif isinstance(instruction, PhaseGadget):
            graph.add_physical_node(num_nodes)
            graph.set_meas_angle(num_nodes, instruction.angle)
            graph.set_meas_plane(num_nodes, Plane.YZ)
            for qubit in instruction.qubits:
                graph.add_physical_edge(front_nodes[qubit], num_nodes)

            gflow[num_nodes] = {num_nodes}

            num_nodes += 1
        else:
            raise ValueError("Invalid instruction")

    for node in front_nodes:
        graph.set_output(node)

    return graph, gflow
