"""Circuit classes for encoding quantum operations.

This module provides:
- BaseCircuit: An abstract base class for quantum circuits.
- MBQCCircuit: A circuit class composed solely of a unit gate set.
- MacroCircuit: A class for circuits that include macro instructions.
- circuit2graph: A function that converts a circuit to a graph state and gflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from graphix_zx.common import Plane
from graphix_zx.gates import CZ, Gate, J, PhaseGadget, UnitGate
from graphix_zx.graphstate import GraphState

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix_zx.flow import FlowLike


class BaseCircuit(ABC):
    """
    Abstract base class for quantum circuits.

    This class defines the interface for quantum circuit objects.
    It enforces implementation of core methods that must be present
    in any subclass representing a specific type of quantum circuit.

    Attributes
    ----------
    num_qubits : int
        The number of qubits in the circuit.
    """

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Get the number of qubits in the circuit.

        Returns
        -------
        int
            The number of qubits in the circuit
        """
        raise NotImplementedError

    @abstractmethod
    def get_instructions(self) -> list[UnitGate]:
        """Get the list of instructions in the circuit.

        Returns
        -------
        list[UnitGate]
            List of unit instructions in the circuit.
        """
        raise NotImplementedError


class MBQCCircuit(BaseCircuit):
    """A circuit class composed solely of a unit gate set.

    Attributes
    ----------
    num_qubits : int
        The number of qubits in the circuit.
    """

    __num_qubits: int
    __gate_instructions: list[UnitGate]

    def __init__(self, num_qubits: int) -> None:
        self.__num_qubits = num_qubits
        self.__gate_instructions = []

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the circuit.

        Returns
        -------
        int
            The number of qubits in the circuit.
        """
        return self.__num_qubits

    def get_instructions(self) -> list[UnitGate]:
        """Get the list of instructions in the circuit.

        Returns
        -------
        list[UnitGate]
            List of unit instructions in the circuit.
        """
        return self.__gate_instructions

    def j(self, qubit: int, angle: float) -> None:
        """Add a J gate to the circuit.

        Parameters
        ----------
        qubit : int
            The qubit index.
        angle : float
            The angle of the J gate.
        """
        self.__gate_instructions.append(J(qubit=qubit, angle=angle))

    def cz(self, qubit1: int, qubit2: int) -> None:
        """Add a CZ gate to the circuit.

        Parameters
        ----------
        qubit1 : int
            The first qubit index.
        qubit2 : int
            The second qubit index.
        """
        self.__gate_instructions.append(CZ(qubits=(qubit1, qubit2)))

    def phase_gadget(self, qubits: Iterable[int], angle: float) -> None:
        """Add a phase gadget to the circuit.

        Parameters
        ----------
        qubits : Iterable[int]
            The qubit indices.
        angle : float
            The angle of the phase gadget
        """
        self.__gate_instructions.append(PhaseGadget(qubits=list(qubits), angle=angle))


class MacroCircuit(BaseCircuit):
    """A class for circuits that include macro instructions.

    Attributes
    ----------
    num_qubits : int
        The number of qubits in the circuit.
    """

    __num_qubits: int
    __macro_gate_instructions: list[Gate]

    def __init__(self, num_qubits: int) -> None:
        self.__num_qubits = num_qubits
        self.__macro_gate_instructions = []

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the circuit.

        Returns
        -------
        int
            The number of qubits in the circuit.
        """
        return self.__num_qubits

    def get_instructions(self) -> list[UnitGate]:
        """Get the list of instructions in the circuit.

        Returns
        -------
        list[UnitGate]
            The list of unit instructions in the circuit.
        """
        gate_instructions = []
        for macro_gate in self.__macro_gate_instructions:
            gate_instructions.extend(macro_gate.get_unit_gates())
        return gate_instructions

    def apply_macro_gate(self, gate: Gate) -> None:
        """Apply a macro gate to the circuit.

        Parameters
        ----------
        gate : Gate
            The macro gate to apply.
        """
        self.__macro_gate_instructions.append(gate)


def circuit2graph(circuit: BaseCircuit) -> tuple[GraphState, FlowLike]:
    """Convert a circuit to a graph state and gflow.

    Parameters
    ----------
    circuit : BaseCircuit
        The quantum circuit to convert.

    Returns
    -------
    tuple[GraphState, FlowLike]
        The graph state and gflow converted from the circuit.

    Raises
    ------
    TypeError
        If the circuit contains an invalid instruction.
    """
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
            msg = f"Invalid instruction {instruction=}"
            raise TypeError(msg)

    for node in front_nodes:
        graph.set_output(node)

    return graph, gflow
