"""Circuit classes for encoding quantum operations.

This module provides:

- `BaseCircuit`: An abstract base class for quantum circuits.
- `MBQCCircuit`: A circuit class composed solely of a unit gate set.
- `Circuit`: A class for circuits that include macro instructions.
- `circuit2graph`: A function that converts a circuit to a graph state and gflow.
"""

from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import typing_extensions

from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.gates import CZ, Gate, J, PhaseGadget, UnitGate
from graphqomb.graphstate import GraphState

if TYPE_CHECKING:
    from collections.abc import Sequence


class BaseCircuit(ABC):
    """
    Abstract base class for quantum circuits.

    This class defines the interface for quantum circuit objects.
    It enforces implementation of core methods that must be present
    in any subclass representing a specific type of quantum circuit.
    """

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Get the number of qubits in the circuit.

        Returns
        -------
        `int`
            The number of qubits in the circuit
        """
        raise NotImplementedError

    @abstractmethod
    def instructions(self) -> list[Gate]:
        r"""Get the list of gate instructions in the circuit.

        Returns
        -------
        `list`\[`Gate`\]
            List of gate instructions in the circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def unit_instructions(self) -> list[UnitGate]:
        r"""Get the list of unit gate instructions in the circuit.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gate instructions in the circuit.
        """
        raise NotImplementedError


class MBQCCircuit(BaseCircuit):
    """A circuit class composed solely of a unit gate set."""

    __num_qubits: int
    __gate_instructions: list[UnitGate]

    def __init__(self, num_qubits: int) -> None:
        self.__num_qubits = num_qubits
        self.__gate_instructions = []

    @property
    @typing_extensions.override
    def num_qubits(self) -> int:
        """Get the number of qubits in the circuit.

        Returns
        -------
        `int`
            The number of qubits in the circuit.
        """
        return self.__num_qubits

    @typing_extensions.override
    def instructions(self) -> list[Gate]:
        r"""Get the list of gate instructions in the circuit.

        Returns
        -------
        `list`\[`Gate`\]
            List of gate instructions in the circuit.
        """
        # For MBQCCircuit, Gate and UnitGate are the same
        return [copy.deepcopy(gate) for gate in self.__gate_instructions]

    @typing_extensions.override
    def unit_instructions(self) -> list[UnitGate]:
        r"""Get the list of unit gate instructions in the circuit.

        Returns
        -------
        `list`\[`UnitGate`\]
            List of unit gate instructions in the circuit.
        """
        return [copy.deepcopy(gate) for gate in self.__gate_instructions]

    def j(self, qubit: int, angle: float) -> None:
        """Add a J gate to the circuit.

        Parameters
        ----------
        qubit : `int`
            The qubit index.
        angle : `float`
            The angle of the J gate.
        """
        self.__gate_instructions.append(J(qubit=qubit, angle=angle))

    def cz(self, qubit1: int, qubit2: int) -> None:
        """Add a CZ gate to the circuit.

        Parameters
        ----------
        qubit1 : `int`
            The first qubit index.
        qubit2 : `int`
            The second qubit index.
        """
        self.__gate_instructions.append(CZ(qubits=(qubit1, qubit2)))

    def phase_gadget(self, qubits: Sequence[int], angle: float) -> None:
        r"""Add a phase gadget to the circuit.

        Parameters
        ----------
        qubits : `collections.abc.Sequence`\[`int`\]
            The qubit indices.
        angle : `float`
            The angle of the phase gadget
        """
        self.__gate_instructions.append(PhaseGadget(qubits=list(qubits), angle=angle))


class Circuit(BaseCircuit):
    """A class for circuits that include macro instructions."""

    __num_qubits: int
    __macro_gate_instructions: list[Gate]

    def __init__(self, num_qubits: int) -> None:
        self.__num_qubits = num_qubits
        self.__macro_gate_instructions = []

    @property
    @typing_extensions.override
    def num_qubits(self) -> int:
        """Get the number of qubits in the circuit.

        Returns
        -------
        `int`
            The number of qubits in the circuit.
        """
        return self.__num_qubits

    @typing_extensions.override
    def instructions(self) -> list[Gate]:
        r"""Get the list of gate instructions in the circuit.

        Returns
        -------
        `list`\[`Gate`\]
            List of gate instructions in the circuit.
        """
        return [copy.deepcopy(gate) for gate in self.__macro_gate_instructions]

    @typing_extensions.override
    def unit_instructions(self) -> list[UnitGate]:
        r"""Get the list of unit gate instructions in the circuit.

        Returns
        -------
        `list`\[`UnitGate`\]
            The list of unit gate instructions in the circuit.
        """
        return list(
            itertools.chain.from_iterable(macro_gate.unit_gates() for macro_gate in self.__macro_gate_instructions)
        )

    def apply_macro_gate(self, gate: Gate) -> None:
        """Apply a macro gate to the circuit.

        Parameters
        ----------
        gate : `Gate`
            The macro gate to apply.
        """
        self.__macro_gate_instructions.append(gate)


def circuit2graph(circuit: BaseCircuit) -> tuple[GraphState, dict[int, set[int]]]:
    r"""Convert a circuit to a graph state and gflow.

    Parameters
    ----------
    circuit : `BaseCircuit`
        The quantum circuit to convert.

    Returns
    -------
    `tuple`\[`GraphState`, `dict`\[`int`, `set`\[`int`\]\]\]
        The graph state and gflow converted from the circuit.

    Raises
    ------
    TypeError
        If the circuit contains an invalid instruction.
    """
    graph = GraphState()
    gflow: dict[int, set[int]] = {}

    qindex2front_nodes: dict[int, int] = {}

    # input nodes
    for i in range(circuit.num_qubits):
        node = graph.add_physical_node()
        graph.register_input(node, i)
        qindex2front_nodes[i] = node

    for instruction in circuit.unit_instructions():
        if isinstance(instruction, J):
            new_node = graph.add_physical_node()
            graph.add_physical_edge(qindex2front_nodes[instruction.qubit], new_node)
            graph.assign_meas_basis(
                qindex2front_nodes[instruction.qubit],
                PlannerMeasBasis(Plane.XY, -instruction.angle),
            )

            gflow[qindex2front_nodes[instruction.qubit]] = {new_node}
            qindex2front_nodes[instruction.qubit] = new_node

        elif isinstance(instruction, CZ):
            graph.add_physical_edge(
                qindex2front_nodes[instruction.qubits[0]],
                qindex2front_nodes[instruction.qubits[1]],
            )
        elif isinstance(instruction, PhaseGadget):
            new_node = graph.add_physical_node()
            graph.assign_meas_basis(new_node, PlannerMeasBasis(Plane.YZ, instruction.angle))
            for qubit in instruction.qubits:
                graph.add_physical_edge(qindex2front_nodes[qubit], new_node)

            gflow[new_node] = {new_node}
        else:
            msg = f"Invalid instruction: {instruction}"
            raise TypeError(msg)

    for qindex, node in qindex2front_nodes.items():
        graph.register_output(node, qindex)

    return graph, gflow
