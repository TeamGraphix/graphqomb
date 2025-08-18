"""Module for simulating circuits and Measurement Patterns.

This module provides:

- `SimulatorBackend` : Enum class for circuit simulator backends.
- `MBQCCircuitSimulator` : Class for simulating MBQC circuits.
- `PatternSimulator` : Class for simulating Measurement Patterns.
- `parse_q_indices` : Parse qubit indices and return permutation to sort the logical qubit indices.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.command import E, M, N, X, Z
from graphix_zx.common import MeasBasis, Plane
from graphix_zx.gates import MultiGate, SingleGate, TwoQubitGate
from graphix_zx.pattern import is_runnable
from graphix_zx.statevec import StateVector

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphix_zx.circuit import BaseCircuit
    from graphix_zx.command import Command
    from graphix_zx.gates import Gate
    from graphix_zx.pattern import Pattern
    from graphix_zx.simulator_backend import BaseSimulatorBackend


class SimulatorBackend(Enum):
    """Enum class for circuit simulator backend.

    Available backends are:
    - StateVector
    - DensityMatrix
    """

    StateVector = auto()
    DensityMatrix = auto()


class MBQCCircuitSimulator:
    """Class for simulating MBQC circuits."""

    __state: BaseSimulatorBackend
    __gate_instructions: list[Gate]

    def __init__(self, mbqc_circuit: BaseCircuit, backend: SimulatorBackend) -> None:
        if backend == SimulatorBackend.StateVector:
            self.__state = StateVector.from_num_qubits(mbqc_circuit.num_qubits)
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

        self.__gate_instructions = mbqc_circuit.instructions()

    @property
    def state(self) -> BaseSimulatorBackend:
        """Get the quantum state as a state vector.

        Returns
        -------
        `BaseSimulatorBackend`
            The quantum state.
        """
        return self.__state

    def apply_gate(self, gate: Gate) -> None:
        """Apply a gate to the circuit.

        Parameters
        ----------
        gate : `Gate`
            The gate to apply.

        Raises
        ------
        TypeError
            If the gate type is not supported.
        """
        operator = gate.matrix()

        # Get qubits that the gate acts on
        if isinstance(gate, SingleGate):
            # Single qubit gate
            qubits = [gate.qubit]
        elif isinstance(gate, (TwoQubitGate, MultiGate)):
            # Multi-qubit gate (both TwoQubitGate and MultiGate have qubits attribute)
            qubits = list(gate.qubits)
        else:
            msg = f"Cannot determine qubits for gate: {gate}"
            raise TypeError(msg)

        self.__state.evolve(operator, qubits)

    def simulate(self) -> None:
        """Simulate the circuit."""
        for gate in self.__gate_instructions:
            self.apply_gate(gate)


class PatternSimulator:
    """Class for simulating Measurement Patterns."""

    __state: BaseSimulatorBackend
    __node_indices: list[int]
    __results: dict[int, bool]
    __calc_prob: bool
    __pattern: Pattern

    def __init__(
        self,
        pattern: Pattern,
        backend: SimulatorBackend,
        *,
        calc_prob: bool = False,
    ) -> None:
        self.__node_indices = list(pattern.input_node_indices.keys())
        self.__results = {}

        self.__calc_prob = calc_prob
        self.__pattern = pattern

        # Pattern runnability check is done via is_runnable function
        is_runnable(self.__pattern)

        if backend == SimulatorBackend.StateVector:
            # Note: deterministic check skipped for now
            self.__state = StateVector.from_num_qubits(len(self.__pattern.input_node_indices))
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

    @property
    def node_indices(self) -> list[int]:
        r"""Get the mapping from qubit index of the state to node index of the pattern.

        Returns
        -------
        `list`\[`int`\]
            The mapping from qubit index of the state to node index of the pattern
        """
        return self.__node_indices

    @property
    def results(self) -> dict[int, bool]:
        r"""Get the map from node index to measurement result.

        Returns
        -------
        `dict`\[`int`, `bool`\]
            The map from node index to measurement result.
        """
        return self.__results

    def apply_cmd(self, cmd: Command) -> None:
        """Apply a command to the state.

        Parameters
        ----------
        cmd : `Command`
            The command to apply.

        Raises
        ------
        TypeError
            If the command is invalid
        """
        if isinstance(cmd, N):
            self._apply_n(cmd)
        elif isinstance(cmd, E):
            self._apply_e(cmd)
        elif isinstance(cmd, M):
            self._apply_m(cmd)
        elif isinstance(cmd, X):
            self._apply_x(cmd)
        elif isinstance(cmd, Z):
            self._apply_z(cmd)
        else:
            msg = f"Invalid command: {cmd}"
            raise TypeError(msg)

    def simulate(self) -> None:
        """Simulate the pattern."""
        for cmd in self.__pattern.commands:
            self.apply_cmd(cmd)

        # Create a mapping from current node indices to output node indices
        output_mapping = {qindex: k for k, qindex in self.__pattern.output_node_indices.items()}
        permutation = [output_mapping.get(node, -1) for node in self.__node_indices]

        # Handle unmapped nodes (ancillas)
        max_output = max(self.__pattern.output_node_indices.values()) if self.__pattern.output_node_indices else -1
        next_ancilla = max_output + 1
        for i, p in enumerate(permutation):
            if p == -1:
                permutation[i] = next_ancilla
                next_ancilla += 1
        new_indices = [-1 for _ in range(len(permutation))]
        for i in range(len(permutation)):
            new_indices[permutation[i]] = self.__node_indices[i]
        self.__node_indices = new_indices
        self.__state.reorder(permutation)

    def get_state(self) -> BaseSimulatorBackend:
        """Get the quantum state in a specified backend.

        Returns
        -------
        BaseSimulatorBackend
            The quantum state in a specified backend.
        """
        return self.__state

    def _apply_n(self, cmd: N) -> None:
        self.__state.add_node(1)
        self.__node_indices.append(cmd.node)

    def _apply_e(self, cmd: E) -> None:
        node_id1 = self.__node_indices.index(cmd.nodes[0])
        node_id2 = self.__node_indices.index(cmd.nodes[1])
        self.__state.entangle(node_id1, node_id2)

    def _apply_m(self, cmd: M) -> None:
        if self.__calc_prob:
            raise NotImplementedError
        rng = np.random.default_rng()
        result = rng.uniform() < 1 / 2

        if cmd.meas_basis.plane == Plane.XY:
            if self.__pattern.pauli_frame.z_pauli[cmd.node]:
                basis: MeasBasis = cmd.meas_basis.flip()
            else:
                basis = cmd.meas_basis
        elif cmd.meas_basis.plane == Plane.YZ:
            basis = cmd.meas_basis.flip() if self.__pattern.pauli_frame.x_pauli[cmd.node] else cmd.meas_basis
        elif self.__pattern.pauli_frame.x_pauli[cmd.node] ^ self.__pattern.pauli_frame.z_pauli[cmd.node]:
            basis = cmd.meas_basis.flip()
        else:
            basis = cmd.meas_basis

        node_id = self.__node_indices.index(cmd.node)
        # Note: measure method requires MeasBasis and result as int
        self.__state.measure(node_id, basis, int(result))
        self.__results[cmd.node] = result
        self.__node_indices.remove(cmd.node)

        self.__state.normalize()

    def _apply_x(self, cmd: X) -> None:
        node_id = self.__node_indices.index(cmd.node)
        if self.__pattern.pauli_frame.x_pauli[cmd.node]:
            self.__state.evolve(np.asarray([[0, 1], [1, 0]]), [node_id])

    def _apply_z(self, cmd: Z) -> None:
        node_id = self.__node_indices.index(cmd.node)
        if self.__pattern.pauli_frame.z_pauli[cmd.node]:
            self.__state.evolve(np.asarray([[1, 0], [0, -1]]), [node_id])


# return permutation
def parse_q_indices(node_indices: Sequence[int], q_indices: Mapping[int, int]) -> list[int]:
    """Parse qubit indices and return permutation to sort the logical qubit indices.

    Parameters
    ----------
    node_indices : Sequence[int]
        mapping from qubit index of the state to node index of the pattern
    q_indices : Mapping[int, int]
        mapping from node index of the pattern to logical qubit index

    Returns
    -------
    list[int]
        The permutation to sort the logical qubit indices

    Raises
    ------
    ValueError
        If the qubit index is invalid
    """
    ancilla = set()
    permutation = [-1 for _ in range(len(node_indices))]
    # check
    for node in node_indices:
        if q_indices[node] == -1:
            ancilla |= {node}
        elif q_indices[node] < 0 or q_indices[node] >= len(node_indices):
            msg = f"Invalid qubit index {q_indices[node]}"
            raise ValueError(msg)
        else:
            permutation[node_indices.index(node)] = q_indices[node]

    # fill ancilla
    ancilla_pos = len(node_indices) - len(ancilla)
    for node in ancilla:
        permutation[node_indices.index(node)] = ancilla_pos
        ancilla_pos += 1

    return permutation
