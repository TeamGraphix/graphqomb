"""Module for simulating circuits and Measurement Patterns.

This module provides:

- `SimulatorBackend` : Enum class for circuit simulator backends.
- `CircuitSimulator` : Class for simulating circuits.
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
    from graphix_zx.simulator_backend import BaseFullStateSimulator


class SimulatorBackend(Enum):
    """Enum class for circuit simulator backend.

    Available backends are:
    - StateVector
    - DensityMatrix
    """

    StateVector = auto()
    DensityMatrix = auto()


class CircuitSimulator:
    r"""Class for simulating circuits.

    Attributes
    ----------
    state : `BaseFullStateSimulator`
        The quantum state of the simulator.
    gate_instructions : `list`\[`Gate`\]
        The list of gate instructions to be applied.
    """

    state: BaseFullStateSimulator
    gate_instructions: list[Gate]

    def __init__(self, mbqc_circuit: BaseCircuit, backend: SimulatorBackend) -> None:
        if backend == SimulatorBackend.StateVector:
            self.state = StateVector.from_num_qubits(mbqc_circuit.num_qubits)
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

        self.gate_instructions = mbqc_circuit.instructions()

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

        self.state.evolve(operator, qubits)

    def simulate(self) -> None:
        """Simulate the circuit."""
        for gate in self.gate_instructions:
            self.apply_gate(gate)


class PatternSimulator:
    r"""Class for simulating Measurement Patterns.

    Attributes
    ----------
    state : `BaseFullStateSimulator`
        The quantum state of the simulator.
    node_indices : `list`\[`int`\]
        The list of node indices in the pattern.
    results : `dict`\[`int`, `bool`\]
        The measurement results for each node.
    calc_prob : `bool`
        Whether to calculate probabilities.
    pattern : `Pattern`
        The measurement pattern being simulated.
    """

    state: BaseFullStateSimulator
    node_indices: list[int]
    results: dict[int, bool]
    calc_prob: bool
    pattern: Pattern

    def __init__(
        self,
        pattern: Pattern,
        backend: SimulatorBackend,
        *,
        calc_prob: bool = False,
    ) -> None:
        self.node_indices = list(pattern.input_node_indices.keys())
        self.results = {}

        self.calc_prob = calc_prob
        self.pattern = pattern

        # Pattern runnability check is done via is_runnable function
        is_runnable(self.pattern)

        if backend == SimulatorBackend.StateVector:
            # Note: deterministic check skipped for now
            self.state = StateVector.from_num_qubits(len(self.pattern.input_node_indices))
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

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
        for cmd in self.pattern.commands:
            self.apply_cmd(cmd)

        # Create a mapping from current node indices to output node indices
        output_mapping = {qindex: k for k, qindex in self.pattern.output_node_indices.items()}
        permutation = [output_mapping.get(node, -1) for node in self.node_indices]

        # Handle unmapped nodes (ancillas)
        max_output = max(self.pattern.output_node_indices.values()) if self.pattern.output_node_indices else -1
        next_ancilla = max_output + 1
        for i, p in enumerate(permutation):
            if p == -1:
                permutation[i] = next_ancilla
                next_ancilla += 1
        new_indices = [-1 for _ in range(len(permutation))]
        for i in range(len(permutation)):
            new_indices[permutation[i]] = self.node_indices[i]
        self.node_indices = new_indices
        self.state.reorder(permutation)

    def _apply_n(self, cmd: N) -> None:
        self.state.add_node(1)
        self.node_indices.append(cmd.node)

    def _apply_e(self, cmd: E) -> None:
        node_id1 = self.node_indices.index(cmd.nodes[0])
        node_id2 = self.node_indices.index(cmd.nodes[1])
        self.state.entangle(node_id1, node_id2)

    def _apply_m(self, cmd: M) -> None:
        if self.calc_prob:
            raise NotImplementedError
        rng = np.random.default_rng()
        result = rng.uniform() < 1 / 2

        if cmd.meas_basis.plane == Plane.XY:
            if self.pattern.pauli_frame.z_pauli[cmd.node]:
                basis: MeasBasis = cmd.meas_basis.flip()
            else:
                basis = cmd.meas_basis
        elif cmd.meas_basis.plane == Plane.YZ:
            basis = cmd.meas_basis.flip() if self.pattern.pauli_frame.x_pauli[cmd.node] else cmd.meas_basis
        elif self.pattern.pauli_frame.x_pauli[cmd.node] ^ self.pattern.pauli_frame.z_pauli[cmd.node]:
            basis = cmd.meas_basis.flip()
        else:
            basis = cmd.meas_basis

        node_id = self.node_indices.index(cmd.node)
        # Note: measure method requires MeasBasis and result as int
        self.state.measure(node_id, basis, int(result))
        self.results[cmd.node] = result
        self.node_indices.remove(cmd.node)

    def _apply_x(self, cmd: X) -> None:
        node_id = self.node_indices.index(cmd.node)
        if self.pattern.pauli_frame.x_pauli[cmd.node]:
            self.state.evolve(np.asarray([[0, 1], [1, 0]]), [node_id])

    def _apply_z(self, cmd: Z) -> None:
        node_id = self.node_indices.index(cmd.node)
        if self.pattern.pauli_frame.z_pauli[cmd.node]:
            self.state.evolve(np.asarray([[1, 0], [0, -1]]), [node_id])


# return permutation
def parse_q_indices(node_indices: Sequence[int], q_indices: Mapping[int, int]) -> list[int]:
    r"""Parse qubit indices and return permutation to sort the logical qubit indices.

    Parameters
    ----------
    node_indices : `collections.abc.Sequence`\[`int`\]
        mapping from qubit index of the state to node index of the pattern
    q_indices : `collections.abc.Mapping`\[`int`, `int`\]
        mapping from node index of the pattern to logical qubit index

    Returns
    -------
    `list`\[`int`\]
        The permutation to sort the logical qubit indices

    Raises
    ------
    ValueError
        If the qubit index is invalid
    """
    ancilla: set[int] = set()
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
