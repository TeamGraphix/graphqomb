"""Module for simulating circuits and Measurement Patterns.

This module provides:

- `SimulatorBackend` : Enum class for circuit simulator backends.
- `CircuitSimulator` : Class for simulating circuits.
- `PatternSimulator` : Class for simulating Measurement Patterns.
"""

from __future__ import annotations

import functools
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from graphqomb.command import TICK, E, M, N, X, Z
from graphqomb.common import MeasBasis, Plane
from graphqomb.gates import MultiGate, SingleGate, TwoQubitGate
from graphqomb.pattern import is_runnable
from graphqomb.rng import ensure_rng
from graphqomb.statevec import StateVector

if TYPE_CHECKING:
    from graphqomb.circuit import BaseCircuit
    from graphqomb.command import Command
    from graphqomb.gates import Gate
    from graphqomb.pattern import Pattern
    from graphqomb.simulator_backend import BaseFullStateSimulator


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
    """

    state: BaseFullStateSimulator
    node_indices: list[int]
    results: dict[int, bool]
    calc_prob: bool
    __pattern: Pattern

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
        self.__pattern = pattern

        # Pattern runnability check is done via is_runnable function
        is_runnable(self.__pattern)

        if backend == SimulatorBackend.StateVector:
            # Note: deterministic check skipped for now
            self.state = StateVector.from_num_qubits(len(self.__pattern.input_node_indices))
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

    @functools.singledispatchmethod
    def apply_cmd(self, cmd: Command, *, rng: np.random.Generator) -> None:
        """Apply a command to the state.

        Parameters
        ----------
        cmd : `Command`
            The command to apply.
        rng : `numpy.random.Generator`
            Random number generator to use.
        """
        self.apply_cmd(cmd, rng=rng)

    @apply_cmd.register
    def _(self, cmd: N, *, rng: np.random.Generator) -> None:  # noqa: ARG002
        self.state.add_node(1)
        self.node_indices.append(cmd.node)

    @apply_cmd.register
    def _(self, cmd: E, *, rng: np.random.Generator) -> None:  # noqa: ARG002
        node_id1 = self.node_indices.index(cmd.nodes[0])
        node_id2 = self.node_indices.index(cmd.nodes[1])
        self.state.entangle(node_id1, node_id2)

    @apply_cmd.register
    def _(self, cmd: M, *, rng: np.random.Generator) -> None:
        if self.calc_prob:
            raise NotImplementedError
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

        node_id = self.node_indices.index(cmd.node)
        self.state.measure(node_id, basis, result)
        self.results[cmd.node] = result
        self.node_indices.remove(cmd.node)

        if result:
            self.__pattern.pauli_frame.meas_flip(cmd.node)

    @apply_cmd.register
    def _(self, cmd: X, *, rng: np.random.Generator) -> None:  # noqa: ARG002
        node_id = self.node_indices.index(cmd.node)
        if self.__pattern.pauli_frame.x_pauli[cmd.node]:
            self.state.evolve(np.asarray([[0, 1], [1, 0]]), node_id)

    @apply_cmd.register
    def _(self, cmd: Z, *, rng: np.random.Generator) -> None:  # noqa: ARG002
        node_id = self.node_indices.index(cmd.node)
        if self.__pattern.pauli_frame.z_pauli[cmd.node]:
            self.state.evolve(np.asarray([[1, 0], [0, -1]]), node_id)

    @apply_cmd.register
    def _(self, cmd: TICK, *, rng: np.random.Generator) -> None:
        # TICK is a time separator that doesn't affect quantum state
        pass

    def simulate(self, rng: np.random.Generator | None = None) -> None:
        """
        Simulate the pattern.

        Parameters
        ----------
        rng : `numpy.random.Generator` | None, optional
            Random number generator to use for measurement outcomes.
            If None, a new generator will be created using the default random source. Default is None.

        """
        rng = ensure_rng(rng)
        for cmd in self.__pattern.commands:
            self.apply_cmd(cmd, rng=rng)

        # Create a mapping from current node indices to output node indices
        permutation = [self.__pattern.output_node_indices[node] for node in self.node_indices]

        self.state.reorder(permutation)
