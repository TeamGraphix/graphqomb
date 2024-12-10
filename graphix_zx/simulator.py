"""Module for simulating circuits and Measurement Patterns.

This module provides:
- BaseCircuitSimulator: Base class for circuit simulators.
- SimulatorBackend: Enum class for circuit
- MBQCCircuitSimulator: Class for simulating MBQC circuits.
- BasePatternSimulator: Base class for pattern simulators.
- PatternSimulator: Class for simulating Measurement Pattern.
- parse_q_indices: Parse qubit indices and return permutation to sort the logical qubit indices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.command import C, E, M, N, X, Z
from graphix_zx.gates import CZ, Gate, J, PhaseGadget, UnitGate
from graphix_zx.statevec import StateVector

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphix_zx.circuit import MBQCCircuit
    from graphix_zx.command import Command
    from graphix_zx.pattern import ImmutablePattern
    from graphix_zx.simulator_backend import BaseSimulatorBackend
    from graphix_zx.statevec import BaseStateVector


class BaseCircuitSimulator(ABC):
    """Base class for circuit simulators."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply_gate(self, gate: Gate) -> None:
        """Apply a gate to the circuit.

        Parameters
        ----------
        gate : Gate
            The gate to apply.
        """
        raise NotImplementedError

    @abstractmethod
    def simulate(self) -> None:
        """Simulate the circuit."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> BaseStateVector:
        """Get the quantum state as a state vector.

        Returns
        -------
        BaseStateVector
            The quantum state as a state vector.
        """
        raise NotImplementedError


class SimulatorBackend(Enum):
    """Enum class for circuit simulator backend.

    Available backends are:
    - StateVector
    - DensityMatrix
    """

    StateVector = auto()
    DensityMatrix = auto()


# NOTE: Currently, only XY plane is supported
class MBQCCircuitSimulator(BaseCircuitSimulator):
    """Class for simulating MBQC circuits.

    Attributes
    ----------
    __state : BaseStateVector
        The quantum state.
    __gate_instructions : list[UnitGate]
        The list of gate instructions in the circuit.
    """

    def __init__(self, mbqc_circuit: MBQCCircuit, backend: SimulatorBackend) -> None:
        if backend == SimulatorBackend.StateVector:
            self.__state = StateVector(mbqc_circuit.num_qubits)
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

        self.__gate_instructions: list[UnitGate] = mbqc_circuit.get_instructions()

    def apply_gate(self, gate: Gate) -> None:
        """Apply a gate to the circuit.

        Parameters
        ----------
        gate : Gate
            The gate to apply.

        Raises
        ------
        TypeError
            If the gate is not a valid gate.
        """
        operator = gate.get_matrix()
        # may be refactored
        if isinstance(gate, J):
            self.__state.evolve(operator, [gate.qubit])
        elif isinstance(gate, (CZ, PhaseGadget)):
            self.__state.evolve(operator, gate.qubits)
        else:
            msg = f"Invalid gate: {gate}"
            raise TypeError(msg)

    def simulate(self) -> None:
        """Simulate the circuit."""
        for gate in self.__gate_instructions:
            self.apply_gate(gate)

    def get_state(self) -> StateVector:
        """Get the quantum state as a state vector.

        Returns
        -------
        StateVector
            The quantum state as a state vector.
        """
        return self.__state


class BasePatternSimulator(ABC):
    """Base class for pattern simulators."""

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def results(self) -> dict[int, bool]:
        """Get the map from node index to measurement result.

        Returns
        -------
        dict[int, bool]
            Map from node index to measurement result.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_cmd(self, cmd: Command) -> None:
        """Apply a command to the pattern.

        Parameters
        ----------
        cmd : Command
            The command to apply.
        """
        raise NotImplementedError

    @abstractmethod
    def simulate(self) -> None:
        """Simulate the pattern."""
        raise NotImplementedError

    @abstractmethod
    def get_state(
        self,
    ) -> BaseSimulatorBackend:
        """Get the quantum state in a specified backend.

        Returns
        -------
        BaseSimulatorBackend
            The quantum state in a specified backend.
        """
        raise NotImplementedError


class PatternSimulator(BasePatternSimulator):
    """Class for simulating Measurement Pattern.

    Attributes
    ----------
    __pattern : ImmutablePattern
        The measurement pattern to simulate.
    __state : SimulatorBackend
        The simulator backend.
    __node_indices : list[int]
        Mapping from qubit index of the state to node index of the pattern.
    __results : dict[int, bool]
        The map from node index to measurement result.
    __calc_prob : bool
        Flag to calculate probability.
    """

    def __init__(
        self,
        pattern: ImmutablePattern,
        backend: SimulatorBackend,
        *,
        calc_prob: bool = False,
    ) -> None:
        self.__node_indices: list[int] = [pattern.q_indices[input_node] for input_node in pattern.input_nodes]
        self.__results: dict[int, bool] = {}

        self.__calc_prob: bool = calc_prob
        self.__pattern = pattern

        if not self.__pattern.is_runnable():
            msg = "Pattern is not runnable"
            raise ValueError(msg)

        if backend == SimulatorBackend.StateVector:
            if not self.__pattern.is_deterministic():
                msg = "Pattern is not deterministic. Please use DensityMatrix backend instead."
                raise ValueError(msg)
            self.__state = StateVector(len(self.__pattern.input_nodes))
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

    @property
    def node_indices(self) -> list[int]:
        """Get the mapping from qubit index of the state to node index of the pattern.

        Returns
        -------
        list[int]
            The mapping from qubit index of the state to node index of the pattern
        """
        return self.__node_indices

    @property
    def results(self) -> dict[int, bool]:
        """Get the map from node index to measurement result.

        Returns
        -------
        dict[int, bool]
            The map from node index to measurement result.
        """
        return self.__results

    def apply_cmd(self, cmd: Command) -> None:
        """Apply a command to the state.

        Parameters
        ----------
        cmd : Command
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
        elif isinstance(cmd, C):
            self._apply_c(cmd)
        else:
            msg = f"Invalid command: {cmd}"
            raise TypeError(msg)

    def simulate(self) -> None:
        """Simulate the pattern."""
        for cmd in self.__pattern.commands:
            self.apply_cmd(cmd)

        permutation = parse_q_indices(self.__node_indices, self.__pattern.q_indices)
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
        self.__state.entangle((node_id1, node_id2))

    def _apply_m(self, cmd: M) -> None:
        if self.__calc_prob:
            raise NotImplementedError
        rng = np.random.default_rng()
        result = rng.uniform() < 1 / 2

        s_bool = 0
        t_bool = 0
        for node in cmd.s_domain:
            s_bool ^= self.__results[node]
        for node in cmd.t_domain:
            t_bool ^= self.__results[node]

        angle = (-1) ** s_bool * cmd.meas_basis.angle + t_bool * np.pi

        node_id = self.__node_indices.index(cmd.node)
        self.__state.measure(node_id, cmd.meas_basis.plane, angle, result)
        self.__results[cmd.node] = result
        self.__node_indices.remove(cmd.node)

        self.__state.normalize()

    def _apply_x(self, cmd: X) -> None:
        node_id = self.__node_indices.index(cmd.node)
        # domain calculation
        result = False
        for node in cmd.domain:
            result ^= self.__results[node]
        if result:
            self.__state.evolve(np.asarray([[0, 1], [1, 0]]), [node_id])
        else:
            pass

    def _apply_z(self, cmd: Z) -> None:
        node_id = self.__node_indices.index(cmd.node)
        # domain calculation
        result = False
        for node in cmd.domain:
            result ^= self.__results[node]
        if result:
            self.__state.evolve(np.asarray([[1, 0], [0, -1]]), [node_id])
        else:
            pass

    def _apply_c(self, cmd: C) -> None:
        clifford = C.local_clifford.get_matrix()
        node_id = self.__node_indices.index(cmd.node)
        self.__state.evolve(clifford, [node_id])


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
