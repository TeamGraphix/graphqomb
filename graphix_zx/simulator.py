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
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply_gate(self, gate: Gate) -> None:
        raise NotImplementedError

    @abstractmethod
    def simulate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> BaseStateVector:
        raise NotImplementedError


class SimulatorBackend(Enum):
    """Enum class for circuit simulator backend"""

    StateVector = auto()
    DensityMatrix = auto()


# NOTE: Currently, only XY plane is supported
class MBQCCircuitSimulator(BaseCircuitSimulator):
    def __init__(self, mbqc_circuit: MBQCCircuit, backend: SimulatorBackend) -> None:
        # NOTE: is it a correct backend switch?
        if backend == SimulatorBackend.StateVector:
            self.__state = StateVector(mbqc_circuit.num_qubits)
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

        self.__gate_instructions: list[UnitGate] = mbqc_circuit.get_instructions()

    def apply_gate(self, gate: Gate) -> None:
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
        for gate in self.__gate_instructions:
            self.apply_gate(gate)

    def get_state(self) -> StateVector:
        return self.__state


class BasePatternSimulator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def results(self) -> dict[int, bool]:
        raise NotImplementedError

    @abstractmethod
    def apply_cmd(self, cmd: Command) -> None:
        raise NotImplementedError

    @abstractmethod
    def simulate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(
        self,
    ) -> BaseSimulatorBackend:
        raise NotImplementedError


class PatternSimulator(BasePatternSimulator):
    def __init__(
        self,
        pattern: ImmutablePattern,
        backend: SimulatorBackend,
        *,
        calc_prob: bool = False,
    ) -> None:
        q_indices = pattern.get_q_indices()
        self.__node_indices: list[int] = [q_indices[input_node] for input_node in pattern.get_input_nodes()]
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
            self.__state = StateVector(len(self.__pattern.get_input_nodes()))
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            msg = f"Invalid backend: {backend}"
            raise ValueError(msg)

    @property
    def node_indices(self) -> list[int]:
        return self.__node_indices

    @property
    def results(self) -> dict[int, bool]:
        return self.__results

    def apply_cmd(self, cmd: Command) -> None:
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
        for cmd in self.__pattern.get_commands():
            self.apply_cmd(cmd)

        permutation = parse_q_indices(self.__node_indices, self.__pattern.get_q_indices())
        new_indices = [-1 for _ in range(len(permutation))]
        for i in range(len(permutation)):
            new_indices[permutation[i]] = self.__node_indices[i]
        self.__node_indices = new_indices
        self.__state.reorder(permutation)

    def get_state(self) -> BaseSimulatorBackend:
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

        angle = (-1) ** s_bool * cmd.angle + t_bool * np.pi

        node_id = self.__node_indices.index(cmd.node)
        self.__state.measure(node_id, cmd.plane, angle, result)
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
        raise NotImplementedError


# return permutation
def parse_q_indices(node_indices: Sequence[int], q_indices: Mapping[int, int]) -> list[int]:
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
