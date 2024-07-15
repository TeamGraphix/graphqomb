from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np

from graphix_zx.circuit import MBQCCircuit, Gate, J, CZ, PhaseGadget
from graphix_zx.command import CommandKind
from graphix_zx.pattern import BasePattern, MutablePattern, ImmutablePattern
from graphix_zx.statevec import BaseStateVector, StateVector


class BaseCircuitSimulator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def apply_gate(self, gate: Gate):
        raise NotImplementedError

    @abstractmethod
    def simulate(self):
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
    def __init__(self, mbqc_circuit: MBQCCircuit, backend: SimulatorBackend):
        # NOTE: is it a correct backend switch?
        if backend == SimulatorBackend.StateVector:
            self.__state = StateVector(mbqc_circuit.num_qubits)
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            raise ValueError("Invalid backend")

        self.__gate_instructions: list[Gate] = mbqc_circuit.get_instructions()

    def apply_gate(self, gate: Gate):
        operator = gate.get_matrix()
        # may be refactored
        if isinstance(gate, J):
            if gate.qubit is None:
                raise ValueError("Invalid qubit")
            self.__state.evolve(operator, [gate.qubit])
        elif isinstance(gate, CZ):
            if gate.qubits is None:
                raise ValueError("Invalid qubits")
            self.__state.evolve(operator, list(gate.qubits))
        elif isinstance(gate, PhaseGadget):
            if gate.qubits is None:
                raise ValueError("Invalid qubit")
            raise NotImplementedError
        else:
            raise ValueError("Invalid gate")

    def simulate(self):
        for gate in self.__gate_instructions:
            self.apply_gate(gate)

    def get_state(self) -> StateVector:
        return self.__state


class BasePatternSimulator(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def results(self):
        raise NotImplementedError

    @abstractmethod
    def apply_cmd(self):
        raise NotImplementedError

    @abstractmethod
    def simulate(self):
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        raise NotImplementedError


class PatternSimulator(BasePatternSimulator):
    def __init__(
        self,
        pattern: BasePattern,
        backend: SimulatorBackend,
        calc_prob: bool = False,
    ):
        self.__node_indices: list[int] = pattern.get_input_nodes()
        self.__results: dict[int, bool] = {}

        self.__calc_prob: bool = calc_prob

        if isinstance(pattern, MutablePattern):
            self.__pattern = pattern.freeze()
        elif isinstance(pattern, ImmutablePattern):
            self.__pattern = pattern

        if not self.__pattern.is_runnable():
            raise ValueError("Pattern is not runnable")

        if backend == SimulatorBackend.StateVector:
            if not self.__pattern.is_deterministic():
                raise ValueError("Pattern is not deterministic. Please use DensityMatrix backend instead.")
            self.__state = StateVector(len(self.__pattern.get_input_nodes()))
        elif backend == SimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            raise ValueError("Invalid backend")

    @property
    def node_indices(self):
        return self.__node_indices

    @property
    def results(self):
        return self.__results

    def apply_cmd(self, cmd):
        if cmd.kind == CommandKind.N:
            self.__state.add_node(1)
            self.__node_indices.append(cmd.node)
        elif cmd.kind == CommandKind.E:
            node_id1 = self.__node_indices.index(cmd.nodes[0])
            node_id2 = self.__node_indices.index(cmd.nodes[1])
            self.__state.entangle((node_id1, node_id2))
        elif cmd.kind == CommandKind.M:
            if self.__calc_prob:
                raise NotImplementedError
            else:
                result = np.random.choice([0, 1])

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

            self.__state.normalize()  # NOTE: necessary?
        elif cmd.kind == CommandKind.X:
            node_id = self.__node_indices.index(cmd.node)
            # domain calculation
            result = False
            for node in cmd.domain:
                try:
                    result ^= self.__results[node]
                except KeyError:
                    raise KeyError(f"node {node} is not measured yet")
            if result:
                self.__state.evolve(np.array([[0, 1], [1, 0]]), [node_id])
            else:
                pass
        elif cmd.kind == CommandKind.Z:
            node_id = self.__node_indices.index(cmd.node)
            # domain calculation
            result = False
            for node in cmd.domain:
                try:
                    result ^= self.__results[node]
                except KeyError:
                    raise KeyError(f"node {node} is not measured yet")
            if result:
                self.__state.evolve(np.array([[1, 0], [0, -1]]), [node_id])
            else:
                pass
        elif cmd.kind == CommandKind.C:
            raise NotImplementedError

    def simulate(self):
        for cmd in self.__pattern.get_commands():
            self.apply_cmd(cmd)

        # permute node indices -> output nodes
        output_nodes = self.__pattern.get_output_nodes()
        permutation = [
            output_nodes.index(node) for node in self.__node_indices
        ]  # TODO: construct permutation based on q_indices
        self.__node_indices = output_nodes
        self.__state.reorder(permutation)

    def get_state(self):
        return self.__state
