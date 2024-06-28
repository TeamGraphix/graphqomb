from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto

from graphix_zx.interface import MBQCCircuit, Gate, GateKind
from graphix_zx.statevec import BaseStateVector, StateVector


class BaseSimulator(ABC):
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


class CircuitSimulatorBackend(Enum):
    """Enum class for circuit simulator backend"""

    StateVector = auto()
    DensityMatrix = auto()


# NOTE: Currently, only XY plane is supported
class MBQCCircuitSimulator(BaseSimulator):
    def __init__(self, backend: CircuitSimulatorBackend, mbqc_circuit: MBQCCircuit):
        # NOTE: is it a correct backend switch?
        if backend == CircuitSimulatorBackend.StateVector:
            self.__state = StateVector(mbqc_circuit.num_qubits)
        elif backend == CircuitSimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            raise ValueError("Invalid backend")

        self.__gate_instructions: list[Gate] = mbqc_circuit.get_instructions()

    def apply_gate(self, gate: Gate):
        operator = gate.get_matrix()
        if gate.kind == GateKind.J:
            self.__state.evolve(operator, [gate.qubit])
        elif gate.kind == GateKind.CZ:
            self.__state.evolve(operator, list(gate.qubits))
        elif gate.kind == GateKind.PhaseGadget:
            raise NotImplementedError
        else:
            raise ValueError("Invalid gate")

    def simulate(self):
        for gate in self.__gate_instructions:
            self.apply_gate(gate)

    def get_state(self) -> StateVector:
        return self.__state
