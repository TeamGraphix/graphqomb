from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from graphix_zx.interface import MBQCCircuit
from graphix_zx.statevec import StateVector


class GateKind(Enum):
    """Enum class for gate kind"""

    J = "J"
    CZ = "CZ"
    PhaseGadget = "PHASE_GADGET"


@dataclass(frozen=True)
class Gate:
    kind: GateKind

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


class J(Gate):
    kind: GateKind = GateKind.J
    qubit: int
    angle: float

    def get_matrix(self) -> NDArray:
        return np.array([[1, np.exp(1j * self.angle)], [1, -np.exp(1j * self.angle)]]) / np.sqrt(2)


class CZ(Gate):
    kind: GateKind = GateKind.CZ
    qubits: tuple[int, int]

    def get_matrix(self) -> NDArray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


class PhaseGadget(Gate):
    kind: GateKind = GateKind.PhaseGadget
    qubits: list[int]
    angle: float

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


class BaseSimulator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def apply_gate(self, gate: Gate):
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def simulate(self) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def get_measurement(self) -> NDArray:
        raise NotImplementedError


class CircuitSimulatorBackend(Enum):
    """Enum class for circuit simulator backend"""

    StateVector = "StateVector"
    DensityMatrix = "DensityMatrix"


# NOTE: Currently, only XY plane is supported
class MBQCCircuitSimulator(BaseSimulator):
    def __init__(self, backend: CircuitSimulatorBackend, mbqc_circuit: MBQCCircuit):
        # NOTE: is it a correct backend switch?
        if backend == CircuitSimulatorBackend.StateVector:
            self.__state_vector = StateVector(mbqc_circuit.num_qubits)
        elif backend == CircuitSimulatorBackend.DensityMatrix:
            raise NotImplementedError
        else:
            raise ValueError("Invalid backend")

        self.__num_qubits = mbqc_circuit.num_qubits
        self.__front_nodes: set[int] = set()

    def apply_gate(self, gate: Gate):
        raise NotImplementedError
