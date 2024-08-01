from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class UnitGate(Enum):
    """Enum class for gate kind"""

    J = auto()
    CZ = auto()
    PhaseGadget = auto()


class Gate(ABC):
    @abstractmethod
    def get_unit_gates(self) -> list[Gate]:
        raise NotImplementedError

    @abstractmethod
    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass(frozen=True)
class J(Gate):
    qubit: int
    angle: float
    kind: UnitGate = UnitGate.J

    def get_unit_gates(self) -> list[Gate]:
        return [self]

    def get_matrix(self) -> NDArray:
        return np.array(
            [[1, np.exp(-1j * self.angle)], [1, -np.exp(-1j * self.angle)]]
        ) / np.sqrt(2)


@dataclass(frozen=True)
class CZ(Gate):
    qubits: tuple[int, int]
    kind: UnitGate = UnitGate.CZ

    def get_unit_gates(self) -> list[Gate]:
        return [self]

    def get_matrix(self) -> NDArray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


@dataclass(frozen=True)
class PhaseGadget(Gate):
    qubits: list[int]
    angle: float
    kind: UnitGate = UnitGate.PhaseGadget

    def get_unit_gates(self) -> list[Gate]:
        return [self]

    def get_matrix(self) -> NDArray:
        def count_ones_in_binary(array):
            count_ones = np.vectorize(lambda x: bin(x).count("1"))
            return count_ones(array)

        index_array = np.arange(2 ** len(self.qubits))
        z_sign = (-1) ** count_ones_in_binary(index_array)
        matrix = np.diag(np.exp(-1j * self.angle / 2 * z_sign))
        return matrix


# Macro gates


class MacroSingleGate(Gate):
    def get_matrix(self) -> NDArray:
        matrix = np.eye(2)
        for unit_gate in self.get_unit_gates():
            matrix = unit_gate.get_matrix() @ matrix
        return matrix


class MacroTwoQubitGate(Gate):
    pass


class MacroMultiGate(Gate):
    pass


@dataclass(frozen=True)
class I(MacroSingleGate):
    qubit: int

    def get_unit_gates(self) -> list[Gate]:
        return [J(self.qubit, 0), J(self.qubit, 0)]


@dataclass(frozen=True)
class X(MacroSingleGate):
    qubit: int

    def get_unit_gates(self) -> list[Gate]:
        return [J(self.qubit, np.pi), J(self.qubit, 0)]


@dataclass(frozen=True)
class Y(MacroSingleGate):
    qubit: int

    def get_unit_gates(self) -> list[Gate]:
        return [
            J(self.qubit, np.pi / 2),
            J(self.qubit, np.pi),
            J(self.qubit, -np.pi / 2),
            J(self.qubit, 0),
        ]


@dataclass(frozen=True)
class Z(MacroSingleGate):
    qubit: int

    def get_unit_gates(self) -> list[Gate]:
        return [J(self.qubit, 0), J(self.qubit, np.pi)]


@dataclass(frozen=True)
class H(MacroSingleGate):
    qubit: int

    def get_unit_gates(self) -> list[Gate]:
        return [J(self.qubit, 0)]


@dataclass(frozen=True)
class S(MacroSingleGate):
    qubit: int

    def get_unit_gates(self) -> list[Gate]:
        return [J(self.qubit, 0), J(self.qubit, np.pi / 2)]


@dataclass(frozen=True)
class Rx(MacroSingleGate):
    qubit: int
    angle: float

    def get_unit_gates(self) -> list[Gate]:
        return [
            J(self.qubit, self.angle),
            J(self.qubit, 0),
        ]


@dataclass(frozen=True)
class Ry(MacroSingleGate):
    qubit: int
    angle: float

    def get_unit_gates(self) -> list[Gate]:
        return [
            J(self.qubit, np.pi / 2),
            J(self.qubit, self.angle),
            J(self.qubit, -np.pi / 2),
            J(self.qubit, 0),
        ]


@dataclass(frozen=True)
class Rz(MacroSingleGate):
    qubit: int
    angle: float

    def get_unit_gates(self) -> list[Gate]:
        return [J(self.qubit, 0), J(self.qubit, self.angle)]


@dataclass(frozen=True)
class U3(MacroSingleGate):
    qubit: int
    angle1: float
    angle2: float
    angle3: float

    def get_unit_gates(self) -> list[Gate]:
        return [
            J(self.qubit, 0),
            J(self.qubit, -self.angle1),
            J(self.qubit, -self.angle2),
            J(self.qubit, -self.angle3),
        ]


@dataclass
class CNOT(MacroMultiGate):
    control: int
    target: int

    def get_unit_gates(self) -> list[Gate]:
        return [
            J(self.target, 0),
            CZ((self.control, self.target)),
            J(self.target, 0),
        ]

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass
class SWAP(MacroMultiGate):
    qubit1: int
    qubit2: int

    def get_unit_gates(self) -> list[Gate]:
        macro_gates = [
            CNOT(self.qubit1, self.qubit2),
            CNOT(self.qubit2, self.qubit1),
            CNOT(self.qubit1, self.qubit2),
        ]
        unit_gates = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass
class CRz(MacroMultiGate):
    control: int
    target: int
    angle: float

    def get_unit_gates(self) -> list[Gate]:
        raise NotImplementedError

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass
class CRx(MacroMultiGate):
    control: int
    target: int
    angle: float

    def get_unit_gates(self) -> list[Gate]:
        raise NotImplementedError

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass
class CU3(MacroMultiGate):
    control: int
    target: int
    angle1: float
    angle2: float
    angle3: float

    def get_unit_gates(self) -> list[Gate]:
        raise NotImplementedError

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass
class CCZ(MacroMultiGate):
    control1: int
    control2: int
    target: int

    def get_unit_gates(self) -> list[Gate]:
        raise NotImplementedError

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass
class Toffoli(MacroMultiGate):
    control1: int
    control2: int
    target: int

    def get_unit_gates(self) -> list[Gate]:
        raise NotImplementedError

    def get_matrix(self) -> NDArray:
        raise NotImplementedError
