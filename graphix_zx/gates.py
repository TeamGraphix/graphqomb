"""Module for gates used in circuit representation.

This module provides:
- UnitGateKind: Enum class for unit gate set.
- Gate: Abstract class for gates.
- UnitGate: Abstract class for unit gates.
- J: Class for the J gate.
- CZ: Class for the CZ gate.
- PhaseGadget: Class for the PhaseGadget gate.
- Identity: Class for the Identity gate.
- X: Class for the X gate.
- Y: Class for the Y gate.
- Z: Class for the Z gate.
- H: Class for the H gate.
- S: Class for the S gate.
- T: Class for the T gate.
- Rx: Class for the Rx gate.
- Ry: Class for the Ry gate.
- Rz: Class for the Rz gate.
- U3: Class for the U3 gate.
- CNOT: Class for the CNOT gate.
- SWAP: Class for the SWAP gate.
- CRz: Class for the CRz gate.
- CRx: Class for the CRx gate.
- CU3: Class for the CU3 gate.
- CCZ: Class for the CCZ gate.
- Toffoli: Class for the Toffoli gate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class UnitGateKind(Enum):
    """Enum class for unit gate set."""

    J = auto()
    CZ = auto()
    PhaseGadget = auto()


class Gate(ABC):
    """Abstract class for gates."""

    @abstractmethod
    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        raise NotImplementedError

    @abstractmethod
    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Matrix representation of the gate.
        """
        raise NotImplementedError


class UnitGate(Gate):
    """Abstract class for unit gates.

    Attributes
    ----------
    kind : UnitGateKind
        Which kind of unit gate it is.
    """

    kind: UnitGateKind


@dataclass(frozen=True)
class J(UnitGate):
    r"""Class for the J gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    angle : float
        The angle of the J gate.

        .. math::
        J = \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix}
        1 & e^{i\\theta} \\\\
        1 & -e^{i\\theta}
        \\end{pmatrix}

    kind : UnitGateKind
        Which kind of unit gate it is.
    """

    qubit: int
    angle: float
    kind: UnitGateKind = UnitGateKind.J

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [self]

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Matrix representation of the gate.
        """
        array: NDArray[np.complex128] = np.asarray(
            [[1, np.exp(1j * self.angle)], [1, -np.exp(1j * self.angle)]]
        ) / np.sqrt(2)
        return array


@dataclass(frozen=True)
class CZ(UnitGate):
    """Class for the CZ gate.

    Attributes
    ----------
    qubits : tuple[int, int]
        The qubits the gate acts on.
    kind : UnitGateKind
        Which kind of unit gate it is.
    """

    qubits: tuple[int, int]
    kind: UnitGateKind = UnitGateKind.CZ

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [self]

    def get_matrix(self) -> NDArray:  # noqa: PLR6301 to align with pyright checks
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray
            Matrix representation of the gate.
        """
        return np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


@dataclass(frozen=True)
class PhaseGadget(UnitGate):
    """Class for the PhaseGadget gate.

    Attributes
    ----------
    qubits : list[int]
        The qubits the gate acts on.
    angle : float
        The angle of the PhaseGadget gate.
    kind : UnitGateKind
        Which kind of unit gate it is.
    """

    qubits: list[int]
    angle: float
    kind: UnitGateKind = UnitGateKind.PhaseGadget

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [self]

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        TODO: Add the matrix representation of the PhaseGadget gate.

        Returns
        -------
        NDArray[np.complex128]
            Matrix representation of the gate.
        """

        def count_ones_in_binary(array: NDArray) -> NDArray[np.uint64]:
            count_ones = np.vectorize(lambda x: bin(x).count("1"))
            binary_array: NDArray[np.uint64] = count_ones(array)
            return binary_array

        index_array = np.arange(2 ** len(self.qubits))
        z_sign = (-1) ** count_ones_in_binary(index_array)
        return np.diag(np.exp(-1j * self.angle / 2 * z_sign))


# Macro gates


class MacroSingleGate(Gate):
    """Base class for single qubit macro gates."""

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate by multiplying the unit gates.

        Returns
        -------
        NDArray[np.complex128]
            Matrix representation of the gate.
        """
        matrix = np.eye(2, dtype=np.complex128)
        for unit_gate in self.get_unit_gates():
            matrix = unit_gate.get_matrix() @ matrix
        return matrix


class MacroTwoQubitGate(Gate):
    """Base class for two qubit macro gates."""


class MacroMultiGate(Gate):
    """Base class for multi qubit macro gates."""


@dataclass(frozen=True)
class Identity(MacroSingleGate):
    """Class for the Identity gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    """

    qubit: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, 0), J(self.qubit, 0)]


@dataclass(frozen=True)
class X(MacroSingleGate):
    """Class for the X gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    """

    qubit: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, np.pi), J(self.qubit, 0)]


@dataclass(frozen=True)
class Y(MacroSingleGate):
    """Class for the Y gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    """

    qubit: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [
            J(self.qubit, np.pi / 2),
            J(self.qubit, np.pi),
            J(self.qubit, -np.pi / 2),
            J(self.qubit, 0),
        ]


@dataclass(frozen=True)
class Z(MacroSingleGate):
    """Class for the Z gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    """

    qubit: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, 0), J(self.qubit, np.pi)]


@dataclass(frozen=True)
class H(MacroSingleGate):
    """Class for the H gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    """

    qubit: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, 0)]


@dataclass(frozen=True)
class S(MacroSingleGate):
    """Class for the S gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    """

    qubit: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, 0), J(self.qubit, np.pi / 2)]


@dataclass(frozen=True)
class T(MacroSingleGate):
    """Class for the T gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    """

    qubit: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, 0), J(self.qubit, np.pi / 4)]


@dataclass(frozen=True)
class Rx(MacroSingleGate):
    """Class for the Rx gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    angle : float
        The angle of the Rx gate.
    """

    qubit: int
    angle: float

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [
            J(self.qubit, self.angle),
            J(self.qubit, 0),
        ]


@dataclass(frozen=True)
class Ry(MacroSingleGate):
    """Class for the Ry gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    angle : float
        The angle of the Ry gate.
    """

    qubit: int
    angle: float

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [
            J(self.qubit, np.pi / 2),
            J(self.qubit, self.angle),
            J(self.qubit, -np.pi / 2),
            J(self.qubit, 0),
        ]


@dataclass(frozen=True)
class Rz(MacroSingleGate):
    """Class for the Rz gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    angle : float
        The angle of the Rz gate.
    """

    qubit: int
    angle: float

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [J(self.qubit, 0), J(self.qubit, self.angle)]


@dataclass(frozen=True)
class U3(MacroSingleGate):
    """Class for the U3 gate.

    Attributes
    ----------
    qubit : int
        The qubit the gate acts on.
    angle1 : float
        The first angle of the U3 gate.
    angle2 : float
        The second angle of the U3 gate.
    angle3 : float
        The third angle of the U3 gate.
    """

    qubit: int
    angle1: float
    angle2: float
    angle3: float

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [
            J(self.qubit, 0),
            J(self.qubit, -self.angle1),
            J(self.qubit, -self.angle2),
            J(self.qubit, -self.angle3),
        ]


@dataclass(frozen=True)
class CNOT(MacroMultiGate):
    """Class for the CNOT gate.

    Attributes
    ----------
    control : int
        The control qubit of the gate.
    target : int
        The target qubit of the gate.
    """

    control: int
    target: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [
            J(self.target, 0),
            CZ((self.control, self.target)),
            J(self.target, 0),
        ]

    def get_matrix(self) -> NDArray:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray
            Matrix representation of the gate.
        """
        return np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


@dataclass(frozen=True)
class SWAP(MacroMultiGate):
    """Class for the SWAP gate.

    Attributes
    ----------
    qubit1 : int
        The first qubit of the gate.
    qubit2 : int
        The second qubit of the gate.
    """

    qubit1: int
    qubit2: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        macro_gates = [
            CNOT(self.qubit1, self.qubit2),
            CNOT(self.qubit2, self.qubit1),
            CNOT(self.qubit1, self.qubit2),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )


@dataclass(frozen=True)
class CRz(MacroMultiGate):
    """Class for the CRz gate.

    Attributes
    ----------
    control : int
        The control qubit of the gate.
    target : int
        The target qubit of the gate.
    angle : float
        The angle of the CRz gate.
    """

    control: int
    target: int
    angle: float

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        macro_gates = [
            Rz(self.target, self.angle / 2),
            CNOT(self.control, self.target),
            Rz(self.target, -self.angle / 2),
            CNOT(self.control, self.target),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, np.exp(-1j * self.angle / 2), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * self.angle / 2)],
            ]
        )


@dataclass(frozen=True)
class CRx(MacroMultiGate):
    """Class for the CRx gate.

    Attributes
    ----------
    control : int
        The control qubit of the gate.
    target : int
        The target qubit of the gate.
    angle : float
        The angle of the CRx gate.
    """

    control: int
    target: int
    angle: float

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        macro_gates = [
            Rz(self.target, np.pi / 2),
            CNOT(self.control, self.target),
            U3(self.target, -self.angle / 2, 0, 0),
            CNOT(self.control, self.target),
            U3(self.target, self.angle / 2, -np.pi / 2, 0),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(self.angle), -1j * np.sin(self.angle)],
                [0, -1j * np.sin(self.angle), 0, np.cos(self.angle)],
            ]
        )


@dataclass(frozen=True)
class CU3(MacroMultiGate):
    """Class for the CU3 gate.

    Attributes
    ----------
    control : int
        The control qubit of the gate.
    target : int
        The target qubit of the gate.
    angle1 : float
        The first angle of the CU3 gate.
    angle2 : float
        The second angle of the CU3 gate.
    angle3 : float
        The third angle of the CU3 gate.
    """

    control: int
    target: int
    angle1: float
    angle2: float
    angle3: float

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        macro_gates = [
            Rz(self.control, self.angle3 / 2 + self.angle2 / 2),
            Rz(self.target, self.angle3 / 2 - self.angle2 / 2),
            CNOT(self.control, self.target),
            U3(self.target, -self.angle1 / 2, 0, -(self.angle2 + self.angle3) / 2),
            CNOT(self.control, self.target),
            U3(self.target, self.angle1 / 2, self.angle2, 0),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [
                    0,
                    0,
                    np.cos(self.angle1),
                    -np.exp(1j * self.angle3) * np.sin(self.angle1),
                ],
                [
                    0,
                    0,
                    np.exp(1j * self.angle2) * np.sin(self.angle1),
                    np.exp(1j * (self.angle2 + self.angle3)) * np.cos(self.angle1),
                ],
            ]
        )


@dataclass(frozen=True)
class CCZ(MacroMultiGate):
    """Class for the CCZ gate.

    Attributes
    ----------
    control1 : int
        The first control qubit of the gate.
    control2 : int
        The second control qubit of the gate.
    target : int
        The target qubit of the gate.
    """

    control1: int
    control2: int
    target: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        macro_gates = [
            CRz(self.control2, self.target, np.pi / 2),
            CNOT(self.control1, self.control2),
            CRz(self.control2, self.target, -np.pi / 2),
            CNOT(self.control1, self.control2),
            CRz(self.control1, self.target, np.pi / 2),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )


@dataclass(frozen=True)
class Toffoli(MacroMultiGate):
    """Class for the Toffoli gate.

    Attributes
    ----------
    control1 : int
        The first control qubit of the gate.
    control2 : int
        The second control qubit of the gate.
    target : int
        The target qubit of the gate.
    """

    control1: int
    control2: int
    target: int

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        macro_gates = [
            H(self.target),
            CRz(self.control2, self.target, np.pi / 2),
            CNOT(self.control1, self.control2),
            CRz(self.control2, self.target, -np.pi / 2),
            CNOT(self.control1, self.control2),
            CRz(self.control1, self.target, np.pi / 2),
            H(self.target),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )
