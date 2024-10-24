"""Module for gates used in circuit representation.

This module provides:
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

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


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


class SingleGate(Gate):
    """Base class for single qubit macro gates."""

    qubit: int


class TwoQubitGate(Gate):
    """Base class for two qubit macro gates."""

    qubits: tuple[int, int]


class MultiGate(Gate):
    """Base class for multi qubit macro gates."""

    qubits: Sequence[int]


@dataclass(frozen=True)
class J(SingleGate):
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
class CZ(TwoQubitGate):
    """Class for the CZ gate.

    Attributes
    ----------
    qubits : tuple[int, int]
        The qubits the gate acts on.
    """

    qubits: tuple[int, int]

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        return [self]

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301 to align with pyright checks
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Matrix representation of the gate.
        """
        return np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex128)


@dataclass(frozen=True)
class PhaseGadget(MultiGate):
    """Class for the PhaseGadget gate.

    Attributes
    ----------
    qubits : Sequence[int]
        The qubits the gate acts on.
    angle : float
        The angle of the PhaseGadget gate.
    """

    qubits: Sequence[int]
    angle: float

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


if sys.version_info >= (3, 10):
    UnitGate = J | CZ | PhaseGadget
else:
    from typing import Union

    UnitGate = Union[J, CZ, PhaseGadget]


@dataclass(frozen=True)
class Identity(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Identity
        """
        return np.eye(2, dtype=np.complex128)


@dataclass(frozen=True)
class X(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            X gate
        """
        return np.asarray([[0, 1], [1, 0]], dtype=np.complex128)


@dataclass(frozen=True)
class Y(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Y gate
        """
        return np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)


@dataclass(frozen=True)
class Z(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Z gate
        """
        return np.asarray([[1, 0], [0, -1]], dtype=np.complex128)


@dataclass(frozen=True)
class H(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            H gate
        """
        array: NDArray[np.complex128] = (1 / np.sqrt(2)) * np.asarray([[1, 1], [1, -1]], dtype=np.complex128)
        return array


@dataclass(frozen=True)
class S(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            S gate
        """
        return np.asarray([[1, 0], [0, 1j]], dtype=np.complex128)


@dataclass(frozen=True)
class T(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            T gate
        """
        return np.asarray([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


@dataclass(frozen=True)
class Rx(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Rx gate
        """
        return np.asarray(
            [
                [np.cos(self.angle / 2), -1j * np.sin(self.angle / 2)],
                [-1j * np.sin(self.angle / 2), np.cos(self.angle / 2)],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class Ry(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Ry gate
        """
        return np.asarray(
            [[np.cos(self.angle / 2), -np.sin(self.angle / 2)], [np.sin(self.angle / 2), np.cos(self.angle / 2)]],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class Rz(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Rz gate
        """
        return np.asarray([[np.exp(-1j * self.angle / 2), 0], [0, np.exp(1j * self.angle / 2)]], dtype=np.complex128)


@dataclass(frozen=True)
class U3(SingleGate):
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

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            U3 gate
        """
        return np.asarray(
            [
                [np.cos(self.angle2 / 2), -np.exp(1j * self.angle3) * np.sin(self.angle2 / 2)],
                [
                    np.exp(1j * self.angle1) * np.sin(self.angle2 / 2),
                    np.exp(1j * (self.angle1 + self.angle3)) * np.cos(self.angle2 / 2),
                ],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CNOT(TwoQubitGate):
    """Class for the CNOT gate.

    Attributes
    ----------
    qubits : tuple[int, int]
        The qubits the gate acts on [control target].
    """

    qubits: tuple[int, int]

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        target = self.qubits[1]
        return [
            J(target, 0),
            CZ((self.qubits[0], self.qubits[1])),
            J(target, 0),
        ]

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Matrix representation of the gate.
        """
        return np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)


@dataclass(frozen=True)
class SWAP(TwoQubitGate):
    """Class for the SWAP gate.

    Attributes
    ----------
    qubits : tuple[int, int]
        The qubits the gate acts on [control target].
    """

    qubits: tuple[int, int]

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        control = self.qubits[0]
        target = self.qubits[1]
        macro_gates = [
            CNOT(self.qubits),
            CNOT((target, control)),
            CNOT(self.qubits),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CRz(TwoQubitGate):
    """Class for the CRz gate.

    Attributes
    ----------
    qubits : tuple[int, int]
        The qubits the gate acts on [control target].
    angle : float
        The angle of the CRz gate.
    """

    qubits: tuple[int, int]
    angle: float

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        target = self.qubits[1]
        macro_gates = [
            Rz(target, self.angle / 2),
            CNOT(self.qubits),
            Rz(target, -self.angle / 2),
            CNOT(self.qubits),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, np.exp(-1j * self.angle / 2), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * self.angle / 2)],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CRx(TwoQubitGate):
    """Class for the CRx gate.

    Attributes
    ----------
    qubits : tuple[int, int]
        The qubits the gate acts on [control target].
    angle : float
        The angle of the CRx gate.
    """

    qubits: tuple[int, int]
    angle: float

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        target = self.qubits[1]
        macro_gates = [
            Rz(target, np.pi / 2),
            CNOT(self.qubits),
            U3(target, -self.angle / 2, 0, 0),
            CNOT(self.qubits),
            U3(target, self.angle / 2, -np.pi / 2, 0),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
            Matrix representation of the gate.
        """
        return np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(self.angle), -1j * np.sin(self.angle)],
                [0, -1j * np.sin(self.angle), 0, np.cos(self.angle)],
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CU3(TwoQubitGate):
    """Class for the CU3 gate.

    Attributes
    ----------
    qubits : tuple[int, int]
        The qubits the gate acts on.
    angle1 : float
        The first angle of the CU3 gate.
    angle2 : float
        The second angle of the CU3 gate.
    angle3 : float
        The third angle of the CU3 gate.
    """

    qubits: tuple[int, int]
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
            Rz(self.qubits[0], self.angle3 / 2 + self.angle2 / 2),
            Rz(self.qubits[1], self.angle3 / 2 - self.angle2 / 2),
            CNOT(self.qubits),
            U3(self.qubits[1], -self.angle1 / 2, 0, -(self.angle2 + self.angle3) / 2),
            CNOT(self.qubits),
            U3(self.qubits[1], self.angle1 / 2, self.angle2, 0),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray[np.complex128]:
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
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
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CCZ(MultiGate):
    """Class for the CCZ gate.

    Attributes
    ----------
    qubits : Sequence[int]
        The qubits the gate acts on [control1, control2, target].
    """

    qubits: Sequence[int]

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        control1 = self.qubits[0]
        control2 = self.qubits[1]
        target = self.qubits[2]
        macro_gates = [
            CRz((control2, target), np.pi / 2),
            CNOT((control1, control2)),
            CRz((control2, target), -np.pi / 2),
            CNOT((control1, control2)),
            CRz((control1, target), np.pi / 2),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
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
            ],
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class Toffoli(MultiGate):
    """Class for the Toffoli gate.

    Attributes
    ----------
    qubits : Sequence[int]
        The qubits the gate acts on [control1, control2, target].
    """

    qubits: Sequence[int]

    def get_unit_gates(self) -> list[UnitGate]:
        """Get the unit gates that make up the gate.

        Returns
        -------
        list[UnitGate]
            List of unit gates that make up the gate.
        """
        control1 = self.qubits[0]
        control2 = self.qubits[1]
        target = self.qubits[2]
        macro_gates = [
            H(target),
            CRz((control2, target), np.pi / 2),
            CNOT((control1, control2)),
            CRz((control2, target), -np.pi / 2),
            CNOT((control1, control2)),
            CRz((control1, target), np.pi / 2),
            H(target),
        ]
        unit_gates: list[UnitGate] = []
        for macro_gate in macro_gates:
            unit_gates.extend(macro_gate.get_unit_gates())
        return unit_gates

    def get_matrix(self) -> NDArray[np.complex128]:  # noqa: PLR6301
        """Get the matrix representation of the gate.

        Returns
        -------
        NDArray[np.complex128]
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
            ],
            dtype=np.complex128,
        )
