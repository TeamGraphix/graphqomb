from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from graphix_zx.common import Plane

CZ_TENSOR = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
)


class BaseStateVector(ABC):
    @abstractmethod
    def __init__(self, num_qubits: int, state: NDArray | None = None):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def evolve(self, operator: NDArray, qubits: list[int]):
        raise NotImplementedError

    @abstractmethod
    def measure(self, qubit: int, plane: Plane, angle: float, result: int):
        raise NotImplementedError

    @abstractmethod
    def tensor_product(self, other: BaseStateVector):
        raise NotImplementedError

    @abstractmethod
    def normalize(self):
        raise NotImplementedError

    @abstractmethod
    def reorder(self, permutation: list[int]):
        raise NotImplementedError

    @abstractmethod
    def get_state_vector(self) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def get_norm(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def expectation_value(self, operator: NDArray, qubits: list[int]) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_density_matrix(self) -> NDArray:
        raise NotImplementedError


class StateVector(BaseStateVector):
    def __init__(self, num_qubits: int, state: NDArray | None = None):
        self.__num_qubits = num_qubits
        if state is not None:
            self.__state = state
        else:
            self.__state = np.ones(2**num_qubits) / np.sqrt(2**num_qubits)

    @property
    def num_qubits(self) -> int:
        return self.__num_qubits

    def evolve(self, operator: NDArray, qubits: list[int]):
        """Apply operator to state

        Args:
            operator (NDArray): quantum operator matrix
            qubits (list[int]): target qubits
        """
        state = self.__state.reshape([2] * self.__num_qubits)
        operator = operator.reshape([2] * len(qubits) * 2)

        axes = (list(range(len(qubits), 2 * len(qubits))), qubits)
        state = np.tensordot(operator, state, axes=axes)

        state = np.moveaxis(state, list(range(len(qubits))), qubits)

        state = state.reshape(2**self.__num_qubits)

        self.__state = state

    def measure(self, qubit: int, plane: Plane, angle: float, result: int):
        """Measure qubit in specified basis

        Args:
            qubit (int): target qubit
            plane (Plane): measurement plane
            angle (float): measurement angle
            result (int): measurement result
        """
        basis = get_basis(plane, angle + np.pi * result)
        state = self.__state.reshape([2] * self.__num_qubits)
        state = np.tensordot(basis.conjugate(), state, axes=(0, qubit))
        state = state.reshape(2 ** (self.__num_qubits - 1))

        self.__state = state
        self.__num_qubits -= 1

    def add_node(self, num_qubits: int):
        """Add |+> state to the end of state vector

        Args:
            num_qubits (int): number of qubits to be added
        """
        self.__state = np.kron(self.__state, np.ones(2**num_qubits) / np.sqrt(2**num_qubits))
        self.__num_qubits += num_qubits

    def entangle(self, qubits: tuple[int, int]):
        """Entangle two qubits

        Args:
            qubits (tuple[int, int]): target qubits
        """
        self.evolve(CZ_TENSOR, list(qubits))

    def tensor_product(self, other: BaseStateVector):
        """Tensor product with other state vector

        Args:
            other (BaseStateVector): other state vector
        """
        self.__state = np.kron(self.__state, other.get_state_vector())
        self.__num_qubits += other.num_qubits

    def normalize(self):
        """Normalize state vector"""
        self.__state /= np.linalg.norm(self.__state)

    def reorder(self, permutation: list[int]):
        """Permute qubits

        if permutation is [2, 0, 1], then
        # [q0, q1, q2] -> [q1, q2, q0]

        Args:
            permutation (list[int]): permutation list
        """
        axes = [permutation.index(i) for i in range(self.__num_qubits)]
        state = self.__state.reshape([2] * self.__num_qubits)
        state = state.transpose(axes).flatten()
        self.__state = state

    def is_isolated(self, qubit: int) -> bool:
        """Check if qubit is isolated(product state)

        Args:
            qubit (int): target qubit

        Returns
        -------
            bool: True if isolated, False otherwise
        """
        state = self.__state.reshape([2] * self.__num_qubits)
        state0 = state.take(indices=0, axis=qubit)
        state1 = state.take(indices=1, axis=qubit)

        # normalize
        state0 /= np.linalg.norm(state0)
        state1 /= np.linalg.norm(state1)

        match_rate = np.dot(state0.conjugate(), state1)

        return np.isclose(match_rate, 1.0)

    def get_state_vector(self) -> NDArray:
        """Get state vector as numpy array

        Returns
        -------
            NDArray: state vector
        """
        return self.__state

    def get_norm(self) -> float:
        """Get norm of state vector

        Returns
        -------
            float: norm of state vector
        """
        return float(np.linalg.norm(self.__state))

    def expectation_value(self, operator: NDArray, qubits: list[int]) -> float:
        """Calculate expectation value of operator

        Args:
            operator (NDArray): Hermitian operator matrix
            qubits (list[int]): target qubits

        Returns
        -------
            float: expectation value
        """
        # TODO: check Hermitian
        state = self.__state.reshape([2] * self.__num_qubits)
        operator = operator.reshape([2] * len(qubits) * 2)

        axes = (list(range(len(qubits), 2 * len(qubits))), qubits)
        state = np.tensordot(operator, state, axes=axes)

        state = np.moveaxis(state, list(range(len(qubits))), qubits).reshape(2**self.__num_qubits)

        return np.dot(self.__state.conjugate(), state) / np.linalg.norm(self.__state) ** 2

    def get_density_matrix(self) -> NDArray:
        """Get density matrix

        Raises
        ------
            NotImplementedError: _description_

        Returns
        -------
            NDArray: density matrix
        """
        raise NotImplementedError


def get_basis(plane: Plane, angle: float) -> NDArray:
    """Get basis

    Args:
        plane (Plane): plane
        angle (float): angle

    Raises
    ------
        ValueError: invalid plane

    Returns
    -------
        NDArray: basis
    """
    if plane == Plane.XY:
        return np.array([1, np.exp(1j * angle)]) / np.sqrt(2)
    if plane == Plane.YZ:
        return np.array([np.cos(angle / 2), 1j * np.sin(angle / 2)])
    if plane == Plane.ZX:
        return np.array([np.cos(angle / 2), np.sin(angle / 2)])
    raise ValueError("Invalid plane")
