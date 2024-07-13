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
        pass

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        pass

    @abstractmethod
    def evolve(self, operator: NDArray, qubits: list[int]):
        pass

    @abstractmethod
    def measure(self, qubit: int, plane: Plane, angle: float, result: int):
        pass

    @abstractmethod
    def tensor_product(self, other: BaseStateVector):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def reorder(self, permutation: list[int]):
        pass

    @abstractmethod
    def get_state_vector(self) -> NDArray:
        pass

    @abstractmethod
    def get_norm(self) -> float:
        pass

    @abstractmethod
    def expectation_value(self, operator: NDArray, qubits: list[int]) -> float:
        pass

    @abstractmethod
    def get_density_matrix(self) -> NDArray:
        pass


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
        state = self.__state.reshape([2] * self.__num_qubits)
        operator = operator.reshape([2] * len(qubits) * 2)

        axes = (list(range(len(qubits), 2 * len(qubits))), qubits)
        state = np.tensordot(operator, state, axes=axes)

        state = np.moveaxis(state, list(range(len(qubits))), qubits).reshape(2**self.__num_qubits)

        state = state.reshape(2**self.__num_qubits)

        self.__state = state

    def measure(self, qubit: int, plane: Plane, angle: float, result: int):
        basis = get_basis(plane, angle + np.pi * result)
        state = self.__state.reshape([2] * self.__num_qubits)
        state = np.tensordot(basis.conjugate(), state, axes=(0, qubit))
        state = state.reshape(2 ** (self.__num_qubits - 1))

        self.__state = state
        self.__num_qubits -= 1

    def add_node(self, num_qubits: int):
        self.__state = np.kron(self.__state, np.ones(2**num_qubits) / np.sqrt(2**num_qubits))
        self.__num_qubits += num_qubits

    def entangle(self, qubits: tuple[int, int]):
        self.evolve(CZ_TENSOR, list(qubits))

    def tensor_product(self, other: BaseStateVector):
        self.__state = np.kron(self.__state, other.get_state_vector())
        self.__num_qubits += other.num_qubits

    def normalize(self):
        self.__state /= np.linalg.norm(self.__state)

    def reorder(self, permutation: list[int]):
        state = self.__state.reshape([2] * self.__num_qubits)
        state = np.transpose(state, permutation)
        self.__state = state.reshape(2**self.__num_qubits)

    # check if qubit is isolated(product state)
    def is_isolated(self, qubit: int) -> bool:
        state = self.__state.reshape([2] * self.__num_qubits)
        state0 = state.take(indices=0, axis=qubit)
        state1 = state.take(indices=1, axis=qubit)

        # normalize
        state0 /= np.linalg.norm(state0)
        state1 /= np.linalg.norm(state1)

        match_rate = np.dot(state0.conjugate(), state1)

        return np.isclose(match_rate, 1.0)

    def get_norm(self) -> float:
        return float(np.linalg.norm(self.__state))

    def get_state_vector(self) -> NDArray:
        return self.__state

    def expectation_value(self, operator: NDArray, qubits: list[int]) -> float:
        # TODO: check Hermitian
        state = self.__state.reshape([2] * self.__num_qubits)
        operator = operator.reshape([2] * len(qubits) * 2)

        axes = (list(range(len(qubits), 2 * len(qubits))), qubits)
        state = np.tensordot(operator, state, axes=axes)

        state = np.moveaxis(state, list(range(len(qubits))), qubits).reshape(2**self.__num_qubits)

        return np.dot(self.__state.conjugate(), state) / np.linalg.norm(self.__state) ** 2

    def get_density_matrix(self) -> NDArray:
        raise NotImplementedError


def get_basis(plane: Plane, angle: float) -> NDArray:
    if plane == Plane.XY:
        return np.array([1, np.exp(1j * angle)]) / np.sqrt(2)
    elif plane == Plane.YZ:
        return np.array([np.cos(angle / 2), np.sin(angle / 2)])
    elif plane == Plane.ZX:
        return np.array([np.cos(angle / 2), 1j * np.sin(angle / 2)])
    else:
        raise ValueError("Invalid plane")
