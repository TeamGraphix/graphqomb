from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import get_meas_basis
from graphix_zx.simulator_backend import BaseSimulatorBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from graphix_zx.common import Plane

CZ_TENSOR = np.asarray(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
)


class BaseStateVector(BaseSimulatorBackend):
    @abstractmethod
    def __init__(self, num_qubits: int, state: NDArray | None = None) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def evolve(self, operator: NDArray, qubits: Sequence[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def measure(self, qubit: int, plane: Plane, angle: float, result: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def tensor_product(self, other: BaseStateVector) -> None:
        raise NotImplementedError

    @abstractmethod
    def normalize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reorder(self, permutation: Sequence[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def expectation_value(self, operator: NDArray, qubits: Sequence[int]) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_density_matrix(self) -> NDArray:
        raise NotImplementedError


class StateVector(BaseStateVector):
    def __init__(self, num_qubits: int, state: NDArray | None = None) -> None:
        self.__num_qubits = num_qubits
        if state is not None:
            # TODO: check state shape
            self.__state = state
        else:
            self.__state = np.ones(2**num_qubits) / np.sqrt(2**num_qubits)

    @property
    def num_qubits(self) -> int:
        return self.__num_qubits

    def evolve(self, operator: NDArray, qubits: Sequence[int]) -> None:
        """Apply operator to state

        Args:
            operator (NDArray): quantum operator matrix
            qubits (list[int]): target qubits
        """
        state = self.__state.reshape([2] * self.__num_qubits)
        operator = operator.reshape([2] * len(qubits) * 2)

        axes = (range(len(qubits), 2 * len(qubits)), qubits)
        state = np.tensordot(operator, state, axes=axes)

        state = np.moveaxis(state, range(len(qubits)), qubits)

        state = state.reshape(2**self.__num_qubits)

        self.__state = state

    def measure(self, qubit: int, plane: Plane, angle: float, result: int) -> None:
        """Measure qubit in specified basis

        Args:
            qubit (int): target qubit
            plane (Plane): measurement plane
            angle (float): measurement angle
            result (int): measurement result
        """
        basis = get_meas_basis(plane, angle + np.pi * result)
        state = self.__state.reshape([2] * self.__num_qubits)
        state = np.tensordot(basis.conjugate(), state, axes=(0, qubit))
        state = state.reshape(2 ** (self.__num_qubits - 1))

        self.__state = state
        self.__num_qubits -= 1

    def add_node(self, num_qubits: int) -> None:
        """Add |+> state to the end of state vector

        Args:
            num_qubits (int): number of qubits to be added
        """
        self.__state = np.kron(self.__state, np.ones(2**num_qubits) / np.sqrt(2**num_qubits))
        self.__num_qubits += num_qubits

    def entangle(self, qubits: tuple[int, int]) -> None:
        """Entangle two qubits

        Args:
            qubits (tuple[int, int]): target qubits
        """
        self.evolve(CZ_TENSOR, qubits)

    def tensor_product(self, other: BaseStateVector) -> None:
        """Tensor product with other state vector

        Args:
            other (BaseStateVector): other state vector
        """
        self.__state = np.kron(self.__state, other.get_array())
        self.__num_qubits += other.num_qubits

    def normalize(self) -> None:
        """Normalize state vector"""
        self.__state /= np.linalg.norm(self.__state)

    def reorder(self, permutation: Sequence[int]) -> None:
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

        return bool(np.isclose(match_rate, 1.0))

    def get_array(self) -> NDArray:
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

    def expectation_value(self, operator: NDArray, qubits: Sequence[int]) -> float:
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

        axes = (range(len(qubits), 2 * len(qubits)), qubits)
        state = np.tensordot(operator, state, axes=axes)

        state = np.moveaxis(state, range(len(qubits)), qubits).reshape(2**self.__num_qubits)

        return float(np.dot(self.__state.conjugate(), state) / np.linalg.norm(self.__state) ** 2)

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
