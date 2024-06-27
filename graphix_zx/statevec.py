from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseStateVector(ABC):
    @abstractmethod
    def __init__(self, num_qubits: int):
        pass

    @abstractmethod
    def evolve(self, operator: NDArray, qubits: list[int]):
        pass

    @abstractmethod
    def measure(self, qubit: int, plane: str, angle: float):
        pass

    @abstractmethod
    def get_state_vector(self) -> NDArray:
        pass

    @abstractmethod
    def expectation_value(self, operator: NDArray) -> float:
        pass

    @abstractmethod
    def get_probabilities(self) -> NDArray:
        pass

    @abstractmethod
    def get_density_matrix(self) -> NDArray:
        pass


class StateVector(BaseStateVector):
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.__state = np.zeros(2**num_qubits, dtype=np.complex128)
        self.__state = np.ones(2**num_qubits) / np.sqrt(2**num_qubits)

    def evolve(self, operator: NDArray, qubits: list[int]):
        raise NotImplementedError

    def measure(self, qubit: int, plane: str, angle: float):
        raise NotImplementedError

    def get_state_vector(self) -> NDArray:
        return self.__state

    def expectation_value(self, operator: NDArray) -> float:
        raise NotImplementedError

    def get_probabilities(self) -> NDArray:
        raise NotImplementedError

    def get_density_matrix(self) -> NDArray:
        raise NotImplementedError
