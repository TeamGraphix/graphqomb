from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from graphix_zx.common import Plane


# backend for all simulator backends
class BaseSimulatorBackend(ABC):
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
    def normalize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_norm(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_array(self) -> NDArray:
        raise NotImplementedError
