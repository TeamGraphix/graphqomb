"""The base class for simulator backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from graphix_zx.common import Plane


# backend for all simulator backends
class BaseSimulatorBackend(ABC):
    """Base class for simulator backends."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Get the number of qubits in the state.

        Returns
        -------
        int
            The number of qubits in the state.
        """
        raise NotImplementedError

    @abstractmethod
    def evolve(self, operator: NDArray, qubits: Sequence[int]) -> None:
        """Evolve the state by applying an operator to a subset of qubits.

        Parameters
        ----------
        operator : NDArray
            The operator to apply.
        qubits : Sequence[int]
            The qubits to apply the operator to.
        """
        raise NotImplementedError

    @abstractmethod
    def measure(self, qubit: int, plane: Plane, angle: float, result: int) -> None:
        """Measure a qubit in a given measurement basis.

        Parameters
        ----------
        qubit : int
            The qubit to measure.
        plane : Plane
            The measurement plane.
        angle : float
            The measurement angle.
        result : int
            The measurement result.
        """
        raise NotImplementedError

    @abstractmethod
    def normalize(self) -> None:
        """Normalize the state."""
        raise NotImplementedError

    @abstractmethod
    def get_norm(self) -> float:
        """Get the norm of the state.

        Returns
        -------
        float
            The norm of the state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_array(self) -> NDArray:
        """Get the state as a numpy array.

        Returns
        -------
        NDArray
            The state as a numpy array.
        """
        raise NotImplementedError
