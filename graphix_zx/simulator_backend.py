"""The base class for simulator backends."""

from __future__ import annotations

import abc
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from graphix_zx.common import MeasBasis


# backend for all simulator backends
class BaseSimulatorBackend(ABC):
    """Base class for simulator backends."""

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Get the number of qubits in the state.

        Returns
        -------
        `int`
            The number of qubits in the state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evolve(self, operator: NDArray[np.complex128], qubits: Sequence[int]) -> None:
        r"""Evolve the state by applying an operator to a subset of qubits.

        Parameters
        ----------
        operator : `numpy.typing.NDArray`\[`numpy.complex128`\]
            The operator to apply.
        qubits : `collections.abc.Sequence`\[`int`\]
            The qubits to apply the operator to.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def measure(self, qubit: int, meas_basis: MeasBasis, result: int) -> None:
        """Measure a qubit in a given measurement basis.

        Parameters
        ----------
        qubit : `int`
            The qubit to measure.
        meas_basis : `MeasBasis`
            The measurement basis to use.
        result : `int`
            The measurement result.
        """
        raise NotImplementedError
