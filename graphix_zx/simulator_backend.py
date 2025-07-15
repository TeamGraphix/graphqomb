"""The base class for simulator backends.

This module provides:

- `QubitIndexManager`: Manages the mapping of external qubit indices to internal indices
- `BaseSimulatorBackend`: Abstract base class for simulator backends.

"""

from __future__ import annotations

import abc
import typing
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from graphix_zx.common import MeasBasis


class QubitIndexManager:
    """Manages the mapping of external qubit indices to internal indices."""

    def __init__(self, initial_indices: Sequence[int]) -> None:
        """Initialize the QubitIndexManager with a list of initial indices."""
        self.__indices = list(initial_indices)

    def add_qubits(self, num_qubits: int) -> None:
        """Add a specified number of qubits to the index manager.

        Parameters
        ----------
        num_qubits : `int`
            The number of qubits to add.
        """
        current_max = max(self.__indices) if self.__indices else -1
        self.__indices.extend(range(current_max + 1, current_max + 1 + num_qubits))

    def remove_qubit(self, qubit: int) -> None:
        r"""Remove specified qubit from the index manager.

        Parameters
        ----------
        qubit : `int`
            The qubit to remove.
        """
        self.__indices = [q if q < qubit else q - 1 for q in self.__indices if q != qubit]

    def match(self, order: Sequence[int]) -> bool:
        r"""Check if the current indices match the given order.

        Parameters
        ----------
        order : `collections.abc.Sequence`\[`int`\]
            A sequence of indices to compare against the current indices.

        Returns
        -------
        `bool`
            True if the current indices match the given order, False otherwise.
        """
        return self.__indices == list(order)

    def reorder(self, permutation: Sequence[int]) -> None:
        r"""Reorder the indices based on a given permutation.

        if permutation is [2, 0, 1], then
        # [q0, q1, q2] -> [q1, q2, q0]

        Parameters
        ----------
        permutation : `collections.abc.Sequence`\[`int`\]
            A sequence of indices that defines the new order of the indices.

        Raises
        ------
        ValueError
            If the length of the permutation does not match the number of indices.
        """
        if len(permutation) != len(self.__indices):
            msg = "Permutation length must match the number of indices."
            raise ValueError(msg)
        self.__indices = [self.__indices[i] for i in permutation]

    def recovery_permutation(self) -> list[int]:
        r"""Get the permutation that would recover the original order of indices.

        Returns
        -------
        `list`\[`int`\]
            A sequence of indices that maps the current order back to the original order.
        """
        return [self.__indices.index(i) for i in range(len(self.__indices))]

    @typing.overload
    def external_to_internal(self, external_qubits: int) -> int: ...

    @typing.overload
    def external_to_internal(self, external_qubits: Sequence[int]) -> tuple[int, ...]: ...

    def external_to_internal(self, external_qubits: int | Sequence[int]) -> int | tuple[int, ...]:
        r"""Convert external qubit indices to internal indices.

        Parameters
        ----------
        external_qubits : `int` | `collections.abc.Sequence`\[`int`\]
            A sequence of external qubit indices.

        Returns
        -------
        `int` | `tuple`\[`int`, ...\]
            A list of internal qubit indices corresponding to the external ones.
        """
        if isinstance(external_qubits, int):
            return self.__indices[external_qubits]
        return tuple(self.__indices[q] for q in external_qubits)


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
    def evolve(self, operator: NDArray[np.complex128], qubits: int | Sequence[int]) -> None:
        r"""Evolve the state by applying an operator to a subset of qubits.

        Parameters
        ----------
        operator : `numpy.typing.NDArray`\[`numpy.complex128`\]
            The operator to apply.
        qubits : `int` | `collections.abc.Sequence`\[`int`\]
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
