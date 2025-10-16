"""The base class for simulator backends.

This module provides:

- `QubitIndexManager`: Manages the mapping of external qubit indices to internal indices
- `BaseSimulatorBackend`: Abstract base class for simulator backends.
- `BaseFullStateSimulator`: Abstract base class for full state simulators.
"""

from __future__ import annotations

import abc
import itertools
import typing
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from graphqomb.common import MeasBasis


class QubitIndexManager:
    """Manages the mapping of external qubit indices to internal indices."""

    def __init__(self, num_qubits: int) -> None:
        """Initialize the QubitIndexManager with a list of initial indices."""
        self.__indices = list(range(num_qubits))

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits managed by this manager.

        Returns
        -------
        `int`
            The number of qubits.
        """
        return len(self.__indices)

    def add_qubits(self, num_qubits: int) -> None:
        """Add a specified number of qubits to the index manager.

        Parameters
        ----------
        num_qubits : `int`
            The number of qubits to add.
        """
        current_max = max(self.__indices, default=-1)
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
        return all(lhs == rhs for lhs, rhs in itertools.zip_longest(self.__indices, order, fillvalue=None))

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

    def inverse_permutation(self) -> list[int]:
        r"""Get the permutation that would recover the original order of indices.

        Returns
        -------
        `list`\[`int`\]
            A sequence of indices that maps the current order back to the original order.
        """
        inverse_perm = [0] * len(self.__indices)
        for i, index in enumerate(self.__indices):
            inverse_perm[index] = i
        return inverse_perm

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

    @typing.overload
    def internal_to_external(self, internal_qubits: int) -> int: ...

    @typing.overload
    def internal_to_external(self, internal_qubits: Sequence[int]) -> tuple[int, ...]: ...

    def internal_to_external(self, internal_qubits: int | Sequence[int]) -> int | tuple[int, ...]:
        r"""Convert internal qubit indices to external indices.

        Parameters
        ----------
        internal_qubits : `int` | `collections.abc.Sequence`\[`int`\]
            A sequence of internal qubit indices.

        Returns
        -------
        `int` | `tuple`\[`int`, ...\]
            A list of external qubit indices corresponding to the internal ones.
        """
        inverse_perm = self.inverse_permutation()
        if isinstance(internal_qubits, int):
            return inverse_perm[internal_qubits]
        return tuple(inverse_perm[q] for q in internal_qubits)


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


class BaseFullStateSimulator(BaseSimulatorBackend):
    """Base class for full state simulators."""

    @abc.abstractmethod
    def state(self) -> NDArray[np.complex128]:
        r"""Get the current state vector.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            The current state vector.
        """

    @abc.abstractmethod
    def norm(self) -> float:
        r"""Get the current state vector norm.

        Returns
        -------
        `float`
            The current state vector norm.
        """

    @abc.abstractmethod
    def add_node(self, num_qubits: int) -> None:
        r"""Add a node to the state.

        Parameters
        ----------
        num_qubits : `int`
            The number of qubits in the new node.
        """

    @abc.abstractmethod
    def entangle(self, qubit1: int, qubit2: int) -> None:
        r"""Entangle two qubits in the state.

        Parameters
        ----------
        qubit1 : `int`
            The first qubit to entangle.
        qubit2 : `int`
            The second qubit to entangle.
        """

    @abc.abstractmethod
    def reorder(self, permutation: list[int]) -> None:
        r"""Reorder the qubits in the state.

        Parameters
        ----------
        permutation : `list`\[`int`\]
            The permutation to apply.
        """

    @abc.abstractmethod
    def expectation(self, operator: NDArray[np.complex128], qubits: int | Sequence[int]) -> float:
        r"""Calculate the expectation value of an operator.

        Parameters
        ----------
        operator : `numpy.typing.NDArray`\[`numpy.complex128`\]
            The operator to calculate the expectation value for.
        qubits : `int` | `collections.abc.Sequence`\[`int`\]
            The qubits to apply the operator to.

        Returns
        -------
        `float`
            The expectation value of the operator.
        """
