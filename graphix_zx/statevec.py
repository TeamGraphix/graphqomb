"""State vector representation module.

This module provides:

- `StateVector`: State vector representation class.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import typing_extensions

from graphix_zx.matrix import is_hermitian
from graphix_zx.simulator_backend import BaseSimulatorBackend, QubitIndexManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from graphix_zx.common import MeasBasis

CZ_TENSOR = np.asarray(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
)


class StateVector(BaseSimulatorBackend):
    r"""State vector representation."""

    def __init__(self, state: ArrayLike | None = None, *, copy: bool = False) -> None:
        if state is not None:
            state = np.array(state, dtype=np.complex128) if copy else np.asarray(state, dtype=np.complex128)
            num_qubits = np.log2(state.size)
            if not np.isclose(num_qubits, int(num_qubits)):
                msg = f"State vector size must be a power of 2, got {state.size}."
                raise ValueError(msg)
            num_qubits = int(num_qubits)
            #  Check if the state is a valid state vector
            self.__state = state.reshape((2,) * num_qubits)
        else:
            num_qubits = 0
            self.__state = np.zeros((2,) * 0, dtype=np.complex128)

        # Internal qubit ordering: maps external qubit index to internal index
        self.__qindex_mng = QubitIndexManager(range(num_qubits))

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the state vector.

        Returns
        -------
        `int`
            The number of qubits in the state vector.
        """
        return self.__state.ndim

    def state(self) -> NDArray[np.complex128]:
        r"""Get the state vector in external qubit order.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            The state vector as a numpy array in external qubit order.
        """
        # If internal order matches external order, return directly
        if self.__qindex_mng.match(range(self.num_qubits)):
            return self.__state

        # Otherwise, reorder to external qubit order
        axes = self.__qindex_mng.inverse_permutation()
        return self.__state.transpose(axes)

    def copy(self) -> StateVector:
        """Create a copy of the state vector.

        Returns
        -------
        `StateVector`
            A new `StateVector` instance with the same state.
        """
        return StateVector(self.state(), copy=True)

    @staticmethod
    def from_num_qubits(num_qubits: int) -> StateVector:
        """Create a state vector in the plus state with given number of qubits.

        Parameters
        ----------
        num_qubits : `int`
            Number of qubits in the state vector.

        Returns
        -------
        `StateVector`
            The resulting state vector in the plus state.

        Raises
        ------
        ValueError
            If num_qubits is negative.
        """
        if num_qubits < 0:
            msg = "Number of qubits must be non-negative."
            raise ValueError(msg)
        return StateVector(np.full((2,) * num_qubits, 1 / math.sqrt(2**num_qubits), dtype=np.complex128))

    @staticmethod
    def tensor_product(a: StateVector, b: StateVector) -> StateVector:
        """Tensor product with other state vector, self ⊗ other.

        Parameters
        ----------
        a : `StateVector`
            first state vector
        b : `StateVector`
            second state vector

        Returns
        -------
        `StateVector`
            The resulting state vector after tensor product.
        """
        return StateVector(np.kron(a.state(), b.state()))

    @typing_extensions.override
    def evolve(self, operator: NDArray[np.complex128], qubits: int | Sequence[int]) -> None:
        r"""Evolve the state by applying an operator to a subset of qubits.

        Parameters
        ----------
        operator : `numpy.typing.NDArray`\[`numpy.complex128`\]
            The operator to apply.
        qubits : `int` | `collections.abc.Sequence`\[`int`\]
            The qubits to apply the operator to.
        """
        # Convert external qubit indices to internal indices
        internal_qubits = self.__qindex_mng.external_to_internal(qubits)
        internal_qubits = (internal_qubits,) if isinstance(internal_qubits, int) else internal_qubits
        rest = tuple(i for i in range(self.num_qubits) if i not in internal_qubits)
        perm = internal_qubits + rest

        state_view = self.__state.transpose(perm)
        state_view = state_view.reshape(2 ** len(internal_qubits), 2 ** len(rest))

        op_view = operator.reshape(2 ** len(internal_qubits), 2 ** len(internal_qubits))

        new_state = op_view @ state_view

        inv_perm = np.argsort(perm)
        new_state = new_state.reshape((2,) * self.num_qubits)
        self.__state = new_state.transpose(inv_perm)

    @typing_extensions.override
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
        # Convert external qubit index to internal index
        internal_qubit = self.__qindex_mng.external_to_internal(qubit)

        meas_basis = meas_basis.flip() if result else meas_basis
        basis_vector = meas_basis.vector()
        new_state = np.tensordot(basis_vector.conjugate(), self.__state, axes=(0, internal_qubit)).astype(np.complex128)
        self.__state = new_state

        # Update qubit order: remove the measured qubit
        self.__qindex_mng.remove_qubit(internal_qubit)

        self.normalize()

    def add_node(self, num_qubits: int) -> None:
        """Add plus state to the end of state vector.

        Parameters
        ----------
        num_qubits : `int`
            number of qubits to add
        """
        flat_state = self.__state.flatten()
        flat_state = np.repeat(flat_state, 1 << num_qubits) / math.sqrt(2**num_qubits)
        self.__state = flat_state.reshape((2,) * (self.num_qubits + num_qubits))
        # Append new qubits to the end of the qubit order
        self.__qindex_mng.add_qubits(num_qubits)

    def entangle(self, qubit1: int, qubit2: int) -> None:
        r"""Entangle two qubits.

        Parameters
        ----------
        qubit1 : `int`
            first qubit index
        qubit2 : `int`
            second qubit index
        """
        self.evolve(CZ_TENSOR, (qubit1, qubit2))

    def normalize(self) -> None:
        """Normalize the state."""
        self.__state /= np.linalg.norm(self.__state)

    def reorder(self, permutation: Sequence[int]) -> None:
        r"""Permute qubits.

        if permutation is [2, 0, 1], then
        # [q0, q1, q2] -> [q1, q2, q0]

        Parameters
        ----------
        permutation : `collections.abc.Sequence`\[`int`\]
            permutation list
        """
        # Update the internal qubit order only (no state reordering)
        self.__qindex_mng.reorder(permutation)

    def norm(self) -> float:
        """Get norm of state vector.

        Returns
        -------
        `float`
            norm of state vector
        """
        return float(np.linalg.norm(self.__state))

    def expectation(self, operator: NDArray[np.complex128], qubits: int | Sequence[int]) -> float:
        r"""Calculate expectation value of operator.

        Parameters
        ----------
        operator : `numpy.typing.NDArray`\[`numpy.complex128`\]
            Hermitian operator matrix
        qubits : `int` | `collections.abc.Sequence`\[`int`\]
            target qubits

        Returns
        -------
        `float`
            expectation value

        Raises
        ------
        ValueError
            if operator is not Hermitian
        """
        if not is_hermitian(operator):
            msg = "Operator must be Hermitian"
            raise ValueError(msg)

        # Convert external qubit indices to internal indices
        internal_qubits = self.__qindex_mng.external_to_internal(qubits)
        internal_qubits = (internal_qubits,) if isinstance(internal_qubits, int) else internal_qubits
        rest = tuple(i for i in range(self.num_qubits) if i not in internal_qubits)
        perm = internal_qubits + rest

        # Reorder both ⟨ψ| and |ψ⟩ for efficient computation
        state_view = self.__state.transpose(perm)
        state_view = state_view.reshape(2 ** len(internal_qubits), 2 ** len(rest))

        op_view = operator.reshape(2 ** len(internal_qubits), 2 ** len(internal_qubits))

        # Apply operator: O|ψ⟩ (reordered)
        transformed_state = op_view @ state_view

        # Calculate expectation value: ⟨ψ|O|ψ⟩ using reordered states
        # No need to restore original order since both sides are reordered identically
        norm_squared = np.real(np.vdot(state_view.ravel(), state_view.ravel()))
        expectation = np.real(np.vdot(state_view.ravel(), transformed_state.ravel()))

        return float(expectation / norm_squared)
