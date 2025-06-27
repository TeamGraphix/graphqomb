"""State vector representation module.

This module provides:

- `StateVector`: State vector representation class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import typing_extensions

from graphix_zx.matrix import is_hermitian
from graphix_zx.simulator_backend import BaseSimulatorBackend

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

    def __init__(self, state: ArrayLike | None = None, *, num_qubits: int = 0) -> None:
        if state is not None:
            state = np.asarray(state, dtype=np.complex128)
            #  Check if the state is a valid state vector
            if state.ndim != 1 or state.size != 2**num_qubits:
                msg = f"State vector must be a 1D array of size {2**num_qubits}, got {state.shape}."
                raise ValueError(msg)
            self.__state = state.reshape((2,) * num_qubits)
        else:
            init_state = np.ones(2**num_qubits, dtype=np.complex128) / np.sqrt(2**num_qubits)
            self.__state = init_state.reshape((2,) * num_qubits)

        # Internal qubit ordering: maps external qubit index to internal index
        self.__qubit_order = list(range(num_qubits))

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the state vector.

        Returns
        -------
        `int`
            The number of qubits in the state vector.
        """
        return len(self.__state.shape)

    @property
    def state(self) -> NDArray[np.complex128]:
        r"""Get the state vector in external qubit order.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            The state vector as a numpy array in external qubit order.
        """
        # If internal order matches external order, return directly
        if self.__qubit_order == list(range(self.num_qubits)):
            return self.__state.flatten()

        # Otherwise, reorder to external qubit order
        axes = [self.__qubit_order.index(i) for i in range(self.num_qubits)]
        return self.__state.transpose(axes).flatten()

    def _external_to_internal_qubits(self, external_qubits: Sequence[int]) -> tuple[int, ...]:
        r"""Convert external qubit indices to internal indices.

        Parameters
        ----------
        external_qubits : `collections.abc.Sequence`\[`int`\]
            External qubit indices.

        Returns
        -------
        `tuple`\[`int`, ...\]
            Internal qubit indices.
        """
        return tuple(self.__qubit_order[q] for q in external_qubits)

    def copy(self) -> StateVector:
        """Create a copy of the state vector.

        Returns
        -------
        `StateVector`
            A new `StateVector` instance with the same state.
        """
        return StateVector(self.state, num_qubits=self.num_qubits)

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
        return StateVector(np.kron(a.state, b.state), num_qubits=a.num_qubits + b.num_qubits)

    @typing_extensions.override
    def evolve(self, operator: NDArray[np.complex128], qubits: Sequence[int]) -> None:
        r"""Evolve the state by applying an operator to a subset of qubits.

        Parameters
        ----------
        operator : `numpy.typing.NDArray`\[`numpy.complex128`\]
            The operator to apply.
        qubits : `collections.abc.Sequence`\[`int`\]
            The qubits to apply the operator to.
        """
        # Convert external qubit indices to internal indices
        internal_qubits = self._external_to_internal_qubits(qubits)
        internal_qubits = tuple(internal_qubits)
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
        internal_qubit = self.__qubit_order[qubit]

        meas_basis = meas_basis.flip() if result else meas_basis
        basis_vector = meas_basis.vector()
        new_state = np.tensordot(basis_vector.conjugate(), self.__state, axes=(0, internal_qubit)).astype(
            np.complex128
        )
        self.__state = new_state

        # Update qubit order: remove the measured qubit
        self.__qubit_order = [q if q < internal_qubit else q - 1 for q in self.__qubit_order if q != internal_qubit]

        self.normalize()

    def add_node(self, num_qubits: int) -> None:
        """Add plus state to the end of state vector.

        Parameters
        ----------
        num_qubits : `int`
            number of qubits to add
        """
        flat_state = self.__state.flatten()
        flat_state = np.repeat(flat_state, 1 << num_qubits) / np.sqrt(2**num_qubits)
        self.__state = flat_state.reshape((2,) * (self.num_qubits + num_qubits))
        # Append new qubits to the end of the qubit order
        current_max = max(self.__qubit_order) if self.__qubit_order else -1
        self.__qubit_order.extend(range(current_max + 1, current_max + 1 + num_qubits))

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
        new_order = [self.__qubit_order[permutation[i]] for i in range(self.num_qubits)]
        self.__qubit_order = new_order

    def norm(self) -> float:
        """Get norm of state vector.

        Returns
        -------
        `float`
            norm of state vector
        """
        return float(np.linalg.norm(self.__state))

    def expectation(self, operator: NDArray[np.complex128], qubits: Sequence[int]) -> float:
        r"""Calculate expectation value of operator.

        Parameters
        ----------
        operator : `numpy.typing.NDArray`\[`numpy.complex128`\]
            Hermitian operator matrix
        qubits : `collections.abc.Sequence`\[`int`\]
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
        internal_qubits = self._external_to_internal_qubits(qubits)
        internal_qubits = tuple(internal_qubits)
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
