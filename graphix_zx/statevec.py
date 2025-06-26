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

    from numpy.typing import NDArray

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
    r"""State vector representation.

    Attributes
    ----------
    state : `numpy.typing.NDArray`\[`numpy.complex128`\]
        The state vector as a numpy array.

    Properties
    ----------
    num_qubits : `int`
        The number of qubits in the state vector.
    """

    _num_qubits: int
    state: NDArray[np.complex128]

    def __init__(self, num_qubits: int, state: NDArray[np.complex128] | None = None) -> None:
        self._num_qubits = num_qubits
        if state is not None:
            #  Check if the state is a valid state vector
            if state.ndim != 1 or state.size != 2**num_qubits:
                msg = f"State vector must be a 1D array of size {2**num_qubits}, got {state.shape}."
                raise ValueError(msg)
            self.state = state
        else:
            self.state = np.ones(2**num_qubits, dtype=np.complex128) / np.sqrt(2**num_qubits)

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the state vector.

        Returns
        -------
        `int`
            The number of qubits in the state vector.
        """
        return self._num_qubits

    def __copy__(self) -> StateVector:
        return StateVector(self._num_qubits, self.state.copy())

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
        self.state = self.state.reshape([2] * self._num_qubits)
        operator = operator.reshape([2] * len(qubits) * 2)

        axes = (range(len(qubits), 2 * len(qubits)), qubits)
        self.state = np.tensordot(operator, self.state, axes=axes).astype(np.complex128)

        self.state = np.moveaxis(self.state, range(len(qubits)), qubits)

        self.state = self.state.reshape(2**self._num_qubits)

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
        meas_basis = meas_basis.flip() if result else meas_basis
        basis_vector = meas_basis.vector()
        self.state = self.state.reshape([2] * self._num_qubits)
        self.state = np.tensordot(basis_vector.conjugate(), self.state, axes=(0, qubit)).astype(np.complex128)
        self.state = self.state.reshape(2 ** (self._num_qubits - 1))

        self._num_qubits -= 1

    def add_node(self, num_qubits: int) -> None:
        """Add |+> state to the end of state vector.

        Parameters
        ----------
        num_qubits : `int`
            number of qubits to add
        """
        self.state = np.kron(self.state, np.ones(2**num_qubits) / np.sqrt(2**num_qubits))
        self._num_qubits += num_qubits

    def entangle(self, qubits: tuple[int, int]) -> None:
        r"""Entangle two qubits.

        Parameters
        ----------
        qubits : `tuple`\[`int`, `int`\]
            qubits to entangle, e.g. (0, 1) for qubit 0 and qubit 1
        """
        self.evolve(CZ_TENSOR, qubits)

    def tensor_product(self, other: StateVector) -> None:
        """Tensor product with other state vector, self ⊗ other.

        Parameters
        ----------
        other : `StateVector`
            other state vector
        """
        self.state = np.kron(self.state, other.state).astype(np.complex128)
        self._num_qubits += other._num_qubits

    def normalize(self) -> None:
        """Normalize the state."""
        self.state /= np.linalg.norm(self.state)

    def reorder(self, permutation: Sequence[int]) -> None:
        r"""Permute qubits.

        if permutation is [2, 0, 1], then
        # [q0, q1, q2] -> [q1, q2, q0]

        Parameters
        ----------
        permutation : `collections.abc.Sequence`\[`int`\]
            permutation list
        """
        axes = [permutation.index(i) for i in range(self._num_qubits)]
        self.state = self.state.reshape([2] * self._num_qubits)
        self.state = self.state.transpose(axes).flatten()

    def is_isolated(self, qubit: int) -> bool:
        """Check if qubit is isolated(product state).

        Parameters
        ----------
        qubit : `int`
            qubit index

        Returns
        -------
        `bool`
            True if isolated, False otherwise
        """
        state = self.state.reshape([2] * self._num_qubits)
        state0 = state.take(indices=0, axis=qubit)
        state1 = state.take(indices=1, axis=qubit)

        # normalize
        state0 /= np.linalg.norm(state0)
        state1 /= np.linalg.norm(state1)

        match_rate = np.dot(state0.conjugate(), state1)

        return bool(np.isclose(match_rate, 1.0))

    def norm(self) -> float:
        """Get norm of state vector.

        Returns
        -------
        `float`
            norm of state vector
        """
        return float(np.linalg.norm(self.state))

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
        state = self.state.reshape([2] * self._num_qubits)
        operator = operator.reshape([2] * len(qubits) * 2)

        axes = (range(len(qubits), 2 * len(qubits)), qubits)
        state = np.tensordot(operator, state, axes=axes).astype(np.complex128)

        state = np.moveaxis(state, range(len(qubits)), qubits).reshape(2**self._num_qubits)

        return float(np.dot(self.state.conjugate(), state) / self.norm() ** 2)
