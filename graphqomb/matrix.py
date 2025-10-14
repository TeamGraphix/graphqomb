"""Matrix utilities.

This module provides:

- `is_unitary`: check if a matrix is unitary.
- `is_hermitian`: check if a matrix is Hermitian.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T", bound=np.number[Any])  # can be removed >= 3.10


def is_unitary(mat: NDArray[T]) -> bool:
    r"""Check if a matrix is unitary.

    Parameters
    ----------
    mat : `numpy.typing.NDArray`\[T\]
        matrix to check

    Returns
    -------
    `bool`
        `True` if unitary, `False` otherwise
    """
    if mat.shape[0] != mat.shape[1]:
        return False
    return np.allclose(np.eye(mat.shape[0]), mat @ mat.T.conj())


def is_hermitian(mat: NDArray[T]) -> bool:
    r"""Check if a matrix is Hermitian.

    Parameters
    ----------
    mat : `numpy.typing.NDArray`\[T\]
        matrix to check

    Returns
    -------
    `bool`
        `True` if Hermitian, `False` otherwise
    """
    return np.allclose(mat, mat.T.conj())
