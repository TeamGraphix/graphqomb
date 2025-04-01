"""Matrix utilities.

This module provides:

- `is_unitary`: check if a matrix is unitary.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    Numeric = np.number
else:
    from typing import Union

    Numeric = Union[np.int64, np.float64, np.complex128]


def is_unitary(mat: NDArray[Numeric]) -> bool:
    r"""Check if a matrix is unitary.

    Parameters
    ----------
    mat : `numpy.typing.NDArray`\[`numpy.number`\]
        matrix to check

    Returns
    -------
    `bool`
        `True` if unitary, `False` otherwise
    """
    if mat.shape[0] != mat.shape[1]:
        return False
    return np.allclose(np.eye(mat.shape[0]), mat @ mat.T.conj())
