"""Matrix utilities.

This module provides:

- `is_unitary`: check if a matrix is unitary.
"""

import numpy as np
from numpy.typing import NDArray


def is_unitary(mat: NDArray[np.complex128]) -> bool:
    r"""Check if a matrix is unitary.

    Parameters
    ----------
    mat : :class:`numpy.typing.NDArray`\[:class:`numpy.complex128`\]
        matrix to check

    Returns
    -------
    `bool`
        True if unitary, False otherwise
    """
    if mat.shape[0] != mat.shape[1]:
        return False
    return np.allclose(np.eye(mat.shape[0]), mat @ mat.T.conj())
