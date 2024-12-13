"""Matrix utilities."""

import numpy as np
from numpy.typing import NDArray


def is_unitary(mat: NDArray) -> bool:
    """Check if a matrix is unitary.

    Parameters
    ----------
    mat : NDArray
        matrix to check

    Returns
    -------
    bool
        True if unitary, False otherwise
    """
    return np.allclose(np.eye(mat.shape[0]), mat @ mat.T.conj())
