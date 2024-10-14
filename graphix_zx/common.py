"""Common classes and functions.

This module provides:
- Plane: Measurement planes for the MBQC.
- get_meas_basis: Function to get the measurement basis vector.
"""

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class Plane(Enum):
    """Measurement planes for the ZX-calculus."""

    XY = auto()
    YZ = auto()
    ZX = auto()
    YX = XY
    ZY = YZ
    XZ = ZX


def get_meas_basis(plane: Plane, angle: float) -> NDArray[np.complex128]:
    """Return the measurement basis vector corresponding to the plane and angle.

    Parameters
    ----------
    plane : Plane
        measurement plane
    angle : float
        measurement angle

    Returns
    -------
    NDArray
        measurement basis vector

    Raises
    ------
    ValueError
        if the plane is not one of XY, YZ, ZX
    """
    basis: NDArray[np.complex128]
    if plane == Plane.XY:
        basis = np.asarray([1, np.exp(1j * angle)]) / np.sqrt(2)
    elif plane == Plane.YZ:
        basis = np.asarray([np.cos(angle / 2), 1j * np.sin(angle / 2)])
    elif plane == Plane.ZX:
        basis = np.asarray([np.cos(angle / 2), np.sin(angle / 2)])
    else:
        msg = "The plane must be one of XY, YZ, ZX"
        raise ValueError(msg)
    return basis
