"""Common classes and functions.

This module provides:
- Plane: Measurement planes for the MBQC.
- MeasBasis: Class to represent a measurement basis.
- get_meas_basis: Function to get the measurement basis vector.
"""

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class Plane(Enum):
    """Measurement planes for the ZX-calculus.

    The measurement planes are:
    - XY(YX)
    - YZ(ZY)
    - ZX(XZ)
    """

    XY = auto()
    YZ = auto()
    ZX = auto()
    YX = XY
    ZY = YZ
    XZ = ZX


class MeasBasis:
    """Class to represent a measurement basis.

    Attributes
    ----------
    plane : Plane
        measurement plane
    angle : float
        measurement angle
    """

    plane: Plane
    angle: float

    def __init__(self, plane: Plane, angle: float) -> None:
        self.plane = plane
        self.angle = angle

    def get_vector(self) -> NDArray:
        """Return the measurement basis vector.

        Returns
        -------
        NDArray
            measurement basis vector
        """
        return get_meas_basis(self.plane, self.angle)


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
        basis = np.asarray([np.cos(angle / 2), np.sin(angle / 2)]).astype(np.complex128)
    else:
        msg = "The plane must be one of XY, YZ, ZX"
        raise ValueError(msg)
    return basis
