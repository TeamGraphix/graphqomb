"""Common classes and functions.

This module provides:
- Plane: Measurement planes for the MBQC.
- Axis: Measurement axis.
- Sign: Measurement sign.
- MeasBasis: Abstract class to represent a measurement basis.
- PlannerMeasBasis: Class to represent a planner measurement basis.
- AxisMeasBasis: Class to represent an axis measurement basis.
- get_meas_basis: Function to get the measurement basis vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Plane(Enum):
    """Measurement planes for MBQC.

    We distinguish the axial measurements from the planar measurements.

    The measurement planes are:
    - XY: Measurement in the XY plane.
    - YZ: Measurement in the YZ plane.
    - XZ: Measurement in the XZ plane.
    """

    XY = auto()
    YZ = auto()
    XZ = auto()


class Axis(Enum):
    """Measurement axis."""

    X = auto()
    Y = auto()
    Z = auto()


class Sign(Enum):
    """Measurement sign."""

    PLUS = auto()
    MINUS = auto()


class MeasBasis(ABC):
    """Abstract class to represent a measurement basis."""

    @property
    @abstractmethod
    def plane(self) -> Plane:
        """Return the measurement plane."""
        raise NotImplementedError

    @property
    @abstractmethod
    def angle(self) -> float:
        """Return the measurement angle."""
        raise NotImplementedError

    @abstractmethod
    def get_vector(self) -> NDArray[np.complex128]:
        """Return the measurement basis vector."""
        raise NotImplementedError


class PlannerMeasBasis(MeasBasis):
    """Class to represent a planner measurement basis.

    Attributes
    ----------
    plane : Plane
        measurement plane
    angle : float
        measurement angle
    """

    def __init__(self, plane: Plane, angle: float) -> None:
        self.__plane = plane
        self.__angle = angle

    @property
    def plane(self) -> Plane:
        """Return the measurement plane.

        Returns
        -------
        Plane
            measurement plane
        """
        return self.__plane

    @property
    def angle(self) -> float:
        """Return the measurement angle.

        Returns
        -------
        float
            measurement angle
        """
        return self.__angle

    def get_vector(self) -> NDArray[np.complex128]:
        """Return the measurement basis vector.

        Returns
        -------
        NDArray
            measurement basis vector
        """
        return get_meas_basis(self.plane, self.angle)

    def conjugate(self) -> PlannerMeasBasis:
        """Return the conjugate of the PlannerMeasBasis object.

        Returns
        -------
        PlannerMeasBasis
            conjugate PlannerMeasBasis

        Raises
        ------
        ValueError
            if the plane is not one of XY, YZ, XZ
        """
        if self.plane == Plane.XY:
            return PlannerMeasBasis(Plane.XY, -self.angle)
        if self.plane == Plane.YZ:
            return PlannerMeasBasis(Plane.YZ, -self.angle)
        if self.plane == Plane.XZ:
            return PlannerMeasBasis(Plane.XZ, self.angle)
        msg = "The plane must be one of XY, YZ, XZ"
        raise ValueError(msg)


class AxisMeasBasis(MeasBasis):
    """Class to represent an axis measurement basis.

    The aim is to pocess the accurate information of the axis measurement.

    Attributes
    ----------
    axis : Axis
        measurement axis
    sign : Sign
        measurement sign
    """

    def __init__(self, axis: Axis, sign: Sign) -> None:
        self.axis = axis
        self.sign = sign

    @property
    def plane(self) -> Plane:
        """Return the measurement plane.

        Returns
        -------
        Plane
            measurement plane

        Raises
        ------
        ValueError
            if the axis is not one of X, Y, Z
        """
        if self.axis == Axis.X:
            plane = Plane.XY
        elif self.axis == Axis.Y:
            plane = Plane.YZ
        elif self.axis == Axis.Z:
            plane = Plane.XZ
        else:
            msg = "The axis must be one of X, Y, Z"
            raise ValueError(msg)
        return plane

    # this could be simpler if we use rotational notation
    @property
    def angle(self) -> float:
        """Return the measurement angle.

        Returns
        -------
        float
            measurement angle

        Raises
        ------
        ValueError
            if the axis is not one of X, Y,
        """
        if self.axis in {Axis.X, Axis.Z}:
            angle = 0 if self.sign == Sign.PLUS else np.pi
        elif self.axis == Axis.Y:
            angle = np.pi / 2 if self.sign == Sign.PLUS else 3 * np.pi / 2
        else:
            msg = "The axis must be one of X, Y, Z"
            raise ValueError(msg)
        return angle

    def get_vector(self) -> NDArray[np.complex128]:
        """Return the measurement basis vector.

        Returns
        -------
        NDArray
            measurement basis vector
        """
        return get_meas_basis(self.plane, self.angle)


def default_meas_basis() -> PlannerMeasBasis:
    """Return the default measurement basis.

    The default measurement basis is the XY plane with angle 0.

    Returns
    -------
    PlannerMeasBasis
        default measurement basis
    """
    return PlannerMeasBasis(Plane.XY, 0.0)


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
        if the plane is not one of XY, YZ, XZ
    """
    basis: NDArray[np.complex128]
    if plane == Plane.XY:
        basis = np.asarray([1, np.exp(1j * angle)]) / np.sqrt(2)
    elif plane == Plane.YZ:
        basis = np.asarray([np.cos(angle / 2), 1j * np.sin(angle / 2)])
    elif plane == Plane.XZ:
        basis = np.asarray([np.cos(angle / 2), np.sin(angle / 2)]).astype(np.complex128)
    else:
        msg = "The plane must be one of XY, YZ, XZ"
        raise ValueError(msg)
    return basis
