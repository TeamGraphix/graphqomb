"""Euler angles and related functions.

This module provides:

- `euler_decomposition`: Decompose a 2x2 unitary matrix into Euler angles.
- `bloch_sphere_coordinates`: Get the Bloch sphere coordinates corresponding to a vector.
- `is_close_angle`: Check if an angle is close to a target angle.
- `is_clifford_angle`: Check if an angle is a Clifford angle.
- `LocalUnitary`: Class to represent a local unitary.
- `LocalClifford`: Class to represent a local Clifford.
- `meas_basis_info`: Return the measurement plane and angle corresponding to a vector.
- `update_lc_lc`: Update a `LocalClifford` object with another `LocalClifford` object.
- `update_lc_basis`: Update a `LocalClifford` object with a MeasBasis object.
"""

from __future__ import annotations

import cmath
from typing import TYPE_CHECKING

import numpy as np
import typing_extensions

from graphix_zx.common import MeasBasis, Plane, PlannerMeasBasis

if TYPE_CHECKING:
    from numpy.typing import NDArray


def euler_decomposition(u: NDArray[np.complex128]) -> tuple[float, float, float]:
    r"""Decompose a 2x2 unitary matrix into Euler angles.

    U -> Rz(gamma)Rx(beta)Rz(alpha)

    Parameters
    ----------
    u : :class:`numpy.typing.NDArray`\[:class:`numpy.complex128`\]
        unitary 2x2 matrix

    Returns
    -------
    :class:`tuple`\[:class:`float`, :class:`float`, :class:`float`\]
        euler angles (alpha, beta, gamma)
    """
    global_phase = cmath.sqrt(np.linalg.det(u))
    u /= global_phase

    if np.isclose(u[1, 0], 0):
        gamma = 2 * np.angle(u[1, 1])
        beta = 0.0
        alpha = 0.0
    elif np.isclose(u[1, 1], 0):
        gamma = 2 * np.angle(u[0, 1] / (-1j))
        beta = np.pi
        alpha = 0.0
    else:
        gamma_p_alpha = np.angle(u[1, 1] / u[0, 0])
        gamma_m_alpha = np.angle(u[1, 0] / u[0, 1])

        gamma = (gamma_p_alpha + gamma_m_alpha) / 2
        alpha = (gamma_p_alpha - gamma_m_alpha) / 2

        cos_term = np.real(u[1, 1] / np.exp(1j * gamma_p_alpha / 2))
        sin_term = np.real(u[1, 0] / (-1j * np.exp(1j * gamma_m_alpha / 2)))

        beta = 2 * np.angle(cos_term + 1j * sin_term)

    return alpha, beta, gamma


def bloch_sphere_coordinates(vector: NDArray[np.complex128]) -> tuple[float, float]:
    r"""Get the Bloch sphere coordinates corresponding to a vector.

    \|psi> = cos(theta/2)\|0> + exp(i*phi)sin(theta/2)|1>

    Parameters
    ----------
    vector : :class:`numpy.typing.NDArray`\[:class:`numpy.complex128`\]
        1 qubit state vector

    Returns
    -------
    :class:`tuple`\[:class:`float`, :class:`float`]
        Bloch sphere coordinates (theta, phi)
    """
    # normalize
    vector /= np.linalg.norm(vector)
    if np.isclose(vector[0], 0):
        theta = np.pi
        phi = np.angle(vector[1])
    else:
        global_phase = np.angle(vector[0])
        vector /= np.exp(1j * global_phase)
        phi = 0 if np.isclose(vector[1], 0) else np.angle(vector[1])
        cos_term = np.real(vector[0])
        sin_term = np.real(vector[1] / np.exp(1j * phi))
        theta = 2 * np.angle(cos_term + 1j * sin_term)
    return theta, phi


def is_close_angle(angle: float, target: float, atol: float = 1e-9) -> bool:
    """Check if an angle is close to a target angle.

    Parameters
    ----------
    angle : `float`
        angle to check
    target : `float`
        target angle
    atol : `float`, optional
        absolute tolerance, by default 1e-9

    Returns
    -------
    `bool`
        True if the angle is close to the target angle
    """
    diff_angle = (angle - target) % (2 * np.pi)

    if diff_angle > np.pi:
        diff_angle = 2 * np.pi - diff_angle
    return bool(np.isclose(diff_angle, 0, atol=atol))


def is_clifford_angle(angle: float, atol: float = 1e-9) -> bool:
    """Check if an angle is a Clifford angle.

    Parameters
    ----------
    angle : `float`
        angle to check
    atol : `float`, optional
        absolute tolerance, by default 1e-9

    Returns
    -------
    `bool`
        True if the angle is a Clifford angle
    """
    angle_preprocessed = angle % (2 * np.pi)
    return any(is_close_angle(angle_preprocessed, target, atol=atol) for target in [0, np.pi / 2, np.pi, 3 * np.pi / 2])


class LocalUnitary:
    """Class to represent signle-qubit unitaries.

    U -> Rz(gamma)Rx(beta)Rz(alpha)

    Attributes
    ----------
    alpha : `float`
        angle for the first Rz, by default 0
    beta : `float`
        angle for the Rx, by default 0
    gamma : `float`
        angle for the last Rz, by default 0
    """

    alpha: float
    beta: float
    gamma: float

    def __init__(self, alpha: float = 0, beta: float = 0, gamma: float = 0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def print_angles(self) -> None:
        """Print the Euler angles."""
        print(f"alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma}")  # noqa: T201

    def conjugate(self) -> LocalUnitary:
        """Return the conjugate of the LocalUnitary object.

        Returns
        -------
        `LocalUnitary`
            conjugate LocalUnitary
        """
        return LocalUnitary(-self.gamma, -self.beta, -self.alpha)

    def matrix(self) -> NDArray[np.complex128]:
        r"""Return the 2x2 unitary matrix corresponding to the Euler angles.

        Returns
        -------
        :class:`numpy.typing.NDArray`\[:class:`numpy.complex128`\]
            2x2 unitary matrix
        """
        return _rz(self.gamma) @ _rx(self.beta) @ _rz(self.alpha)


class LocalClifford(LocalUnitary):
    """Class to represent a local Clifford.

    U -> Rz(alpha)Rx(beta)Rz(gamma)
    Each angle must be integer multiples of pi/2.

    Attributes
    ----------
    alpha : `float`
        angle for the first Rz. The angle must be a multiple of pi/2, by default 0
    beta : `float`
        angle for the Rx. The angle must be a multiple of pi/2, by default 0
    gamma : `float`
        angle for the last Rz. The angle must be a multiple of pi/2, by default 0
    """

    alpha: float
    beta: float
    gamma: float

    def __init__(self, alpha: float = 0, beta: float = 0, gamma: float = 0) -> None:
        self._angle_check(alpha, beta, gamma)
        super().__init__(alpha, beta, gamma)

    @classmethod
    def _angle_check(cls, alpha: float, beta: float, gamma: float, atol: float = 1e-9) -> None:
        """Check if the angles are Clifford angles.

        Parameters
        ----------
        alpha : `float`
            angle for the first Rz
        beta : `float`
            angle for the Rx
        gamma : `float`
            angle for the last Rz
        atol : `float`, optional
            absolute tolerance, by default 1e-9

        Raises
        ------
        ValueError
            if any of the angles is not a Clifford angle
        """
        if not all(is_clifford_angle(angle, atol=atol) for angle in [alpha, beta, gamma]):
            msg = "The angles must be integer multiples of pi/2"
            raise ValueError(msg)

    @typing_extensions.override
    def conjugate(self) -> LocalClifford:
        """Return the conjugate of the `LocalClifford` object.

        Returns
        -------
        `LocalClifford`
            conjugate `LocalClifford`
        """
        return LocalClifford(-self.gamma, -self.beta, -self.alpha)


def meas_basis_info(vector: NDArray[np.complex128]) -> tuple[Plane, float]:
    r"""Return the measurement plane and angle corresponding to a vector.

    Parameters
    ----------
    vector : :class:`numpy.typing.NDArray`\[:class:`numpy.complex128`\]
        1 qubit state vector

    Returns
    -------
    :class:`tuple`\[:class:`Plane`, :class:`float`]
        measurement plane and angle

    Raises
    ------
    ValueError
        if the vector does not lie on any of 3 planes
    """
    theta, phi = bloch_sphere_coordinates(vector)
    if is_clifford_angle(phi):
        # YZ or XZ plane
        if is_clifford_angle(phi / 2):  # 0 or pi
            if is_close_angle(phi, np.pi):
                theta = -theta
            return Plane.XZ, theta
        if is_close_angle(phi, 3 * np.pi / 2):
            theta = -theta
        return Plane.YZ, theta
    if is_clifford_angle(theta) and not is_clifford_angle(theta / 2):
        # XY plane
        if is_close_angle(theta, 3 * np.pi / 2):
            phi += np.pi
        return Plane.XY, phi
    msg = "The vector does not lie on any of 3 planes"
    raise ValueError(msg)


# TODO(masa10-f): Algebraic backend for this computation(#023)
def update_lc_lc(lc1: LocalClifford, lc2: LocalClifford) -> LocalClifford:
    """Update a `LocalClifford` object with another `LocalClifford` object.

    Parameters
    ----------
    lc1 : `LocalClifford`
        left `LocalClifford`
    lc2 : `LocalClifford`
        right `LocalClifford`

    Returns
    -------
    `LocalClifford`
        multiplied `LocalClifford`
    """
    matrix1 = lc1.matrix()
    matrix2 = lc2.matrix()

    matrix = matrix1 @ matrix2
    alpha, beta, gamma = euler_decomposition(matrix)
    return LocalClifford(alpha, beta, gamma)


# TODO(masa10-f): Algebraic backend for this computation(#023)
def update_lc_basis(lc: LocalClifford, basis: MeasBasis) -> PlannerMeasBasis:
    """Update a `MeasBasis` object with an action of `LocalClifford` object.

    Parameters
    ----------
    lc : `LocalClifford`
        `LocalClifford`
    basis : `MeasBasis`
        `MeasBasis`

    Returns
    -------
    `PlannerMeasBasis`
        updated `PlannerMeasBasis`
    """
    matrix = lc.matrix()
    vector = basis.vector()

    vector = matrix @ vector
    plane, angle = meas_basis_info(vector)
    return PlannerMeasBasis(plane, angle)


def _rx(angle: float) -> NDArray[np.complex128]:
    return np.asarray(
        [
            [np.cos(angle / 2), -1j * np.sin(angle / 2)],
            [-1j * np.sin(angle / 2), np.cos(angle / 2)],
        ],
        dtype=np.complex128,
    )


def _rz(angle: float) -> NDArray[np.complex128]:
    return np.asarray(
        [
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)],
        ],
        dtype=np.complex128,
    )
