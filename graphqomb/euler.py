"""Euler angles and related functions.

This module provides:

- `euler_decomposition`: Decompose a 2x2 unitary matrix into Euler angles.
- `bloch_sphere_coordinates`: Get the Bloch sphere coordinates corresponding to a vector.
- `LocalUnitary`: Class to represent a local unitary.
- `LocalClifford`: Class to represent a local Clifford.
- `meas_basis_info`: Return the measurement plane and angle corresponding to a vector.
- `update_lc_lc`: Update a `LocalClifford` object with another `LocalClifford` object.
- `update_lc_basis`: Update a `LocalClifford` object with a MeasBasis object.
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING

import numpy as np
import typing_extensions

from graphqomb.common import MeasBasis, Plane, PlannerMeasBasis, is_clifford_angle, is_close_angle

if TYPE_CHECKING:
    from numpy.typing import NDArray


def euler_decomposition(u: NDArray[np.complex128]) -> tuple[float, float, float]:
    r"""Decompose a 2x2 unitary matrix into Euler angles.

    :math:`U \rightarrow R_z(\gamma)R_x(\beta)R_z(\alpha)`

    Parameters
    ----------
    u : `numpy.typing.NDArray`\[`numpy.complex128`\]
        unitary 2x2 matrix

    Returns
    -------
    `tuple`\[`float`, `float`, `float`\]
        euler angles (:math:`\alpha`, :math:`\beta`, :math:`\gamma`)
    """
    global_phase = cmath.sqrt(np.linalg.det(u))
    u /= global_phase
    u00 = complex(u[0, 0])
    u01 = complex(u[0, 1])
    u10 = complex(u[1, 0])
    u11 = complex(u[1, 1])

    if np.isclose(u10, 0):
        gamma = 2 * cmath.phase(u11)
        beta = 0.0
        alpha = 0.0
    elif np.isclose(u11, 0):
        gamma = 2 * cmath.phase(u01 / (-1j))
        beta = math.pi
        alpha = 0.0
    else:
        gamma_p_alpha = cmath.phase(u11 / u00)
        gamma_m_alpha = cmath.phase(u10 / u01)

        gamma = (gamma_p_alpha + gamma_m_alpha) / 2
        alpha = (gamma_p_alpha - gamma_m_alpha) / 2

        cos_term = (u11 / cmath.exp(1j * gamma_p_alpha / 2)).real
        sin_term = (u10 / (-1j * cmath.exp(1j * gamma_m_alpha / 2))).real

        beta = 2 * cmath.phase(cos_term + 1j * sin_term)

    return alpha, beta, gamma


def bloch_sphere_coordinates(vector: NDArray[np.complex128]) -> tuple[float, float]:
    r"""Get the Bloch sphere coordinates corresponding to a vector.

    :math:`|\psi\rangle = \cos(\theta/2)|0\rangle + \exp(i\phi)\sin(\theta/2)|1\rangle`

    Parameters
    ----------
    vector : `numpy.typing.NDArray`\[`numpy.complex128`\]
        1 qubit state vector

    Returns
    -------
    `tuple`\[`float`, `float`]
        Bloch sphere coordinates (:math:`\theta`, :math:`\phi`)
    """
    # normalize
    vector /= np.linalg.norm(vector)
    v0 = complex(vector[0])
    v1 = complex(vector[1])
    if np.isclose(v0, 0):
        theta = math.pi
        phi = cmath.phase(v1)
    else:
        global_phase = cmath.phase(v0)
        v0 /= cmath.exp(1j * global_phase)
        v1 /= cmath.exp(1j * global_phase)
        phi = 0 if np.isclose(v1, 0) else cmath.phase(v1)
        cos_term = v0.real
        sin_term = (v1 / cmath.exp(1j * phi)).real
        theta = 2 * cmath.phase(cos_term + 1j * sin_term)
    return theta, phi


class LocalUnitary:
    r"""Class to represent signle-qubit unitaries.

    :math:`U(\alpha, \beta, \gamma) = R_z(\gamma)R_x(\beta)R_z(\alpha)`

    Attributes
    ----------
    alpha : `float`
        angle for the first :math:`R_z`, by default 0
    beta : `float`
        angle for the :math:`R_x`, by default 0
    gamma : `float`
        angle for the last :math:`R_z`, by default 0
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
        """Return the conjugate of the `LocalUnitary` object.

        Returns
        -------
        `LocalUnitary`
            conjugate `LocalUnitary`
        """
        return LocalUnitary(-self.gamma, -self.beta, -self.alpha)

    def matrix(self) -> NDArray[np.complex128]:
        r"""Return the 2x2 unitary matrix corresponding to the Euler angles.

        Returns
        -------
        `numpy.typing.NDArray`\[`numpy.complex128`\]
            2x2 unitary matrix
        """
        return np.asarray(_rz(self.gamma) @ _rx(self.beta) @ _rz(self.alpha), dtype=np.complex128)


class LocalClifford(LocalUnitary):
    r"""Class to represent a local Clifford.

    :math:`U(\alpha, \beta, \gamma) = R_z(\gamma)R_x(\beta)R_z(\alpha)`
    Each angle must be integer multiples of :math:`\pi/2`.

    Attributes
    ----------
    alpha : `float`
        angle for the first :math:`R_z`. The angle must be a multiple of :math:`\pi/2`, by default 0
    beta : `float`
        angle for the :math:`R_x`. The angle must be a multiple of :math:`\pi/2`, by default 0
    gamma : `float`
        angle for the last :math:`R_z`. The angle must be a multiple of :math:`\pi/2`, by default 0
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
    vector : `numpy.typing.NDArray`\[`numpy.complex128`\]
        1 qubit state vector

    Returns
    -------
    `tuple`\[`Plane`, `float`]
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
            if is_close_angle(phi, math.pi):
                theta = -theta
            return Plane.XZ, theta
        if is_close_angle(phi, 3 * math.pi / 2):
            theta = -theta
        return Plane.YZ, theta
    if is_clifford_angle(theta) and not is_clifford_angle(theta / 2):
        # XY plane
        if is_close_angle(theta, 3 * math.pi / 2):
            phi += math.pi
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

    matrix = np.asarray(matrix1 @ matrix2, dtype=np.complex128)
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

    updated_vector = np.asarray(matrix @ vector, dtype=np.complex128)
    plane, angle = meas_basis_info(updated_vector)
    return PlannerMeasBasis(plane, angle)


def _rx(angle: float) -> NDArray[np.complex128]:
    return np.asarray(
        [
            [math.cos(angle / 2), -1j * math.sin(angle / 2)],
            [-1j * math.sin(angle / 2), math.cos(angle / 2)],
        ],
        dtype=np.complex128,
    )


def _rz(angle: float) -> NDArray[np.complex128]:
    return np.asarray(
        [
            [cmath.exp(-1j * angle / 2), 0],
            [0, cmath.exp(1j * angle / 2)],
        ],
        dtype=np.complex128,
    )
