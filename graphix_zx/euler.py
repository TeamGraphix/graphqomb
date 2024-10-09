from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def euler_decomposition(u: NDArray) -> tuple[float, float, float]:
    """Decompose a 2x2 unitary matrix into Euler angles.

    U -> Rz(alpha)Rx(beta)Rz(gamma)

    Parameters
    ----------
    u : NDArray
        unitary 2x2 matrix

    Returns
    -------
    tuple[float, float, float]
        euler angles (alpha, beta, gamma)
    """
    global_phase = np.angle(u[0, 0])
    u *= np.exp(-1j * global_phase)

    alpha = np.angle(u[1, 0]) + np.angle(u[0, 0])
    beta = 2 * np.arccos(np.abs(u[0, 0]))
    gamma = np.angle(u[0, 1]) - np.angle(u[1, 1])

    return alpha, beta, gamma


# TODO: there is room to improve the data type for angles
class LocalUnitary:
    alpha: float
    beta: float
    gamma: float

    def __init__(self, alpha: float = 0, beta: float = 0, gamma: float = 0) -> None:
        """Initialize the Euler angles.

        U -> Rz(alpha)Rx(beta)Rz(gamma)

        Parameters
        ----------
        alpha : float, optional
            angle for the first Rz, by default 0
        beta : float, optional
            angle for the Rx, by default 0
        gamma : float, optional
            angle for the last Rz, by default 0
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_matrix(self) -> NDArray:
        """Return the 2x2 unitary matrix corresponding to the Euler angles.

        Returns
        -------
        NDArray
            2x2 unitary matrix
        """
        return np.asarray(
            [
                [
                    np.cos(self.beta / 2) * np.exp(-1j * (self.alpha + self.gamma) / 2),
                    -1j * np.sin(self.beta / 2) * np.exp(-1j * (self.alpha - self.gamma) / 2),
                ],
                [
                    -1j * np.sin(self.beta / 2) * np.exp(1j * (self.alpha - self.gamma) / 2),
                    np.cos(self.beta / 2) * np.exp(1j * (self.alpha + self.gamma) / 2),
                ],
            ]
        )


def is_clifford_angle(angle: float, atol: float = 1e-9) -> bool:
    """Check if an angle is a Clifford angle.

    Parameters
    ----------
    angle : float
        angle to check
    atol : float, optional
        absolute tolerance, by default 1e-9

    Returns
    -------
    bool
        True if the angle is a Clifford angle
    """
    return bool(np.isclose(angle % (np.pi / 2), 0, atol=atol))


class LocalClifford(LocalUnitary):
    alpha: float
    beta: float
    gamma: float

    def __init__(self, alpha: float = 0, beta: float = 0, gamma: float = 0) -> None:
        """Initialize the Euler angles for Clifford gates.

        Parameters
        ----------
        alpha : float, optional
            angle for the first Rz. The angle must be a multiple of pi/2, by default 0
        beta : float, optional
            angle for the Rx. The angle must be a multiple of pi/2, by default 0
        gamma : float, optional
            angle for the last Rz. The angle must be a multiple of pi/2, by default 0
        """
        self._angle_check(alpha, beta, gamma)
        super().__init__(alpha, beta, gamma)

    @classmethod
    def _angle_check(cls, alpha: float, beta: float, gamma: float, atol: float = 1e-9) -> None:
        """Check if the angles are Clifford angles.

        Parameters
        ----------
        alpha : float
            angle for the first Rz
        beta : float
            angle for the Rx
        gamma : float
            angle for the last Rz
        atol : float, optional
            absolute tolerance, by default 1e-9

        Raises
        ------
        ValueError
            if any of the angles is not a Clifford angle
        """
        if not any(is_clifford_angle(angle, atol=atol) for angle in [alpha, beta, gamma]):
            msg = "The angles must be multiples of pi/2"
            raise ValueError(msg)
