from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphqomb.common import Plane, PlannerMeasBasis, is_clifford_angle, is_close_angle, meas_basis
from graphqomb.euler import (
    LocalClifford,
    LocalUnitary,
    bloch_sphere_coordinates,
    euler_decomposition,
    meas_basis_info,
    update_lc_basis,
    update_lc_lc,
)
from graphqomb.matrix import is_unitary

if TYPE_CHECKING:
    from typing import Callable


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


@pytest.fixture
def random_angles(rng: np.random.Generator) -> tuple[float, float, float]:
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    return float(a), float(b), float(c)


@pytest.fixture
def random_clifford_angles(rng: np.random.Generator) -> tuple[float, float, float]:
    a, b, c = rng.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], 3)
    return float(a), float(b), float(c)


def test_identity() -> None:
    lc = LocalUnitary(0, 0, 0)
    assert np.allclose(lc.matrix(), np.eye(2))


def test_unitary(random_angles: tuple[float, float, float]) -> None:
    lc = LocalUnitary(*random_angles)
    assert is_unitary(lc.matrix())


def test_lu_conjugate(random_angles: tuple[float, float, float]) -> None:
    lu = LocalUnitary(*random_angles)
    lu_conj = lu.conjugate()
    assert np.allclose(lu.matrix(), lu_conj.matrix().conj().T)


def test_euler_decomposition(random_angles: tuple[float, float, float]) -> None:
    array = LocalUnitary(*random_angles).matrix()
    alpha, beta, gamma = euler_decomposition(array)

    array_reconstructed = LocalUnitary(alpha, beta, gamma).matrix()
    assert np.allclose(array, array_reconstructed)


@pytest.mark.parametrize("angles", [(0, 0, 0), (np.pi, 0, 0), (0, np.pi, 0), (0, 0, np.pi)])
def test_euler_decomposition_corner(angles: tuple[float, float, float]) -> None:
    array = LocalUnitary(*angles).matrix()
    alpha, beta, gamma = euler_decomposition(array)

    array_reconstructed = LocalUnitary(alpha, beta, gamma).matrix()
    assert np.allclose(array, array_reconstructed)


@pytest.mark.parametrize("plane", list(Plane))
def test_bloch_sphere_coordinates(plane: Plane, rng: np.random.Generator) -> None:
    angle = rng.uniform(0, 2 * np.pi)
    basis = meas_basis(plane, angle)
    theta, phi = bloch_sphere_coordinates(basis)
    reconst_vec = np.asarray([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    inner_product = abs(np.vdot(reconst_vec, basis))
    assert np.isclose(inner_product, 1)


@pytest.mark.parametrize("plane", list(Plane))
@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi])
def test_bloch_sphere_coordinates_corner(plane: Plane, angle: float) -> None:
    basis = meas_basis(plane, angle)
    theta, phi = bloch_sphere_coordinates(basis)
    reconst_vec = np.asarray([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    inner_product = abs(np.vdot(reconst_vec, basis))
    assert np.isclose(inner_product, 1)


@pytest.mark.parametrize("plane", list(Plane))
def test_meas_basis_info(plane: Plane, rng: np.random.Generator) -> None:
    angle = rng.uniform(0, 2 * np.pi)
    basis = meas_basis(plane, angle)
    plane_get, angle_get = meas_basis_info(basis)
    assert plane == plane_get, f"Expected {plane}, got {plane_get}"
    assert is_close_angle(angle, angle_get), f"Expected {angle}, got {angle_get}"


def test_local_clifford(random_clifford_angles: tuple[float, float, float]) -> None:
    lc = LocalClifford(*random_clifford_angles)
    assert is_unitary(lc.matrix())

    assert is_clifford_angle(lc.alpha)
    assert is_clifford_angle(lc.beta)
    assert is_clifford_angle(lc.gamma)


def test_lc_lc_update(random_clifford_angles: tuple[float, float, float]) -> None:
    lc1 = LocalClifford(*random_clifford_angles)
    lc2 = LocalClifford(*random_clifford_angles)
    lc = update_lc_lc(lc1, lc2)
    assert is_unitary(lc.matrix())

    assert is_clifford_angle(lc.alpha)
    assert is_clifford_angle(lc.beta)
    assert is_clifford_angle(lc.gamma)


@pytest.mark.parametrize("plane", list(Plane))
def test_lc_basis_update(
    plane: Plane,
    random_clifford_angles: tuple[float, float, float],
    rng: np.random.Generator,
) -> None:
    lc = LocalClifford(*random_clifford_angles)
    angle = rng.uniform(0, 2 * np.pi)
    basis = PlannerMeasBasis(plane, angle)
    basis_updated = update_lc_basis(lc, basis)
    ref_updated_basis = lc.matrix() @ basis.vector()
    inner_product = abs(np.vdot(basis_updated.vector(), ref_updated_basis))
    assert np.isclose(inner_product, 1)


@pytest.mark.parametrize("plane", list(Plane))
def test_local_complement_target_update(plane: Plane, rng: np.random.Generator) -> None:
    lc = LocalClifford(0, np.pi / 2, 0)
    measurement_action: dict[Plane, tuple[Plane, Callable[[float], float]]] = {
        Plane.XY: (Plane.XZ, lambda angle: angle + np.pi / 2),
        Plane.XZ: (Plane.XY, lambda angle: np.pi / 2 - angle),
        Plane.YZ: (Plane.YZ, lambda angle: angle + np.pi / 2),
    }

    angle = rng.random() * 2 * np.pi

    meas_basis = PlannerMeasBasis(plane, angle)
    result_basis = update_lc_basis(lc.conjugate(), meas_basis)
    ref_plane, ref_angle_func = measurement_action[plane]
    ref_angle = ref_angle_func(angle)

    assert result_basis.plane == ref_plane
    assert is_close_angle(result_basis.angle, ref_angle)


@pytest.mark.parametrize("plane", list(Plane))
def test_local_complement_neighbors(plane: Plane, rng: np.random.Generator) -> None:
    lc = LocalClifford(-np.pi / 2, 0, 0)
    measurement_action: dict[Plane, tuple[Plane, Callable[[float], float]]] = {
        Plane.XY: (Plane.XY, lambda angle: angle + np.pi / 2),
        Plane.XZ: (Plane.YZ, lambda angle: angle),
        Plane.YZ: (Plane.XZ, lambda angle: -1 * angle),
    }

    angle = rng.random() * 2 * np.pi

    meas_basis = PlannerMeasBasis(plane, angle)
    result_basis = update_lc_basis(lc.conjugate(), meas_basis)
    ref_plane, ref_angle_func = measurement_action[plane]
    ref_angle = ref_angle_func(angle)

    assert result_basis.plane == ref_plane
    assert is_close_angle(result_basis.angle, ref_angle)
