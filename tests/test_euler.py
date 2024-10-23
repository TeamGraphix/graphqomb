import numpy as np
import pytest

from graphix_zx.common import MeasBasis, Plane, get_meas_basis
from graphix_zx.euler import (
    LocalClifford,
    LocalUnitary,
    _get_meas_basis_info,
    _is_close_angle,
    euler_decomposition,
    get_bloch_sphere_coordinates,
    is_clifford_angle,
    update_lc_basis,
    update_lc_lc,
)
from graphix_zx.matrix import is_unitary


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


@pytest.fixture
def random_angles(rng: np.random.Generator) -> tuple[float, float, float]:
    return tuple(rng.uniform(0, 2 * np.pi, 3))


@pytest.fixture
def random_clifford_angles(rng: np.random.Generator) -> tuple[float, float, float]:
    return tuple(rng.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], 3))


def test_is_clifford_angle() -> None:
    assert is_clifford_angle(0)
    assert is_clifford_angle(np.pi / 2)
    assert is_clifford_angle(np.pi)
    assert is_clifford_angle(3 * np.pi / 2)
    assert not is_clifford_angle(np.pi / 3)
    assert not is_clifford_angle(np.pi / 4)
    assert not is_clifford_angle(np.pi / 6)


def test_is_close_angle() -> None:
    assert _is_close_angle(0, 0)
    assert _is_close_angle(np.pi / 2, np.pi / 2)
    assert _is_close_angle(np.pi, np.pi)
    assert _is_close_angle(3 * np.pi / 2, 3 * np.pi / 2)
    assert not _is_close_angle(0, np.pi / 2)
    assert not _is_close_angle(np.pi / 2, np.pi)
    assert not _is_close_angle(np.pi, 3 * np.pi / 2)
    assert not _is_close_angle(3 * np.pi / 2, 0)

    # add 2 * np.pi to the second angle
    assert _is_close_angle(0, 2 * np.pi)
    assert _is_close_angle(np.pi / 2, 2 * np.pi + np.pi / 2)
    assert _is_close_angle(np.pi, 2 * np.pi + np.pi)
    assert _is_close_angle(3 * np.pi / 2, 2 * np.pi + 3 * np.pi / 2)

    # minus 2 * np.pi to the second angle
    assert _is_close_angle(0, -2 * np.pi)
    assert _is_close_angle(np.pi / 2, -2 * np.pi + np.pi / 2)
    assert _is_close_angle(np.pi, -2 * np.pi + np.pi)
    assert _is_close_angle(3 * np.pi / 2, -2 * np.pi + 3 * np.pi / 2)

    # boundary cases
    assert _is_close_angle(-1e-10, 1e-10)


def test_identity() -> None:
    lc = LocalUnitary(0, 0, 0)
    assert np.allclose(lc.get_matrix(), np.eye(2))


def test_unitary(random_angles: tuple[float, float, float]) -> None:
    lc = LocalUnitary(*random_angles)
    assert is_unitary(lc.get_matrix())


def test_euler_decomposition(random_angles: tuple[float, float, float]) -> None:
    array = LocalUnitary(*random_angles).get_matrix()
    alpha, beta, gamma = euler_decomposition(array)

    array_reconstructed = LocalUnitary(alpha, beta, gamma).get_matrix()
    assert np.allclose(array, array_reconstructed)


@pytest.mark.parametrize("angles", [(0, 0, 0), (np.pi, 0, 0), (0, np.pi, 0), (0, 0, np.pi)])
def test_euler_decomposition_corner(angles: tuple[float, float, float]) -> None:
    array = LocalUnitary(*angles).get_matrix()
    alpha, beta, gamma = euler_decomposition(array)

    array_reconstructed = LocalUnitary(alpha, beta, gamma).get_matrix()
    assert np.allclose(array, array_reconstructed)


@pytest.mark.parametrize("plane", [Plane.XY, Plane.YZ, Plane.ZX])
def test_get_bloch_sphere_coordinates(plane: Plane, rng: np.random.Generator) -> None:
    angle = rng.uniform(0, 2 * np.pi)
    basis = get_meas_basis(plane, angle)
    theta, phi = get_bloch_sphere_coordinates(basis)
    reconst_vec = np.asarray([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    inner_product = np.abs(np.vdot(reconst_vec, basis))
    assert np.allclose(inner_product, 1)


@pytest.mark.parametrize("plane", [Plane.XY, Plane.YZ, Plane.ZX])
@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi])
def test_get_bloch_sphere_coordinates_corner(plane: Plane, angle: float) -> None:
    basis = get_meas_basis(plane, angle)
    theta, phi = get_bloch_sphere_coordinates(basis)
    reconst_vec = np.asarray([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    inner_product = np.abs(np.vdot(reconst_vec, basis))
    assert np.allclose(inner_product, 1)


@pytest.mark.parametrize("plane", [Plane.XY, Plane.YZ, Plane.ZX])
def test_get_meas_basis_info(plane: Plane, rng: np.random.Generator) -> None:
    angle = rng.uniform(0, 2 * np.pi)
    basis = get_meas_basis(plane, angle)
    plane_get, angle_get = _get_meas_basis_info(basis)
    assert plane == plane_get, f"Expected {plane}, got {plane_get}"
    assert _is_close_angle(angle, angle_get), f"Expected {angle}, got {angle_get}"


def test_local_clifford(random_clifford_angles: tuple[float, float, float]) -> None:
    lc = LocalClifford(*random_clifford_angles)
    assert is_unitary(lc.get_matrix())

    assert is_clifford_angle(lc.alpha)
    assert is_clifford_angle(lc.beta)
    assert is_clifford_angle(lc.gamma)


@pytest.mark.skip
def test_lc_lc_update(random_angles: tuple[float, float, float]) -> None:
    lc1 = LocalClifford(*random_angles)
    lc2 = LocalClifford(*random_angles)
    lc = update_lc_lc(lc1, lc2)
    assert is_unitary(lc.get_matrix())

    assert is_clifford_angle(lc.alpha)
    assert is_clifford_angle(lc.beta)
    assert is_clifford_angle(lc.gamma)


@pytest.mark.skip
@pytest.mark.parametrize("plane", [Plane.XY, Plane.YZ, Plane.ZX])
def test_lc_basis_update(plane: Plane, random_angles: tuple[float, float, float], rng: np.random.Generator) -> None:
    lc = LocalClifford(*random_angles)
    angle = rng.uniform(0, 2 * np.pi)
    basis = MeasBasis(plane, angle)
    basis_updated = update_lc_basis(lc, basis)
    assert np.allclose(basis_updated.get_vector(), lc.get_matrix() @ basis.get_vector())
    assert basis_updated.plane == plane
    assert _is_close_angle(basis_updated.angle, angle)
