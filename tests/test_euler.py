from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphqomb.common import Plane, PlannerMeasBasis, is_clifford_angle, is_close_angle, meas_basis
from graphqomb.euler import (
    LocalClifford,
    LocalUnitary,
    _meas_basis_candidates,
    bloch_sphere_coordinates,
    euler_decomposition,
    meas_basis_info,
    update_lc_basis,
    update_lc_lc,
)
from graphqomb.matrix import is_unitary

if TYPE_CHECKING:
    from collections.abc import Callable


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


DEGENERATE_CASES: list[tuple[Plane, float, Plane, float]] = [
    (Plane.XZ, 0.0, Plane.YZ, 0.0),
    (Plane.XZ, np.pi, Plane.YZ, np.pi),
    (Plane.XZ, 0.5 * np.pi, Plane.XY, 0.0),
    (Plane.XZ, 1.5 * np.pi, Plane.XY, np.pi),
    (Plane.YZ, 0.5 * np.pi, Plane.XY, 0.5 * np.pi),
    (Plane.YZ, 1.5 * np.pi, Plane.XY, 1.5 * np.pi),
]


def test_equivalence() -> None:
    for plane1, angle1, plane2, angle2 in DEGENERATE_CASES:
        basis1 = meas_basis(plane1, angle1)
        basis2 = meas_basis(plane2, angle2)
        inner_product = abs(np.vdot(basis1, basis2))
        assert np.isclose(inner_product, 1)


@pytest.mark.parametrize("case", DEGENERATE_CASES)
def test_meas_basis_candidates(case: tuple[Plane, float, Plane, float]) -> None:
    plane1, angle1, plane2, angle2 = case
    basis = meas_basis(plane1, angle1)
    candidates = _meas_basis_candidates(basis)
    expected_candidates = [(plane1, angle1), (plane2, angle2)]
    assert len(candidates) == len(expected_candidates)
    for expected in expected_candidates:
        assert any(
            candidate[0] == expected[0] and is_close_angle(candidate[1], expected[1]) for candidate in candidates
        )


@pytest.mark.parametrize("plane", list(Plane))
def test_meas_basis_info(plane: Plane, rng: np.random.Generator) -> None:
    angle = rng.uniform(0, 2 * np.pi)
    basis = meas_basis(plane, angle)
    plane_get, angle_get = meas_basis_info(basis)
    assert plane == plane_get, f"Expected {plane}, got {plane_get}"
    assert is_close_angle(angle, angle_get), f"Expected {angle}, got {angle_get}"


@pytest.mark.parametrize("case", DEGENERATE_CASES)
def test_meas_basis_info_degenerate(case: tuple[Plane, float, Plane, float]) -> None:
    plane1, angle1, plane2, angle2 = case
    basis = meas_basis(plane1, angle1)

    for expected_plane, expected_angle in ((plane1, angle1), (plane2, angle2)):
        plane_get, angle_get = meas_basis_info(basis, expected_plane=expected_plane)
        assert expected_plane == plane_get
        assert is_close_angle(expected_angle, angle_get)


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
    ref_updated_vector = lc.conjugate().matrix() @ basis.vector()
    inner_product = abs(np.vdot(basis_updated.vector(), ref_updated_vector))
    assert np.isclose(inner_product, 1)


@pytest.mark.parametrize("plane", list(Plane))
def test_local_complement_target_update(plane: Plane, rng: np.random.Generator) -> None:
    lc = LocalClifford(0, np.pi / 2, 0)
    measurement_action: dict[Plane, tuple[Plane, Callable[[float], float]]] = {
        Plane.XY: (Plane.XZ, lambda angle: angle + np.pi / 2),
        Plane.XZ: (Plane.XY, lambda angle: -angle + np.pi / 2),
        Plane.YZ: (Plane.YZ, lambda angle: angle + np.pi / 2),
    }

    angle = rng.random() * 2 * np.pi

    meas_basis = PlannerMeasBasis(plane, angle)
    result_basis = update_lc_basis(lc, meas_basis)
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
    result_basis = update_lc_basis(lc, meas_basis)
    ref_plane, ref_angle_func = measurement_action[plane]
    ref_angle = ref_angle_func(angle)

    assert result_basis.plane == ref_plane
    assert is_close_angle(result_basis.angle, ref_angle)


@pytest.mark.parametrize("plane", list(Plane))
def test_pivot_target_update(plane: Plane, rng: np.random.Generator) -> None:
    lc = LocalClifford(np.pi / 2, np.pi / 2, np.pi / 2)
    measurement_action: dict[Plane, tuple[Plane, Callable[[float], float]]] = {
        Plane.XY: (Plane.YZ, lambda angle: -1 * angle),
        Plane.XZ: (Plane.XZ, lambda angle: (np.pi / 2 - angle)),
        Plane.YZ: (Plane.XY, lambda angle: -1 * angle),
    }

    angle = rng.random() * 2 * np.pi

    meas_basis = PlannerMeasBasis(plane, angle)
    result_basis = update_lc_basis(lc, meas_basis)
    ref_plane, ref_angle_func = measurement_action[plane]
    ref_angle = ref_angle_func(angle)

    assert result_basis.plane == ref_plane
    assert is_close_angle(result_basis.angle, ref_angle)


@pytest.mark.parametrize("plane", list(Plane))
def test_pivot_neighbors(plane: Plane, rng: np.random.Generator) -> None:
    lc = LocalClifford(np.pi, 0, 0)
    measurement_action: dict[Plane, tuple[Plane, Callable[[float], float]]] = {
        Plane.XY: (Plane.XY, lambda angle: (angle + np.pi) % (2.0 * np.pi)),
        Plane.XZ: (Plane.XZ, lambda angle: -1 * angle),
        Plane.YZ: (Plane.YZ, lambda angle: -1 * angle),
    }

    angle = rng.random() * 2 * np.pi

    meas_basis = PlannerMeasBasis(plane, angle)
    result_basis = update_lc_basis(lc, meas_basis)
    ref_plane, ref_angle_func = measurement_action[plane]
    ref_angle = ref_angle_func(angle)

    assert result_basis.plane == ref_plane
    assert is_close_angle(result_basis.angle, ref_angle)


@pytest.mark.parametrize("plane", list(Plane))
def test_remove_clifford_update(plane: Plane, rng: np.random.Generator) -> None:
    measurement_action: dict[Plane, tuple[Plane, Callable[[float, float], float]]] = {
        Plane.XY: (
            Plane.XY,
            lambda a_pi, alpha: (alpha if is_close_angle(a_pi, 0) else alpha + np.pi) % (2.0 * np.pi),
        ),
        Plane.XZ: (
            Plane.XZ,
            lambda a_pi, alpha: (alpha if is_close_angle(a_pi, 0) else -alpha) % (2.0 * np.pi),
        ),
        Plane.YZ: (
            Plane.YZ,
            lambda a_pi, alpha: (alpha if is_close_angle(a_pi, 0) else -alpha) % (2.0 * np.pi),
        ),
    }

    angle = rng.random() * 2 * np.pi

    a_pi = np.pi
    for a_pi in (0.0, np.pi):
        lc = LocalClifford(a_pi, 0, 0)
        meas_basis = PlannerMeasBasis(plane, angle)
        result_basis = update_lc_basis(lc, meas_basis)
        ref_plane, ref_angle_func = measurement_action[plane]
        ref_angle = ref_angle_func(a_pi, angle)

        assert result_basis.plane == ref_plane
        assert is_close_angle(result_basis.angle, ref_angle)
