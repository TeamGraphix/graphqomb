from __future__ import annotations

import math

import pytest

from graphqomb.common import Plane, PlannerMeasBasis, is_clifford_angle, is_close_angle


def test_inverse_order_plane() -> None:
    with pytest.raises(AttributeError):
        _ = PlannerMeasBasis(Plane.YX, 0)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = PlannerMeasBasis(Plane.ZX, 0)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = PlannerMeasBasis(Plane.ZY, 0)  # type: ignore[attr-defined]


def test_is_clifford_angle() -> None:
    assert is_clifford_angle(0)
    assert is_clifford_angle(math.pi / 2)
    assert is_clifford_angle(math.pi)
    assert is_clifford_angle(3 * math.pi / 2)
    assert not is_clifford_angle(math.pi / 3)
    assert not is_clifford_angle(math.pi / 4)
    assert not is_clifford_angle(math.pi / 6)


def test_is_close_angle() -> None:
    assert is_close_angle(0, 0)
    assert is_close_angle(math.pi / 2, math.pi / 2)
    assert is_close_angle(math.pi, math.pi)
    assert is_close_angle(3 * math.pi / 2, 3 * math.pi / 2)
    assert not is_close_angle(0, math.pi / 2)
    assert not is_close_angle(math.pi / 2, math.pi)
    assert not is_close_angle(math.pi, 3 * math.pi / 2)
    assert not is_close_angle(3 * math.pi / 2, 0)

    # add 2 * math.pi to the second angle
    assert is_close_angle(0, 2 * math.pi)
    assert is_close_angle(math.pi / 2, 2 * math.pi + math.pi / 2)
    assert is_close_angle(math.pi, 2 * math.pi + math.pi)
    assert is_close_angle(3 * math.pi / 2, 2 * math.pi + 3 * math.pi / 2)

    # minus 2 * math.pi to the second angle
    assert is_close_angle(0, -2 * math.pi)
    assert is_close_angle(math.pi / 2, -2 * math.pi + math.pi / 2)
    assert is_close_angle(math.pi, -2 * math.pi + math.pi)
    assert is_close_angle(3 * math.pi / 2, -2 * math.pi + 3 * math.pi / 2)

    # boundary cases
    assert is_close_angle(-1e-10, 1e-10)
