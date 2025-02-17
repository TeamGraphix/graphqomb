import pytest

from graphix_zx.common import Plane, PlannerMeasBasis


def test_inverse_order_plane() -> None:
    with pytest.raises(AttributeError):
        _ = PlannerMeasBasis(Plane.YX, 0)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = PlannerMeasBasis(Plane.ZX, 0)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = PlannerMeasBasis(Plane.ZY, 0)  # type: ignore[attr-defined]
