"""Measurement planes for the ZX-calculus."""

from enum import Enum, auto


class Plane(Enum):
    """Measurement planes for the ZX-calculus."""

    XY = auto()
    YZ = auto()
    ZX = auto()
    YX = XY
    ZY = YZ
    XZ = ZX
