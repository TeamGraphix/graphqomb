from enum import Enum, auto


class Plane(Enum):
    XY = auto()
    YZ = auto()
    ZX = auto()
    YX = XY
    ZY = YZ
    XZ = ZX
