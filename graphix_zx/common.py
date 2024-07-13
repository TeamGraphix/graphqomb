import enum


class Plane(enum.Enum):
    XY = "XY"
    YZ = "YZ"
    ZX = "ZX"
    YX = XY
    ZY = YZ
    XZ = ZX
