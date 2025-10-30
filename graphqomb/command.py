"""Command module for measurement pattern.

This module provides:

- `N`: Preparation command.
- `E`: Entanglement command.
- `M`: Measurement command.
- `X`: X correction command.
- `Z`: Z correction command.
- `TICK`: Time slice separator command.
- `Command`: Type alias of all commands.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from graphqomb.common import MeasBasis


@dataclasses.dataclass
class N:
    """Preparation command.

    Attributes
    ----------
    node : `int`
        The node index to be prepared.
    """

    node: int

    def __str__(self) -> str:
        return f"N: node={self.node}"


@dataclasses.dataclass
class E:
    r"""Entanglement command.

    Attributes
    ----------
    nodes : `tuple`\[`int`, `int`\]
        The node indices to be entangled.
    """

    nodes: tuple[int, int]

    def __str__(self) -> str:
        return f"E: nodes={self.nodes}"


@dataclasses.dataclass
class M:
    """Measurement command.

    Attributes
    ----------
    node : `int`
        The node index to be measured.
    meas_basis : MeasBasis
        The measurement basis.
    """

    node: int
    meas_basis: MeasBasis

    def __str__(self) -> str:
        return f"M: node={self.node}, plane={self.meas_basis.plane}, angle={self.meas_basis.angle}"


@dataclasses.dataclass
class _Correction:
    node: int


@dataclasses.dataclass
class X(_Correction):
    """X correction command.

    Attributes
    ----------
    node : `int`
        The node index to apply the correction.
    """

    def __str__(self) -> str:
        return f"X: node={self.node}"


@dataclasses.dataclass
class Z(_Correction):
    """Z correction command.

    Attributes
    ----------
    node : `int`
        The node index to apply the correction.
    """

    def __str__(self) -> str:
        return f"Z: node={self.node}"


@dataclasses.dataclass
class TICK:
    """Time slice separator command.

    Marks the boundary between time slices. Commands between two consecutive
    TICK commands can be executed in parallel within the same time slice.
    """

    def __str__(self) -> str:
        return "TICK"


if sys.version_info >= (3, 10):
    Command = N | E | M | X | Z | TICK
else:
    Command = Union[N, E, M, X, Z, TICK]
