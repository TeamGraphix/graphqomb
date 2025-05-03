"""Command module for measurement pattern.

This module provides:

- `N`: Preparation command.
- `M`: Measurement command.
- `E`: Entanglement command.
- `C`: Clifford command.
- `Correction`: Base class for Pauli correction command.
- `X`: X correction command.
- `Z`: Z correction command.
- `Command`: Type alias of all commands.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from graphix_zx.common import MeasBasis, default_meas_basis

if TYPE_CHECKING:
    from graphix_zx.euler import LocalClifford


@dataclass
class N:
    """Preparation command.

    Attributes
    ----------
    node : `int`
        The node index to be prepared.
    """

    node: int


@dataclass
class M:
    """Measurement command.

    Attributes
    ----------
    node : `int`
        The node index to be measured.
    meas_basis : MeasBasis
        The measurement basis.
    s_flag : `bool`
        The s_flag of the measurement.
    t_flag : `bool`
        The t_flag of the measurement.
    """

    node: int
    meas_basis: MeasBasis = field(default_factory=default_meas_basis)
    s_flag: bool = False
    t_flag: bool = False


@dataclass
class E:
    r"""Entanglement command.

    Attributes
    ----------
    nodes : `tuple`\[`int`, `int`\]
        The node indices to be entangled.
    """

    nodes: tuple[int, int]


@dataclass
class C:
    """Clifford command.

    Attributes
    ----------
    node : `int`
        The node index to apply the local Clifford.
    local_clifford : `LocalClifford`
        The local Clifford to apply
    """

    node: int
    local_clifford: LocalClifford


@dataclass
class Correction:
    """Correction command.Either X or Z.

    Attributes
    ----------
    node : `int`
        The node index to apply the correction.
    flag : `bool`
        The domain of the correction
    """

    node: int
    flag: bool = False


@dataclass
class X(Correction):
    """X correction command."""


@dataclass
class Z(Correction):
    """Z correction command."""


if sys.version_info >= (3, 10):
    Command = N | M | E | C | X | Z
else:
    Command = Union[N, M, E, C, X, Z]
