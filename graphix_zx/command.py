"""Command module for measurement pattern.

This module provides:

- `N`: Preparation command.
- `E`: Entanglement command.
- `M`: Measurement command.
- `X`: X correction command.
- `Z`: Z correction command.
- `Clifford`: Clifford command.
- `D`: Decode command.
- `Command`: Type alias of all commands.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from graphix_zx.common import MeasBasis, default_meas_basis

if TYPE_CHECKING:
    from graphix_zx.decoder_backend import BaseDecoder
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
class E:
    r"""Entanglement command.

    Attributes
    ----------
    nodes : `tuple`\[`int`, `int`\]
        The node indices to be entangled.
    """

    nodes: tuple[int, int]


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
class _Correction:
    """Base Correction command. Either X or Z.

    Attributes
    ----------
    node : `int`
        The node index to apply the correction.
    flag : `bool`
        The domain of the correction.
    """

    node: int
    flag: bool = False


@dataclass
class X(_Correction):
    """X correction command."""


@dataclass
class Z(_Correction):
    """Z correction command."""


@dataclass
class Clifford:
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
class D:
    r"""Decode command.

    Attributes
    ----------
    input_cbits : `dict`\[`int`, `bool`\]
        A dictionary mapping classical bit indices to their boolean values.
    output_cbits : `list`\[`int`\]
        A list of classical bit indices that will store the decoding results.
    decoder : `BaseDecoder`
        The decoder instance used to process the input classical bits and produce the output.
    """

    input_cbits: dict[int, bool]
    output_cbits: list[int]
    decoder: BaseDecoder


if sys.version_info >= (3, 10):
    Command = N | E | M | X | Z | Clifford | D
else:
    Command = Union[N, E, M, X, Z, Clifford, D]
