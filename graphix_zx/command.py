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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from graphix_zx.common import MeasBasis
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
    s_cbit : `int`
        The index of s_domain control classical bit.
    t_cbit : `int`
        The index of t_domain control classical bit.
    """

    node: int
    meas_basis: MeasBasis
    s_cbit: int
    t_cbit: int


@dataclass
class _Correction:
    node: int
    cbit: int


@dataclass
class X(_Correction):
    """X correction command.

    Attributes
    ----------
    node : `int`
        The node index to apply the correction.
    cbit : `int`
        The index of the classical bit to control the correction.
    """


@dataclass
class Z(_Correction):
    """Z correction command.

    Attributes
    ----------
    node : `int`
        The node index to apply the correction.
    cbit : `int`
        The index of the classical bit to control the correction.
    """


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
