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

import dataclasses
import sys
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from graphix_zx.common import MeasBasis
    from graphix_zx.decoder_backend import BaseDecoder
    from graphix_zx.euler import LocalClifford


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
    s_cbit : `int` | `None`
        The index of s_domain control classical bit.
        Default is None, meaning the flag is always False.
    t_cbit : `int` | `None`
        The index of t_domain control classical bit.
        Default is None, meaning the flag is always False.
    """

    node: int
    meas_basis: MeasBasis
    s_cbit: int | None = None
    t_cbit: int | None = None

    def __str__(self) -> str:
        return (
            f"M: node={self.node}, plane={self.meas_basis.plane}, "
            f"angle={self.meas_basis.angle}, s_cbit={self.s_cbit}, t_cbit={self.t_cbit}"
        )


@dataclasses.dataclass
class _Correction:
    node: int
    cbit: int | None = None


@dataclasses.dataclass
class X(_Correction):
    """X correction command.

    Attributes
    ----------
    node : `int`
        The node index to apply the correction.
    cbit : `int` | `None`
        The index of the classical bit to control the correction.
        If cbit is None, the flag is always False, meaning the correction will not be applied.
    """

    def __str__(self) -> str:
        return f"X: node={self.node}, cbit={self.cbit}"


@dataclasses.dataclass
class Z(_Correction):
    """Z correction command.

    Attributes
    ----------
    node : `int`
        The node index to apply the correction.
    cbit : `int` | `None`
        The index of the classical bit to control the correction.
        If cbit is None, the flag is always False, meaning the correction will not be applied.
    """

    def __str__(self) -> str:
        return f"Z: node={self.node}, cbit={self.cbit}"


@dataclasses.dataclass
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

    def __str__(self) -> str:
        return (
            f"Clifford: node={self.node}, alpha={self.local_clifford.alpha}, "
            f"beta={self.local_clifford.beta}, gamma={self.local_clifford.gamma}"
        )


@dataclasses.dataclass
class D:
    r"""Decode command.

    Attributes
    ----------
    input_cbits : `list`\[`int`\]
        A list of classical bit indices that will serve as input to the decoder.
    output_cbits : `list`\[`int`\]
        A list of classical bit indices that will store the decoding results.
    decoder : `BaseDecoder`
        The decoder instance used to process the input classical bits and produce the output.
    """

    input_cbits: list[int]
    output_cbits: list[int]
    decoder: BaseDecoder

    def __str__(self) -> str:
        return f"D: input_cbits={self.input_cbits}, output_cbits={self.output_cbits}, decoder={self.decoder}"


if sys.version_info >= (3, 10):
    Command = N | E | M | X | Z | Clifford | D
else:
    Command = Union[N, E, M, X, Z, Clifford, D]
