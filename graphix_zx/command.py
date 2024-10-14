"""Command module for measurement pattern.

This module provides:
- N: Preparation command.
- M: Measurement command.
- E: Entanglement command.
- C: Clifford command.
- Correction: Base class for Pauli correction command.
- X: X correction command.
- Z: Z correction command.
- Command: Type alias of all commands.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from graphix_zx.common import Plane

if TYPE_CHECKING:
    from graphix_zx.euler import LocalClifford

Node = int


@dataclass
class N:
    """Preparation command.

    Attributes
    ----------
    node : int
        The node index to be prepared.
    q_index : int
        The logical qubit index.
        Default is -1, which means the node is an ancilla.
    """

    node: Node
    q_index: int = -1


@dataclass
class M:
    """Measurement command.

    Attributes
    ----------
    node : int
        The node index to be measured.
    plane : Plane
        The measurement plane.
    angle : float
        The measurement angle.
    s_domain : set[int]
        The s_domain of the measurement.
    t_domain : set[int]
        The t_domain of the measurement.

    See Also
    --------
    graphix_zx.simulator.PatternSimulator._apply_m :
        The s_domain and t_domain follow the same definition in the original paper
        Journal of the ACM (JACM), Volume 54, Issue 2.

    """

    node: Node
    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: set[Node] = field(default_factory=set)
    t_domain: set[Node] = field(default_factory=set)


@dataclass
class E:
    """Entanglement command.

    Attributes
    ----------
    nodes : tuple[int, int]
        The node indices to be entangled.
    """

    nodes: tuple[Node, Node]


@dataclass
class C:
    """Clifford command.

    Attributes
    ----------
    node : int
        The node index to apply the local Clifford.
    local_clifford : LocalClifford
        The local Clifford to apply
    """

    node: Node
    local_clifford: LocalClifford


@dataclass
class Correction:
    """Correction command.Either X or Z.

    Attributes
    ----------
    node : int
        The node index to apply the correction.
    domain : set[int]
        The domain of the correction
    """

    node: Node
    domain: set[Node] = field(default_factory=set)


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
