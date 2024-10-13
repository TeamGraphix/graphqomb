"""Data validator command classes."""

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
    """Preparation command."""

    node: Node
    q_index: int = -1


@dataclass
class M:
    """Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop."""

    node: Node
    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: set[Node] = field(default_factory=set)
    t_domain: set[Node] = field(default_factory=set)


@dataclass
class E:
    """Entanglement command."""

    nodes: tuple[Node, Node]


@dataclass
class C:
    """Clifford command."""

    node: Node
    local_clifford: LocalClifford


@dataclass
class Correction:
    """Correction command.Either X or Z."""

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
