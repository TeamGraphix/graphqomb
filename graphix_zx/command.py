"""Data validator command classes."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from graphix_zx.common import Plane

Node = int


class CommandKind(str, enum.Enum):
    N = "N"
    M = "M"
    E = "E"
    C = "C"
    X = "X"
    Z = "Z"


class Command:
    """Base command class."""

    kind: CommandKind | None = None


@dataclass
class N(Command):
    """Preparation command."""

    node: Node
    q_index: int = -1
    kind: CommandKind = CommandKind.N


@dataclass
class M(Command):
    """Measurement command. By default the plane is set to 'XY', the angle to 0, empty domains and identity vop."""

    node: Node
    plane: Plane = Plane.XY
    angle: float = 0.0
    s_domain: set[Node] = field(default_factory=set)
    t_domain: set[Node] = field(default_factory=set)
    kind: CommandKind = CommandKind.M


@dataclass
class E(Command):
    """Entanglement command."""

    nodes: tuple[Node, Node]
    kind: CommandKind = CommandKind.E


@dataclass
class C(Command):
    """Clifford command."""

    node: Node
    cliff_index: int
    kind: CommandKind = CommandKind.C


@dataclass
class Correction(Command):
    """Correction command.Either X or Z."""

    node: Node
    domain: set[Node] = field(default_factory=set)


@dataclass
class X(Correction):
    """X correction command."""

    kind: CommandKind = CommandKind.X


@dataclass
class Z(Correction):
    """Z correction command."""

    kind: CommandKind = CommandKind.Z
