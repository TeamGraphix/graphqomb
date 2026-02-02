"""Noise model interface for Stim circuit compilation."""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphqomb.common import Axis


class NoiseKind(enum.Enum):
    """Kind of noise injection event."""

    PREPARE = enum.auto()
    ENTANGLE = enum.auto()
    MEASURE = enum.auto()
    IDLE = enum.auto()


@dataclass(frozen=True)
class NoiseOp:
    r"""A single Stim instruction plus its measurement record delta.

    Parameters
    ----------
    text : `str`
        A single Stim instruction line (without trailing newline).
    record_delta : `int`, optional
        The number of measurement records appended by this instruction.
        For example, HERALDED_* instructions add one record per target.
    """

    text: str
    record_delta: int = 0


@dataclass(frozen=True)
class NoiseEvent:
    """Context describing where noise should be injected."""

    kind: NoiseKind
    tick: int
    nodes: tuple[int, ...]
    edge: tuple[int, int] | None
    coords: tuple[tuple[float, float] | None, ...]
    axis: Axis | None
    duration: float | None = None
    is_input: bool = False


class NoiseModel(abc.ABC):
    """Abstract base class for custom noise injection during Stim compilation."""

    @abc.abstractmethod
    def emit(self, event: NoiseEvent) -> Iterable[NoiseOp]:
        r"""Return Stim instructions to inject for the given event."""
        raise NotImplementedError
