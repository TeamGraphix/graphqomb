r"""Noise model interface for Stim circuit compilation.

This module provides an abstract interface for injecting custom noise into
Stim circuits during pattern compilation. Users can implement the
:class:`NoiseModel` abstract base class to define noise behavior for
different events (preparation, entanglement, measurement, and idle).

Examples
--------
Create a simple depolarizing noise model:

>>> from graphqomb.noise_model import NoiseModel, NoiseEvent, NoiseKind, NoiseOp
>>>
>>> class DepolarizingNoise(NoiseModel):
...     def __init__(self, p: float) -> None:
...         self.p = p
...
...     def emit(self, event: NoiseEvent) -> list[NoiseOp]:
...         if event.kind == NoiseKind.PREPARE:
...             node = event.nodes[0]
...             return [NoiseOp(f"DEPOLARIZE1({self.p}) {node}")]
...         elif event.kind == NoiseKind.ENTANGLE:
...             n0, n1 = event.nodes
...             return [NoiseOp(f"DEPOLARIZE2({self.p}) {n0} {n1}")]
...         return []

Use a heralded noise model that adds measurement records:

>>> class HeraldedNoise(NoiseModel):
...     def emit(self, event: NoiseEvent) -> list[NoiseOp]:
...         if event.kind == NoiseKind.MEASURE:
...             node = event.nodes[0]
...             # HERALDED_PAULI_CHANNEL_1 adds one measurement record per target
...             return [NoiseOp(f"HERALDED_PAULI_CHANNEL_1(0,0,0,0.1) {node}", record_delta=1)]
...         return []

Pass the noise model to stim_compile:

>>> from graphqomb.stim_compiler import stim_compile
>>> # pattern = ...  # your compiled pattern
>>> # stim_str = stim_compile(pattern, noise_model=DepolarizingNoise(0.001))
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphqomb.common import Axis


class NoiseKind(enum.Enum):
    """Kind of noise injection event.

    - ``PREPARE``: Noise injected after qubit preparation (N command).
    - ``ENTANGLE``: Noise injected after entanglement (E command).
    - ``MEASURE``: Noise injected before measurement (M command).
    - ``IDLE``: Noise injected for qubits that are idle during a TICK.
    """

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
        For example: ``"DEPOLARIZE1(0.001) 0"`` or ``"X_ERROR(0.01) 5"``.
    record_delta : `int`, optional
        The number of measurement records appended by this instruction.
        Most noise instructions do not add records (default 0).
        However, ``HERALDED_*`` instructions add one record per target qubit.

    Examples
    --------
    >>> NoiseOp("DEPOLARIZE1(0.001) 0")
    NoiseOp(text='DEPOLARIZE1(0.001) 0', record_delta=0)

    >>> NoiseOp("HERALDED_PAULI_CHANNEL_1(0,0,0,0.1) 5", record_delta=1)
    NoiseOp(text='HERALDED_PAULI_CHANNEL_1(0,0,0,0.1) 5', record_delta=1)
    """

    text: str
    record_delta: int = 0


@dataclass(frozen=True)
class NoiseEvent:
    r"""Context describing where noise should be injected.

    This dataclass is passed to :meth:`NoiseModel.emit` to provide full context
    about the current compilation state and the operation being performed.

    Parameters
    ----------
    kind : `NoiseKind`
        The type of event triggering noise injection.
    tick : `int`
        The current tick (time step) in the pattern execution.
        Starts at 0 and increments with each TICK command.
    nodes : `tuple`\[`int`, ...]
        The node indices involved in this event.
        For PREPARE/MEASURE: single node ``(node,)``.
        For ENTANGLE: two nodes ``(node1, node2)``.
        For IDLE: all idle nodes ``(node1, node2, ...)``.
    edge : `tuple`\[`int`, `int`\] | `None`
        For ENTANGLE events, the edge as ``(min_node, max_node)``.
        None for other event kinds.
    coords : `tuple`\[`tuple`\[`float`, `float`\] | `None`, ...]
        The (x, y) coordinates for each node in ``nodes``, if available.
        None for nodes without assigned coordinates.
    axis : `Axis` | `None`
        For MEASURE events, the measurement axis (X, Y, or Z).
        None for other event kinds.
    duration : `float` | `None`, optional
        For IDLE events, the duration of the idle period (from ``tick_duration``).
        None for other event kinds. Default is None.
    is_input : `bool`, optional
        For PREPARE events, whether this is an input node of the pattern.
        Input nodes may require different noise treatment. Default is False.

    Examples
    --------
    A preparation event for node 5 at tick 0:

    >>> from graphqomb.noise_model import NoiseEvent, NoiseKind
    >>> event = NoiseEvent(
    ...     kind=NoiseKind.PREPARE,
    ...     tick=0,
    ...     nodes=(5,),
    ...     edge=None,
    ...     coords=((1.0, 2.0),),
    ...     axis=None,
    ...     is_input=True,
    ... )

    An entanglement event between nodes 3 and 7:

    >>> event = NoiseEvent(
    ...     kind=NoiseKind.ENTANGLE,
    ...     tick=1,
    ...     nodes=(3, 7),
    ...     edge=(3, 7),
    ...     coords=((0.0, 0.0), (1.0, 0.0)),
    ...     axis=None,
    ... )
    """

    kind: NoiseKind
    tick: int
    nodes: tuple[int, ...]
    edge: tuple[int, int] | None
    coords: tuple[tuple[float, float] | None, ...]
    axis: Axis | None
    duration: float | None = None
    is_input: bool = False


class NoiseModel(abc.ABC):
    r"""Abstract base class for custom noise injection during Stim compilation.

    Subclass this to define custom noise behavior. The :meth:`emit` method
    is called at each event during pattern compilation, allowing you to
    inject arbitrary Stim noise instructions.

    Examples
    --------
    A noise model that adds different noise for each event type:

    >>> from graphqomb.noise_model import NoiseModel, NoiseEvent, NoiseKind, NoiseOp
    >>>
    >>> class CustomNoise(NoiseModel):
    ...     def emit(self, event: NoiseEvent) -> list[NoiseOp]:
    ...         ops: list[NoiseOp] = []
    ...         if event.kind == NoiseKind.PREPARE:
    ...             # Add preparation error
    ...             ops.append(NoiseOp(f"X_ERROR(0.001) {event.nodes[0]}"))
    ...         elif event.kind == NoiseKind.ENTANGLE:
    ...             # Add two-qubit depolarizing noise
    ...             n0, n1 = event.nodes
    ...             ops.append(NoiseOp(f"DEPOLARIZE2(0.01) {n0} {n1}"))
    ...         elif event.kind == NoiseKind.MEASURE:
    ...             # Add measurement error via heralded channel
    ...             node = event.nodes[0]
    ...             ops.append(NoiseOp(f"HERALDED_PAULI_CHANNEL_1(0,0,0,0.005) {node}", record_delta=1))
    ...         elif event.kind == NoiseKind.IDLE:
    ...             # Add idle noise based on duration
    ...             if event.duration is not None:
    ...                 p = 0.0001 * event.duration
    ...                 for node in event.nodes:
    ...                     ops.append(NoiseOp(f"DEPOLARIZE1({p}) {node}"))
    ...         return ops

    Notes
    -----
    When using ``HERALDED_*`` instructions or other measurement-like operations,
    set ``record_delta`` appropriately so that detector indices are computed
    correctly. Each ``HERALDED_*`` instruction adds one measurement record
    per target qubit.

    See Also
    --------
    stim_compile : The main compilation function that accepts a NoiseModel.
    """

    @abc.abstractmethod
    def emit(self, event: NoiseEvent) -> Iterable[NoiseOp]:
        r"""Return Stim instructions to inject for the given event.

        Parameters
        ----------
        event : `NoiseEvent`
            The context describing where noise should be injected.

        Returns
        -------
        `collections.abc.Iterable`\[`NoiseOp`\]
            Zero or more Stim instructions to insert at this point.
            Return an empty iterable to skip noise injection for this event.

        Examples
        --------
        >>> class SimpleNoise(NoiseModel):
        ...     def emit(self, event: NoiseEvent) -> list[NoiseOp]:
        ...         if event.kind == NoiseKind.PREPARE:
        ...             return [NoiseOp(f"DEPOLARIZE1(0.001) {event.nodes[0]}")]
        ...         return []
        """
        raise NotImplementedError
