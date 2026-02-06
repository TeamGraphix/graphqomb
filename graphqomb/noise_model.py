r"""Noise model interface for Stim circuit compilation.

This module provides:

- `NoisePlacement`: Enum for noise placement.
- `Coordinate`: N-dimensional coordinate dataclass.
- `NodeInfo`: Node identifier with optional coordinate.
- `PrepareEvent`, `EntangleEvent`, `MeasureEvent`, `IdleEvent`: Event dataclasses.
- `NoiseEvent`: Union type of all event types.
- `PauliChannel1`, `PauliChannel2`, `HeraldedPauliChannel1`, `HeraldedErase`, `RawStimOp`,
  `MeasurementFlip`: NoiseOp types.
- `NoiseOp`: Union type of all noise operation types.
- `NoiseModel`: Base class for noise models.
- `DepolarizingNoiseModel`, `MeasurementFlipNoiseModel`: Built-in noise models.
- `noise_op_to_stim`: Conversion function.
- `PAULI_CHANNEL_2_ORDER`: Constant for Pauli channel order.

Examples
--------
Create a simple depolarizing noise model:

>>> from graphqomb.noise_model import (
...     NoiseModel, PrepareEvent, EntangleEvent, PauliChannel1, PauliChannel2
... )
>>>
>>> class DepolarizingNoise(NoiseModel):
...     def __init__(self, p1: float, p2: float) -> None:
...         self.p1 = p1  # Single-qubit depolarizing probability
...         self.p2 = p2  # Two-qubit depolarizing probability
...
...     def on_prepare(self, event: PrepareEvent) -> list[PauliChannel1]:
...         p = self.p1 / 3  # Equal probability for X, Y, Z
...         return [PauliChannel1(px=p, py=p, pz=p, targets=[event.node.id])]
...
...     def on_entangle(self, event: EntangleEvent) -> list[PauliChannel2]:
...         # Simplified: only add ZZ error
...         return [PauliChannel2(
...             probabilities={"ZZ": self.p2},
...             targets=[(event.node0.id, event.node1.id)]
...         )]

Use with stim_compile:

>>> from graphqomb.stim_compiler import stim_compile
>>> # pattern = ...  # your compiled pattern
>>> # stim_str = stim_compile(pattern, noise_model=DepolarizingNoise(0.001, 0.01))

Use heralded noise that adds measurement records:

>>> from graphqomb.noise_model import NoiseModel, MeasureEvent, HeraldedPauliChannel1
>>>
>>> class HeraldedMeasurementNoise(NoiseModel):
...     def on_measure(self, event: MeasureEvent) -> list[HeraldedPauliChannel1]:
...         # Heralded erasure with 10% probability
...         return [HeraldedPauliChannel1(
...             pi=0.1, px=0.0, py=0.0, pz=0.0,
...             targets=[event.node.id]
...         )]

Notes
-----
- **Placement control**: Each `NoiseOp` has a `placement` attribute.
  `AUTO` defers to :meth:`NoiseModel.default_placement`, while
  `BEFORE`/`AFTER` force insertion side.

- **Record delta**: Heralded instructions (`HeraldedPauliChannel1`,
  `HeraldedErase`) add measurement records. The compiler automatically
  tracks these to compute correct detector indices.

- **Coordinate access**: Events provide `NodeInfo` objects with optional
  coordinates, useful for position-dependent noise models.

See Also
--------
stim_compile : The main compilation function that accepts a NoiseModel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphqomb.common import Axis


PAULI_CHANNEL_2_ORDER: tuple[str, ...] = (
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
)


class NoisePlacement(Enum):
    """Where to insert noise relative to the main operation."""

    AUTO = auto()
    BEFORE = auto()
    AFTER = auto()


@dataclass(frozen=True)
class Coordinate:
    """N-dimensional coordinate for a node.

    Parameters
    ----------
    values : tuple[float, ...]
        The coordinate values as a tuple of floats.

    Examples
    --------
    >>> coord = Coordinate((1.0, 2.0, 3.0))
    >>> coord.xy
    (1.0, 2.0)
    >>> coord.xyz
    (1.0, 2.0, 3.0)
    """

    values: tuple[float, ...]

    @property
    def xy(self) -> tuple[float, float] | None:
        """Return the first two dimensions as (x, y), or None if fewer than 2 dimensions."""
        if len(self.values) < 2:  # noqa: PLR2004
            return None
        return (self.values[0], self.values[1])

    @property
    def xyz(self) -> tuple[float, float, float] | None:
        """Return the first three dimensions as (x, y, z), or None if fewer than 3 dimensions."""
        if len(self.values) < 3:  # noqa: PLR2004
            return None
        return (self.values[0], self.values[1], self.values[2])


@dataclass(frozen=True)
class NodeInfo:
    """Node identifier with optional coordinate.

    Parameters
    ----------
    id : int
        The unique node index in the pattern.
    coord : Coordinate | None
        The spatial coordinate of the node, if available.
    """

    id: int
    coord: Coordinate | None


@dataclass(frozen=True)
class PrepareEvent:
    """Event emitted when a qubit is prepared (N command).

    Parameters
    ----------
    time : int
        The current tick (time step) in the pattern execution.
    node : NodeInfo
        Information about the node being prepared.
    is_input : bool
        Whether this node is an input node of the pattern.
        Input nodes may require different noise treatment.
    """

    time: int
    node: NodeInfo
    is_input: bool


@dataclass(frozen=True)
class EntangleEvent:
    """Event emitted when two qubits are entangled (E command / CZ gate).

    Parameters
    ----------
    time : int
        The current tick (time step) in the pattern execution.
    node0 : NodeInfo
        Information about the first node in the entanglement.
    node1 : NodeInfo
        Information about the second node in the entanglement.
    edge : tuple[int, int]
        The edge as `(min_node_id, max_node_id)`.
    """

    time: int
    node0: NodeInfo
    node1: NodeInfo
    edge: tuple[int, int]


@dataclass(frozen=True)
class MeasureEvent:
    """Event emitted when a qubit is measured (M command).

    Parameters
    ----------
    time : int
        The current tick (time step) in the pattern execution.
    node : NodeInfo
        Information about the node being measured.
    axis : Axis
        The measurement axis (X, Y, or Z).
    """

    time: int
    node: NodeInfo
    axis: Axis


@dataclass(frozen=True)
class IdleEvent:
    """Event emitted for qubits that are idle during a TICK.

    Parameters
    ----------
    time : int
        The current tick (time step) in the pattern execution.
    nodes : Sequence[NodeInfo]
        Information about all nodes that are idle during this tick.
    duration : float
        The duration of the idle period (from `tick_duration` parameter).
    """

    time: int
    nodes: Sequence[NodeInfo]
    duration: float


NoiseEvent = PrepareEvent | EntangleEvent | MeasureEvent | IdleEvent
"""Union type of all noise event types."""


@dataclass(frozen=True)
class PauliChannel1:
    """Single-qubit Pauli channel noise operation.

    Applies independent X, Y, Z errors with given probabilities.
    Corresponds to Stim's `PAULI_CHANNEL_1` instruction.

    Parameters
    ----------
    px : float
        Probability of X error.
    py : float
        Probability of Y error.
    pz : float
        Probability of Z error.
    targets : Sequence[int]
        Target qubit indices.
    placement : NoisePlacement
        Whether to insert before or after the main operation.
        `AUTO` defers to the model's default placement for the event.

    Examples
    --------
    >>> op = PauliChannel1(px=0.01, py=0.01, pz=0.01, targets=[0, 1])
    >>> noise_op_to_stim(op)
    ('PAULI_CHANNEL_1(0.01,0.01,0.01) 0 1', 0)
    """

    px: float
    py: float
    pz: float
    targets: Sequence[int]
    placement: NoisePlacement = NoisePlacement.AUTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))


@dataclass(frozen=True)
class PauliChannel2:
    """Two-qubit Pauli channel noise operation.

    Applies correlated two-qubit Pauli errors.
    Corresponds to Stim's `PAULI_CHANNEL_2` instruction.

    Parameters
    ----------
    probabilities : Sequence[float] | Mapping[str, float]
        Either a sequence of 15 probabilities in the order
        (IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ),
        or a mapping from Pauli string keys to probabilities.
        Missing keys default to 0.
    targets : Sequence[tuple[int, int]]
        Target qubit pairs as `[(q0, q1), ...]`.
    placement : NoisePlacement
        Whether to insert before or after the main operation.
        `AUTO` defers to the model's default placement for the event.

    Examples
    --------
    Using a mapping (recommended for sparse errors):

    >>> op = PauliChannel2(probabilities={"ZZ": 0.01}, targets=[(0, 1)])
    >>> text, delta = noise_op_to_stim(op)
    >>> "PAULI_CHANNEL_2" in text
    True

    Using a full probability sequence:

    >>> probs = [0.0] * 14 + [0.01]  # Only ZZ error
    >>> op = PauliChannel2(probabilities=probs, targets=[(2, 3)])
    """

    probabilities: Sequence[float] | Mapping[str, float]
    targets: Sequence[tuple[int, int]]
    placement: NoisePlacement = NoisePlacement.AUTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(tuple(pair) for pair in self.targets))


@dataclass(frozen=True)
class HeraldedPauliChannel1:
    """Heralded single-qubit Pauli channel noise operation.

    Similar to `PauliChannel1` but produces a herald measurement record
    indicating whether an error occurred. The herald outcome is 1 if any
    error occurred (including identity with probability `pi`).
    Corresponds to Stim's `HERALDED_PAULI_CHANNEL_1` instruction.

    Parameters
    ----------
    pi : float
        Probability of heralded identity (no error but flagged).
    px : float
        Probability of heralded X error.
    py : float
        Probability of heralded Y error.
    pz : float
        Probability of heralded Z error.
    targets : Sequence[int]
        Target qubit indices.
    placement : NoisePlacement
        Whether to insert before or after the main operation.
        `AUTO` defers to the model's default placement for the event.

    Notes
    -----
    This instruction adds one measurement record per target qubit.
    The compiler automatically tracks this when computing detector indices.

    Examples
    --------
    >>> op = HeraldedPauliChannel1(pi=0.0, px=0.01, py=0.0, pz=0.0, targets=[5])
    >>> text, delta = noise_op_to_stim(op)
    >>> text
    'HERALDED_PAULI_CHANNEL_1(0.0,0.01,0.0,0.0) 5'
    >>> delta  # One record added per target
    1
    """

    pi: float
    px: float
    py: float
    pz: float
    targets: Sequence[int]
    placement: NoisePlacement = NoisePlacement.AUTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))


@dataclass(frozen=True)
class HeraldedErase:
    """Heralded erasure noise operation.

    Models photon loss or erasure errors with a herald signal.
    Corresponds to Stim's `HERALDED_ERASE` instruction.

    Parameters
    ----------
    p : float
        Probability of erasure.
    targets : Sequence[int]
        Target qubit indices.
    placement : NoisePlacement
        Whether to insert before or after the main operation.
        `AUTO` defers to the model's default placement for the event.

    Notes
    -----
    This instruction adds one measurement record per target qubit.
    The compiler automatically tracks this when computing detector indices.

    Examples
    --------
    >>> op = HeraldedErase(p=0.05, targets=[0, 1, 2])
    >>> text, delta = noise_op_to_stim(op)
    >>> text
    'HERALDED_ERASE(0.05) 0 1 2'
    >>> delta  # One record added per target
    3
    """

    p: float
    targets: Sequence[int]
    placement: NoisePlacement = NoisePlacement.AUTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))


@dataclass(frozen=True)
class RawStimOp:
    """Raw Stim instruction for advanced use cases.

    Use this when the typed noise operations don't cover your use case.
    The text is inserted directly into the Stim circuit.

    Parameters
    ----------
    text : str
        A single Stim instruction line (without trailing newline).
    record_delta : int
        The number of measurement records added by this instruction.
        Most noise instructions do not add records (default 0).
    placement : NoisePlacement
        Whether to insert before or after the main operation.
        `AUTO` defers to the model's default placement for the event.

    Examples
    --------
    >>> op = RawStimOp("X_ERROR(0.001) 0 1 2")
    >>> noise_op_to_stim(op)
    ('X_ERROR(0.001) 0 1 2', 0)

    With custom record delta for measurement-like instructions:

    >>> op = RawStimOp("MR 5", record_delta=1)
    >>> noise_op_to_stim(op)
    ('MR 5', 1)
    """

    text: str
    record_delta: int = 0
    placement: NoisePlacement = NoisePlacement.AUTO


@dataclass(frozen=True)
class MeasurementFlip:
    """Measurement flip error applied to measurement instruction.

    Unlike other NoiseOp types that insert separate instructions,
    this modifies the measurement instruction itself to use Stim's
    built-in measurement error probability: MX(p) instead of MX.

    Parameters
    ----------
    p : float
        Probability of measurement result flip.
    target : int
        Target qubit index (must match the measurement target).
    placement : NoisePlacement
        Placement attribute for compatibility (ignored, as this modifies
        the measurement instruction itself).
    """

    p: float
    target: int
    placement: NoisePlacement = NoisePlacement.AUTO


NoiseOp = PauliChannel1 | PauliChannel2 | HeraldedPauliChannel1 | HeraldedErase | RawStimOp | MeasurementFlip
"""Union type of all noise operation types."""


class NoiseModel:
    """Base class for custom noise injection during Stim compilation.

    Subclass this to define custom noise behavior by overriding one or more
    of the event handler methods. Each method receives an event object with
    context about the current operation and returns noise operations to inject.

    Examples
    --------
    >>> class SimpleNoise(NoiseModel):
    ...     def on_prepare(self, event: PrepareEvent) -> list[PauliChannel1]:
    ...         # Add depolarizing noise after preparation
    ...         p = 0.001 / 3
    ...         return [PauliChannel1(px=p, py=p, pz=p, targets=[event.node.id])]
    ...
    ...     def on_measure(self, event: MeasureEvent) -> list[PauliChannel1]:
    ...         # Add bit-flip noise before measurement
    ...         return [PauliChannel1(
    ...             px=0.01, py=0.0, pz=0.0,
    ...             targets=[event.node.id],
    ...             placement=NoisePlacement.BEFORE
    ...         )]

    See Also
    --------
    stim_compile : The main compilation function that accepts a NoiseModel.
    """

    def on_prepare(self, event: PrepareEvent) -> Sequence[NoiseOp]:  # noqa: ARG002, PLR6301
        """Return noise operations to inject at qubit preparation.

        Parameters
        ----------
        event : PrepareEvent
            Context about the preparation operation.

        Returns
        -------
        Sequence[NoiseOp]
            Zero or more noise operations to inject.
        """
        return []

    def on_entangle(self, event: EntangleEvent) -> Sequence[NoiseOp]:  # noqa: ARG002, PLR6301
        """Return noise operations to inject at entanglement.

        Parameters
        ----------
        event : EntangleEvent
            Context about the entanglement operation.

        Returns
        -------
        Sequence[NoiseOp]
            Zero or more noise operations to inject.
        """
        return []

    def on_measure(self, event: MeasureEvent) -> Sequence[NoiseOp]:  # noqa: ARG002, PLR6301
        """Return noise operations to inject at measurement.

        Parameters
        ----------
        event : MeasureEvent
            Context about the measurement operation.

        Returns
        -------
        Sequence[NoiseOp]
            Zero or more noise operations to inject.
        """
        return []

    def on_idle(self, event: IdleEvent) -> Sequence[NoiseOp]:  # noqa: ARG002, PLR6301
        """Return noise operations to inject during idle periods.

        Parameters
        ----------
        event : IdleEvent
            Context about the idle period.

        Returns
        -------
        Sequence[NoiseOp]
            Zero or more noise operations to inject.
        """
        return []

    def default_placement(self, event: NoiseEvent) -> NoisePlacement:  # noqa: PLR6301
        """Return the default placement for AUTO noise operations.

        By default, measurement noise is injected before measurement and all
        other noise is injected after the main operation.

        Parameters
        ----------
        event : NoiseEvent
            The event for which to determine the default placement.

        Returns
        -------
        NoisePlacement
            `BEFORE` for measurement events, `AFTER` for all others.
        """
        if isinstance(event, MeasureEvent):
            return NoisePlacement.BEFORE
        return NoisePlacement.AFTER


def noise_op_to_stim(op: NoiseOp) -> tuple[str, int]:  # noqa: PLR0911, C901
    """Convert a NoiseOp into a Stim instruction line and record delta.

    Parameters
    ----------
    op : NoiseOp
        The noise operation to convert.

    Returns
    -------
    tuple[str, int]
        A tuple of `(stim_instruction, record_delta)` where
        `stim_instruction` is a single line of Stim code and
        `record_delta` is the number of measurement records added.

    Raises
    ------
    TypeError
        If `op` is not a recognized NoiseOp type.

    Examples
    --------
    >>> op = PauliChannel1(px=0.01, py=0.02, pz=0.03, targets=[0])
    >>> noise_op_to_stim(op)
    ('PAULI_CHANNEL_1(0.01,0.02,0.03) 0', 0)
    """
    if isinstance(op, RawStimOp):
        return op.text, op.record_delta

    if isinstance(op, PauliChannel1):
        if not op.targets:
            return "", 0
        targets = " ".join(str(t) for t in op.targets)
        return f"PAULI_CHANNEL_1({op.px},{op.py},{op.pz}) {targets}", 0

    if isinstance(op, PauliChannel2):
        if not op.targets:
            return "", 0
        args = _pauli_channel_2_args(op.probabilities)
        flat_targets = _flatten_pairs(op.targets)
        targets_str = " ".join(str(t) for t in flat_targets)
        args_str = ",".join(str(v) for v in args)
        return f"PAULI_CHANNEL_2({args_str}) {targets_str}", 0

    if isinstance(op, HeraldedPauliChannel1):
        if not op.targets:
            return "", 0
        targets = " ".join(str(t) for t in op.targets)
        return (
            f"HERALDED_PAULI_CHANNEL_1({op.pi},{op.px},{op.py},{op.pz}) {targets}",
            len(op.targets),
        )

    if isinstance(op, HeraldedErase):
        if not op.targets:
            return "", 0
        targets = " ".join(str(t) for t in op.targets)
        return f"HERALDED_ERASE({op.p}) {targets}", len(op.targets)

    if isinstance(op, MeasurementFlip):
        # MeasurementFlip is handled specially in the compiler by modifying
        # the measurement instruction. It should not be emitted as a separate op.
        return "", 0

    msg = f"Unsupported noise op type: {type(op)!r}"
    raise TypeError(msg)


def _pauli_channel_2_args(probabilities: Sequence[float] | Mapping[str, float]) -> tuple[float, ...]:
    if isinstance(probabilities, Mapping):
        unknown = set(probabilities) - set(PAULI_CHANNEL_2_ORDER)
        if unknown:
            msg = f"Unknown PAULI_CHANNEL_2 keys: {sorted(unknown)}"
            raise ValueError(msg)
        return tuple(float(probabilities.get(key, 0.0)) for key in PAULI_CHANNEL_2_ORDER)
    values = tuple(float(v) for v in probabilities)
    if len(values) != len(PAULI_CHANNEL_2_ORDER):
        msg = f"PAULI_CHANNEL_2 expects {len(PAULI_CHANNEL_2_ORDER)} probabilities, got {len(values)}"
        raise ValueError(msg)
    return values


def _flatten_pairs(pairs: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    flat: list[int] = []
    for pair in pairs:
        if len(pair) != 2:  # noqa: PLR2004
            msg = f"PAULI_CHANNEL_2 targets must be pairs, got: {pair!r}"
            raise ValueError(msg)
        flat.extend(pair)
    return tuple(flat)


# ---- Built-in NoiseModel implementations ----


class DepolarizingNoiseModel(NoiseModel):
    """Depolarizing noise after single and two-qubit gates.

    This model adds depolarizing noise after qubit preparation (RX) and
    entanglement (CZ) operations.

    Parameters
    ----------
    p1 : float
        Single-qubit depolarizing probability (after RX preparation).
    p2 : float | None
        Two-qubit depolarizing probability (after CZ).
        If None, defaults to p1.

    Examples
    --------
    >>> from graphqomb.noise_model import DepolarizingNoiseModel
    >>> model = DepolarizingNoiseModel(p1=0.001, p2=0.01)
    >>> # Use with stim_compile:
    >>> # stim_compile(pattern, noise_models=[model])
    """

    def __init__(self, p1: float, p2: float | None = None) -> None:
        self._p1 = p1
        self._p2 = p2 if p2 is not None else p1

    def on_prepare(self, event: PrepareEvent) -> Sequence[NoiseOp]:
        """Add single-qubit depolarizing noise after preparation.

        Returns
        -------
        Sequence[NoiseOp]
            A tuple containing DEPOLARIZE1 instruction, or empty if p1 <= 0.
        """
        if self._p1 <= 0:
            return ()
        return (RawStimOp(f"DEPOLARIZE1({self._p1}) {event.node.id}"),)

    def on_entangle(self, event: EntangleEvent) -> Sequence[NoiseOp]:
        """Add two-qubit depolarizing noise after entanglement.

        Returns
        -------
        Sequence[NoiseOp]
            A tuple containing DEPOLARIZE2 instruction, or empty if p2 <= 0.
        """
        if self._p2 <= 0:
            return ()
        return (RawStimOp(f"DEPOLARIZE2({self._p2}) {event.node0.id} {event.node1.id}"),)


class MeasurementFlipNoiseModel(NoiseModel):
    """Measurement bit-flip noise using Stim's built-in measurement error.

    This model produces MX(p), MY(p), MZ(p) instead of MX, MY, MZ,
    which adds measurement flip error with probability p.

    Parameters
    ----------
    p : float
        Probability of measurement result flip.

    Examples
    --------
    >>> from graphqomb.noise_model import MeasurementFlipNoiseModel
    >>> model = MeasurementFlipNoiseModel(p=0.001)
    >>> # Use with stim_compile:
    >>> # stim_compile(pattern, noise_models=[model])
    """

    def __init__(self, p: float) -> None:
        self._p = p

    def on_measure(self, event: MeasureEvent) -> Sequence[NoiseOp]:
        """Add measurement flip error.

        Returns
        -------
        Sequence[NoiseOp]
            A tuple containing MeasurementFlip operation, or empty if p <= 0.
        """
        if self._p <= 0:
            return ()
        return (MeasurementFlip(p=self._p, target=event.node.id),)
