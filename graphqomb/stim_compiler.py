"""Pattern to stim compiler.

This module provides:

- `stim_compile`: Function to compile a pattern into stim format.
"""

from __future__ import annotations

import math
from io import StringIO
from typing import TYPE_CHECKING
from warnings import warn

from graphqomb.command import TICK, E, M, N, X, Z
from graphqomb.common import Axis, MeasBasis, determine_pauli_axis
from graphqomb.noise_model import (
    Coordinate,
    DepolarizingNoiseModel,
    EntangleEvent,
    IdleEvent,
    MeasureEvent,
    MeasurementFlip,
    MeasurementFlipNoiseModel,
    NodeInfo,
    NoiseModel,
    NoiseOp,
    NoisePlacement,
    PrepareEvent,
    default_noise_placement,
    noise_op_to_stim,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    from graphqomb.pattern import Pattern


class _StimCompiler:
    def __init__(
        self,
        pattern: Pattern,
        *,
        emit_qubit_coords: bool,
        noise_models: Sequence[NoiseModel],
        tick_duration: float,
    ) -> None:
        self._pattern = pattern
        self._pframe = pattern.pauli_frame
        self._coord_lookup = pattern.coordinates
        self._emit_qubit_coords = emit_qubit_coords
        self._noise_models = noise_models
        self._tick_duration = tick_duration
        self._stim_io = StringIO()
        self._meas_order: dict[int, int] = {}
        self._rec_index = 0
        self._alive_nodes: set[int] = set(pattern.input_node_indices)
        self._touched_nodes: set[int] = set()
        self._tick = 0

    def compile(self, logical_observables: Mapping[int, Collection[int]] | None) -> str:
        self._emit_input_nodes()
        self._process_commands()
        total = self._rec_index
        self._emit_detectors(total)
        effective_observables = (
            logical_observables if logical_observables is not None else self._pframe.logical_observables
        )
        if effective_observables:
            self._emit_observables(effective_observables, total)
        return self._stim_io.getvalue().strip()

    def _emit_detectors(self, total_measurements: int) -> None:
        for checks in self._pframe.detector_groups():
            targets = [f"rec[{self._meas_order[c] - total_measurements}]" for c in checks]
            self._stim_io.write(f"DETECTOR {' '.join(targets)}\n")

    def _emit_observables(self, logical_observables: Mapping[int, Collection[int]], total_measurements: int) -> None:
        for log_idx, obs in sorted(logical_observables.items()):
            group = self._pframe.logical_observables_group(obs)
            targets = [f"rec[{self._meas_order[n] - total_measurements}]" for n in group]
            self._stim_io.write(f"OBSERVABLE_INCLUDE({log_idx}) {' '.join(targets)}\n")

    def _emit_input_nodes(self) -> None:
        coordinates = self._pattern.input_coordinates if self._emit_qubit_coords else None
        for node in self._pattern.input_node_indices:
            coord = coordinates.get(node) if coordinates else None
            self._process_prepare(node, coord, is_input=True)

    def _process_commands(self) -> None:
        for cmd in self._pattern:
            if isinstance(cmd, N):
                self._process_prepare(cmd.node, cmd.coordinate, is_input=False)
            elif isinstance(cmd, E):
                self._handle_entangle(cmd.nodes)
            elif isinstance(cmd, M):
                self._handle_measure(cmd.node, cmd.meas_basis)
            elif isinstance(cmd, TICK):
                self._handle_tick()
            elif isinstance(cmd, (X, Z)):
                cmd_name = type(cmd).__name__
                msg = (
                    f"Unsupported command for stim compilation: {cmd_name}. X/Z correction commands are not supported."
                )
                raise NotImplementedError(msg)
            else:
                msg = f"Unsupported command for stim compilation: {type(cmd).__name__}"
                raise TypeError(msg)

    def _process_prepare(self, node: int, coordinate: tuple[float, ...] | None, *, is_input: bool) -> None:
        event = PrepareEvent(time=self._tick, node=self._node_info(node), is_input=is_input)
        ops = self._collect_noise_ops_from_models(lambda m: m.on_prepare(event))
        default_placement = default_noise_placement(event)
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.BEFORE, default_placement)

        coord = coordinate if self._emit_qubit_coords else None
        if coord is not None:
            self._stim_io.write(f"QUBIT_COORDS({', '.join(str(c) for c in coord)}) {node}\n")
        self._stim_io.write(f"RX {node}\n")

        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.AFTER, default_placement)
        self._alive_nodes.add(node)
        self._touched_nodes.add(node)

    def _handle_entangle(self, nodes: tuple[int, int]) -> None:
        n0, n1 = nodes
        edge: tuple[int, int] = (n0, n1) if n0 < n1 else (n1, n0)
        event = EntangleEvent(time=self._tick, node0=self._node_info(n0), node1=self._node_info(n1), edge=edge)
        ops = self._collect_noise_ops_from_models(lambda m: m.on_entangle(event))
        default_placement = default_noise_placement(event)
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.BEFORE, default_placement)

        self._stim_io.write(f"CZ {n0} {n1}\n")
        self._touched_nodes.update(nodes)
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.AFTER, default_placement)

    def _handle_measure(self, node: int, meas_basis: MeasBasis) -> None:
        axis = determine_pauli_axis(meas_basis)
        if axis is None:
            msg = f"Unsupported measurement basis: {meas_basis.plane, meas_basis.angle}"
            raise ValueError(msg)
        event = MeasureEvent(time=self._tick, node=self._node_info(node), axis=axis)
        ops = self._collect_noise_ops_from_models(lambda m: m.on_measure(event))

        # Separate MeasurementFlip from other noise ops
        meas_flip_probs: list[float] = []
        other_ops: list[NoiseOp] = []
        for op in ops:
            if isinstance(op, MeasurementFlip):
                if op.target != node:
                    msg = (
                        f"MeasurementFlip target mismatch: measurement on node {node}, "
                        f"but flip targets node {op.target}"
                    )
                    raise ValueError(msg)
                meas_flip_probs.append(op.p)
            else:
                other_ops.append(op)
        if not meas_flip_probs:
            meas_flip_p = 0.0
        elif len(meas_flip_probs) == 1:
            meas_flip_p = meas_flip_probs[0]
        else:
            meas_flip_p = 1.0 - math.prod(1.0 - p for p in meas_flip_probs)
        meas_flip_p = min(max(meas_flip_p, 0.0), 1.0)

        default_placement = default_noise_placement(event)
        self._rec_index += self._emit_noise_ops(other_ops, NoisePlacement.BEFORE, default_placement)

        # Emit measurement with optional flip probability
        meas_instr = {Axis.X: "MX", Axis.Y: "MY", Axis.Z: "MZ"}[axis]
        if meas_flip_p > 0.0:
            self._stim_io.write(f"{meas_instr}({meas_flip_p}) {node}\n")
        else:
            self._stim_io.write(f"{meas_instr} {node}\n")

        self._meas_order[node] = self._rec_index
        self._rec_index += 1
        self._alive_nodes.discard(node)
        self._touched_nodes.add(node)
        self._rec_index += self._emit_noise_ops(other_ops, NoisePlacement.AFTER, default_placement)

    def _handle_tick(self) -> None:
        idle_nodes = sorted(self._alive_nodes - self._touched_nodes)
        if idle_nodes and self._noise_models:
            event = IdleEvent(
                time=self._tick,
                nodes=tuple(self._node_info(node) for node in idle_nodes),
                duration=self._tick_duration,
            )
            ops = self._collect_noise_ops_from_models(lambda m: m.on_idle(event))
            default_placement = default_noise_placement(event)
        else:
            ops = ()
            default_placement = NoisePlacement.AFTER
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.BEFORE, default_placement)
        self._stim_io.write("TICK\n")
        self._rec_index += self._emit_noise_ops(ops, NoisePlacement.AFTER, default_placement)
        self._touched_nodes.clear()
        self._tick += 1

    def _node_info(self, node: int) -> NodeInfo:
        coord_raw = self._coord_lookup.get(node)
        coord = Coordinate(tuple(coord_raw)) if coord_raw is not None else None
        return NodeInfo(id=node, coord=coord)

    def _collect_noise_ops_from_models(self, get_ops: Callable[[NoiseModel], Iterable[NoiseOp]]) -> tuple[NoiseOp, ...]:
        ops: list[NoiseOp] = []
        for model in self._noise_models:
            ops.extend(get_ops(model))
        return tuple(ops)

    def _emit_noise_ops(
        self, ops: Iterable[NoiseOp], placement: NoisePlacement, default_placement: NoisePlacement
    ) -> int:
        record_delta = 0
        for op in ops:
            op_placement = op.placement
            if op_placement is NoisePlacement.AUTO:
                op_placement = default_placement
            if op_placement is not placement:
                continue
            text, delta = noise_op_to_stim(op)
            if text:
                self._stim_io.write(f"{text}\n")
            record_delta += delta
        return record_delta


def _validate_probability_parameter(name: str, value: float) -> None:
    """Validate that a probability parameter is within [0, 1].

    Parameters
    ----------
    name : `str`
        Parameter name used in error messages.
    value : `float`
        Probability value to validate.

    Raises
    ------
    ValueError
        If ``value`` is outside the inclusive range [0, 1].
    """
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be within [0, 1], got {value}"
        raise ValueError(msg)


def _normalize_noise_models(
    noise_models: Sequence[NoiseModel] | None,
    p_depol_after_clifford: float | None,
    p_before_meas_flip: float | None,
) -> tuple[NoiseModel, ...]:
    r"""Normalize legacy noise parameters and new noise model API.

    Parameters
    ----------
    noise_models : `collections.abc.Sequence`\[`NoiseModel`\] | `None`
        New noise model API input.
    p_depol_after_clifford : `float` | `None`
        Legacy depolarizing noise parameter.
    p_before_meas_flip : `float` | `None`
        Legacy measurement flip noise parameter.

    Returns
    -------
    `tuple`\[`NoiseModel`, ...\]
        Normalized noise model sequence.

    Raises
    ------
    ValueError
        If legacy parameters are invalid or mixed with ``noise_models``.
    """
    used_legacy: list[str] = []
    legacy_models: list[NoiseModel] = []

    if p_depol_after_clifford is not None:
        _validate_probability_parameter("p_depol_after_clifford", p_depol_after_clifford)
        used_legacy.append("p_depol_after_clifford")
        if p_depol_after_clifford > 0.0:
            legacy_models.append(DepolarizingNoiseModel(p1=p_depol_after_clifford, p2=p_depol_after_clifford))

    if p_before_meas_flip is not None:
        _validate_probability_parameter("p_before_meas_flip", p_before_meas_flip)
        used_legacy.append("p_before_meas_flip")
        if p_before_meas_flip > 0.0:
            legacy_models.append(MeasurementFlipNoiseModel(p=p_before_meas_flip))

    if noise_models is not None and used_legacy:
        legacy_args = ", ".join(used_legacy)
        msg = f"{legacy_args} cannot be used together with noise_models."
        raise ValueError(msg)

    if used_legacy:
        warn(
            "p_depol_after_clifford and p_before_meas_flip are deprecated in 0.3.0 and will be removed in 0.4.0. "
            "Use noise_models with DepolarizingNoiseModel and MeasurementFlipNoiseModel instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    if noise_models is not None:
        return tuple(noise_models)
    return tuple(legacy_models)


def stim_compile(  # noqa: PLR0913
    pattern: Pattern,
    logical_observables: Mapping[int, Collection[int]] | None = None,
    *,
    p_depol_after_clifford: float | None = None,
    p_before_meas_flip: float | None = None,
    emit_qubit_coords: bool = True,
    noise_models: Sequence[NoiseModel] | None = None,
    tick_duration: float = 1.0,
) -> str:
    r"""Compile a pattern to stim format.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to compile.
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Collection`\[`int`\]\], optional
        A mapping from logical observable index to a collection of node indices.
        If `None`, logical observables stored in `pattern.pauli_frame.logical_observables`
        are used. By default `None`.
    p_depol_after_clifford : `float` | `None`, optional
        Legacy depolarizing probability after Clifford gates. Deprecated in 0.3.0.
        Use ``noise_models=[DepolarizingNoiseModel(p1=..., p2=...)]`` instead.
    p_before_meas_flip : `float` | `None`, optional
        Legacy measurement bit-flip probability. Deprecated in 0.3.0.
        Use ``noise_models=[MeasurementFlipNoiseModel(p=...)]`` instead.
    emit_qubit_coords : `bool`, optional
        Whether to emit QUBIT_COORDS instructions for nodes with coordinates,
        by default True.
    noise_models : `collections.abc.Sequence`\[`NoiseModel`\] | `None`, optional
        Custom noise models for injecting Stim noise instructions, by default None.
        Use `DepolarizingNoiseModel` for gate noise and `MeasurementFlipNoiseModel`
        for measurement errors.
    tick_duration : `float`, optional
        Duration associated with each TICK for idle noise, by default 1.0.

    Returns
    -------
    `str`
        The compiled stim string.

    Notes
    -----
    Deprecated parameters ``p_depol_after_clifford`` and ``p_before_meas_flip`` emit
    a `DeprecationWarning` and will be removed in 0.4.0.
    Legacy parameters cannot be mixed with ``noise_models``.
    Stim only supports Clifford gates, therefore this compiler only supports
    Pauli measurements (X, Y, Z basis) which correspond to Clifford operations.
    Non-Pauli measurements will raise a ValueError.
    Patterns containing X or Z correction commands will raise a NotImplementedError.

    Examples
    --------
    Basic compilation without noise:

    >>> # stim_str = stim_compile(pattern)

    With depolarizing and measurement flip noise:

    >>> from graphqomb.noise_model import DepolarizingNoiseModel, MeasurementFlipNoiseModel
    >>> # stim_str = stim_compile(
    >>> #     pattern,
    >>> #     noise_models=[
    >>> #         DepolarizingNoiseModel(p1=0.001, p2=0.01),
    >>> #         MeasurementFlipNoiseModel(p=0.001)
    >>> #     ]
    >>> # )
    """
    normalized_noise_models = _normalize_noise_models(
        noise_models=noise_models,
        p_depol_after_clifford=p_depol_after_clifford,
        p_before_meas_flip=p_before_meas_flip,
    )
    compiler = _StimCompiler(
        pattern,
        emit_qubit_coords=emit_qubit_coords,
        noise_models=normalized_noise_models,
        tick_duration=tick_duration,
    )
    return compiler.compile(logical_observables)
