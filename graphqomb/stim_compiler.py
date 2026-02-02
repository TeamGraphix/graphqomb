"""Pattern to stim compiler.

This module provides:

- `stim_compile`: Function to compile a pattern into stim format.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import typing_extensions

from graphqomb.command import TICK, E, M, N
from graphqomb.common import Axis, MeasBasis, determine_pauli_axis
from graphqomb.noise_model import NoiseEvent, NoiseKind, NoiseModel

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

    from graphqomb.pattern import Pattern
    from graphqomb.pauli_frame import PauliFrame


def _emit_qubit_coords(
    stim_io: StringIO,
    node: int,
    coordinate: tuple[float, ...] | None,
) -> None:
    r"""Emit QUBIT_COORDS instruction if coordinate is available.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    node : `int`
        The qubit index.
    coordinate : `tuple`\[`float`, ...\] | `None`
        The coordinate tuple (2D or 3D), or None if no coordinate.
    """
    if coordinate is not None:
        coords_str = ", ".join(str(c) for c in coordinate)
        stim_io.write(f"QUBIT_COORDS({coords_str}) {node}\n")


def _prepare_nodes(
    stim_io: StringIO,
    nodes: int | Iterable[int],
    p_depol_after_clifford: float,
    coordinates: Mapping[int, tuple[float, ...]] | None = None,
    emit_qubit_coords: bool = True,
) -> None:
    r"""Prepare nodes in |+> state.

    This function handles both single nodes (N command) and multiple nodes
    (input nodes initialization).

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    nodes : `int` | `collections.abc.Iterable`\[`int`\]
        A single node index or an iterable of node indices to prepare.
    p_depol_after_clifford : `float`
        The probability of depolarization after Clifford gates.
    coordinates : `collections.abc.Mapping`\[`int`, `tuple`\[`float`, ...\]\] | `None`, optional
        Coordinates for nodes, by default None.
    emit_qubit_coords : `bool`, optional
        Whether to emit QUBIT_COORDS instructions, by default True.
    """
    if isinstance(nodes, int):
        nodes = [nodes]
    for node in nodes:
        coord = coordinates.get(node) if coordinates else None
        if emit_qubit_coords:
            _emit_qubit_coords(stim_io, node, coord)
        stim_io.write(f"RX {node}\n")
        if p_depol_after_clifford > 0.0:
            stim_io.write(f"DEPOLARIZE1({p_depol_after_clifford}) {node}\n")


def _entangle_nodes(
    stim_io: StringIO,
    nodes: tuple[int, int],
    p_depol_after_clifford: float,
) -> None:
    r"""Entangle two nodes with CZ gate (E command).

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    nodes : `tuple`\[`int`, `int`\]
        The pair of nodes to entangle.
    p_depol_after_clifford : `float`
        The probability of depolarization after Clifford gates.
    """
    q1, q2 = nodes
    stim_io.write(f"CZ {q1} {q2}\n")
    if p_depol_after_clifford > 0.0:
        stim_io.write(f"DEPOLARIZE2({p_depol_after_clifford}) {q1} {q2}\n")


def _emit_measurement_noise(
    stim_io: StringIO,
    axis: Axis,
    node: int,
    p_before_meas_flip: float,
) -> None:
    r"""Emit measurement noise before a measurement operation."""
    if axis == Axis.X:
        if p_before_meas_flip > 0.0:
            stim_io.write(f"Z_ERROR({p_before_meas_flip}) {node}\n")
    elif axis == Axis.Y:
        if p_before_meas_flip > 0.0:
            stim_io.write(f"X_ERROR({p_before_meas_flip}) {node}\n")
            stim_io.write(f"Z_ERROR({p_before_meas_flip}) {node}\n")
    elif axis == Axis.Z:
        if p_before_meas_flip > 0.0:
            stim_io.write(f"X_ERROR({p_before_meas_flip}) {node}\n")
    else:
        typing_extensions.assert_never(axis)


def _emit_measurement(
    stim_io: StringIO,
    axis: Axis,
    node: int,
) -> None:
    r"""Emit a measurement operation."""
    if axis == Axis.X:
        stim_io.write(f"MX {node}\n")
    elif axis == Axis.Y:
        stim_io.write(f"MY {node}\n")
    elif axis == Axis.Z:
        stim_io.write(f"MZ {node}\n")
    else:
        typing_extensions.assert_never(axis)


def _emit_noise(
    stim_io: StringIO,
    noise_model: NoiseModel | None,
    event: NoiseEvent,
) -> int:
    if noise_model is None:
        return 0
    record_delta = 0
    for op in noise_model.emit(event):
        if op.text:
            stim_io.write(f"{op.text}\n")
        record_delta += op.record_delta
    return record_delta


_MIN_COORD_DIMS = 2


def _xy_coords(coordinate: tuple[float, ...] | None) -> tuple[float, float] | None:
    if coordinate is None or len(coordinate) < _MIN_COORD_DIMS:
        return None
    return (coordinate[0], coordinate[1])


def _add_detectors(
    stim_io: StringIO,
    check_groups: Sequence[Collection[int]],
    meas_order: Mapping[int, int],
    total_measurements: int,
) -> None:
    r"""Add detector declarations to the circuit.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    check_groups : `collections.abc.Sequence`\[`collections.abc.Collection`\[`int`\]\]
        The parity check groups for detectors.
    meas_order : `collections.abc.Mapping`\[`int`, `int`\]
        The measurement order lookup dict mapping node to measurement index.
    total_measurements : `int`
        The total number of measurements.
    """
    for checks in check_groups:
        targets = [f"rec[{meas_order[check] - total_measurements}]" for check in checks]
        stim_io.write(f"DETECTOR {' '.join(targets)}\n")


def _add_observables(
    stim_io: StringIO,
    logical_observables: Mapping[int, Collection[int]],
    pframe: PauliFrame,
    meas_order: Mapping[int, int],
    total_measurements: int,
) -> None:
    r"""Add logical observable declarations to the circuit.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Collection`\[`int`\]\]
        A mapping from logical observable index to a collection of node indices.
    pframe : `PauliFrame`
        The Pauli frame object.
    meas_order : `collections.abc.Mapping`\[`int`, `int`\]
        The measurement order lookup dict mapping node to measurement index.
    total_measurements : `int`
        The total number of measurements.
    """
    for log_idx, obs in logical_observables.items():
        logical_observables_group = pframe.logical_observables_group(obs)
        targets = [f"rec[{meas_order[node] - total_measurements}]" for node in logical_observables_group]
        stim_io.write(f"OBSERVABLE_INCLUDE({log_idx}) {' '.join(targets)}\n")


class _StimCompiler:
    def __init__(  # noqa: PLR0913
        self,
        pattern: Pattern,
        *,
        p_depol_after_clifford: float,
        p_before_meas_flip: float,
        emit_qubit_coords: bool,
        noise_model: NoiseModel | None,
        tick_duration: float,
    ) -> None:
        self._pattern = pattern
        self._pframe = pattern.pauli_frame
        self._coord_lookup = pattern.coordinates
        self._p_depol_after_clifford = p_depol_after_clifford
        self._p_before_meas_flip = p_before_meas_flip
        self._emit_qubit_coords = emit_qubit_coords
        self._noise_model = noise_model
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
        total_measurements = self._rec_index
        _add_detectors(self._stim_io, self._pframe.detector_groups(), self._meas_order, total_measurements)
        if logical_observables is not None:
            _add_observables(
                self._stim_io,
                logical_observables,
                self._pframe,
                self._meas_order,
                total_measurements,
            )
        return self._stim_io.getvalue().strip()

    def _emit_input_nodes(self) -> None:
        coordinates = self._pattern.input_coordinates if self._emit_qubit_coords else None
        for node in self._pattern.input_node_indices:
            _prepare_nodes(
                self._stim_io,
                node,
                self._p_depol_after_clifford,
                coordinates=coordinates,
                emit_qubit_coords=self._emit_qubit_coords,
            )
            self._after_prepare(node, is_input=True)

    def _process_commands(self) -> None:
        for cmd in self._pattern:
            if isinstance(cmd, N):
                self._handle_prepare(cmd.node, cmd.coordinate)
            elif isinstance(cmd, E):
                self._handle_entangle(cmd.nodes)
            elif isinstance(cmd, M):
                self._handle_measure(cmd.node, cmd.meas_basis)
            elif isinstance(cmd, TICK):
                self._handle_tick()

    def _handle_prepare(self, node: int, coordinate: tuple[float, ...] | None) -> None:
        coordinates = {node: coordinate} if self._emit_qubit_coords and coordinate is not None else None
        _prepare_nodes(
            self._stim_io,
            node,
            self._p_depol_after_clifford,
            coordinates=coordinates,
            emit_qubit_coords=self._emit_qubit_coords,
        )
        self._after_prepare(node, is_input=False)

    def _after_prepare(self, node: int, *, is_input: bool) -> None:
        self._alive_nodes.add(node)
        self._touched_nodes.add(node)
        self._apply_noise(
            NoiseEvent(
                kind=NoiseKind.PREPARE,
                tick=self._tick,
                nodes=(node,),
                edge=None,
                coords=(self._coords_for(node),),
                axis=None,
                is_input=is_input,
            ),
        )

    def _handle_entangle(self, nodes: tuple[int, int]) -> None:
        _entangle_nodes(self._stim_io, nodes, self._p_depol_after_clifford)
        self._touched_nodes.update(nodes)
        n0, n1 = nodes
        edge: tuple[int, int] = (n0, n1) if n0 < n1 else (n1, n0)
        self._apply_noise(
            NoiseEvent(
                kind=NoiseKind.ENTANGLE,
                tick=self._tick,
                nodes=nodes,
                edge=edge,
                coords=(self._coords_for(nodes[0]), self._coords_for(nodes[1])),
                axis=None,
            ),
        )

    def _handle_measure(self, node: int, meas_basis: MeasBasis) -> None:
        axis = determine_pauli_axis(meas_basis)
        if axis is None:
            msg = f"Unsupported measurement basis: {meas_basis.plane, meas_basis.angle}"
            raise ValueError(msg)
        _emit_measurement_noise(self._stim_io, axis, node, self._p_before_meas_flip)
        self._apply_noise(
            NoiseEvent(
                kind=NoiseKind.MEASURE,
                tick=self._tick,
                nodes=(node,),
                edge=None,
                coords=(self._coords_for(node),),
                axis=axis,
            ),
        )
        _emit_measurement(self._stim_io, axis, node)
        self._meas_order[node] = self._rec_index
        self._rec_index += 1
        self._alive_nodes.discard(node)
        self._touched_nodes.add(node)

    def _handle_tick(self) -> None:
        if self._noise_model is not None:
            idle_nodes = sorted(self._alive_nodes - self._touched_nodes)
            if idle_nodes:
                self._apply_noise(
                    NoiseEvent(
                        kind=NoiseKind.IDLE,
                        tick=self._tick,
                        nodes=tuple(idle_nodes),
                        edge=None,
                        coords=tuple(self._coords_for(node) for node in idle_nodes),
                        axis=None,
                        duration=self._tick_duration,
                    ),
                )
        self._stim_io.write("TICK\n")
        self._touched_nodes.clear()
        self._tick += 1

    def _apply_noise(self, event: NoiseEvent) -> None:
        self._rec_index += _emit_noise(self._stim_io, self._noise_model, event)

    def _coords_for(self, node: int) -> tuple[float, float] | None:
        return _xy_coords(self._coord_lookup.get(node))


def stim_compile(  # noqa: PLR0913
    pattern: Pattern,
    logical_observables: Mapping[int, Collection[int]] | None = None,
    *,
    p_depol_after_clifford: float = 0.0,
    p_before_meas_flip: float = 0.0,
    emit_qubit_coords: bool = True,
    noise_model: NoiseModel | None = None,
    tick_duration: float = 1.0,
) -> str:
    r"""Compile a pattern to stim format.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to compile.
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Collection`\[`int`\]\], optional
        A mapping from logical observable index to a collection of node indices, by default None.
    p_depol_after_clifford : `float`, optional
        The probability of depolarization after a Clifford gate, by default 0.0.
    p_before_meas_flip : `float`, optional
        The probability of flipping a measurement result before measurement, by default 0.0.
    emit_qubit_coords : `bool`, optional
        Whether to emit QUBIT_COORDS instructions for nodes with coordinates,
        by default True.
    noise_model : `NoiseModel` | `None`, optional
        Custom noise model for injecting Stim noise instructions, by default None.
    tick_duration : `float`, optional
        Duration associated with each TICK for idle noise, by default 1.0.

    Returns
    -------
    `str`
        The compiled stim string.

    Notes
    -----
    Stim only supports Clifford gates, therefore this compiler only supports
    Pauli measurements (X, Y, Z basis) which correspond to Clifford operations.
    Non-Pauli measurements will raise a ValueError.
    """
    compiler = _StimCompiler(
        pattern,
        p_depol_after_clifford=p_depol_after_clifford,
        p_before_meas_flip=p_before_meas_flip,
        emit_qubit_coords=emit_qubit_coords,
        noise_model=noise_model,
        tick_duration=tick_duration,
    )
    return compiler.compile(logical_observables)
