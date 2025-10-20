"""Pattern to stim compiler.

This module provides:

- `stim_compile`: Function to compile a pattern into stim format.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import typing_extensions

from graphqomb.command import E, M, N
from graphqomb.common import Axis, MeasBasis, determine_pauli_axis

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from graphqomb.pattern import Pattern
    from graphqomb.pauli_frame import PauliFrame


def _write_input_nodes(
    stim_io: StringIO,
    input_node_indices: Mapping[int, int],
    after_clifford_depolarization: float,
) -> None:
    """Write input node initialization to stim format.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    input_node_indices : `collections.abc.Mapping`[`int`, `int`]
        The input node indices mapping to initialize.
    after_clifford_depolarization : `float`
        The probability of depolarization after Clifford gates.
    """
    for input_node in input_node_indices:
        stim_io.write(f"RX {input_node}\n")
        if after_clifford_depolarization > 0.0:
            stim_io.write(f"DEPOLARIZE1({after_clifford_depolarization}) {input_node}\n")


def _write_node_preparation(
    stim_io: StringIO,
    node: int,
    after_clifford_depolarization: float,
) -> None:
    """Write node preparation (N command) to stim format.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    node : `int`
        The node to prepare in |+> state.
    after_clifford_depolarization : `float`
        The probability of depolarization after Clifford gates.
    """
    stim_io.write(f"RX {node}\n")
    if after_clifford_depolarization > 0.0:
        stim_io.write(f"DEPOLARIZE1({after_clifford_depolarization}) {node}\n")


def _write_entanglement(
    stim_io: StringIO,
    nodes: tuple[int, int],
    after_clifford_depolarization: float,
) -> None:
    """Write entanglement operation (E command) to stim format.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    nodes : `tuple`[`int`, `int`]
        The pair of nodes to entangle.
    after_clifford_depolarization : `float`
        The probability of depolarization after Clifford gates.
    """
    q1, q2 = nodes
    stim_io.write(f"CZ {q1} {q2}\n")
    if after_clifford_depolarization > 0.0:
        stim_io.write(f"DEPOLARIZE2({after_clifford_depolarization}) {q1} {q2}\n")


def _write_measurement(
    stim_io: StringIO,
    meas_basis: MeasBasis,
    node: int,
    before_measure_flip_probability: float,
    meas_order: list[int],
) -> None:
    """Write measurement operation (M command) to stim format.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    meas_basis : `MeasBasis`
        The measurement basis.
    node : `int`
        The node to measure.
    before_measure_flip_probability : `float`
        The probability of flipping a measurement result before measurement.
    meas_order : `list`[`int`]
        The list tracking measurement order (modified in-place).

    Raises
    ------
    ValueError
        If an unsupported measurement basis is encountered.
    """
    axis = determine_pauli_axis(meas_basis)
    if axis is None:
        msg = f"Unsupported measurement basis: {meas_basis.plane, meas_basis.angle}"
        raise ValueError(msg)

    if axis == Axis.X:
        if before_measure_flip_probability > 0.0:
            stim_io.write(f"Z_ERROR({before_measure_flip_probability}) {node}\n")
        stim_io.write(f"MX {node}\n")
        meas_order.append(node)
    elif axis == Axis.Y:
        if before_measure_flip_probability > 0.0:
            stim_io.write(f"X_ERROR({before_measure_flip_probability}) {node}\n")
            stim_io.write(f"Z_ERROR({before_measure_flip_probability}) {node}\n")
        stim_io.write(f"MY {node}\n")
        meas_order.append(node)
    elif axis == Axis.Z:
        if before_measure_flip_probability > 0.0:
            stim_io.write(f"X_ERROR({before_measure_flip_probability}) {node}\n")
        stim_io.write(f"MZ {node}\n")
        meas_order.append(node)
    else:
        typing_extensions.assert_never(axis)


def _write_detectors(
    stim_io: StringIO,
    check_groups: Sequence[Collection[int]],
    meas_order: list[int],
) -> None:
    """Write detector declarations to stim format.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    check_groups : `collections.abc.Sequence`[`collections.abc.Collection`[`int`]]
        The parity check groups for detectors.
    meas_order : `list`[`int`]
        The measurement order list.
    """
    for checks in check_groups:
        targets = [f"rec[{meas_order.index(check) - len(meas_order)}]" for check in checks]
        stim_io.write(f"DETECTOR {' '.join(targets)}\n")


def _write_observables(
    stim_io: StringIO,
    logical_observables: Mapping[int, Collection[int]],
    pframe: PauliFrame,
    meas_order: list[int],
) -> None:
    """Write logical observable declarations to stim format.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    logical_observables : `collections.abc.Mapping`[`int`, `collections.abc.Collection`[`int`]]
        A mapping from logical observable index to a collection of node indices.
    pframe : `PauliFrame`
        The Pauli frame object.
    meas_order : `list`[`int`]
        The measurement order list.
    """
    for log_idx, obs in logical_observables.items():
        logical_observables_group = pframe.logical_observables_group(obs)
        targets = [f"rec[{meas_order.index(node) - len(meas_order)}]" for node in logical_observables_group]
        stim_io.write(f"OBSERVABLE_INCLUDE({log_idx}) {' '.join(targets)}\n")


def stim_compile(
    pattern: Pattern,
    logical_observables: Mapping[int, Collection[int]] | None = None,
    *,
    after_clifford_depolarization: float = 0.0,
    before_measure_flip_probability: float = 0.0,
) -> str:
    r"""Compile a pattern to stim format.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to compile.
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Collection`\[`int`\]\], optional
        A mapping from logical observable index to a collection of node indices, by default None.
    after_clifford_depolarization : `float`, optional
        The probability of depolarization after a Clifford gate, by default 0.0.
    before_measure_flip_probability : `float`, optional
        The probability of flipping a measurement result before measurement, by default 0.0.

    Returns
    -------
    `str`
        The compiled stim string.

    Notes
    -----
    Stim only supports Clifford gates, therefore this compiler only supports
    Pauli measurements (X, Y, Z basis) which correspond to Clifford operations.
    Non-Pauli measurements will raise a ValueError.

    Raises
    ------
    ValueError
        If an unsupported measurement basis is encountered.
    """
    stim_io = StringIO()
    meas_order: list[int] = []
    pframe = pattern.pauli_frame

    # Initialize input nodes
    _write_input_nodes(stim_io, pattern.input_node_indices, after_clifford_depolarization)

    # Process pattern commands
    for cmd in pattern:
        if isinstance(cmd, N):
            _write_node_preparation(stim_io, cmd.node, after_clifford_depolarization)
        elif isinstance(cmd, E):
            _write_entanglement(stim_io, cmd.nodes, after_clifford_depolarization)
        elif isinstance(cmd, M):
            _write_measurement(stim_io, cmd.meas_basis, cmd.node, before_measure_flip_probability, meas_order)

    # Write detectors
    check_groups = pframe.detector_groups()
    _write_detectors(stim_io, check_groups, meas_order)

    # Write logical observables
    if logical_observables is not None:
        _write_observables(stim_io, logical_observables, pframe, meas_order)

    return stim_io.getvalue().strip()
