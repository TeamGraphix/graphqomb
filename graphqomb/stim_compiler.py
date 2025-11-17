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

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from graphqomb.pattern import Pattern
    from graphqomb.pauli_frame import PauliFrame


def _initialize_nodes(
    stim_io: StringIO,
    node_indices: Mapping[int, int],
    p_depol_after_clifford: float,
) -> None:
    r"""Initialize nodes in the stim circuit.

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    node_indices : `collections.abc.Mapping`\[`int`, `int`\]
        The node indices mapping to initialize.
    p_depol_after_clifford : `float`
        The probability of depolarization after Clifford gates.
    """
    for node in node_indices:
        stim_io.write(f"RX {node}\n")
        if p_depol_after_clifford > 0.0:
            stim_io.write(f"DEPOLARIZE1({p_depol_after_clifford}) {node}\n")


def _prepare_node(
    stim_io: StringIO,
    node: int,
    p_depol_after_clifford: float,
) -> None:
    r"""Prepare a node in |+> state (N command).

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    node : `int`
        The node to prepare in |+> state.
    p_depol_after_clifford : `float`
        The probability of depolarization after Clifford gates.
    """
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


def _measure_node(
    stim_io: StringIO,
    meas_basis: MeasBasis,
    node: int,
    p_before_meas_flip: float,
) -> None:
    r"""Measure a node in the specified basis (M command).

    Parameters
    ----------
    stim_io : `StringIO`
        The output stream to write to.
    meas_basis : `MeasBasis`
        The measurement basis.
    node : `int`
        The node to measure.
    p_before_meas_flip : `float`
        The probability of flipping a measurement result before measurement.

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
        if p_before_meas_flip > 0.0:
            stim_io.write(f"Z_ERROR({p_before_meas_flip}) {node}\n")
        stim_io.write(f"MX {node}\n")
    elif axis == Axis.Y:
        if p_before_meas_flip > 0.0:
            stim_io.write(f"X_ERROR({p_before_meas_flip}) {node}\n")
            stim_io.write(f"Z_ERROR({p_before_meas_flip}) {node}\n")
        stim_io.write(f"MY {node}\n")
    elif axis == Axis.Z:
        if p_before_meas_flip > 0.0:
            stim_io.write(f"X_ERROR({p_before_meas_flip}) {node}\n")
        stim_io.write(f"MZ {node}\n")
    else:
        typing_extensions.assert_never(axis)


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


def stim_compile(
    pattern: Pattern,
    logical_observables: Mapping[int, Collection[int]] | None = None,
    *,
    p_depol_after_clifford: float = 0.0,
    p_before_meas_flip: float = 0.0,
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
    stim_io = StringIO()
    pframe = pattern.pauli_frame

    # Build measurement order lookup dict
    meas_order: dict[int, int] = {}
    meas_idx = 0
    for cmd in pattern:
        if isinstance(cmd, M):
            meas_order[cmd.node] = meas_idx
            meas_idx += 1
    total_measurements = meas_idx

    # Initialize input nodes
    _initialize_nodes(stim_io, pattern.input_node_indices, p_depol_after_clifford)

    # Process pattern commands
    for cmd in pattern:
        if isinstance(cmd, N):
            _prepare_node(stim_io, cmd.node, p_depol_after_clifford)
        elif isinstance(cmd, E):
            _entangle_nodes(stim_io, cmd.nodes, p_depol_after_clifford)
        elif isinstance(cmd, M):
            _measure_node(stim_io, cmd.meas_basis, cmd.node, p_before_meas_flip)
        elif isinstance(cmd, TICK):
            stim_io.write("TICK\n")

    # Add detectors
    check_groups = pframe.detector_groups()
    _add_detectors(stim_io, check_groups, meas_order, total_measurements)

    # Add logical observables
    if logical_observables is not None:
        _add_observables(stim_io, logical_observables, pframe, meas_order, total_measurements)

    return stim_io.getvalue().strip()
