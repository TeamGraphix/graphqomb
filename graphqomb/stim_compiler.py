"""Pattern to stim compiler."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from graphqomb.command import E, M, N
from graphqomb.common import Axis, determine_pauli_axis

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from graphqomb.pattern import Pattern


def stim_compile(  # noqa: C901, PLR0912
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

    Raises
    ------
    ValueError
        If an unsupported measurement basis is encountered.
    """
    stim_io = StringIO()
    meas_order: list[int] = []
    pframe = pattern.pauli_frame
    for input_node in pattern.input_node_indices:
        stim_io.write(f"RX {input_node}\n")
        if after_clifford_depolarization > 0.0:
            stim_io.write(f"DEPOLARIZE1({after_clifford_depolarization}) {input_node}\n")
    for cmd in pattern:
        if isinstance(cmd, N):
            # prepare node in |+> state
            stim_io.write(f"RX {cmd.node}\n")
            if after_clifford_depolarization > 0.0:
                stim_io.write(f"DEPOLARIZE1({after_clifford_depolarization}) {cmd.node}\n")
        if isinstance(cmd, E):
            q1, q2 = cmd.nodes
            stim_io.write(f"CZ {q1} {q2}\n")
            if after_clifford_depolarization > 0.0:
                stim_io.write(f"DEPOLARIZE2({after_clifford_depolarization}) {q1} {q2}\n")
        if isinstance(cmd, M):
            axis = determine_pauli_axis(cmd.meas_basis)
            if axis is None:
                msg = f"Unsupported measurement basis: {cmd.meas_basis.plane, cmd.meas_basis.angle}"
                raise ValueError(msg)

            if axis == Axis.X:
                if before_measure_flip_probability > 0.0:
                    stim_io.write(f"Z_ERROR({before_measure_flip_probability}) {cmd.node}\n")
                stim_io.write(f"MX {cmd.node}\n")
                meas_order.append(cmd.node)
            elif axis == Axis.Y:
                if before_measure_flip_probability > 0.0:
                    stim_io.write(f"X_ERROR({before_measure_flip_probability}) {cmd.node}\n")
                    stim_io.write(f"Z_ERROR({before_measure_flip_probability}) {cmd.node}\n")
                stim_io.write(f"MY {cmd.node}\n")
                meas_order.append(cmd.node)
            elif axis == Axis.Z:
                if before_measure_flip_probability > 0.0:
                    stim_io.write(f"X_ERROR({before_measure_flip_probability}) {cmd.node}\n")
                stim_io.write(f"MZ {cmd.node}\n")
                meas_order.append(cmd.node)

    check_groups = pframe.detector_groups()
    for checks in check_groups:
        targets = [f"rec[{meas_order.index(check) - len(meas_order)}]" for check in checks]
        stim_io.write(f"DETECTOR {' '.join(targets)}\n")

    # logical observables
    if logical_observables is not None:
        for log_idx, obs in logical_observables.items():
            logical_observables_group = pframe.logical_observables_group(obs)
            targets = [f"rec[{meas_order.index(node) - len(meas_order)}]" for node in logical_observables_group]
            stim_io.write(f"OBSERVABLE_INCLUDE({log_idx}) {' '.join(targets)}\n")

    return stim_io.getvalue().strip()
