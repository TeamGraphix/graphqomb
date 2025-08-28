"""Pattern to stim compiler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.command import E, M, N
from graphix_zx.common import Axis

if TYPE_CHECKING:
    from collections.abc import Mapping

    from graphix_zx.pattern import Pattern


def stim_compile(  # noqa: C901, PLR0912
    pattern: Pattern,
    logical_observables: Mapping[int, Mapping[int, Axis]] | None = None,
    *,
    after_clifford_depolarization: float = 0.0,
    before_measure_flip_probability: float = 0.0,
) -> str:
    r"""Compile a pattern to stim format.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to compile.
    logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Mapping`\[`int`, `Axis`\]\]`, optional
        A mapping from logical observable index to a mapping of q_index to measurement axis, by default None.
    after_clifford_depolarization : `float`, optional
        The probability of depolarization after a Clifford gate, by default 0.0.
    before_measure_flip_probability : `float`, optional
        The probability of flipping a measurement result before measurement, by default 0.0.

    Returns
    -------
    `str`
        The compiled stim string.
    """
    stim_str = ""
    meas_order: list[int] = []
    pframe = pattern.pauli_frame
    for input_node in pattern.input_node_indices:
        stim_str += f"RX {input_node}\n"
        if after_clifford_depolarization > 0.0:
            stim_str += f"DEPOLARIZE1({after_clifford_depolarization}) {input_node}\n"
    for cmd in pattern:
        if isinstance(cmd, N):
            # prepare node in |+> state
            stim_str += f"RX {cmd.node}\n"
            if after_clifford_depolarization > 0.0:
                stim_str += f"DEPOLARIZE1({after_clifford_depolarization}) {cmd.node}\n"
        if isinstance(cmd, E):
            q1, q2 = cmd.nodes
            stim_str += f"CZ {q1} {q2}\n"
            if after_clifford_depolarization > 0.0:
                stim_str += f"DEPOLARIZE2({after_clifford_depolarization}) {q1} {q2}\n"
        if isinstance(cmd, M):
            # need X/Z switch
            if before_measure_flip_probability > 0.0:
                stim_str += f"Z_ERROR({before_measure_flip_probability}) {cmd.node}\n"
            stim_str += f"MX {cmd.node}\n"
            meas_order.append(cmd.node)

    # measure output qubits
    for output_node, q_index in pattern.output_node_indices.items():
        axis: Axis | None = None
        if logical_observables is not None:
            for obs_map in logical_observables.values():
                if q_index in obs_map:
                    axis = obs_map[q_index]
                    break

        if axis is None or axis == Axis.X:
            if before_measure_flip_probability > 0.0:
                stim_str += f"Z_ERROR({before_measure_flip_probability}) {output_node}\n"
            stim_str += f"MX {output_node}\n"
            meas_order.append(output_node)
        elif axis == Axis.Y:
            if before_measure_flip_probability > 0.0:
                stim_str += f"X_ERROR({before_measure_flip_probability}) {output_node}\n"
                stim_str += f"Z_ERROR({before_measure_flip_probability}) {output_node}\n"
            stim_str += f"MY {output_node}\n"
            meas_order.append(output_node)
        elif axis == Axis.Z:
            if before_measure_flip_probability > 0.0:
                stim_str += f"X_ERROR({before_measure_flip_probability}) {output_node}\n"
            stim_str += f"MZ {output_node}\n"
            meas_order.append(output_node)

    x_check_groups, z_check_groups = pframe.detector_groups()
    for x_checks in x_check_groups:
        target_str = ""
        for x_check in x_checks:
            target_str += f"rec[{meas_order.index(x_check) - len(meas_order)}] "
        stim_str += f"DETECTOR {target_str.strip()}\n"
    for z_checks in z_check_groups:
        target_str = ""
        for z_check in z_checks:
            target_str += f"rec[{meas_order.index(z_check) - len(meas_order)}] "
        stim_str += f"DETECTOR {target_str.strip()}\n"

    # logical observables
    if logical_observables is not None:
        qindex_to_output = {q: i for i, q in pattern.output_node_indices.items()}
        for log_idx, obs in logical_observables.items():
            target_nodes = {qindex_to_output[q_index] for q_index in obs}
            logical_observables_group = pframe.logical_observables_group(target_nodes)
            target_str = ""
            for node in logical_observables_group:
                target_str += f"rec[{meas_order.index(node) - len(meas_order)}] "
            stim_str += f"OBSERVABLE_INCLUDE({log_idx}) {target_str.strip()}\n"

    return stim_str.strip()
