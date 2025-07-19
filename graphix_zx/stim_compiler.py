"""Pattern to stim compiler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.command import E, M, N

if TYPE_CHECKING:
    from graphix_zx.pattern import Pattern


def stim_compile(  # noqa: C901, PLR0912
    pattern: Pattern,
    *,
    after_clifford_depolarization: float = 0.0,
    before_measure_flip_probability: float = 0.0,
    logical_observables: dict[int, set[int]] | None = None,
) -> str:
    """Compile a pattern to stim format.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to compile.
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
    meas_order = []
    pframe = pattern.pauli_frame
    for cmd in pattern:
        if isinstance(cmd, N):
            # prepare node in |+> state
            stim_str += f"H {cmd.node}\n"
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

    x_check_groups, z_check_groups = pframe.detector_groups()
    for x_checks in x_check_groups:
        target_str = ""
        for x_check in x_checks:
            target_str += f"rec[{meas_order.index(x_check)}] "
        stim_str += f"DETECTOR {target_str.strip()}\n"
    for z_checks in z_check_groups:
        target_str = ""
        for z_check in z_checks:
            target_str += f"rec[{meas_order.index(z_check)}] "
        stim_str += f"DETECTOR {target_str.strip()}\n"

    # measure output qubits
    for output_node in pattern.output_node_indices:
        if before_measure_flip_probability > 0.0:
            stim_str += f"Z_ERROR({before_measure_flip_probability}) {output_node}\n"
        stim_str += f"MX {output_node}\n"
        meas_order.append(output_node)

    # logical observables
    if logical_observables is not None:
        for log_idx, obs in logical_observables.items():
            target_str = ""
            for node in obs:
                target_str += f"rec[{meas_order.index(node)}] "
            stim_str += f"OBSERVABLE_INCLUDE({log_idx}) {target_str.strip()}\n"

    return stim_str.strip()
