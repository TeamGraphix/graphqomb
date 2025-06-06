"""Quantum Compiler(qompiler) module for Measurement-Based Quantum Computation (MBQC).

note: `compile` is used in Python built-in functions, so we use `qompile` instead.

This module provides:

- `qompile_from_flow`: Compile graph state into pattern with gflow.
- `qompile_from_xz_flow`: Compile graph state into pattern with x/z correction flows.
- `qompile`: Compile graph state into pattern with correctionmaps and directed acyclic graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING

from graphix_zx.command import Clifford, E, M, N, X, Z
from graphix_zx.common import MeasBasis, Plane
from graphix_zx.feedforward import check_causality
from graphix_zx.pattern import Pattern

if TYPE_CHECKING:
    from graphix_zx.graphstate import BaseGraphState


# extended MBQC
def _generate_m_cmd(
    node: int,
    meas_basis: MeasBasis,
    x_correction: AbstractSet[int],
    z_correction: AbstractSet[int],
) -> M:
    r"""Generate a measurement command.

    Parameters
    ----------
    node : `int`
        node to be measured
    meas_basis : `MeasBasis`
        measurement basis
    x_correction : `collections.abc.AbstractSet`\[`int`\]
        x correction applied to the node
    z_correction : `collections.abc.AbstractSet`\[`int`\]
        z correction applied to the node

    Raises
    ------
    ValueError
        invalid measurement plane

    Returns
    -------
    `M`
        measurement command
    """
    if meas_basis.plane == Plane.XY:
        s_domain = x_correction
        t_domain = z_correction
    elif meas_basis.plane == Plane.XZ:
        s_domain = x_correction | z_correction
        t_domain = x_correction
    elif meas_basis.plane == Plane.YZ:
        s_domain = z_correction
        t_domain = x_correction
    else:  # NOTE: possible to include Pauli simplification.
        msg = "Invalid measurement plane"
        raise ValueError(msg)
    return M(
        node=node,
        meas_basis=meas_basis,
        s_domain=set(s_domain),
        t_domain=set(t_domain),
    )


def _generate_corrections(graph: BaseGraphState, flowlike: FlowLike) -> CorrectionMap:
    """Generate correction from flowlike object.

    Parameters
    ----------
    graph : BaseGraphState
        graph state
    flowlike : FlowLike
        flowlike object

    Returns
    -------
    CorrectionMap
        correction mapping
    """
    corrections: dict[int, set[int]] = {node: set() for node in graph.physical_nodes}

    for node in flowlike:
        for correction in flowlike[node]:
            corrections[correction] |= {node}

    # remove self-corrections
    for node in corrections:
        corrections[node] -= {node}

    return corrections


def qompile_from_flow(graph: BaseGraphState, gflow: FlowLike, *, correct_output: bool = True) -> ImmutablePattern:
    """Compile graph state into pattern with gflow.

    Parameters
    ----------
    graph : BaseGraphState
        graph state
    gflow : FlowLike
        gflow
    correct_output : bool, optional
        whether to correct outputs or not. Defaults to True.

    Returns
    -------
    ImmutablePattern
        immutable pattern

    Raises
    ------
    ValueError
        if the flow is invalid
    """
    # TODO: check the validity of the flows
    if not check_causality(graph, gflow):
        msg = "Invalid flow"
        raise ValueError(msg)

    # generate corrections
    x_flow = gflow
    z_flow = {node: oddneighbors(gflow[node], graph) for node in gflow}
    return qompile_from_xz_flow(graph, x_flow, z_flow, correct_output=correct_output)


def qompile_from_xz_flow(
    graph: BaseGraphState,
    x_flow: FlowLike,
    z_flow: FlowLike,
    *,
    correct_output: bool = True,
) -> ImmutablePattern:
    """Compile graph state into pattern with x/z correction flows.

    Parameters
    ----------
    graph : BaseGraphState
        graph state
    x_flow : FlowLike
        x correction flow
    z_flow : FlowLike
        z correction flow
    correct_output : bool, optional
        whether to correct outputs or not, by default True

    Returns
    -------
    ImmutablePattern
        immutable pattern
    """
    x_corrections = _generate_corrections(graph, x_flow)
    z_corrections = _generate_corrections(graph, z_flow)

    dag = {node: (x_flow.get(node, set()) | z_flow.get(node, set())) - {node} for node in x_flow}
    for node in graph.output_nodes:
        dag[node] = set()

    pattern = qompile(graph, x_corrections, z_corrections, dag, correct_output=correct_output)
    pattern.mark_runnable()
    pattern.mark_deterministic()
    return pattern.freeze()


def qompile(
    graph: BaseGraphState,
    x_corrections: CorrectionMap,
    z_corrections: CorrectionMap,
    dag: Mapping[int, AbstractSet[int]],
    *,
    correct_output: bool = True,
) -> MutablePattern:
    """Compile graph state into pattern with correctionmaps and directed acyclic graph.

    Parameters
    ----------
    graph : BaseGraphState
        graph state
    x_corrections : CorrectionMap
        x corrections
    z_corrections : CorrectionMap
        z corrections
    dag : Mappinh[int, AbstratcSet[int]]
        directed acyclic graph representation of the causality of flow
    correct_output : bool, optional
        whether to correct outputs or not, by default True

    Returns
    -------
    MutablePattern
        mutable pattern
    """
    # TODO: check the validity of graph(appropriate input, output, meas_bases, etc.)

    input_nodes = graph.input_nodes
    output_nodes = graph.output_nodes
    meas_bases = graph.meas_bases
    q_indices = graph.q_indices
    local_cliffords = graph.local_cliffords

    input_q_indices = {node: q_indices[node] for node in input_nodes}

    internal_nodes = graph.physical_nodes - input_nodes - output_nodes

    topo_order = topological_sort_kahn(dag)

    pattern = MutablePattern(input_nodes=input_nodes, q_indices=input_q_indices)
    pattern.extend(N(node=node, q_index=q_indices[node]) for node in internal_nodes)
    pattern.extend(N(node=node, q_index=q_indices[node]) for node in output_nodes - input_nodes)
    pattern.extend(E(nodes=edge) for edge in graph.physical_edges)
    # TODO: local clifford on input nodes if we want to have arbitrary input states
    pattern.extend(
        _generate_m_cmd(
            node,
            meas_bases[node],
            x_corrections[node],
            z_corrections[node],
        )
        for node in topo_order
        if node not in output_nodes
    )
    if correct_output:
        pattern.extend(X(node=node, domain=set(x_corrections[node])) for node in output_nodes)
        pattern.extend(Z(node=node, domain=set(z_corrections[node])) for node in output_nodes)
    pattern.extend(
        C(node=node, local_clifford=local_cliffords[node])
        for node in output_nodes
        if local_cliffords.get(node, None) is not None
    )

    return pattern
