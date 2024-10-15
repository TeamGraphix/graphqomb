"""Quantum Compiler(qompiler) module for Measurement-Based Quantum Computation (MBQC).

note: `compile` is used in Python built-in functions, so we use `qompile` instead.

This module provides:
- generate_m_cmd: Generate a measurement command.
- generate_corrections: Generate correction from flowlike object.
- qompile_from_flow: Compile graph state into pattern with gflow.
- qompile_from_xz_flow: Compile graph state into pattern with x/z correction flows.
- qompile: Compile graph state into pattern with correctionmaps and directed acyclic graph.
- qompile_from_subgraphs: Compile graph state into pattern with subgraph sequence and gflow.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING

from graphix_zx.command import C, E, M, N, X, Z
from graphix_zx.common import Plane
from graphix_zx.flow import check_causality, oddneighbors
from graphix_zx.focus_flow import topological_sort_kahn
from graphix_zx.pattern import MutablePattern

if TYPE_CHECKING:
    from graphix_zx.flow import FlowLike
    from graphix_zx.graphstate import BaseGraphState
    from graphix_zx.pattern import ImmutablePattern

CorrectionSet = AbstractSet[int]
CorrectionMap = Mapping[int, CorrectionSet]


# extended MBQC
def generate_m_cmd(
    node: int,
    meas_plane: Plane,
    meas_angle: float,
    x_correction: CorrectionSet,
    z_correction: CorrectionSet,
) -> M:
    """Generate a measurement command.

    Parameters
    ----------
    node : int
        node to be measured
    meas_plane : Plane
        measurement plane
    meas_angle : float
        measurement angle
    x_correction : Correction
        x correction applied to the node
    z_correction : Correction
        z correction applied to the node

    Raises
    ------
    ValueError
        invalid measurement plane

    Returns
    -------
    M
        measurement command
    """
    if meas_plane == Plane.XY:
        s_domain = x_correction
        t_domain = z_correction
    elif meas_plane == Plane.ZX:
        s_domain = x_correction | z_correction
        t_domain = x_correction
    elif meas_plane == Plane.YZ:
        s_domain = z_correction
        t_domain = x_correction
    else:  # NOTE: possible to include Pauli simplification.
        msg = "Invalid measurement plane"
        raise ValueError(msg)
    return M(
        node=node,
        plane=meas_plane,
        angle=meas_angle,
        s_domain=set(s_domain),
        t_domain=set(t_domain),
    )


def generate_corrections(graph: BaseGraphState, flowlike: FlowLike) -> CorrectionMap:
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
    x_corrections = generate_corrections(graph, x_flow)
    z_corrections = generate_corrections(graph, z_flow)

    dag = {node: (x_flow[node] | z_flow[node]) - {node} for node in x_flow}
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
    dag : dict[int, set[int]]
        directed acyclic graph representation of the causality of flow
    correct_output : bool, optional
        whether to correct outputs or not, by default True

    Returns
    -------
    MutablePattern
        mutable pattern
    """
    input_nodes = graph.input_nodes
    output_nodes = graph.output_nodes
    meas_planes = graph.meas_planes
    meas_angles = graph.meas_angles
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
        generate_m_cmd(
            node,
            meas_planes[node],
            meas_angles[node],
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


def qompile_from_subgraphs(
    graph: BaseGraphState,
    subgraphs: Iterable[BaseGraphState],
    gflow: FlowLike,
) -> ImmutablePattern:
    """Compile graph state into pattern with subgraph sequence and gflow.

    Parameters
    ----------
    graph : BaseGraphState
        graph state
    subgraphs : Iterable[BaseGraphState]
        sequence of graph structures at each step of the computation.
    gflow : FlowLike
        gflow

    Returns
    -------
    ImmutablePattern
        immutable pattern

    Raises
    ------
    ValueError
        if the flow is invalid
    """
    if not check_causality(graph, gflow):
        msg = "Invalid flow"
        raise ValueError(msg)
    # stabilizer check?

    pattern = MutablePattern()

    xflow = gflow
    zflow = {node: oddneighbors(gflow[node], graph) for node in gflow}
    x_corrections = generate_corrections(graph, xflow)
    z_corrections = generate_corrections(graph, zflow)

    for subgraph in subgraphs:
        sub_nodes = subgraph.physical_nodes

        sub_x_corrections = {node: x_corrections[node] for node in subgraph.physical_nodes}
        sub_z_corrections = {node: z_corrections[node] for node in subgraph.physical_nodes}

        sub_dag = {
            node: ((xflow[node] | zflow[node]) - {node}) & sub_nodes for node in sub_nodes - subgraph.output_nodes
        }
        for node in subgraph.output_nodes:
            sub_dag[node] = set()

        sub_pattern = qompile(
            subgraph,
            sub_x_corrections,
            sub_z_corrections,
            sub_dag,
            correct_output=False,
        )

        pattern = pattern.append_pattern(sub_pattern)

    for output_node in graph.output_nodes:
        pattern.add(X(node=output_node, domain=set(x_corrections[output_node])))
        pattern.add(Z(node=output_node, domain=set(z_corrections[output_node])))

    pattern.mark_runnable()
    pattern.mark_deterministic()

    return pattern.freeze()
