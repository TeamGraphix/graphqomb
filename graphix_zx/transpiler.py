"""generate standardized or resource optimized pattern from underlying graph and gflow. extended MC is included"""

from __future__ import annotations

from typing import Dict, Set

from graphix_zx.command import E, M, N, X, Z
from graphix_zx.common import Plane
from graphix_zx.flow import FlowLike, check_causality, oddneighbors
from graphix_zx.focus_flow import (
    topological_sort_kahn,
)
from graphix_zx.graphstate import BaseGraphState
from graphix_zx.pattern import ImmutablePattern, MutablePattern

Correction = Set[int]
CorrectionMap = Dict[int, Correction]


# extended MBQC
def generate_m_cmd(
    node: int,
    meas_plane: Plane,
    meas_angle: float,
    x_correction: Correction,
    z_correction: Correction,
) -> M:
    """Generate a measurement command.

    Args:
        node (int): node to be measured
        meas_plane (Plane): measurement plane
        meas_angle (float): measurement angle
        x_correction (Correction): x correction applied to the node
        z_correction (Correction): z correction applied to the node

    Raises
    ------
        ValueError: invalid measurement plane

    Returns
    -------
        M: measurement command
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
        raise ValueError("Invalid measurement plane")
    return M(
        node=node,
        plane=meas_plane,
        angle=meas_angle,
        s_domain=s_domain,
        t_domain=t_domain,
    )


def generate_corrections(graph: BaseGraphState, flowlike: FlowLike) -> CorrectionMap:
    """Generate correction from flowlike object

    Args:
        graph (BaseGraphState): graph state
        flowlike (FlowLike): flowlike object

    Returns
    -------
        CorrectionMap : correction dictionary
    """
    corrections: dict[int, set[int]] = {node: set() for node in graph.get_physical_nodes()}

    for node in flowlike:
        for correction in flowlike[node]:
            corrections[correction] |= {node}

    # remove self-corrections
    for node in corrections:
        corrections[node] -= {node}

    return corrections


def transpile_from_flow(graph: BaseGraphState, gflow: FlowLike, correct_output: bool = True) -> ImmutablePattern:
    """Transpile pattern from gflow object

    Args:
        graph (BaseGraphState): graph state
        gflow (FlowLike): gflow
        correct_output (bool, optional): whether to correct outputs or not. Defaults to True.

    Returns
    -------
        ImmutablePattern: immutable pattern
    """
    # TODO: check the validity of the flows
    if not check_causality(graph, gflow):
        raise ValueError("Invalid flow")
    # stabilizer check?

    # generate corrections
    x_flow = gflow
    z_flow = {node: oddneighbors(gflow[node], graph) for node in gflow}
    x_corrections = generate_corrections(graph, x_flow)
    z_corrections = generate_corrections(graph, z_flow)

    dag = {node: (x_flow[node] | z_flow[node]) - {node} for node in x_flow}
    for node in graph.output_nodes:
        dag[node] = set()

    pattern = transpile(graph, x_corrections, z_corrections, dag, correct_output)
    pattern.mark_runnable()
    pattern.mark_deterministic()
    return pattern.freeze()


def transpile(
    graph: BaseGraphState,
    x_corrections: CorrectionMap,
    z_corrections: CorrectionMap,
    dag: dict[int, set[int]],
    correct_output: bool = True,
) -> MutablePattern:
    """Transpile pattern from graph, corrections, and dag

    Args:
        graph (BaseGraphState): graph state
        x_corrections (CorrectionMap): x corrections
        z_corrections (CorrectionMap): z corrections
        dag (dict[int, set[int]]): directed acyclic graph
        correct_output (bool, optional): whether to correct outputs or not. Defaults to True.

    Returns
    -------
        MutablePattern: mutable pattern
    """
    input_nodes = graph.input_nodes
    output_nodes = graph.output_nodes
    meas_planes = graph.get_meas_planes()
    meas_angles = graph.get_meas_angles()
    q_indices = graph.get_q_indices()

    input_q_indices = {node: q_indices[node] for node in input_nodes}

    internal_nodes = set(graph.get_physical_nodes()) - set(input_nodes) - set(output_nodes)

    topo_order = topological_sort_kahn(dag)

    pattern = MutablePattern(input_nodes=input_nodes, q_indices=input_q_indices)
    pattern.extend([N(node=node, q_index=q_indices[node]) for node in internal_nodes])
    pattern.extend([N(node=node, q_index=q_indices[node]) for node in output_nodes - input_nodes])
    pattern.extend([E(nodes=edge) for edge in graph.get_physical_edges()])
    pattern.extend(
        [
            generate_m_cmd(
                node,
                meas_planes[node],
                meas_angles[node],
                x_corrections[node],
                z_corrections[node],
            )
            for node in topo_order
            if node not in output_nodes
        ]
    )
    if correct_output:
        pattern.extend([X(node=node, domain=x_corrections[node]) for node in output_nodes])
        pattern.extend([Z(node=node, domain=z_corrections[node]) for node in output_nodes])
    # TODO: add Clifford commands on the output nodes

    return pattern


def transpile_from_subgraphs(
    graph: BaseGraphState,
    subgraphs: list[BaseGraphState],
    gflow: FlowLike,
) -> ImmutablePattern:
    """Generate a pattern from subgraph sequence

    Args:
        graph (BaseGraphState): graph state
        subgraphs (list[BaseGraphState]): subgraph sequence
        gflow (FlowLike): gflow

    Returns
    -------
        ImmutablePattern: immutable pattern
    """
    if not check_causality(graph, gflow):
        raise ValueError("Invalid flow")
    # stabilizer check?

    pattern = MutablePattern()

    xflow = gflow
    zflow = {node: oddneighbors(gflow[node], graph) for node in gflow}
    x_corrections = generate_corrections(graph, xflow)
    z_corrections = generate_corrections(graph, zflow)

    for subgraph in subgraphs:
        sub_nodes = subgraph.get_physical_nodes()

        sub_x_corrections = {node: x_corrections[node] for node in subgraph.get_physical_nodes()}
        sub_z_corrections = {node: z_corrections[node] for node in subgraph.get_physical_nodes()}

        sub_dag = {
            node: ((xflow[node] | zflow[node]) - {node}) & sub_nodes for node in sub_nodes - subgraph.output_nodes
        }
        for node in subgraph.output_nodes:
            sub_dag[node] = set()

        sub_pattern = transpile(
            subgraph,
            sub_x_corrections,
            sub_z_corrections,
            sub_dag,
            correct_output=False,
        )

        pattern = pattern.append_pattern(sub_pattern)

    for output_node in graph.output_nodes:
        pattern.extend([X(node=output_node, domain=x_corrections[output_node])])
        pattern.extend([Z(node=output_node, domain=z_corrections[output_node])])

    pattern.mark_runnable()
    pattern.mark_deterministic()

    return pattern.freeze()
