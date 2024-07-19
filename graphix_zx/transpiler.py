"""generate standardized or resource optimized pattern from underlying graph and gflow. extended MC is included"""

from __future__ import annotations

from graphix_zx.common import Plane
from graphix_zx.command import E, M, N, X, Z
from graphix_zx.pattern import MutablePattern
from graphix_zx.flow import FlowLike
from graphix_zx.focus_flow import (
    oddneighbors,
    topological_sort_kahn,
)
from graphix_zx.graphstate import BaseGraphState


# extended MBQC
def generate_m_cmd(
    node: int,
    meas_plane: Plane,
    meas_angle: float,
    x_correction: set[int],
    z_correction: set[int],
) -> M:
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


# generate signal lists
def generate_corrections(graph: BaseGraphState, flow: FlowLike) -> dict[int, set[int]]:
    corrections: dict[int, set[int]] = {node: set() for node in graph.get_physical_nodes()}

    for node in flow.keys():
        for correction in flow[node]:
            corrections[correction] |= {node}

    # remove self-corrections
    for node in corrections.keys():
        corrections[node] -= {node}

    return corrections


def transpile_from_flow(graph: BaseGraphState, gflow: FlowLike, correct_output: bool = True) -> MutablePattern:
    # generate corrections
    x_flow = gflow
    z_flow = {node: oddneighbors(gflow[node], graph) for node in gflow.keys()}
    x_corrections = generate_corrections(graph, x_flow)
    z_corrections = generate_corrections(graph, z_flow)

    dag = {node: (x_flow[node] | z_flow[node]) - {node} for node in x_flow.keys()}
    for node in graph.output_nodes:
        dag[node] = set()

    pattern = transpile(graph, x_corrections, z_corrections, dag, correct_output)
    pattern.mark_runnable()
    pattern.mark_deterministic()
    return pattern


# generate standardized pattern from underlying graph and gflow
def transpile(
    graph: BaseGraphState,
    x_corrections: dict[int, set[int]],
    z_corrections: dict[int, set[int]],
    dag: dict[int, set[int]],
    correct_output: bool = True,
) -> MutablePattern:
    # TODO : check the validity of the flows
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
) -> MutablePattern:
    pattern = MutablePattern()

    xflow = gflow
    zflow = {node: oddneighbors(gflow[node], graph) for node in gflow.keys()}
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

    return pattern
