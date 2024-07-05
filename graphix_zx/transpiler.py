from __future__ import annotations

from graphix_zx.command import E, M, N, Pattern, X, Z
from graphix_zx.focus_flow import (
    GFlow,
    oddneighbors,
    topological_sort_kahn,
)
from graphix_zx.interface import GraphState


# extended MBQC
def generate_m_cmd(
    node: int,
    meas_plane: str,
    meas_angle: float,
    x_correction: list[int],
    z_correction: list[int],
) -> M:
    if meas_plane == "XY":
        s_domain = x_correction
        t_domain = z_correction
    elif meas_plane == "XZ":
        s_domain = x_correction.extend(z_correction)
        t_domain = x_correction
    elif meas_plane == "YZ":
        s_domain = z_correction
        t_domain = x_correction
    else:
        raise ValueError("Invalid measurement plane")
    return M(
        node=node,
        plane=meas_plane,
        angle=meas_angle,
        s_domain=s_domain,
        t_domain=t_domain,
    )


# generate signal lists
def generate_corrections(graph: GraphState, flow: GFlow) -> dict[int, set[int]]:
    corrections: dict[str, list[int]] = {node: set() for node in graph.get_physical_nodes()}

    for node in flow.keys():
        for correction in flow[node]:
            corrections[correction] |= {node}

    # remove self-corrections
    for node in corrections.keys():
        corrections[node] -= {node}

    return corrections


def transpile_from_flow(graph: GraphState, gflow: GFlow, correct_output: bool = False) -> Pattern:
    # generate corrections
    x_flow = gflow
    z_flow = {node: oddneighbors(gflow[node], graph) for node in gflow.keys()}
    return transpile(graph, x_flow, z_flow, correct_output)


# generate standardized pattern from underlying graph and gflow
def transpile(
    graph: GraphState,
    x_flow: dict[int, set[int]],
    z_flow: dict[int, set[int]],
    correct_output: bool = True,
) -> Pattern:
    # TODO : check the validity of the gflow
    input_nodes = graph.input_nodes
    output_nodes = graph.output_nodes
    meas_planes = graph.get_meas_planes()
    meas_angles = graph.get_meas_angles()

    internal_nodes = set(graph.get_physical_nodes()) - set(input_nodes) - set(output_nodes)

    x_corrections = generate_corrections(graph, x_flow)
    z_corrections = generate_corrections(graph, z_flow)

    dag = {node: (x_flow[node] | z_flow[node]) - {node} for node in x_flow.keys()}
    for output in output_nodes:
        dag[output] = set()
    topo_order = topological_sort_kahn(dag)

    pattern = Pattern(input_nodes=input_nodes)
    pattern.extend([N(node=node) for node in internal_nodes])
    pattern.extend([N(node=node) for node in output_nodes])
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
    subgraphs: list[GraphState],
    input_nodes: list[int],
    gflow: GFlow,
) -> Pattern:
    pattern = Pattern(input_nodes=input_nodes)
    for subgraph in subgraphs:
        sub_input_nodes = subgraph.input_nodes
        sub_output_nodes = subgraph.output_nodes

        sub_internal_nodes = set(subgraph.nodes) - set(sub_input_nodes) - set(sub_output_nodes)
        sub_gflow = {node: gflow[node] for node in set(sub_input_nodes) | sub_internal_nodes}

        sub_pattern = transpile(subgraph, sub_gflow, correct_output=False)

        # TODO: corrections on output

        pattern += sub_pattern

    return pattern
