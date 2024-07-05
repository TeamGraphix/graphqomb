from __future__ import annotations

from graphix_zx.interface import GraphState, BasicGraphState
from graphix_zx.focus_flow import GFlow, construct_dag


def get_subgraph_sequences(graph: GraphState, meas_order: list[int]) -> list[GraphState]:
    """Get the subgraph sequences."""
    subgraphs = []
    activated_nodes = graph.input_nodes
    for target_node in meas_order:
        subgraph = BasicGraphState()
        neighbors = graph.get_neighbors(target_node)
        activation_nodes = get_activation_nodes(graph, target_node, activated_nodes)
        prepared_nodes = neighbors - activation_nodes

        if target_node in activated_nodes:
            subgraph.add_physical_node(target_node, is_input=True)
        else:
            subgraph.add_physical_node(target_node)
        subgraph.set_meas_angle(target_node, graph.get_meas_angles()[target_node])
        subgraph.set_meas_plane(target_node, graph.get_meas_planes()[target_node])

        for prepared_node in prepared_nodes:
            subgraph.add_physical_node(prepared_node, is_input=True)
            subgraph.add_physical_edge(prepared_node, target_node)

        for activation_node in activation_nodes:
            subgraph.add_physical_node(activation_node, is_output=True)
            subgraph.add_physical_edge(activation_node, target_node)

        subgraphs.append(subgraph)

    return subgraphs


def get_minimized_sp_meas_order(graph: GraphState, gflow: GFlow) -> list[int]:
    """Get the minimized space measurement order."""
    inverted_dag = get_dependencies(graph, gflow)
    activated_nodes = set(graph.input_nodes)
    unmeasured_nodes = graph.get_physical_nodes() - activated_nodes - set(graph.output_nodes)

    meas_order = []

    while unmeasured_nodes:
        meas_candidates = set()
        for node in unmeasured_nodes:
            if len(inverted_dag[node]) == 0:
                meas_candidates |= {node}

        if len(meas_candidates) == 0:
            raise ValueError("Cycle detected in the graph")

        # evaluate activation cost
        activation_costs = {node: count_activation_cost(graph, node, activated_nodes) for node in meas_candidates}
        # select the node with the minimum activation cost
        target_node = min(activation_costs, key=activation_costs.get)
        meas_order.append(target_node)
        activated_nodes |= get_activation_nodes(graph, target_node, activated_nodes) | {target_node}
        unmeasured_nodes -= {target_node}
    return meas_order


def get_dependencies(graph: GraphState, gflow: GFlow) -> dict[int, set[int]]:
    """Get the dependencies."""
    dag = construct_dag(gflow, graph)
    inverted_dag: dict[int, set[int]] = {node: set() for node in dag.keys()}
    for node, children in dag.items():
        for child in children:
            inverted_dag[child].add(node)
    return inverted_dag


def get_activation_nodes(graph: GraphState, target_node: int, activated_nodes: set[int]) -> set[int]:
    """Get the nodes to be activated."""
    neighbors = set(graph.get_neighbors(target_node))
    activation_nodes = neighbors - activated_nodes
    return activation_nodes


def count_activation_cost(graph: GraphState, target_node: int, activated_nodes: set[int]) -> int:
    """Count the activation cost."""
    activation_nodes = get_activation_nodes(graph, target_node, activated_nodes)
    return len(activation_nodes)
