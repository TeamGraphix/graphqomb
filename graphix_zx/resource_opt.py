"""same functionaly as minimize_space in the original repo."""

from __future__ import annotations

from graphix_zx.flow import FlowLike, construct_dag
from graphix_zx.graphstate import BaseGraphState, GraphState


def get_subgraph_sequences(graph: BaseGraphState, meas_order: list[int]) -> list[GraphState]:
    """Get the subgraph sequences."""
    subgraphs = []
    q_indices = graph.get_q_indices()
    # initial graph
    initial_graph = GraphState()
    for node in graph.input_nodes:
        initial_graph.add_physical_node(node, q_index=q_indices[node], is_input=True, is_output=True)
    subgraphs.append(initial_graph)
    activated_nodes = graph.input_nodes

    # screen shots of graph states before each measurement
    for target_node in meas_order:
        subgraph = GraphState()
        neighbors = graph.get_neighbors(target_node)
        activation_nodes = get_activation_nodes(graph, target_node, activated_nodes)
        neighboring_prepared_nodes = neighbors - activation_nodes

        if target_node in activated_nodes:
            subgraph.add_physical_node(target_node, q_index=q_indices[node], is_input=True)
        else:
            subgraph.add_physical_node(target_node, q_index=q_indices[target_node])
        activated_nodes |= {target_node}
        subgraph.set_meas_angle(target_node, graph.get_meas_angles()[target_node])
        subgraph.set_meas_plane(target_node, graph.get_meas_planes()[target_node])

        for prepared_node in neighboring_prepared_nodes:
            subgraph.add_physical_node(
                prepared_node,
                q_index=q_indices[prepared_node],
                is_input=True,
                is_output=True,
            )
            subgraph.add_physical_edge(prepared_node, target_node)

        for activation_node in activation_nodes:
            subgraph.add_physical_node(activation_node, q_index=q_indices[activation_node], is_output=True)
            subgraph.add_physical_edge(activation_node, target_node)
            activated_nodes |= {activation_node}

        subgraphs.append(subgraph)

    # last graph
    last_graph = GraphState()
    for node in graph.output_nodes:
        last_graph.add_physical_node(node, q_index=q_indices[node], is_input=True, is_output=True)

    # add edge between output nodes
    # the other edges are already prepared before
    _prepared_outputs = set()
    for node in graph.output_nodes:
        for neighbor in graph.get_neighbors(node):
            if neighbor in _prepared_outputs:
                last_graph.add_physical_edge(node, neighbor)
        _prepared_outputs.add(node)

    subgraphs.append(last_graph)

    return subgraphs


def get_minimized_sp_meas_order(graph: BaseGraphState, gflow: FlowLike) -> list[int]:
    """Get the minimized space measurement order."""
    inverted_dag = get_dependencies(graph, gflow)
    activated_nodes = set(graph.input_nodes)
    unmeasured_nodes = graph.get_physical_nodes() - set(graph.output_nodes)

    meas_order = []

    while unmeasured_nodes:
        meas_candidates: set[int] = set()
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
        for node, children in inverted_dag.items():
            if target_node in children:
                children.remove(target_node)
    return meas_order


def get_dependencies(graph: BaseGraphState, gflow: FlowLike) -> dict[int, set[int]]:
    """Get the dependencies."""
    dag = construct_dag(gflow, graph)
    inverted_dag: dict[int, set[int]] = {node: set() for node in dag.keys()}
    for node, children in dag.items():
        for child in children:
            inverted_dag[child].add(node)
    return inverted_dag


def get_activation_nodes(graph: BaseGraphState, target_node: int, activated_nodes: set[int]) -> set[int]:
    """Get the nodes to be activated."""
    neighbors = set(graph.get_neighbors(target_node))
    activation_nodes = neighbors - activated_nodes
    return activation_nodes


def count_activation_cost(graph: BaseGraphState, target_node: int, activated_nodes: set[int]) -> int:
    """Count the activation cost."""
    activation_nodes = get_activation_nodes(graph, target_node, activated_nodes)
    return len(activation_nodes)
