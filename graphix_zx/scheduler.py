"""Get the sequence of graph structures at each step of the computation to optimize the resource.

This module provides:
- get_subgraph_sequences: Get the sequence of graph structures at each step of the computation.
- get_reduced_space_meas_order: Naive algorithm to get a measurement order reducing the space of pattern.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.flow import FlowLike, construct_dag
from graphix_zx.graphstate import BaseGraphState, GraphState

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet


def get_subgraph_sequences(graph: BaseGraphState, meas_order: Iterable[int]) -> list[GraphState]:
    """Get the sequence of graph structures at each step of the computation.

    Parameters
    ----------
    graph : BaseGraphState
        Whole graph state
    meas_order : Iterable[int]
        Measurement order to get the subgraph sequence

    Returns
    -------
    list[GraphState]
        Sequence of subgraphs
    """
    subgraphs = []
    q_indices = graph.q_indices
    # initial graph
    initial_graph = GraphState()
    for node in graph.input_nodes:
        initial_graph.add_physical_node(node, q_index=q_indices[node], is_input=True, is_output=True)
    subgraphs.append(initial_graph)
    prepared_nodes = graph.input_nodes

    # snap shots of graph structure at each step
    for target_node in meas_order:
        subgraph = GraphState()
        neighbors = graph.get_neighbors(target_node)
        node_set_to_activate = _get_node_set_to_activate(graph, target_node, prepared_nodes)
        neighboring_prepared_nodes = neighbors - node_set_to_activate

        if target_node in prepared_nodes:
            subgraph.add_physical_node(target_node, q_index=q_indices[target_node], is_input=True)
        else:
            subgraph.add_physical_node(target_node, q_index=q_indices[target_node])
        prepared_nodes |= {target_node}
        subgraph.set_meas_angle(target_node, graph.meas_angles[target_node])
        subgraph.set_meas_plane(target_node, graph.meas_planes[target_node])

        for prepared_node in neighboring_prepared_nodes:
            subgraph.add_physical_node(
                prepared_node,
                q_index=q_indices[prepared_node],
                is_input=True,
                is_output=True,
            )
            subgraph.add_physical_edge(prepared_node, target_node)

        for node_to_activate in node_set_to_activate:
            subgraph.add_physical_node(node_to_activate, q_index=q_indices[node_to_activate], is_output=True)
            subgraph.add_physical_edge(node_to_activate, target_node)
            prepared_nodes |= {node_to_activate}

        subgraphs.append(subgraph)

    # last graph
    last_graph = GraphState()
    for node in graph.output_nodes:
        last_graph.add_physical_node(node, q_index=q_indices[node], is_input=True, is_output=True)

    # add edge between output nodes
    # the other edges are already prepared before
    prepared_outputs = set()
    for node in graph.output_nodes:
        for neighbor in graph.get_neighbors(node):
            if neighbor in prepared_outputs:
                last_graph.add_physical_edge(node, neighbor)
        prepared_outputs.add(node)

    subgraphs.append(last_graph)

    return subgraphs


def get_reduced_space_meas_order(graph: BaseGraphState, gflow: FlowLike) -> list[int]:
    """Naive algorithm to get a measurement order reducing the space of pattern.

    Parameters
    ----------
    graph : BaseGraphState
        The graph state.
    gflow : FlowLike
        The flowlike object.

    Returns
    -------
    list[int]
        Measurement order to reduce the space of pattern.

    Raises
    ------
    ValueError
        If a cycle is detected in the graph.
    """
    inverted_dag = _get_dependencies(graph, gflow)
    prepared_nodes = set(graph.input_nodes)
    unmeasured_nodes = graph.physical_nodes - graph.output_nodes

    meas_order = []

    while unmeasured_nodes:
        meas_candidates: set[int] = set()
        for node in unmeasured_nodes:
            if len(inverted_dag[node]) == 0:
                meas_candidates |= {node}

        if len(meas_candidates) == 0:
            msg = "Cycle detected in the graph"
            raise ValueError(msg)

        # evaluate activation cost
        activation_costs = {node: _count_activation_cost(graph, node, prepared_nodes) for node in meas_candidates}
        # select the node with the minimum activation cost
        target_node = _get_min_from_dict(activation_costs)
        meas_order.append(target_node)
        prepared_nodes |= _get_node_set_to_activate(graph, target_node, prepared_nodes) | {target_node}
        unmeasured_nodes -= {target_node}
        for children in inverted_dag.values():
            if target_node in children:
                children.remove(target_node)
    return meas_order


def _count_activation_cost(graph: BaseGraphState, target_node: int, prepared_nodes: AbstractSet[int]) -> int:
    """Count the activation cost.

    Parameters
    ----------
    graph : BaseGraphState
        The graph state.
    target_node : int
        The target node to measure.
    prepared_nodes : AbstractSet[int]
        The prepared nodes.

    Returns
    -------
    int
        The activation cost to measure the target node.
    """
    node_set_to_activate = _get_node_set_to_activate(graph, target_node, prepared_nodes)
    return len(node_set_to_activate)


def _get_dependencies(graph: BaseGraphState, gflow: FlowLike) -> dict[int, set[int]]:
    """Get the dependencies of each node.

    Parameters
    ----------
    graph : BaseGraphState
        The graph state.
    gflow : FlowLike
        The flowlike object.

    Returns
    -------
    dict[int, set[int]]
        The dependencies of each node
    """
    dag = construct_dag(gflow, graph)
    invert_dag: dict[int, set[int]] = {node: set() for node in dag}
    for node, children in dag.items():
        for child in children:
            invert_dag[child].add(node)
    return invert_dag


def _get_node_set_to_activate(graph: BaseGraphState, target_node: int, prepared_nodes: AbstractSet[int]) -> set[int]:
    """Get the node set to activate when measuring the target node.

    Parameters
    ----------
    graph : BaseGraphState
        The graph state.
    target_node : int
        The target node to measure.
    prepared_nodes : AbstractSet[int]
        The prepared nodes.

    Returns
    -------
    set[int]
        The node set to activate.
    """
    return graph.get_neighbors(target_node) - prepared_nodes


def _get_min_from_dict(d: Mapping[int, int]) -> int:
    """Get the key with the minimum value from a dictionary.

    Parameters
    ----------
    d : Mapping[int, int]
        The dictionary.

    Returns
    -------
    int
        The key with the minimum value.

    Raises
    ------
    ValueError
        If no minimum value is found.
    """
    min_index = None
    min_value = float("inf")
    for key, value in d.items():
        if value < min_value:
            min_index = key
            min_value = value
    if min_index is None:
        msg = "No minimum value found"
        raise ValueError(msg)
    return min_index
