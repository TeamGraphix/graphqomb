"""We will use fastflow once published.

This module provides:
- FlowLike: Type alias for a flowlike object.
- Layer: Type alias for a layer object.
- oddneighbors: Return the set of odd neighbors of a set of nodes.
- construct_dag: Construct a directed acyclic graph (DAG) from a flowlike object.
- check_causality: Check if the flowlike object is causal with respect to the graph state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

    from graphix_zx.graphstate import BaseGraphState


FlowLike = dict[int, set[int]]
Layer = dict[int, int]


def oddneighbors(nodes: AbstractSet[int], graph: BaseGraphState) -> set[int]:
    """Return the set of odd neighbors of a set of nodes.

    Parameters
    ----------
    nodes : AbstractSet[int]
        A set of nodes, of which we want to find the odd neighbors.
    graph : BaseGraphState
        The graph state.

    Returns
    -------
    set[int]
        Odd neighbors of the nodes.
    """
    odd_neighbors: set[int] = set()
    for node in nodes:
        odd_neighbors ^= graph.get_neighbors(node)
    return odd_neighbors


def construct_dag(gflow: FlowLike, graph: BaseGraphState, *, check: bool = True) -> dict[int, set[int]]:
    """Construct a directed acyclic graph (DAG) from a flowlike object.

    Parameters
    ----------
    gflow : FlowLike
        A flowlike object
    graph : BaseGraphState
        The graph state
    check : bool, optional
        Raise an error if a cycle is detected, by default True

    Returns
    -------
    dict[int, set[int]]
        The directed acyclic graph

    Raises
    ------
    ValueError
        If a cycle is detected
    """
    dag = {}
    outputs = graph.physical_nodes - gflow.keys()
    for node in gflow:
        dag[node] = (gflow[node] | oddneighbors(gflow[node], graph)) - {node}
    for output in outputs:
        dag[output] = set()

    if check and not _check_dag(dag):
        msg = "Cycle detected in the graph"
        raise ValueError(msg)

    return dag


def check_causality(
    graph: BaseGraphState,
    gflow: FlowLike,
) -> bool:
    """Check if the flowlike object is causal with respect to the graph state.

    Parameters
    ----------
    graph : BaseGraphState
        The graph state
    gflow : FlowLike
        The flowlike object

    Returns
    -------
    bool
        True if the flowlike object is causal, False otherwise
    """
    dag = construct_dag(gflow, graph)
    return _check_dag(dag)


def _check_dag(dag: Mapping[int, Iterable[int]]) -> bool:
    """Check if a directed acyclic graph (DAG) does not contain a cycle.

    Parameters
    ----------
    dag : Mapping[int, Iterable[int]]
        directed acyclic graph

    Returns
    -------
    bool
        True if the graph is valid, False otherwise
    """
    for node, children in dag.items():
        for child in children:
            if node in dag[child]:
                return False
    return True
