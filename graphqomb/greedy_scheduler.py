"""Greedy heuristic scheduler for fast MBQC pattern scheduling.

This module provides fast greedy scheduling algorithms as an alternative to
CP-SAT based optimization. The greedy algorithms provide approximate solutions
with speedup compared to CP-SAT, making them suitable for large-scale
graphs or when optimality is not critical.

This module provides:

- `greedy_minimize_time`: Fast greedy scheduler optimizing for minimal execution time
- `greedy_minimize_space`: Fast greedy scheduler optimizing for minimal qubit usage
"""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


def greedy_minimize_time(
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
) -> tuple[dict[int, int], dict[int, int]]:
    r"""Fast greedy scheduler optimizing for minimal execution time (makespan).

    This algorithm uses a straightforward greedy approach:
    1. At each time step, measure all nodes that can be measured
    2. Prepare all neighbors of measured nodes just before measurement

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state to schedule
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The directed acyclic graph representing measurement dependencies

    Returns
    -------
    `tuple`\[`dict`\[`int`, `int`\], `dict`\[`int`, `int`\]\]
        A tuple of (prepare_time, measure_time) dictionaries

    Raises
    ------
    RuntimeError
        If no nodes can be measured at a given time step, indicating a possible
    """
    prepare_time: dict[int, int] = {}
    measure_time: dict[int, int] = {}

    unmeasured = graph.physical_nodes - graph.output_node_indices.keys()

    # Build inverse DAG: for each node, track which nodes must be measured before it
    inv_dag: dict[int, set[int]] = {node: set() for node in graph.physical_nodes}
    for parent, children in dag.items():
        for child in children:
            inv_dag[child].add(parent)

    prepared: set[int] = set(graph.input_node_indices.keys())
    current_time = 0

    while unmeasured:
        to_measure = set()
        for node in unmeasured:
            if len(inv_dag[node]) == 0:
                to_measure.add(node)

        if not to_measure:
            msg = "No nodes can be measured; possible cyclic dependency or incomplete preparation."
            raise RuntimeError(msg)

        needs_prep = False
        # Prepare neighbors at current_time
        for node in to_measure:
            for neighbor in graph.neighbors(node):
                if neighbor not in prepared:
                    prepare_time[neighbor] = current_time
                    prepared.add(neighbor)
                    needs_prep = True

        # Measure at current_time if no prep needed, otherwise at current_time + 1
        meas_time = current_time + 1 if needs_prep else current_time
        for node in to_measure:
            measure_time[node] = meas_time
            unmeasured.remove(node)
            # Remove measured node from dependencies of all its children in the DAG
            for child in dag.get(node, set()):
                if child in inv_dag:
                    inv_dag[child].remove(node)

        current_time = meas_time + 1

    return prepare_time, measure_time


def greedy_minimize_space(  # noqa: C901, PLR0912
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
) -> tuple[dict[int, int], dict[int, int]]:
    """Fast greedy scheduler optimizing for minimal qubit usage (space).

    This algorithm uses a greedy approach to minimize the number of active
    qubits at each time step:
    1. At each time step, select the next node to measure that minimizes the
       number of new qubits that need to be prepared.
    2. Prepare neighbors of the measured node just before measurement.

    Parameters
    ----------
    graph : BaseGraphState
        The graph state to schedule
    dag : Mapping[int, AbstractSet[int]]
        The directed acyclic graph representing measurement dependencies

    Returns
    -------
    tuple[dict[int, int], dict[int, int]]
        A tuple of (prepare_time, measure_time) dictionaries

    Raises
    ------
    RuntimeError
        If no nodes can be measured at a given time step, indicating a possible
        cyclic dependency or incomplete preparation.
    """
    prepare_time: dict[int, int] = {}
    measure_time: dict[int, int] = {}

    unmeasured = graph.physical_nodes - graph.output_node_indices.keys()

    topo_order = list(TopologicalSorter(dag).static_order())
    topo_order.reverse()  # from parents to children

    # Build inverse DAG: for each node, track which nodes must be measured before it
    inv_dag: dict[int, set[int]] = {node: set() for node in graph.physical_nodes}
    for parent, children in dag.items():
        for child in children:
            inv_dag[child].add(parent)

    prepared: set[int] = set(graph.input_node_indices.keys())
    alive: set[int] = set(graph.input_node_indices.keys())
    current_time = 0

    while unmeasured:
        candidate_nodes = set()
        for node in alive:
            if len(inv_dag[node]) == 0:
                candidate_nodes.add(node)

        if not candidate_nodes:
            # If no alive nodes can be measured, pick from unmeasured
            for node in unmeasured - alive:
                if len(inv_dag[node]) == 0:
                    candidate_nodes.add(node)

        if not candidate_nodes:
            msg = "No nodes can be measured; possible cyclic dependency or incomplete preparation."
            raise RuntimeError(msg)

        # calculate costs and pick the best node to measure
        best_node_candidate: set[int] = set()
        best_cost = float("inf")
        for node in candidate_nodes:
            cost = _calc_activate_cost(node, graph, prepared)
            if cost < best_cost:
                best_cost = cost
                best_node_candidate = {node}
            elif cost == best_cost:
                best_node_candidate.add(node)

        # tie-breaker: choose the node that appears first in topological order
        best_node = min(best_node_candidate, key=topo_order.index)

        # Prepare neighbors at current_time
        needs_prep = False
        for neighbor in graph.neighbors(best_node):
            if neighbor not in prepared:
                prepare_time[neighbor] = current_time
                prepared.add(neighbor)
                alive.add(neighbor)
                needs_prep = True

        # Measure at current_time if no prep needed, otherwise at current_time + 1
        meas_time = current_time + 1 if needs_prep else current_time
        measure_time[best_node] = meas_time
        unmeasured.remove(best_node)
        alive.remove(best_node)

        # Remove measured node from dependencies of all its children in the DAG
        for child in dag.get(best_node, set()):
            if child in inv_dag:
                inv_dag[child].remove(best_node)

        current_time = meas_time + 1

    return prepare_time, measure_time


def _calc_activate_cost(
    node: int,
    graph: BaseGraphState,
    prepared: set[int],
) -> int:
    """Calculate the cost of activating (preparing) a node.

    The cost is defined as the number of new qubits that would become active
    (prepared but not yet measured) if this node were to be measured next.

    Parameters
    ----------
    node : int
        The node to evaluate.
    graph : BaseGraphState
        The graph state.
    prepared : set[int]
        The set of currently prepared nodes.

    Returns
    -------
    int
        The activation cost for the node.
    """
    return len(graph.neighbors(node) - prepared)


def solve_greedy_schedule(
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
    minimize_space: bool = False,
) -> tuple[dict[int, int], dict[int, int]] | None:
    """Solve scheduling using greedy heuristics.

    This is a convenience wrapper that selects the appropriate greedy algorithm
    based on the optimization objective.

    Parameters
    ----------
    graph : BaseGraphState
        The graph state to schedule
    dag : Mapping[int, AbstractSet[int]]
        The directed acyclic graph representing measurement dependencies
    minimize_space : bool, default=False
        If True, optimize for minimal qubit usage (space).
        If False, optimize for minimal execution time.

    Returns
    -------
    tuple[dict[int, int], dict[int, int]] | None
        A tuple of (prepare_time, measure_time) dictionaries if successful,
        None if scheduling fails (should rarely happen for valid inputs)
    """
    if minimize_space:
        return greedy_minimize_space(graph, dag)
    return greedy_minimize_time(graph, dag)
