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

from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


def greedy_minimize_time(  # noqa: C901, PLR0912
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
    max_qubit_count: int | None = None,
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
    max_qubit_count : `int` | `None`, optional
        Maximum allowed number of active qubits. If None, no limit is enforced.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `int`\], `dict`\[`int`, `int`\]\]
        A tuple of (prepare_time, measure_time) dictionaries

    Raises
    ------
    RuntimeError
        If no nodes can be measured at a given time step, indicating a possible
        cyclic dependency or incomplete preparation, or if max_qubit_count
        is too small to make progress.
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
    alive: set[int] = set(graph.input_node_indices.keys())

    if max_qubit_count is not None and len(alive) > max_qubit_count:
        msg = "Initial number of active qubits exceeds max_qubit_count."
        raise RuntimeError(msg)

    current_time = 0

    # Nodes whose dependencies are all resolved and are not yet measured
    measure_candidates: set[int] = {node for node in unmeasured if not inv_dag[node]}

    # Cache neighbors to avoid repeated set constructions in tight loops
    neighbors_map = {node: graph.neighbors(node) for node in graph.physical_nodes}

    while unmeasured:  # noqa: PLR1702
        if not measure_candidates:
            msg = "No nodes can be measured; possible cyclic dependency or incomplete preparation."
            raise RuntimeError(msg)

        # Track which nodes have neighbors being prepared at current_time
        nodes_with_prep: set[int] = set()

        if max_qubit_count is not None:
            # Choose measurement nodes from measure_candidates while respecting max_qubit_count
            to_measure, to_prepare = _determine_measure_nodes(
                neighbors_map,
                measure_candidates,
                prepared,
                alive,
                max_qubit_count,
            )
            for neighbor in to_prepare:
                prepare_time[neighbor] = current_time
                # If this neighbor already had no dependencies, it becomes measure candidate
                if not inv_dag[neighbor] and neighbor in unmeasured:
                    measure_candidates.add(neighbor)
                # Record which measurement nodes have this neighbor
                for node in to_measure:
                    if neighbor in neighbors_map[node]:
                        nodes_with_prep.add(node)
            prepared.update(to_prepare)
            alive.update(to_prepare)
        else:
            # Without a qubit limit, measure all currently measure candidates
            to_measure = set(measure_candidates)
            for node in to_measure:
                for neighbor in neighbors_map[node]:
                    if neighbor not in prepared:
                        prepare_time[neighbor] = current_time
                        prepared.add(neighbor)
                        nodes_with_prep.add(node)

                        if not inv_dag[neighbor] and neighbor in unmeasured:
                            measure_candidates.add(neighbor)

        # Measure at current_time if no prep needed for that node, otherwise at current_time + 1
        max_meas_time = current_time
        for node in to_measure:
            if node in nodes_with_prep:
                measure_time[node] = current_time + 1
                max_meas_time = current_time + 1
            else:
                measure_time[node] = current_time

            if max_qubit_count is not None:
                alive.remove(node)
            unmeasured.remove(node)
            measure_candidates.remove(node)

            # Remove measured node from dependencies of all its children in the DAG
            for child in dag.get(node, ()):
                inv_dag[child].remove(node)
                if not inv_dag[child] and child in unmeasured:
                    measure_candidates.add(child)

        current_time = max_meas_time + 1

    return prepare_time, measure_time


def _determine_measure_nodes(
    neighbors_map: Mapping[int, AbstractSet[int]],
    measure_candidates: AbstractSet[int],
    prepared: AbstractSet[int],
    alive: AbstractSet[int],
    max_qubit_count: int,
) -> tuple[set[int], set[int]]:
    r"""Determine which nodes to measure without exceeding max qubit count.

    Parameters
    ----------
    neighbors_map : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        Mapping from node to its neighbors.
    measure_candidates : `collections.abc.Set`\[`int`\]
        The candidate nodes available for measurement.
    prepared : `collections.abc.Set`\[`int`\]
        The set of currently prepared nodes.
    alive : `collections.abc.Set`\[`int`\]
        The set of currently active (prepared but not yet measured) nodes.
    max_qubit_count : `int`
        The maximum allowed number of active qubits.

    Returns
    -------
    `tuple`\[`set`\[`int`\], `set`\[`int`\]\]
        A tuple of (to_measure, to_prepare) sets indicating which nodes to measure and prepare.

    Raises
    ------
    RuntimeError
        If no nodes can be measured without exceeding the max qubit count.
    """
    to_measure: set[int] = set()
    to_prepare: set[int] = set()

    for node in measure_candidates:
        # Neighbors that still need to be prepared for this node
        new_neighbors = neighbors_map[node] - prepared
        additional_to_prepare = new_neighbors - to_prepare

        # Projected number of active qubits after preparing these neighbors
        projected_active = len(alive) + len(to_prepare) + len(additional_to_prepare)

        if projected_active <= max_qubit_count:
            to_measure.add(node)
            to_prepare |= new_neighbors

    if not to_measure:
        msg = "Cannot schedule more measurements without exceeding max qubit count. Please increase max_qubit_count."
        raise RuntimeError(msg)

    return to_measure, to_prepare


def greedy_minimize_space(  # noqa: C901, PLR0914
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
) -> tuple[dict[int, int], dict[int, int]]:
    r"""Fast greedy scheduler optimizing for minimal qubit usage (space).

    This algorithm uses a greedy approach to minimize the number of active
    qubits at each time step:
    1. At each time step, select the next node to measure that minimizes the
       projected number of alive qubits after any required preparations.
    2. Prepare neighbors of the measured node just before measurement.

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
        cyclic dependency or incomplete preparation.
    """
    prepare_time: dict[int, int] = {}
    measure_time: dict[int, int] = {}

    unmeasured = graph.physical_nodes - graph.output_node_indices.keys()

    try:
        topo_order = list(TopologicalSorter(dag).static_order())
    except CycleError as exc:
        msg = "No nodes can be measured; possible cyclic dependency or incomplete preparation."
        raise RuntimeError(msg) from exc
    topo_order.reverse()  # from parents to children
    topo_rank = {node: i for i, node in enumerate(topo_order)}

    # Build inverse DAG: for each node, track which nodes must be measured before it
    inv_dag: dict[int, set[int]] = {node: set() for node in graph.physical_nodes}
    for parent, children in dag.items():
        for child in children:
            inv_dag[child].add(parent)

    prepared: set[int] = set(graph.input_node_indices.keys())
    alive: set[int] = set(graph.input_node_indices.keys())
    current_time = 0

    # Cache neighbors once as the graph is static during scheduling
    neighbors_map = {node: graph.neighbors(node) for node in graph.physical_nodes}

    measure_candidates: set[int] = {node for node in unmeasured if not inv_dag[node]}

    while unmeasured:
        if not measure_candidates:
            msg = "No nodes can be measured; possible cyclic dependency or incomplete preparation."
            raise RuntimeError(msg)

        # calculate costs and pick the best node to measure
        default_rank = len(topo_rank)
        candidates = iter(measure_candidates)
        best_node = next(candidates)
        best_cost = _calc_activate_cost(best_node, neighbors_map, prepared, alive)
        best_rank = topo_rank.get(best_node, default_rank)
        for node in candidates:
            cost = _calc_activate_cost(node, neighbors_map, prepared, alive)
            rank = topo_rank.get(node, default_rank)
            if cost < best_cost or (cost == best_cost and rank < best_rank):
                best_cost = cost
                best_rank = rank
                best_node = node

        # Prepare neighbors at current_time
        new_neighbors = neighbors_map[best_node] - prepared
        needs_prep = bool(new_neighbors)
        if needs_prep:
            for neighbor in new_neighbors:
                prepare_time[neighbor] = current_time
            prepared.update(new_neighbors)
            alive.update(new_neighbors)

        # Measure at current_time if no prep needed, otherwise at current_time + 1
        meas_time = current_time + 1 if needs_prep else current_time
        measure_time[best_node] = meas_time
        unmeasured.remove(best_node)
        alive.remove(best_node)

        measure_candidates.remove(best_node)

        # Remove measured node from dependencies of all its children in the DAG
        for child in dag.get(best_node, ()):
            inv_dag[child].remove(best_node)
            if not inv_dag[child] and child in unmeasured:
                measure_candidates.add(child)

        current_time = meas_time + 1

    return prepare_time, measure_time


def _calc_activate_cost(
    node: int,
    neighbors_map: Mapping[int, AbstractSet[int]],
    prepared: AbstractSet[int],
    alive: AbstractSet[int],
) -> int:
    r"""Calculate the projected number of alive qubits if measuring this node next.

    If neighbors must be prepared, they become alive at the current time slice
    while the node itself remains alive until the next slice. If no preparation
    is needed, the node is measured in the current slice and removed.

    Parameters
    ----------
    node : `int`
        The node to evaluate.
    neighbors_map : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        Cached neighbor sets for graph nodes.
    prepared : `collections.abc.Set`\[`int`\]
        The set of currently prepared nodes.
    alive : `collections.abc.Set`\[`int`\]
        The set of currently active (prepared but not yet measured) nodes.

    Returns
    -------
    `int`
        The activation cost for the node.
    """
    new_neighbors = neighbors_map[node] - prepared
    if new_neighbors:
        return len(alive) + len(new_neighbors)
    # No preparation needed -> node is measured in the current slice, so alive decreases by 1.
    return max(len(alive) - 1, 0)
