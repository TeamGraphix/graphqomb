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


def greedy_minimize_time(
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
    max_qubit_count: int | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    r"""Fast greedy scheduler optimizing for minimal execution time (makespan).

    This algorithm uses different strategies based on max_qubit_count:
    - Without qubit limit: Prepare all nodes at time=0, measure in ASAP order
    - With qubit limit: Use slice-by-slice scheduling with slack-filling

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
    """
    unmeasured = graph.physical_nodes - graph.output_node_indices.keys()
    input_nodes = set(graph.input_node_indices.keys())
    output_nodes = set(graph.output_node_indices.keys())

    # Build inverse DAG: for each node, track which nodes must be measured before it
    inv_dag: dict[int, set[int]] = {node: set() for node in graph.physical_nodes}
    for parent, children in dag.items():
        for child in children:
            inv_dag[child].add(parent)

    # Cache neighbors to avoid repeated set constructions in tight loops
    neighbors_map = {node: graph.neighbors(node) for node in graph.physical_nodes}

    if max_qubit_count is None:
        # Optimal strategy: prepare all nodes at time=0, measure in ASAP order
        return _greedy_minimize_time_unlimited(
            graph, inv_dag, neighbors_map, input_nodes, output_nodes
        )

    # With qubit limit: use slice-by-slice scheduling with slack-filling
    return _greedy_minimize_time_limited(
        graph,
        dag,
        inv_dag,
        neighbors_map,
        unmeasured,
        input_nodes,
        output_nodes,
        max_qubit_count,
    )


def _greedy_minimize_time_unlimited(
    graph: BaseGraphState,
    inv_dag: Mapping[int, AbstractSet[int]],
    neighbors_map: Mapping[int, AbstractSet[int]],
    input_nodes: AbstractSet[int],
    output_nodes: AbstractSet[int],
) -> tuple[dict[int, int], dict[int, int]]:
    prepare_time: dict[int, int] = {}
    measure_time: dict[int, int] = {}

    # 1. Prepare all non-input nodes at time=0
    for node in graph.physical_nodes:
        if node not in input_nodes:
            prepare_time[node] = 0

    # 2. Compute ASAP measurement times using topological order
    # Each node can be measured at max(parent_meas_times) + 1
    # TopologicalSorter expects {node: dependencies}, which is inv_dag
    try:
        topo_order = list(TopologicalSorter(inv_dag).static_order())
    except CycleError as exc:
        msg = "No nodes can be measured; possible cyclic dependency or incomplete preparation."
        raise RuntimeError(msg) from exc

    for node in topo_order:
        if node in output_nodes:
            continue
        # Find the latest measurement time among parents
        parent_times = [measure_time[p] for p in inv_dag[node] if p in measure_time]
        # Find the latest preparation time among neighbors (constraint: prep[neighbor] < meas[node])
        neighbor_prep_times = [
            prepare_time.get(n, -1) for n in neighbors_map[node]
        ]
        # Measure at the next time slot after all parents are measured
        # AND after all neighbors are prepared
        measure_time[node] = max(
            max(parent_times, default=-1) + 1,
            max(neighbor_prep_times, default=-1) + 1,
        )

    return prepare_time, measure_time


def _greedy_minimize_time_limited(  # noqa: C901, PLR0912, PLR0913, PLR0917
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
    inv_dag: Mapping[int, AbstractSet[int]],
    neighbors_map: Mapping[int, AbstractSet[int]],
    unmeasured: AbstractSet[int],
    input_nodes: AbstractSet[int],
    output_nodes: AbstractSet[int],
    max_qubit_count: int,
) -> tuple[dict[int, int], dict[int, int]]:
    prepare_time: dict[int, int] = {}
    measure_time: dict[int, int] = {}

    # Make mutable copies
    inv_dag_mut: dict[int, set[int]] = {
        node: set(parents) for node, parents in inv_dag.items()
    }
    unmeasured_mut: set[int] = set(unmeasured)

    prepared: set[int] = set(input_nodes)
    alive: set[int] = set(input_nodes)

    if len(alive) > max_qubit_count:
        msg = "Initial number of active qubits exceeds max_qubit_count."
        raise RuntimeError(msg)

    # Compute criticality for prioritizing preparations
    criticality = _compute_criticality(dag, output_nodes)

    current_time = 0

    while unmeasured_mut:
        # Phase 1: Measure all ready nodes
        # A node is ready if:
        # - DAG dependencies are resolved (inv_dag_mut[node] is empty)
        # - All neighbors are prepared
        # - The node itself is prepared (if not an input node)
        ready_to_measure: set[int] = set()
        for node in unmeasured_mut:
            if inv_dag_mut[node]:
                continue  # DAG dependencies not resolved
            if not neighbors_map[node] <= prepared:
                continue  # Neighbors not prepared
            if node not in input_nodes and node not in prepared:
                continue  # Self not prepared
            ready_to_measure.add(node)

        for node in ready_to_measure:
            measure_time[node] = current_time
            unmeasured_mut.remove(node)
            alive.discard(node)

            # Update DAG dependencies
            for child in dag.get(node, ()):
                inv_dag_mut[child].discard(node)

        # Phase 2: Prepare nodes using free capacity (slack-filling)
        free_capacity = max_qubit_count - len(alive)

        if free_capacity > 0:
            # Get unprepared nodes with their priority scores
            unprepared = graph.physical_nodes - prepared
            if unprepared:
                prep_candidates = _get_prep_candidates_with_priority(
                    unprepared,
                    inv_dag_mut,
                    neighbors_map,
                    prepared,
                    unmeasured_mut,
                    output_nodes,
                    criticality,
                )
                # Prepare top candidates within free capacity
                for candidate, _score in prep_candidates[:free_capacity]:
                    prepare_time[candidate] = current_time
                    prepared.add(candidate)
                    alive.add(candidate)

        # Check if we made progress
        if not ready_to_measure and free_capacity == 0 and unmeasured_mut:
            # No measurements and no room to prepare - stuck
            msg = (
                "Cannot schedule more measurements without exceeding max qubit count. "
                "Please increase max_qubit_count."
            )
            raise RuntimeError(msg)

        current_time += 1

        # Safety check for infinite loops
        if current_time > len(graph.physical_nodes) * 2:
            msg = "Scheduling did not converge; possible cyclic dependency."
            raise RuntimeError(msg)

    return prepare_time, measure_time


def _compute_criticality(
    dag: Mapping[int, AbstractSet[int]],
    output_nodes: AbstractSet[int],
) -> dict[int, int]:
    # Compute criticality (remaining DAG depth) for each node.
    # Nodes with higher criticality should be prioritized for unblocking.
    criticality: dict[int, int] = {}

    # TopologicalSorter(dag) returns nodes with no "dependencies" first.
    # Since dag is {parent: children}, nodes with empty children come first (leaves).
    # This is the correct order for computing criticality (leaves before roots).
    try:
        topo_order = list(TopologicalSorter(dag).static_order())
    except CycleError:
        return {}

    for node in topo_order:
        children_crits = [criticality.get(c, 0) for c in dag.get(node, ())]
        criticality[node] = 1 + max(children_crits, default=0)

    # Output nodes have criticality 0 (they don't need to be measured)
    for node in output_nodes:
        criticality[node] = 0

    return criticality


def _get_prep_candidates_with_priority(  # noqa: PLR0913, PLR0917
    unprepared: AbstractSet[int],
    inv_dag: Mapping[int, AbstractSet[int]],
    neighbors_map: Mapping[int, AbstractSet[int]],
    prepared: AbstractSet[int],
    unmeasured: AbstractSet[int],
    output_nodes: AbstractSet[int],
    criticality: Mapping[int, int],
) -> list[tuple[int, float]]:
    # Get preparation candidates sorted by priority score.
    # Priority is based on how much preparing a node helps unblock measurements.
    # Find nodes that are DAG-ready but blocked by missing neighbors
    dag_ready_blocked: set[int] = set()
    missing_map: dict[int, set[int]] = {}

    for node in unmeasured:
        if inv_dag[node]:
            continue  # Not DAG-ready
        missing = set(neighbors_map[node]) - set(prepared)
        # Also check if the node itself needs preparation
        if node not in prepared:
            missing.add(node)
        if missing:
            dag_ready_blocked.add(node)
            missing_map[node] = missing

    # Score each unprepared node
    scores: list[tuple[int, float]] = []
    for candidate in unprepared:
        score = 0.0
        for blocked_node in dag_ready_blocked:
            if candidate in missing_map[blocked_node]:
                crit = criticality.get(blocked_node, 1)
                score += crit / len(missing_map[blocked_node])

        # Apply penalty for output nodes (they stay alive forever)
        if candidate in output_nodes:
            score *= 0.5

        scores.append((candidate, score))

    # Sort by score descending (higher score = higher priority)
    scores.sort(key=lambda x: -x[1])

    return scores


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
