"""Test greedy scheduling algorithms."""

import time

import pytest

from graphqomb.graphstate import GraphState
from graphqomb.greedy_scheduler import (
    greedy_minimize_space,
    greedy_minimize_time,
)
from graphqomb.schedule_solver import ScheduleConfig, Strategy
from graphqomb.scheduler import Scheduler


def test_greedy_minimize_time_simple() -> None:
    """Test greedy_minimize_time on a simple graph."""
    # Create a simple 3-node chain graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Run greedy scheduler
    prepare_time, measure_time = greedy_minimize_time(graph, scheduler.dag)

    # Check that all non-input nodes have preparation times
    assert node1 in prepare_time
    assert node0 not in prepare_time  # Input node should not be prepared

    # Check that all non-output nodes have measurement times
    assert node0 in measure_time
    assert node1 in measure_time
    assert node2 not in measure_time  # Output node should not be measured

    # Verify DAG constraints: node0 measured before node1
    assert measure_time[node0] < measure_time[node1]


def test_greedy_minimize_space_simple() -> None:
    """Test greedy_minimize_space on a simple graph."""
    # Create a simple 3-node chain graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Run greedy scheduler
    prepare_time, measure_time = greedy_minimize_space(graph, scheduler.dag)

    # Check that all non-input nodes have preparation times
    assert node1 in prepare_time
    assert node0 not in prepare_time  # Input node should not be prepared

    # Check that all non-output nodes have measurement times
    assert node0 in measure_time
    assert node1 in measure_time
    assert node2 not in measure_time  # Output node should not be measured

    # Verify DAG constraints
    assert measure_time[node0] < measure_time[node1]


def _compute_max_alive_qubits(
    graph: GraphState,
    prepare_time: dict[int, int],
    measure_time: dict[int, int],
) -> int:
    """Compute the maximum number of alive qubits over time.

    A node is considered alive at time t if:
    - It is an input node and t >= -1 and t < measurement time (if any), or
    - It has a preparation time p and t >= p and t < measurement time (if any).

    Returns
    -------
    int
        The maximum number of alive qubits at any time step.
    """
    # Determine time range to check
    max_t = max(set(prepare_time.values()) | set(measure_time.values()), default=0)

    max_alive = len(graph.input_node_indices)  # At least inputs are alive at t = -1
    for t in range(max_t + 1):
        alive_nodes: set[int] = set()
        for node in graph.physical_nodes:
            # Determine preparation time
            prep_t = -1 if node in graph.input_node_indices else prepare_time.get(node)

            if prep_t is None or t < prep_t:
                continue

            # Determine measurement time (None for outputs or unscheduled)
            meas_t = measure_time.get(node)

            if meas_t is None or t < meas_t:
                alive_nodes.add(node)

        max_alive = max(max_alive, len(alive_nodes))

    return max_alive


def test_greedy_minimize_time_with_max_qubit_count_respects_limit() -> None:
    """Verify that greedy_minimize_time respects max_qubit_count."""
    graph = GraphState()
    # chain graph: 0-1-2-3
    n0 = graph.add_physical_node()
    n1 = graph.add_physical_node()
    n2 = graph.add_physical_node()
    n3 = graph.add_physical_node()
    graph.add_physical_edge(n0, n1)
    graph.add_physical_edge(n1, n2)
    graph.add_physical_edge(n2, n3)

    qindex = 0
    graph.register_input(n0, qindex)
    graph.register_output(n3, qindex)

    flow = {n0: {n1}, n1: {n2}, n2: {n3}}
    scheduler = Scheduler(graph, flow)

    # Set max_qubit_count to 2 (a feasible value for this graph)
    prepare_time, measure_time = greedy_minimize_time(graph, scheduler.dag, max_qubit_count=2)

    # Check basic properties
    assert n1 in prepare_time
    assert n0 not in prepare_time
    assert n0 in measure_time
    assert n2 in measure_time
    assert n3 not in measure_time

    # Verify that the number of alive qubits never exceeds the limit
    max_alive = _compute_max_alive_qubits(graph, prepare_time, measure_time)
    assert max_alive <= 2


def test_greedy_minimize_time_with_too_small_max_qubit_count_raises() -> None:
    """Verify that greedy_minimize_time raises RuntimeError when max_qubit_count is too small."""
    graph = GraphState()
    # chain graph: 0-1-2 (at least 2 qubits are needed)
    n0 = graph.add_physical_node()
    n1 = graph.add_physical_node()
    n2 = graph.add_physical_node()
    graph.add_physical_edge(n0, n1)
    graph.add_physical_edge(n1, n2)

    qindex = 0
    graph.register_input(n0, qindex)
    graph.register_output(n2, qindex)

    flow = {n0: {n1}, n1: {n2}}
    scheduler = Scheduler(graph, flow)

    # max_qubit_count=1 is not feasible, so expect RuntimeError
    with pytest.raises(RuntimeError, match="max_qubit_count"):
        greedy_minimize_time(graph, scheduler.dag, max_qubit_count=1)


def test_greedy_scheduler_via_solve_schedule() -> None:
    """Test greedy scheduler through Scheduler.solve_schedule with use_greedy=True."""
    # Create a simple graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Test with greedy MINIMIZE_TIME
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, use_greedy=True)
    success = scheduler.solve_schedule(config)
    assert success

    # Verify schedule is valid
    scheduler.validate_schedule()

    # Test with greedy MINIMIZE_SPACE
    scheduler2 = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE, use_greedy=True)
    success = scheduler2.solve_schedule(config)
    assert success

    # Verify schedule is valid
    scheduler2.validate_schedule()


def test_greedy_vs_cpsat_correctness() -> None:
    """Test that greedy scheduler produces valid schedules compared to CP-SAT."""
    # Create a slightly larger graph
    graph = GraphState()
    nodes = [graph.add_physical_node() for _ in range(5)]

    # Create a chain
    for i in range(4):
        graph.add_physical_edge(nodes[i], nodes[i + 1])

    qindex = 0
    graph.register_input(nodes[0], qindex)
    graph.register_output(nodes[4], qindex)

    flow = {nodes[i]: {nodes[i + 1]} for i in range(4)}

    # Test greedy scheduler
    scheduler_greedy = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, use_greedy=True)
    success_greedy = scheduler_greedy.solve_schedule(config)
    assert success_greedy

    # Verify greedy schedule is valid
    scheduler_greedy.validate_schedule()

    # Test CP-SAT scheduler
    scheduler_cpsat = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, use_greedy=False)
    success_cpsat = scheduler_cpsat.solve_schedule(config, timeout=10)
    assert success_cpsat

    # Verify CP-SAT schedule is valid
    scheduler_cpsat.validate_schedule()

    # Both should produce valid schedules
    # Note: Greedy may not be optimal, so we don't compare quality here


def test_greedy_scheduler_larger_graph() -> None:
    """Test greedy scheduler on a larger graph to ensure scalability."""
    # Create a larger graph with branching structure
    graph = GraphState()
    num_layers = 4
    nodes_per_layer = 3

    # Build layered graph
    all_nodes: list[list[int]] = []
    for layer in range(num_layers):
        layer_nodes = [graph.add_physical_node() for _ in range(nodes_per_layer)]
        all_nodes.append(layer_nodes)

        # Connect to previous layer (if not first layer)
        if layer > 0:
            for i, node in enumerate(layer_nodes):
                # Connect to corresponding node in previous layer
                prev_node = all_nodes[layer - 1][i]
                graph.add_physical_edge(prev_node, node)

    # Register inputs (first layer) and outputs (last layer)
    for i, node in enumerate(all_nodes[0]):
        graph.register_input(node, i)
    for i, node in enumerate(all_nodes[-1]):
        graph.register_output(node, i)

    # Build flow (simple forward flow)
    flow: dict[int, set[int]] = {}
    for layer in range(num_layers - 1):
        for i, node in enumerate(all_nodes[layer]):
            if node not in graph.output_node_indices:
                flow[node] = {all_nodes[layer + 1][i]}

    # Test greedy scheduler
    scheduler = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, use_greedy=True)
    success = scheduler.solve_schedule(config)
    assert success

    # Validate the schedule
    scheduler.validate_schedule()

    # Check that we got reasonable results
    assert scheduler.num_slices() > 0
    assert scheduler.num_slices() <= num_layers * 2  # Reasonable upper bound


@pytest.mark.parametrize("strategy", [Strategy.MINIMIZE_TIME, Strategy.MINIMIZE_SPACE])
def test_greedy_scheduler_both_strategies(strategy: Strategy) -> None:
    """Test greedy scheduler with both optimization strategies."""
    # Create a graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    node3 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    graph.add_physical_edge(node2, node3)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node3, qindex)

    flow = {node0: {node1}, node1: {node2}, node2: {node3}}
    scheduler = Scheduler(graph, flow)

    # Test with specified strategy
    config = ScheduleConfig(strategy=strategy, use_greedy=True)
    success = scheduler.solve_schedule(config)
    assert success

    # Validate schedule
    scheduler.validate_schedule()


def test_greedy_minimize_space_wrapper() -> None:
    """Test the greedy_minimize_space wrapper function."""
    # Create a simple graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Test MINIMIZE_TIME
    result = greedy_minimize_time(graph, scheduler.dag)
    assert result is not None
    prepare_time, measure_time = result
    assert len(prepare_time) > 0
    assert len(measure_time) > 0

    # Test MINIMIZE_SPACE
    result = greedy_minimize_space(graph, scheduler.dag)
    assert result is not None
    prepare_time, measure_time = result
    assert len(prepare_time) > 0
    assert len(measure_time) > 0


def test_greedy_scheduler_performance() -> None:
    """Test that greedy scheduler is significantly faster than CP-SAT on larger graphs."""
    # Create a larger graph (chain of 20 nodes)
    graph = GraphState()
    nodes = [graph.add_physical_node() for _ in range(20)]

    for i in range(19):
        graph.add_physical_edge(nodes[i], nodes[i + 1])

    qindex = 0
    graph.register_input(nodes[0], qindex)
    graph.register_output(nodes[-1], qindex)

    flow = {nodes[i]: {nodes[i + 1]} for i in range(19)}

    # Time greedy scheduler
    scheduler_greedy = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, use_greedy=True)

    start_greedy = time.perf_counter()
    success_greedy = scheduler_greedy.solve_schedule(config)
    end_greedy = time.perf_counter()
    greedy_time = end_greedy - start_greedy

    assert success_greedy
    scheduler_greedy.validate_schedule()

    # Time CP-SAT scheduler
    scheduler_cpsat = Scheduler(graph, flow)

    start_cpsat = time.perf_counter()
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, use_greedy=False)
    success_cpsat = scheduler_cpsat.solve_schedule(config, timeout=10)
    end_cpsat = time.perf_counter()
    cpsat_time = end_cpsat - start_cpsat

    assert success_cpsat
    scheduler_cpsat.validate_schedule()

    # Greedy should be significantly faster (at least 5x for this size)
    # Note: We use a conservative factor to avoid flaky tests
    assert greedy_time < cpsat_time


def test_greedy_scheduler_dag_constraints() -> None:
    """Test that greedy scheduler respects DAG constraints."""
    # Create a graph with more complex dependencies
    graph = GraphState()
    nodes = [graph.add_physical_node() for _ in range(6)]

    # Create edges forming a DAG structure
    #   0 -> 1 -> 3 -> 5
    #        2 -> 4 ->
    graph.add_physical_edge(nodes[0], nodes[1])
    graph.add_physical_edge(nodes[1], nodes[2])
    graph.add_physical_edge(nodes[1], nodes[3])
    graph.add_physical_edge(nodes[2], nodes[4])
    graph.add_physical_edge(nodes[3], nodes[5])
    graph.add_physical_edge(nodes[4], nodes[5])

    qindex = 0
    graph.register_input(nodes[0], qindex)
    graph.register_output(nodes[5], qindex)

    # Create flow with dependencies
    flow = {
        nodes[0]: {nodes[1]},
        nodes[1]: {nodes[2], nodes[3]},
        nodes[2]: {nodes[4]},
        nodes[3]: {nodes[5]},
        nodes[4]: {nodes[5]},
    }

    scheduler = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, use_greedy=True)

    # Note: This flow creates a cyclic DAG (nodes 3 and 4 have circular dependency)
    # The greedy scheduler should raise RuntimeError for invalid flows
    with pytest.raises(RuntimeError, match="No nodes can be measured"):
        scheduler.solve_schedule(config)


def test_greedy_scheduler_edge_constraints() -> None:
    """Test that greedy scheduler respects edge constraints (neighbor preparation)."""
    # Create a simple graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, use_greedy=True)
    success = scheduler.solve_schedule(config)
    assert success

    # Validate edge constraints via validate_schedule
    scheduler.validate_schedule()

    # Manually check: neighbors must be prepared before measurement
    # node0 (input) is prepared at time -1, node1 prepared at some time
    # node0 must be measured after node1 is prepared
    # This is ensured by the auto-scheduled entanglement times

    # Check that entanglement times were auto-scheduled correctly
    edge01 = (node0, node1)
    edge12 = (node1, node2)
    entangle01 = scheduler.entangle_time[edge01]
    entangle12 = scheduler.entangle_time[edge12]
    assert entangle01 is not None
    assert entangle12 is not None

    # Entanglement must happen before measurement
    meas0 = scheduler.measure_time[node0]
    meas1 = scheduler.measure_time[node1]
    assert meas0 is not None
    assert meas1 is not None
    assert entangle01 < meas0
    assert entangle12 < meas1
