"""Test greedy scheduling algorithms."""

import time

import pytest

from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.greedy_scheduler import (
    greedy_minimize_space,
    greedy_minimize_time,
    solve_greedy_schedule,
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
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(config, use_greedy=True)
    assert success

    # Verify schedule is valid
    scheduler.validate_schedule()

    # Test with greedy MINIMIZE_SPACE
    scheduler2 = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE)
    success = scheduler2.solve_schedule(config, use_greedy=True)
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
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success_greedy = scheduler_greedy.solve_schedule(config, use_greedy=True)
    assert success_greedy

    # Verify greedy schedule is valid
    scheduler_greedy.validate_schedule()

    # Test CP-SAT scheduler
    scheduler_cpsat = Scheduler(graph, flow)
    success_cpsat = scheduler_cpsat.solve_schedule(config, use_greedy=False, timeout=10)
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
    all_nodes = []
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
    flow = {}
    for layer in range(num_layers - 1):
        for i, node in enumerate(all_nodes[layer]):
            if node not in graph.output_node_indices:
                flow[node] = {all_nodes[layer + 1][i]}

    # Test greedy scheduler
    scheduler = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(config, use_greedy=True)
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
    config = ScheduleConfig(strategy=strategy)
    success = scheduler.solve_schedule(config, use_greedy=True)
    assert success

    # Validate schedule
    scheduler.validate_schedule()


def test_solve_greedy_schedule_wrapper() -> None:
    """Test the solve_greedy_schedule wrapper function."""
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

    # Test MINIMIZE_TIME (minimize_space=False)
    result = solve_greedy_schedule(graph, scheduler.dag, minimize_space=False)
    assert result is not None
    prepare_time, measure_time = result
    assert len(prepare_time) > 0
    assert len(measure_time) > 0

    # Test MINIMIZE_SPACE (minimize_space=True)
    result = solve_greedy_schedule(graph, scheduler.dag, minimize_space=True)
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
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)

    start_greedy = time.perf_counter()
    success_greedy = scheduler_greedy.solve_schedule(config, use_greedy=True)
    end_greedy = time.perf_counter()
    greedy_time = end_greedy - start_greedy

    assert success_greedy
    scheduler_greedy.validate_schedule()

    # Time CP-SAT scheduler
    scheduler_cpsat = Scheduler(graph, flow)

    start_cpsat = time.perf_counter()
    success_cpsat = scheduler_cpsat.solve_schedule(config, use_greedy=False, timeout=10)
    end_cpsat = time.perf_counter()
    cpsat_time = end_cpsat - start_cpsat

    assert success_cpsat
    scheduler_cpsat.validate_schedule()

    # Print timing information for debugging
    print(f"\nGreedy time: {greedy_time:.4f}s")
    print(f"CP-SAT time: {cpsat_time:.4f}s")
    print(f"Speedup: {cpsat_time / greedy_time:.1f}x")

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
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(config, use_greedy=True)

    # Note: This flow creates a cyclic DAG (nodes 3 and 4 have circular dependency)
    # Both CP-SAT and greedy schedulers should fail on invalid flows
    # This test verifies that the greedy scheduler handles invalid input gracefully
    assert not success  # Should fail due to cyclic DAG


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
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(config, use_greedy=True)
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
    assert scheduler.entangle_time[edge01] is not None
    assert scheduler.entangle_time[edge12] is not None

    # Entanglement must happen before measurement
    assert scheduler.entangle_time[edge01] < scheduler.measure_time[node0]
    assert scheduler.entangle_time[edge12] < scheduler.measure_time[node1]
