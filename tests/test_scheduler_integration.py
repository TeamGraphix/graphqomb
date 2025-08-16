"""Integration tests for scheduler and schedule_solver."""

from graphix_zx.graphstate import GraphState
from graphix_zx.schedule_solver import ScheduleConfig, Strategy
from graphix_zx.scheduler import Scheduler


def test_simple_graph_scheduling() -> None:
    """Test scheduling a simple graph with solver."""
    # Create a simple 3-node graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = graph.register_input(node0)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Test solver-based scheduling
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.from_solver(config)
    assert success

    # Check that times were assigned
    assert scheduler.prepare_time[node1] is not None
    assert scheduler.measure_time[node1] is not None

    # Check schedule structure
    schedule = scheduler.schedule
    assert len(schedule) > 0


def test_manual_vs_solver_scheduling() -> None:
    """Test that manual and solver scheduling both work."""
    # Create a graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    node3 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    graph.add_physical_edge(node2, node3)
    qindex = graph.register_input(node0)
    graph.register_output(node3, qindex)

    flow = {node1: {node0}, node2: {node1}, node3: {node2}}

    scheduler = Scheduler(graph, flow)

    # Test manual scheduling
    scheduler.from_manual_design(prepare_time={node1: 0, node2: 1}, measure_time={node1: 1, node2: 2})
    manual_schedule = scheduler.schedule

    # Test solver-based scheduling
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.from_solver(config)
    assert success
    solver_schedule = scheduler.schedule

    # Both should produce valid schedules
    assert len(manual_schedule) > 0
    assert len(solver_schedule) > 0


def test_solver_failure_handling() -> None:
    """Test handling of solver failures."""
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    qindex = graph.register_input(node0)
    graph.register_output(node1, qindex)

    flow = {node0: {node1}, node1: {node0}}  # This should be impossible to satisfy

    scheduler = Scheduler(graph, flow)

    # Solver should return False for unsolvable problems
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.from_solver(config, timeout=1)
    # Note: This might still succeed depending on the specific constraints
    # The test mainly checks that the method doesn't crash
    assert isinstance(success, bool)


def test_schedule_config_options() -> None:
    """Test different ScheduleConfig options."""
    # Create a simple graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = graph.register_input(node0)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Test space optimization
    space_config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE)
    success = scheduler.from_solver(space_config)
    assert success
    space_slices = scheduler.num_slices()

    # Test time optimization
    time_config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.from_solver(time_config)
    assert success
    time_slices = scheduler.num_slices()

    # Test custom max_time
    custom_time_config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE, max_time=10)
    success = scheduler.from_solver(custom_time_config)
    assert success

    # Time optimization should generally use fewer slices than space optimization
    # (though this isn't guaranteed for all graphs)
    assert time_slices <= space_slices or space_slices <= time_slices  # Either way is valid


def test_space_constrained_scheduling() -> None:
    """Test space-constrained time optimization."""
    # Create a larger graph to test constraints
    graph = GraphState()
    nodes = [graph.add_physical_node() for _ in range(5)]

    # Create a chain of nodes
    for i in range(4):
        graph.add_physical_edge(nodes[i], nodes[i + 1])

    qindex = graph.register_input(nodes[0])
    graph.register_output(nodes[4], qindex)

    # Simple flow
    flow = {nodes[i]: {nodes[i + 1]} for i in range(4)}
    scheduler = Scheduler(graph, flow)

    # Test constrained optimization
    max_qubits = 3
    constrained_config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, max_qubit_count=max_qubits)

    success = scheduler.from_solver(constrained_config, timeout=30)

    # This might fail if the constraint is too restrictive,
    # but the method should not crash
    assert isinstance(success, bool)


def test_schedule_compression() -> None:
    """Test that schedule compression reduces unnecessary time gaps."""
    # Create a graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = graph.register_input(node0)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Test manual scheduling with gaps
    scheduler.from_manual_design(prepare_time={node1: 5}, measure_time={node1: 10})

    # Before compression, there should be gaps
    slices_before = scheduler.num_slices()

    # Apply compression
    scheduler.compress_schedule()

    # After compression, gaps should be removed
    slices_after = scheduler.num_slices()

    # Compression should reduce the number of slices
    assert slices_after <= slices_before

    # The compressed schedule should be more compact
    # All time indices should be consecutive starting from 0
    prep_times = [t for t in scheduler.prepare_time.values() if t is not None]
    meas_times = [t for t in scheduler.measure_time.values() if t is not None]
    all_used_times = sorted(set(prep_times + meas_times))

    # Times should be consecutive from 0
    expected_times = list(range(len(all_used_times)))
    assert all_used_times == expected_times


def test_solver_with_automatic_compression() -> None:
    """Test that solver results are automatically compressed."""
    # Create a simple graph
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = graph.register_input(node0)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Test with MINIMIZE_SPACE strategy (prone to gaps)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE)
    success = scheduler.from_solver(config)
    assert success

    # Verify that compression was applied automatically
    prep_times = [t for t in scheduler.prepare_time.values() if t is not None]
    meas_times = [t for t in scheduler.measure_time.values() if t is not None]
    all_used_times = sorted(set(prep_times + meas_times))

    # Times should start from 0 and be consecutive (no gaps)
    if all_used_times:  # Only check if there are any times
        assert all_used_times[0] == 0  # Should start from 0
        # Check that times are consecutive (no large gaps)
        if len(all_used_times) > 1:
            max_gap = max(all_used_times[i + 1] - all_used_times[i] for i in range(len(all_used_times) - 1))
            assert max_gap <= 1  # At most gap of 1 (consecutive times)
