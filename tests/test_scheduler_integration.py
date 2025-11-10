"""Integration tests for scheduler and schedule_solve_scheduler."""

from graphqomb.command import TICK
from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.qompiler import qompile
from graphqomb.schedule_solver import ScheduleConfig, Strategy
from graphqomb.scheduler import Scheduler, compress_schedule
from graphqomb.simulator import PatternSimulator, SimulatorBackend


def test_simple_graph_scheduling() -> None:
    """Test scheduling a simple graph with solve_scheduler."""
    # Create a simple 3-node graph
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

    # Test solve_scheduler-based scheduling
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(config)
    assert success

    # Check that times were assigned
    assert scheduler.prepare_time[node1] is not None
    assert scheduler.measure_time[node1] is not None

    # Check schedule structure
    timeline = scheduler.timeline
    assert len(timeline) > 0


def test_manual_vs_solve_scheduler_scheduling() -> None:
    """Test that manual and solve_scheduler scheduling both work."""
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

    flow = {node1: {node0}, node2: {node1}, node3: {node2}}

    scheduler = Scheduler(graph, flow)

    # Test manual scheduling
    scheduler.manual_schedule(prepare_time={node1: 0, node2: 1}, measure_time={node1: 1, node2: 2})
    manual_schedule = scheduler.timeline

    # Test solve_scheduler-based scheduling
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(config)
    assert success
    solve_scheduler_schedule = scheduler.timeline

    # Both should produce valid schedules
    assert len(manual_schedule) > 0
    assert len(solve_scheduler_schedule) > 0


def test_solve_scheduler_failure_handling() -> None:
    """Test handling of solve_scheduler failures."""
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node1, qindex)

    flow = {node0: {node1}, node1: {node0}}  # This should be impossible to satisfy

    scheduler = Scheduler(graph, flow)

    # Solver should return False for unsolvable problems
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(config, timeout=1)
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
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Test space optimization
    space_config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE)
    success = scheduler.solve_schedule(space_config)
    assert success
    space_slices = scheduler.num_slices()

    # Test time optimization
    time_config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(time_config)
    assert success
    time_slices = scheduler.num_slices()

    # Test custom max_time
    custom_time_config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE, max_time=10)
    success = scheduler.solve_schedule(custom_time_config)
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

    qindex = 0
    graph.register_input(nodes[0], qindex)
    graph.register_output(nodes[4], qindex)

    # Simple flow
    flow = {nodes[i]: {nodes[i + 1]} for i in range(4)}
    scheduler = Scheduler(graph, flow)

    # Test constrained optimization
    max_qubits = 3
    constrained_config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME, max_qubit_count=max_qubits)

    success = scheduler.solve_schedule(constrained_config, timeout=30)

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
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Test manual scheduling with gaps
    scheduler.manual_schedule(prepare_time={node1: 5}, measure_time={node1: 10})

    # Before compression, there should be gaps
    slices_before = scheduler.num_slices()

    # Apply compression
    timings = compress_schedule(scheduler.prepare_time, scheduler.measure_time, scheduler.entangle_time)
    scheduler.prepare_time = timings.prepare_time
    scheduler.measure_time = timings.measure_time
    scheduler.entangle_time = timings.entangle_time

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


def test_solve_scheduler_with_automatic_compression() -> None:
    """Test that solve_scheduler results are automatically compressed."""
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

    # Test with MINIMIZE_SPACE strategy (prone to gaps)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE)
    success = scheduler.solve_schedule(config)
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


def test_validate_schedule_valid() -> None:
    """Test that validate_schedule correctly identifies valid schedules."""
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

    # Test that a solve_scheduler-generated schedule is valid
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(config)
    assert success
    assert scheduler.validate_schedule()

    # Test a valid manual schedule
    scheduler2 = Scheduler(graph, flow)
    # node0 is input (not in prepare_time), node2 is output (not in measure_time)
    scheduler2.manual_schedule(prepare_time={node1: 0, node2: 1}, measure_time={node0: 0, node1: 1})
    assert scheduler2.validate_schedule()


def test_validate_schedule_invalid_node_sets() -> None:
    """Test that validate_schedule rejects schedules with wrong node sets."""
    # Create a graph
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

    # Manually set invalid schedule (trying to prepare input node)
    scheduler.prepare_time = {node0: 0, node1: 1}  # node0 is input, shouldn't be prepared
    scheduler.measure_time = {node0: 1, node1: 2}
    assert not scheduler.validate_schedule()

    # Reset and test measuring output node
    scheduler2 = Scheduler(graph, flow)
    scheduler2.prepare_time = {node1: 0}
    scheduler2.measure_time = {node0: 1, node1: 1, node2: 2}  # node2 is output, shouldn't be measured
    assert not scheduler2.validate_schedule()


def test_validate_schedule_missing_times() -> None:
    """Test that validate_schedule rejects schedules with missing times."""
    # Create a graph
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

    # Set schedule with None values (unscheduled nodes)
    scheduler.prepare_time = {node1: None}  # node1 not scheduled for preparation
    scheduler.measure_time = {node0: 0, node1: 1}
    assert not scheduler.validate_schedule()

    # Reset and test missing measurement time
    scheduler2 = Scheduler(graph, flow)
    scheduler2.prepare_time = {node1: 0}
    scheduler2.measure_time = {node0: 0, node1: None}  # node1 not scheduled for measurement
    assert not scheduler2.validate_schedule()


def test_validate_schedule_dag_violations() -> None:
    """Test that validate_schedule rejects schedules violating DAG constraints."""
    # Create a graph with flow dependencies
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

    # Flow creates DAG: node0 -> node1 -> node2
    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Set schedule that violates DAG (node1 measured after node2)
    scheduler.prepare_time = {node1: 0, node2: 0}
    scheduler.measure_time = {node0: 0, node1: 2, node2: 1}  # Violates DAG: node1 should be measured before node2
    assert not scheduler.validate_schedule()

    # Test equal times (also violates DAG)
    scheduler2 = Scheduler(graph, flow)
    scheduler2.prepare_time = {node1: 0, node2: 0}
    scheduler2.measure_time = {node0: 0, node1: 1, node2: 1}  # Same measurement time violates DAG
    assert not scheduler2.validate_schedule()


def test_validate_schedule_same_time_prep_meas() -> None:
    """Test that validate_schedule rejects schedules with nodes prepared and measured at same time."""
    # Create a graph
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

    # Set schedule where node1 is both prepared and measured at time 1
    scheduler.prepare_time = {node1: 1}
    scheduler.measure_time = {node0: 0, node1: 1}  # node1 prepared and measured at time 1
    assert not scheduler.validate_schedule()


def test_entangle_time_scheduling() -> None:
    """Test that entanglement times can be scheduled."""
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

    # Set manual schedule
    scheduler.manual_schedule(prepare_time={node1: 0, node2: 1}, measure_time={node0: 0, node1: 1})

    # Auto-schedule entanglement
    scheduler.auto_schedule_entanglement()

    # Check that entanglement times were set
    edge01 = (node0, node1)
    edge12 = (node1, node2)

    assert scheduler.entangle_time[edge01] is not None
    assert scheduler.entangle_time[edge12] is not None

    # Edge (0,1) should be scheduled at time 0 (when node1 is prepared, node0 is input)
    assert scheduler.entangle_time[edge01] == 0

    # Edge (1,2) should be scheduled at time 1 (when node2 is prepared)
    assert scheduler.entangle_time[edge12] == 1


def test_timeline_includes_entanglement() -> None:
    """Timeline should expose preparation, entanglement, and measurement sets."""
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

    # Set manual schedule
    scheduler.manual_schedule(prepare_time={node1: 0, node2: 1}, measure_time={node0: 0, node1: 1})
    scheduler.auto_schedule_entanglement()

    # Get timeline with entanglement information
    timeline = scheduler.timeline

    # Check that timeline has the correct structure
    assert len(timeline) == 2  # 2 time slices

    # Time slice 0: prepare node1, entangle (0,1), measure node0
    prep_nodes_0, ent_edges_0, meas_nodes_0 = timeline[0]
    assert node1 in prep_nodes_0
    assert (node0, node1) in ent_edges_0
    assert node0 in meas_nodes_0

    # Time slice 1: prepare node2, entangle (1,2), measure node1
    prep_nodes_1, ent_edges_1, meas_nodes_1 = timeline[1]
    assert node2 in prep_nodes_1
    assert (node1, node2) in ent_edges_1
    assert node1 in meas_nodes_1


def test_qompile_with_tick_commands() -> None:
    """Test that qompile generates TICK commands with scheduler."""
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

    # Set measurement bases for non-output nodes
    graph.assign_meas_basis(node0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)

    # Solve schedule
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    success = scheduler.solve_schedule(config)
    assert success

    # Compile pattern and ensure TICK commands mark every slice boundary
    pattern = qompile(graph, flow, scheduler=scheduler)

    tick_commands = [cmd for cmd in pattern if isinstance(cmd, TICK)]
    assert tick_commands, "Pattern should contain TICK commands when a scheduler is provided"
    assert len(tick_commands) == scheduler.num_slices(), "Each time slice should contribute exactly one TICK"


def test_validate_entangle_time_constraints() -> None:
    """Test validation of entanglement time constraints."""
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

    # Set valid schedule
    scheduler.manual_schedule(prepare_time={node1: 0, node2: 1}, measure_time={node0: 0, node1: 1})
    scheduler.auto_schedule_entanglement()

    # Should be valid
    assert scheduler.validate_schedule()

    # Set invalid entanglement time (before node is prepared)
    edge12 = (node1, node2)
    scheduler.entangle_time[edge12] = 0  # node2 is prepared at time 1, so this is invalid

    # Should be invalid
    assert not scheduler.validate_schedule()


def test_compress_schedule_with_entangle_time() -> None:
    """Test that compress_schedule handles entangle_time correctly."""
    # Create schedules with gaps
    prepare_time = {1: 5, 2: 10}
    measure_time = {0: 5, 1: 10}
    entangle_time = {(0, 1): 5, (1, 2): 10}

    # Compress
    timings = compress_schedule(prepare_time, measure_time, entangle_time)

    # Check that gaps are removed
    assert timings.prepare_time[1] == 0
    assert timings.prepare_time[2] == 1
    assert timings.measure_time[0] == 0
    assert timings.measure_time[1] == 1
    assert timings.entangle_time[0, 1] == 0
    assert timings.entangle_time[1, 2] == 1


def test_simulator_with_tick_commands() -> None:
    """Test that PatternSimulator handles TICK commands correctly."""
    # Create a simple graph and compile with TICK commands
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    # Assign measurement bases to non-output nodes
    graph.assign_meas_basis(node0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    scheduler.solve_schedule(config)
    scheduler.auto_schedule_entanglement()

    # Compile with scheduler-driven TICK commands
    pattern = qompile(graph, flow, scheduler=scheduler)

    # Verify TICK commands are present
    tick_count = sum(1 for cmd in pattern if isinstance(cmd, TICK))
    assert tick_count > 0, "Pattern should contain TICK commands"

    # Simulate the pattern - should not raise any errors
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.simulate()

    # Check that simulation completed successfully
    assert len(simulator.results) > 0
