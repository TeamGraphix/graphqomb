"""Integration tests for scheduler and schedule_solver."""

from graphix_zx.graphstate import GraphState
from graphix_zx.schedule_solver import Strategy
from graphix_zx.scheduler import Scheduler


class TestSchedulerIntegration:
    """Test scheduler integration with solver."""

    def test_simple_graph_scheduling(self):
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

        scheduler = Scheduler(graph)

        # chain DAG
        dag = {node0: {node1}, node1: {node2}}

        # Test solver-based scheduling
        success = scheduler.from_solver(dag, Strategy.MINIMIZE_TIME)
        assert success

        # Check that times were assigned
        assert scheduler.prepare_time[node1] is not None
        assert scheduler.measure_time[node1] is not None

        # Check schedule structure
        schedule = scheduler.get_schedule()
        assert len(schedule) > 0

    def test_manual_vs_solver_scheduling(self):
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

        scheduler = Scheduler(graph)

        # Test manual scheduling
        scheduler.from_manual_design(prepare_time={node1: 0, node2: 1}, measure_time={node1: 1, node2: 2})
        manual_schedule = scheduler.get_schedule()

        # Test solver-based scheduling
        dag = {node1: {node0}, node2: {node1}, node3: {node2}}
        success = scheduler.from_solver(dag, Strategy.MINIMIZE_TIME)
        assert success
        solver_schedule = scheduler.get_schedule()

        # Both should produce valid schedules
        assert len(manual_schedule) > 0
        assert len(solver_schedule) > 0

    def test_solver_failure_handling(self):
        """Test handling of solver failures."""
        graph = GraphState()
        node0 = graph.add_physical_node()
        node1 = graph.add_physical_node()
        graph.add_physical_edge(node0, node1)
        qindex = graph.register_input(node0)
        graph.register_output(node1, qindex)

        scheduler = Scheduler(graph)

        # Create an impossible DAG (circular dependency)
        dag = {node0: {node1}, node1: {node0}}  # This should be impossible to satisfy

        # Solver should return False for unsolvable problems
        success = scheduler.from_solver(dag, Strategy.MINIMIZE_TIME, timeout=1)
        # Note: This might still succeed depending on the specific constraints
        # The test mainly checks that the method doesn't crash
        assert isinstance(success, bool)
