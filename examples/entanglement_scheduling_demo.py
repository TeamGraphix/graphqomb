"""Entanglement Scheduling and TICK Command Demo
================================================

This example demonstrates the new entanglement scheduling functionality
and TICK commands in graphqomb.
"""

from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.pattern import print_pattern
from graphqomb.qompiler import qompile
from graphqomb.schedule_solver import ScheduleConfig, Strategy
from graphqomb.scheduler import Scheduler

# Create a simple graph state
print("=== Entanglement Scheduling Demo ===\n")
print("1. Creating graph state...")
graph = GraphState()
node0 = graph.add_physical_node()
node1 = graph.add_physical_node()
node2 = graph.add_physical_node()
node3 = graph.add_physical_node()

# Add edges
graph.add_physical_edge(node0, node1)
graph.add_physical_edge(node1, node2)
graph.add_physical_edge(node2, node3)

# Set input/output
qindex = 0
graph.register_input(node0, qindex)
graph.register_output(node3, qindex)

# Set measurement bases for non-output nodes
graph.assign_meas_basis(node0, PlannerMeasBasis(Plane.XY, 0.0))
graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))
graph.assign_meas_basis(node2, PlannerMeasBasis(Plane.XY, 0.0))

print(f"   Nodes: {list(graph.physical_nodes)}")
print(f"   Input: {list(graph.input_node_indices.keys())}")
print(f"   Output: {list(graph.output_node_indices.keys())}")
print(f"   Edges: {list(graph.physical_edges)}")

# Define flow
flow = {node0: {node1}, node1: {node2}, node2: {node3}}

# Create scheduler
print("\n2. Creating scheduler and solving schedule...")
scheduler = Scheduler(graph, flow)
config = ScheduleConfig(strategy=Strategy.MINIMIZE_SPACE)
success = scheduler.solve_schedule(config)

if success:
    print(f"   Scheduling successful!")
    print(f"   Number of time slices: {scheduler.num_slices()}")

    # Show preparation times
    prep_times = {k: v for k, v in scheduler.prepare_time.items() if v is not None}
    print(f"   Preparation times: {prep_times}")

    # Show measurement times
    meas_times = {k: v for k, v in scheduler.measure_time.items() if v is not None}
    print(f"   Measurement times: {meas_times}")

    # Auto-schedule entanglement
    print("\n3. Auto-scheduling entanglement operations...")
    scheduler.auto_schedule_entanglement()

    # Show entanglement times
    ent_times = {
        tuple(sorted(edge)): time
        for edge, time in scheduler.entangle_time.items()
        if time is not None
    }
    print(f"   Entanglement times: {ent_times}")

    # Show detailed timeline
    print("\n4. Detailed timeline (Prep, Entangle, Measure):")
    timeline = scheduler.detailed_timeline
    for time_idx, (prep_nodes, ent_edges, meas_nodes) in enumerate(timeline):
        print(f"   Time {time_idx}:")
        if prep_nodes:
            print(f"     Prepare: {sorted(prep_nodes)}")
        if ent_edges:
            edges_str = [f"({min(e)},{max(e)})" for e in ent_edges]
            print(f"     Entangle: {', '.join(edges_str)}")
        if meas_nodes:
            print(f"     Measure: {sorted(meas_nodes)}")

    # Compile pattern with TICK commands
    print("\n5. Compiling pattern with TICK commands...")
    pattern = qompile(graph, flow, scheduler=scheduler, insert_tick=True)
    print(f"   Pattern has {len(pattern.commands)} commands")
    print(f"   Maximum space usage: {pattern.max_space} qubits")

    print("\n   Pattern commands:")
    print_pattern(pattern, lim=30)

    # Compile pattern without TICK commands
    print("\n6. Compiling pattern without TICK commands...")
    pattern_no_tick = qompile(graph, flow, scheduler=scheduler, insert_tick=False)
    print(f"   Pattern has {len(pattern_no_tick.commands)} commands")

    print("\n   Pattern commands (first 15):")
    print_pattern(pattern_no_tick, lim=15)

    # Manual entanglement scheduling example
    print("\n7. Manual entanglement scheduling example...")
    scheduler2 = Scheduler(graph, flow)
    scheduler2.manual_schedule(
        prepare_time={node1: 0, node2: 1, node3: 2},
        measure_time={node0: 0, node1: 1, node2: 2},
        entangle_time={
            frozenset({node0, node1}): 0,
            frozenset({node1, node2}): 1,
            frozenset({node2, node3}): 2,
        },
    )

    print(f"   Manual schedule is valid: {scheduler2.validate_schedule()}")
    print(f"   Number of time slices: {scheduler2.num_slices()}")

    pattern_manual = qompile(graph, flow, scheduler=scheduler2, insert_tick=True)
    print(f"   Pattern has {len(pattern_manual.commands)} commands")
    print(f"   Maximum space usage: {pattern_manual.max_space} qubits")

print("\nDemo completed successfully!")
