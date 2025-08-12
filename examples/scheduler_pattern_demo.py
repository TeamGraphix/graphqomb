"""
Scheduler-based Pattern Generation Demo
=======================================

This example demonstrates how to use the scheduler to generate optimized
measurement patterns from graph states using different scheduling strategies.
"""

# %%
from graphix_zx.pattern import print_pattern
from graphix_zx.qompiler import qompile
from graphix_zx.random_objects import generate_random_flow_graph
from graphix_zx.schedule_solver import Strategy
from graphix_zx.scheduler import Scheduler

# %%
# Create sample graph state and flow
print("=== Scheduler-based Pattern Generation Demo ===\n")
print("1. Creating sample graph state...")
graph, xflow = generate_random_flow_graph(width=3, depth=3, edge_p=0.7)
print(f"   Graph has {len(graph.physical_nodes)} nodes")
print(f"   Input nodes: {list(graph.input_node_indices.keys())}")
print(f"   Output nodes: {list(graph.output_node_indices.keys())}")
print(f"   Edges: {len(graph.physical_edges)}")

# %%
# Demonstrate space-optimized scheduling
print("\n2. Space-optimized Scheduling:")
print("=" * 40)

scheduler_space = Scheduler(graph, xflow)
success_space = scheduler_space.from_solver(Strategy.MINIMIZE_SPACE, timeout=10)
pattern_space = None

if success_space:
    print("   Scheduling successful!")
    print(f"   Number of time slices: {scheduler_space.num_slices()}")

    prep_times = {k: v for k, v in scheduler_space.prepare_time.items() if v is not None}
    if prep_times:
        print(f"   Preparation times: {prep_times}")

    meas_times = {k: v for k, v in scheduler_space.measure_time.items() if v is not None}
    if meas_times:
        print(f"   Measurement times: {meas_times}")

    print("\n   Generated Pattern (Space-optimized):")
    pattern_space = qompile(graph, xflow, scheduler=scheduler_space)
    print(f"   Pattern has {len(pattern_space.commands)} commands")
    print(f"   Maximum space usage: {pattern_space.max_space} qubits")
    print(f"   Space usage over time: {pattern_space.space}")

    print("\n   Pattern commands:")
    print_pattern(pattern_space, lim=10)
else:
    print("   Failed to find solution for Space-optimized strategy")

# %%
# Demonstrate time-optimized scheduling
print("\n3. Time-optimized Scheduling:")
print("=" * 40)

scheduler_time = Scheduler(graph, xflow)
success_time = scheduler_time.from_solver(Strategy.MINIMIZE_TIME, timeout=10)
pattern_time = None

if success_time:
    print("   Scheduling successful!")
    print(f"   Number of time slices: {scheduler_time.num_slices()}")

    prep_times = {k: v for k, v in scheduler_time.prepare_time.items() if v is not None}
    if prep_times:
        print(f"   Preparation times: {prep_times}")

    meas_times = {k: v for k, v in scheduler_time.measure_time.items() if v is not None}
    if meas_times:
        print(f"   Measurement times: {meas_times}")

    print("\n   Generated Pattern (Time-optimized):")
    pattern_time = qompile(graph, xflow, scheduler=scheduler_time)
    print(f"   Pattern has {len(pattern_time.commands)} commands")
    print(f"   Maximum space usage: {pattern_time.max_space} qubits")
    print(f"   Space usage over time: {pattern_time.space}")

    print("\n   Pattern commands:")
    print_pattern(pattern_time, lim=10)
else:
    print("   Failed to find solution for Time-optimized strategy")

# %%
# Compare strategies
print("\n4. Strategy Comparison:")
print("=" * 40)

if success_space and success_time and pattern_space and pattern_time:
    print(
        f"   Space-optimized: {scheduler_space.num_slices()} slices, "
        f"{pattern_space.max_space} max qubits, {len(pattern_space.commands)} commands"
    )
    print(
        f"   Time-optimized:  {scheduler_time.num_slices()} slices, "
        f"{pattern_time.max_space} max qubits, {len(pattern_time.commands)} commands"
    )

    if pattern_space.max_space < pattern_time.max_space:
        print("   → Space optimization reduced qubit usage")
    if scheduler_time.num_slices() < scheduler_space.num_slices():
        print("   → Time optimization reduced execution time")

print("\nDemo completed successfully!")
