"""
Graph Visualizer
================

Simple example to draw a GraphState in graphix-zx.
"""

# %%
import numpy as np

from graphix_zx.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign
from graphix_zx.graphstate import GraphState
from graphix_zx.random_objects import generate_random_flow_graph
from graphix_zx.visualizer import visualize

# Create a random flow graph
random_graph, flow = generate_random_flow_graph(5, 5)


# Visualize the flow graph
visualize(random_graph, save=False, filename="flow_graph.png")

# %%
# Create a demo graph with different measurement planes and input/output nodes

demo_graph = GraphState()

# Add input nodes
input_node1 = demo_graph.add_physical_node()
input_node2 = demo_graph.add_physical_node()
demo_graph.register_input(input_node1, 0)
demo_graph.register_input(input_node2, 1)

# Set measurement bases for input nodes (XY plane with different angles)
demo_graph.assign_meas_basis(input_node1, AxisMeasBasis(Axis.X, Sign.PLUS))
demo_graph.assign_meas_basis(input_node2, PlannerMeasBasis(Plane.XY, np.pi / 6))

# Add internal nodes with different measurement planes
internal_node1 = demo_graph.add_physical_node()
internal_node2 = demo_graph.add_physical_node()
internal_node3 = demo_graph.add_physical_node()

# Set measurement bases for internal nodes
# XZ plane (blue) with angle π/4
demo_graph.assign_meas_basis(internal_node1, PlannerMeasBasis(Plane.XZ, np.pi / 4))
# YZ plane (red) with angle π/3
demo_graph.assign_meas_basis(internal_node2, PlannerMeasBasis(Plane.YZ, np.pi / 3))
# XZ plane (blue) with angle π/2
demo_graph.assign_meas_basis(internal_node3, PlannerMeasBasis(Plane.XZ, np.pi / 2))

# Add output nodes
output_node1 = demo_graph.add_physical_node()
output_node2 = demo_graph.add_physical_node()
demo_graph.register_output(output_node1, 0)
demo_graph.register_output(output_node2, 1)

# Create edges to connect the graph
demo_graph.add_physical_edge(input_node1, internal_node1)
demo_graph.add_physical_edge(input_node2, internal_node2)
demo_graph.add_physical_edge(internal_node1, internal_node3)
demo_graph.add_physical_edge(internal_node2, internal_node3)
demo_graph.add_physical_edge(internal_node3, output_node1)
demo_graph.add_physical_edge(internal_node1, output_node2)

print("Demo graph with XZ and YZ measurement planes:")
print(f"Input nodes: {list(demo_graph.input_node_indices.keys())}")
print(f"Output nodes: {list(demo_graph.output_node_indices.keys())}")
print(f"All physical nodes: {demo_graph.physical_nodes}")
print("Internal nodes with measurement bases:")
for node, basis in demo_graph.meas_bases.items():
    print(f"  Node {node}: {basis.plane.name} plane, angle={basis.angle:.3f}")

# Visualize the demo graph with labels
visualize(demo_graph, save=False, filename="demo_graph.png", show_node_labels=True)

# Visualize without labels to see just the colored patterns
print("\n--- Same graph without node labels ---")
visualize(demo_graph, save=False, filename="demo_graph_no_labels.png", show_node_labels=False)

# %%
# Create another demo graph with Pauli measurements (θ=0, π)
pauli_demo_graph = GraphState()

# Add nodes for Pauli measurements
pauli_input = pauli_demo_graph.add_physical_node()
pauli_demo_graph.register_input(pauli_input, 0)

# Create internal nodes with Pauli measurements
x_measurement_node = pauli_demo_graph.add_physical_node()  # X measurement: XY plane, θ=0
y_measurement_node = pauli_demo_graph.add_physical_node()  # Y measurement: YZ plane, θ=π/2
z_measurement_node = pauli_demo_graph.add_physical_node()  # Z measurement: XZ plane, θ=π

# Set Pauli measurement bases
pauli_demo_graph.assign_meas_basis(pauli_input, AxisMeasBasis(Axis.X, Sign.PLUS))  # X+
pauli_demo_graph.assign_meas_basis(x_measurement_node, AxisMeasBasis(Axis.X, Sign.PLUS))  # X+
pauli_demo_graph.assign_meas_basis(y_measurement_node, AxisMeasBasis(Axis.Y, Sign.PLUS))  # Y+
pauli_demo_graph.assign_meas_basis(z_measurement_node, AxisMeasBasis(Axis.Z, Sign.MINUS))  # Z-

# Add output node
pauli_output = pauli_demo_graph.add_physical_node()
pauli_demo_graph.register_output(pauli_output, 0)

# Connect nodes
pauli_demo_graph.add_physical_edge(pauli_input, x_measurement_node)
pauli_demo_graph.add_physical_edge(x_measurement_node, y_measurement_node)
pauli_demo_graph.add_physical_edge(y_measurement_node, z_measurement_node)
pauli_demo_graph.add_physical_edge(z_measurement_node, pauli_output)

print("\\nPauli measurement demo graph:")
print(f"Input nodes: {list(pauli_demo_graph.input_node_indices.keys())}")
print(f"Output nodes: {list(pauli_demo_graph.output_node_indices.keys())}")
print("Pauli measurement nodes (will show bordered patterns):")
print("  - X measurement (θ=0°): Green center + Blue border (XY+XZ planes)")
print("  - Y measurement (θ=90°): Red center + Green border (YZ+XY planes)")
print("  - Z measurement (θ=180°): Blue center + Red border (XZ+YZ planes)")
print("Individual nodes:")
for node, basis in pauli_demo_graph.meas_bases.items():
    plane_name = basis.plane.name
    angle_deg = basis.angle * 180 / np.pi
    print(f"  Node {node}: {plane_name} plane, angle={basis.angle:.3f} ({angle_deg:.1f}°)")

# Visualize the Pauli demo graph (using bordered-node visualization)
visualize(pauli_demo_graph, save=False, filename="pauli_demo_graph.png", show_node_labels=True)

# Demo with larger nodes and no labels
print("\n--- Larger nodes without labels ---")
visualize(pauli_demo_graph, save=False, filename="pauli_demo_large.png", show_node_labels=False)

# Demo without legend to avoid overlap
print("\n--- Without legend to avoid overlap ---")
visualize(pauli_demo_graph, save=False, filename="pauli_demo_no_legend.png", show_node_labels=True, show_legend=False)

# %%
