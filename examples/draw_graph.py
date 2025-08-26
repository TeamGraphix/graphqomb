"""
Graph Visualizer
================

Simple example to draw a GraphState in graphix-zx.
"""

# %%
import numpy as np

from graphix_zx.common import Plane, PlannerMeasBasis
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
q_index1 = demo_graph.register_input(input_node1)
q_index2 = demo_graph.register_input(input_node2)

# Set measurement bases for input nodes (XY plane with different angles)
demo_graph.assign_meas_basis(input_node1, PlannerMeasBasis(Plane.XY, 0.0))
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

# Visualize the demo graph
visualize(demo_graph, save=False, filename="demo_graph.png")
