"""
Custom Coordinate Visualization
================================

Examples of visualizing GraphState with custom spatial coordinates.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from graphix_zx.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign
from graphix_zx.graphstate import GraphState
from graphix_zx.visualizer import visualize

# %%
# Example 1: Grid Layout
# Create a simple graph with a regular grid layout
print("Example 1: Grid Layout")
print("=" * 50)

grid_graph = GraphState()

# Create a 3x3 grid of nodes
grid_positions = {}
node_grid = {}
for i in range(3):
    for j in range(3):
        node_idx = grid_graph.add_physical_node()
        node_grid[i, j] = node_idx
        grid_positions[node_idx] = (float(i), float(j))

# Register input and output nodes first
grid_graph.register_input(node_grid[0, 0])
grid_graph.register_input(node_grid[0, 1])
grid_graph.register_output(node_grid[2, 1], 0)
grid_graph.register_output(node_grid[2, 2], 1)

# Assign measurement basis only to non-output nodes
for i in range(3):
    for j in range(3):
        node_idx = node_grid[i, j]
        # Skip output nodes
        if node_idx in grid_graph.output_node_indices:
            continue
        # Assign measurement basis
        if (i + j) % 2 == 0:
            grid_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XY, np.pi / 4))
        else:
            grid_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XZ, np.pi / 3))

# Connect neighbors in grid
for i in range(3):
    for j in range(3):
        # Connect to right neighbor
        if i < 2:
            grid_graph.add_physical_edge(node_grid[i, j], node_grid[i + 1, j])
        # Connect to upper neighbor
        if j < 2:
            grid_graph.add_physical_edge(node_grid[i, j], node_grid[i, j + 1])

print(f"Created grid graph with {len(grid_positions)} nodes")
print(f"Input nodes: {list(grid_graph.input_node_indices.keys())}")
print(f"Output nodes: {list(grid_graph.output_node_indices.keys())}")

# Visualize with custom grid coordinates
ax = visualize(grid_graph, node_positions=grid_positions, show_node_labels=True)
plt.title("Grid Layout (Custom Coordinates)")
plt.show()
print("Displayed grid layout\n")

# %%
# Example 2: Circular Layout
# Arrange nodes in a circular pattern
print("Example 2: Circular Layout")
print("=" * 50)

circular_graph = GraphState()

# Create nodes arranged in a circle
n_nodes = 8
circular_positions = {}
nodes = []

for i in range(n_nodes):
    node_idx = circular_graph.add_physical_node()
    nodes.append(node_idx)
    # Calculate position on circle
    angle = 2 * np.pi * i / n_nodes
    x = np.cos(angle) * 2.0
    y = np.sin(angle) * 2.0
    circular_positions[node_idx] = (float(x), float(y))

# Register input/output first
circular_graph.register_input(nodes[0])
circular_graph.register_output(nodes[n_nodes // 2], 0)

# Assign different measurement planes (skip output nodes)
for i in range(n_nodes):
    node_idx = nodes[i]
    if node_idx in circular_graph.output_node_indices:
        continue
    angle = 2 * np.pi * i / n_nodes
    if i % 3 == 0:
        circular_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XY, angle))
    elif i % 3 == 1:
        circular_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.YZ, angle))
    else:
        circular_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XZ, angle))

# Connect nodes in a ring and add some cross connections
for i in range(n_nodes):
    # Connect to next node in circle
    circular_graph.add_physical_edge(nodes[i], nodes[(i + 1) % n_nodes])
    # Connect to opposite node
    if i < n_nodes // 2:
        circular_graph.add_physical_edge(nodes[i], nodes[(i + n_nodes // 2) % n_nodes])

print(f"Created circular graph with {n_nodes} nodes")
print(f"Input nodes: {list(circular_graph.input_node_indices.keys())}")
print(f"Output nodes: {list(circular_graph.output_node_indices.keys())}")

# Visualize with circular coordinates
ax = visualize(circular_graph, node_positions=circular_positions, show_node_labels=True, node_size=400)
plt.title("Circular Layout (Custom Coordinates)")
plt.show()
print("Displayed circular layout\n")

# %%
# Example 3: Layered Layout (Quantum Circuit-like)
# Arrange nodes in layers similar to a quantum circuit
print("Example 3: Layered Layout (Quantum Circuit-like)")
print("=" * 50)

layered_graph = GraphState()

n_qubits = 4
n_layers = 5
layered_positions = {}
layer_nodes = []

for layer in range(n_layers):
    layer_list = []
    for qubit in range(n_qubits):
        node_idx = layered_graph.add_physical_node()
        layer_list.append(node_idx)
        # Position: x = layer, y = -qubit (negative for top-to-bottom)
        layered_positions[node_idx] = (float(layer), -float(qubit))

    layer_nodes.append(layer_list)

# Register inputs (first layer) and outputs (last layer)
for qubit in range(n_qubits):
    layered_graph.register_input(layer_nodes[0][qubit])
    layered_graph.register_output(layer_nodes[-1][qubit], qubit)

# Assign measurement basis based on layer (skip input/output nodes)
for layer in range(n_layers):
    for qubit in range(n_qubits):
        node_idx = layer_nodes[layer][qubit]
        if node_idx in layered_graph.output_node_indices or node_idx in layered_graph.input_node_indices:
            continue
        if layer % 3 == 0:
            layered_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XY, np.pi / 4))
        elif layer % 3 == 1:
            layered_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XZ, np.pi / 3))
        else:
            layered_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.YZ, np.pi / 6))

# Connect horizontally within each layer and vertically between layers
for layer in range(n_layers):
    for qubit in range(n_qubits):
        # Connect to next layer (same qubit)
        if layer < n_layers - 1:
            layered_graph.add_physical_edge(layer_nodes[layer][qubit], layer_nodes[layer + 1][qubit])

        # Connect to adjacent qubit in same layer
        if qubit < n_qubits - 1:
            layered_graph.add_physical_edge(layer_nodes[layer][qubit], layer_nodes[layer][qubit + 1])

print(f"Created layered graph with {n_qubits} qubits and {n_layers} layers")
print(f"Total nodes: {n_qubits * n_layers}")
print(f"Input nodes: {list(layered_graph.input_node_indices.keys())}")
print(f"Output nodes: {list(layered_graph.output_node_indices.keys())}")

# Visualize with layered coordinates
ax = visualize(layered_graph, node_positions=layered_positions, show_node_labels=True, node_size=250)
plt.title("Layered Layout (Circuit-like)")
plt.show()
print("Displayed layered layout\n")

# %%
# Example 4: 3D to 2D Projection
# Simulate 3D coordinates projected onto 2D plane
print("Example 4: 3D to 2D Projection")
print("=" * 50)

projection_graph = GraphState()

# Create nodes with 3D coordinates
n_3d_nodes = 12
nodes_3d = []
coords_3d = []

for i in range(n_3d_nodes):
    node_idx = projection_graph.add_physical_node()
    nodes_3d.append(node_idx)

    # Generate 3D coordinates on a helix
    t = i / n_3d_nodes * 4 * np.pi
    x = np.cos(t) * 2.0
    y = np.sin(t) * 2.0
    z = t / (2 * np.pi)
    coords_3d.append((x, y, z))

# Register input/output first
projection_graph.register_input(nodes_3d[0])
projection_graph.register_output(nodes_3d[-1], 0)

# Assign measurement basis (skip input/output nodes)
for i in range(n_3d_nodes):
    node_idx = nodes_3d[i]
    if node_idx in projection_graph.output_node_indices or node_idx in projection_graph.input_node_indices:
        continue
    t = i / n_3d_nodes * 4 * np.pi
    if i % 4 == 0:
        projection_graph.assign_meas_basis(node_idx, AxisMeasBasis(Axis.X, Sign.PLUS))
    elif i % 4 == 1:
        projection_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.YZ, t))
    elif i % 4 == 2:
        projection_graph.assign_meas_basis(node_idx, AxisMeasBasis(Axis.Z, Sign.MINUS))
    else:
        projection_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XY, t))

# Connect adjacent nodes in the helix
for i in range(n_3d_nodes - 1):
    projection_graph.add_physical_edge(nodes_3d[i], nodes_3d[i + 1])

# Add some cross connections
for i in range(0, n_3d_nodes - 3, 3):
    projection_graph.add_physical_edge(nodes_3d[i], nodes_3d[i + 3])

# Project to 2D using different projections
# XY projection (view from top)
xy_positions = {node: (coords_3d[i][0], coords_3d[i][1]) for i, node in enumerate(nodes_3d)}

# XZ projection (view from side)
xz_positions = {node: (coords_3d[i][0], coords_3d[i][2]) for i, node in enumerate(nodes_3d)}

# Isometric projection
iso_positions = {}
for i, node in enumerate(nodes_3d):
    x, y, z = coords_3d[i]
    # Isometric projection formula
    iso_x = x - z
    iso_y = y + (x + z) / 2
    iso_positions[node] = (iso_x, iso_y)

print(f"Created 3D helix graph with {n_3d_nodes} nodes")
print(f"Input nodes: {list(projection_graph.input_node_indices.keys())}")
print(f"Output nodes: {list(projection_graph.output_node_indices.keys())}")
print("Showing three different 2D projections of the same 3D structure")

# Create figure with subplots for different projections
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

visualize(projection_graph, ax=axes[0], node_positions=xy_positions, show_node_labels=True, node_size=300)
axes[0].set_title("XY Projection (Top View)")

visualize(projection_graph, ax=axes[1], node_positions=xz_positions, show_node_labels=True, node_size=300)
axes[1].set_title("XZ Projection (Side View)")

visualize(projection_graph, ax=axes[2], node_positions=iso_positions, show_node_labels=True, node_size=300)
axes[2].set_title("Isometric Projection")

plt.tight_layout()
plt.show()
print("Displayed 3D projections\n")

# %%
# Example 5: Comparison with Automatic Layout
# Compare custom layout with automatic layout
print("Example 5: Comparison with Automatic Layout")
print("=" * 50)

comparison_graph = GraphState()

# Create a simple graph
nodes_comp = []
for _ in range(6):
    node_idx = comparison_graph.add_physical_node()
    nodes_comp.append(node_idx)

# Register input/output
comparison_graph.register_input(nodes_comp[0])
comparison_graph.register_output(nodes_comp[-1], 0)

# Assign measurement basis (skip input/output nodes)
for i in range(6):
    node_idx = nodes_comp[i]
    if node_idx in comparison_graph.output_node_indices or node_idx in comparison_graph.input_node_indices:
        continue
    if i % 2 == 0:
        comparison_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XZ, np.pi / 4))
    else:
        comparison_graph.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.YZ, np.pi / 3))

# Create edges
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (1, 4)]
for src, dst in edges:
    comparison_graph.add_physical_edge(nodes_comp[src], nodes_comp[dst])

# Define custom positions (tree-like layout)
custom_positions = {
    nodes_comp[0]: (0.0, 0.0),
    nodes_comp[1]: (1.0, 1.0),
    nodes_comp[2]: (1.0, -1.0),
    nodes_comp[3]: (2.0, 0.0),
    nodes_comp[4]: (3.0, 0.5),
    nodes_comp[5]: (4.0, 0.0),
}

print(f"Created comparison graph with {len(nodes_comp)} nodes")
print(f"Edges: {edges}")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Automatic layout
visualize(comparison_graph, ax=axes[0], show_node_labels=True, node_size=400)
axes[0].set_title("Automatic Layout")

# Custom layout
visualize(comparison_graph, ax=axes[1], node_positions=custom_positions, show_node_labels=True, node_size=400)
axes[1].set_title("Custom Layout")

plt.tight_layout()
plt.show()
print("Displayed automatic vs custom layout comparison\n")

# %%
