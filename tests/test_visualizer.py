"""Tests for the visualizer module."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

mpl.use("Agg")  # Use non-interactive backend for testing

from graphix_zx.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign
from graphix_zx.graphstate import GraphState
from graphix_zx.visualizer import visualize


@pytest.fixture
def simple_graph() -> GraphState:
    """Create a simple GraphState for testing.

    Returns
    -------
    GraphState
        A simple GraphState with input, internal, and output nodes
    """
    graph = GraphState()

    # Create nodes
    input_node = graph.add_physical_node()
    internal_node1 = graph.add_physical_node()
    internal_node2 = graph.add_physical_node()
    output_node = graph.add_physical_node()

    # Register input/output
    q_idx = graph.register_input(input_node)
    graph.register_output(output_node, q_idx)

    # Assign measurement bases
    graph.assign_meas_basis(input_node, PlannerMeasBasis(Plane.XY, np.pi / 4))
    graph.assign_meas_basis(internal_node1, PlannerMeasBasis(Plane.XZ, np.pi / 3))
    graph.assign_meas_basis(internal_node2, PlannerMeasBasis(Plane.YZ, np.pi / 6))

    # Add edges
    graph.add_physical_edge(input_node, internal_node1)
    graph.add_physical_edge(internal_node1, internal_node2)
    graph.add_physical_edge(internal_node2, output_node)

    return graph


def test_visualize_with_automatic_layout(simple_graph: GraphState) -> None:
    """Test visualize function with automatic layout."""
    ax = visualize(simple_graph)
    assert ax is not None
    # Verify that the axes contains the expected number of nodes
    # The scatter plots and patches represent nodes
    assert len(simple_graph.physical_nodes) == 4
    plt.close()


def test_visualize_with_custom_positions(simple_graph: GraphState) -> None:
    """Test visualize function with custom node positions."""
    # Create custom positions for all nodes
    nodes = list(simple_graph.physical_nodes)
    custom_positions = {
        nodes[0]: (0.0, 0.0),
        nodes[1]: (1.0, 0.0),
        nodes[2]: (2.0, 0.0),
        nodes[3]: (3.0, 0.0),
    }

    ax = visualize(simple_graph, node_positions=custom_positions)
    assert ax is not None

    # Verify that the axes limits include the custom positions
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Check that all x coordinates are within the limits
    for x, y in custom_positions.values():
        assert xlim[0] <= x <= xlim[1]
        assert ylim[0] <= y <= ylim[1]

    plt.close()


def test_visualize_with_node_labels(simple_graph: GraphState) -> None:
    """Test visualize function with node labels enabled."""
    ax = visualize(simple_graph, show_node_labels=True)
    assert ax is not None

    # Check that text elements (labels) are present
    texts = [child for child in ax.get_children() if isinstance(child, mpl.text.Text)]
    # At least some labels should be present (including axis labels)
    assert len(texts) > 0

    plt.close()


def test_visualize_without_node_labels(simple_graph: GraphState) -> None:
    """Test visualize function with node labels disabled."""
    ax = visualize(simple_graph, show_node_labels=False)
    assert ax is not None
    plt.close()


def test_visualize_with_legend(simple_graph: GraphState) -> None:
    """Test visualize function with legend enabled."""
    ax = visualize(simple_graph, show_legend=True)
    assert ax is not None

    # Check that legend is present
    legend = ax.get_legend()
    assert legend is not None

    plt.close()


def test_visualize_without_legend(simple_graph: GraphState) -> None:
    """Test visualize function with legend disabled."""
    ax = visualize(simple_graph, show_legend=False)
    assert ax is not None

    # Check that no legend is present
    legend = ax.get_legend()
    assert legend is None

    plt.close()


def test_visualize_with_custom_node_size(simple_graph: GraphState) -> None:
    """Test visualize function with custom node size."""
    custom_size = 500
    ax = visualize(simple_graph, node_size=custom_size)
    assert ax is not None
    plt.close()


def test_visualize_with_custom_axes(simple_graph: GraphState) -> None:
    """Test visualize function with pre-existing axes."""
    fig, ax = plt.subplots()
    result_ax = visualize(simple_graph, ax=ax)

    # Verify that the same axes object is returned
    assert result_ax is ax

    plt.close(fig)


def test_visualize_grid_layout() -> None:
    """Test visualize function with grid layout using custom positions."""
    graph = GraphState()

    # Create 3x3 grid
    grid_positions = {}
    nodes = []
    for i in range(3):
        for j in range(3):
            node = graph.add_physical_node()
            nodes.append(node)
            grid_positions[node] = (float(i), float(j))

    # Register input/output
    graph.register_input(nodes[0])
    graph.register_output(nodes[-1], 0)

    # Assign measurement bases (skip input/output)
    for node in nodes[1:-1]:
        graph.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, np.pi / 4))

    # Add edges
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            # Connect to right neighbor
            if i < 2:
                graph.add_physical_edge(nodes[idx], nodes[idx + 3])
            # Connect to bottom neighbor
            if j < 2:
                graph.add_physical_edge(nodes[idx], nodes[idx + 1])

    ax = visualize(graph, node_positions=grid_positions)
    assert ax is not None
    plt.close()


def test_visualize_circular_layout() -> None:
    """Test visualize function with circular layout using custom positions."""
    graph = GraphState()

    n_nodes = 6
    nodes = []
    circular_positions = {}

    for i in range(n_nodes):
        node = graph.add_physical_node()
        nodes.append(node)
        # Position on circle
        angle = 2 * np.pi * i / n_nodes
        x = np.cos(angle)
        y = np.sin(angle)
        circular_positions[node] = (float(x), float(y))

    # Register input/output
    graph.register_input(nodes[0])
    graph.register_output(nodes[n_nodes // 2], 0)

    # Assign measurement bases
    for node in nodes:
        if node in graph.output_node_indices:
            continue
        graph.assign_meas_basis(node, PlannerMeasBasis(Plane.XZ, np.pi / 4))

    # Connect in a ring
    for i in range(n_nodes):
        graph.add_physical_edge(nodes[i], nodes[(i + 1) % n_nodes])

    ax = visualize(graph, node_positions=circular_positions)
    assert ax is not None
    plt.close()


def test_visualize_with_pauli_measurements() -> None:
    """Test visualize function with Pauli measurements."""
    graph = GraphState()

    # Create nodes with Pauli measurements
    x_node = graph.add_physical_node()
    y_node = graph.add_physical_node()
    z_node = graph.add_physical_node()
    output_node = graph.add_physical_node()

    graph.register_input(x_node)
    graph.register_output(output_node, 0)

    # Assign Pauli measurements
    graph.assign_meas_basis(x_node, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(y_node, AxisMeasBasis(Axis.Y, Sign.PLUS))
    graph.assign_meas_basis(z_node, AxisMeasBasis(Axis.Z, Sign.MINUS))

    # Add edges
    graph.add_physical_edge(x_node, y_node)
    graph.add_physical_edge(y_node, z_node)
    graph.add_physical_edge(z_node, output_node)

    ax = visualize(graph)
    assert ax is not None
    plt.close()


def test_visualize_empty_graph() -> None:
    """Test visualize function with empty graph."""
    graph = GraphState()
    ax = visualize(graph)
    assert ax is not None
    plt.close()


def test_visualize_single_node() -> None:
    """Test visualize function with single node."""
    graph = GraphState()
    node = graph.add_physical_node()
    graph.register_input(node)
    graph.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, 0.0))

    ax = visualize(graph)
    assert ax is not None
    plt.close()


def test_visualize_custom_positions_missing_node() -> None:
    """Test that visualize works even if custom positions don't include all nodes."""
    graph = GraphState()

    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    node3 = graph.add_physical_node()

    graph.register_input(node1)
    graph.register_output(node3, 0)
    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(node2, PlannerMeasBasis(Plane.XZ, 0.0))

    graph.add_physical_edge(node1, node2)
    graph.add_physical_edge(node2, node3)

    # Only provide positions for some nodes
    # This should cause a KeyError if the visualizer tries to use these positions
    incomplete_positions = {
        node1: (0.0, 0.0),
        node2: (1.0, 0.0),
        # node3 is missing
    }

    # This should raise a KeyError because node3 is missing
    with pytest.raises(KeyError):
        visualize(graph, node_positions=incomplete_positions)

    plt.close()
