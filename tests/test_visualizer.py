"""Tests for the visualizer module."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.visualizer import visualize

# Use non-interactive backend for tests
mpl.use("Agg")


@pytest.fixture
def graph_with_coordinates() -> GraphState:
    """Create a graph with coordinates.

    Returns
    -------
    GraphState
        A graph with nodes at coordinates (0,0), (1,0), (2,0).
    """
    graph = GraphState()
    node1 = graph.add_physical_node(coordinate=(0.0, 0.0))
    node2 = graph.add_physical_node(coordinate=(1.0, 0.0))
    node3 = graph.add_physical_node(coordinate=(2.0, 0.0))

    graph.add_physical_edge(node1, node2)
    graph.add_physical_edge(node2, node3)

    graph.register_input(node1, 0)
    graph.register_output(node3, 0)

    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(node2, PlannerMeasBasis(Plane.XY, 0.0))

    return graph


def test_visualize_with_graph_coordinates(graph_with_coordinates: GraphState) -> None:
    """Test that visualize uses graph coordinates when available."""
    ax = visualize(graph_with_coordinates, use_graph_coordinates=True)
    assert ax is not None
    plt.close("all")


def test_visualize_without_graph_coordinates(graph_with_coordinates: GraphState) -> None:
    """Test that visualize can use auto-calculated positions."""
    ax = visualize(graph_with_coordinates, use_graph_coordinates=False)
    assert ax is not None
    plt.close("all")


def test_visualize_with_partial_coordinates() -> None:
    """Test that visualize handles graphs with partial coordinates."""
    graph = GraphState()
    # Only some nodes have coordinates
    node1 = graph.add_physical_node(coordinate=(0.0, 0.0))
    node2 = graph.add_physical_node()  # No coordinate
    node3 = graph.add_physical_node(coordinate=(2.0, 0.0))

    graph.add_physical_edge(node1, node2)
    graph.add_physical_edge(node2, node3)

    graph.register_input(node1, 0)
    graph.register_output(node3, 0)

    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(node2, PlannerMeasBasis(Plane.XY, 0.0))

    ax = visualize(graph, use_graph_coordinates=True)
    assert ax is not None
    plt.close("all")


def test_visualize_with_3d_coordinates() -> None:
    """Test that visualize projects 3D coordinates to 2D."""
    graph = GraphState()
    node1 = graph.add_physical_node(coordinate=(0.0, 0.0, 0.0))
    node2 = graph.add_physical_node(coordinate=(1.0, 1.0, 1.0))

    graph.add_physical_edge(node1, node2)

    graph.register_input(node1, 0)
    graph.register_output(node2, 0)

    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    ax = visualize(graph, use_graph_coordinates=True)
    assert ax is not None
    plt.close("all")


def test_visualize_empty_coordinates() -> None:
    """Test that visualize works with empty graph coordinates."""
    graph = GraphState()
    node1 = graph.add_physical_node()  # No coordinate
    node2 = graph.add_physical_node()  # No coordinate

    graph.add_physical_edge(node1, node2)

    graph.register_input(node1, 0)
    graph.register_output(node2, 0)

    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    # Should fall back to auto-calculated positions
    ax = visualize(graph, use_graph_coordinates=True)
    assert ax is not None
    plt.close("all")
