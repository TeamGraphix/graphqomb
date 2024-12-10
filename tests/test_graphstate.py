"""Tests for the GraphState class."""

from __future__ import annotations

import numpy as np
import pytest

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import GraphState, ZXGraphState


@pytest.fixture
def graph() -> GraphState:
    """Generate an empty GraphState object.

    Returns
    -------
        GraphState: An empty GraphState object.
    """
    return GraphState()


@pytest.fixture
def zx_graph() -> ZXGraphState:
    """Generate an empty ZXGraphState object.

    Returns
    -------
        ZXGraphState: An empty ZXGraphState object.
    """
    return ZXGraphState()


def test_add_physical_node(graph: GraphState) -> None:
    """Test adding a physical node to the graph."""
    graph.add_physical_node(1)
    assert 1 in graph.physical_nodes
    assert graph.num_physical_nodes == 1


def test_add_physical_node_input_output(graph: GraphState) -> None:
    """Test adding a physical node as input and output."""
    graph.add_physical_node(1, is_input=True, is_output=True)
    assert 1 in graph.input_nodes
    assert 1 in graph.output_nodes


def test_add_duplicate_physical_node(graph: GraphState) -> None:
    """Test adding a duplicate physical node to the graph."""
    graph.add_physical_node(1)
    with pytest.raises(Exception, match="Node already exists"):
        graph.add_physical_node(1)


def test_ensure_node_exists_raises(graph: GraphState) -> None:
    """Test ensuring a node exists in the graph."""
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.ensure_node_exists(1)


def test_ensure_node_exists(graph: GraphState) -> None:
    """Test ensuring a node exists in the graph."""
    graph.add_physical_node(1)
    graph.ensure_node_exists(1)


def test_get_neighbors(graph: GraphState) -> None:
    """Test getting the neighbors of a node in the graph."""
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_node(3)
    graph.add_physical_edge(1, 2)
    graph.add_physical_edge(2, 3)
    assert graph.get_neighbors(1) == {2}
    assert graph.get_neighbors(2) == {1, 3}
    assert graph.get_neighbors(3) == {2}


def test_add_physical_edge(graph: GraphState) -> None:
    """Test adding a physical edge to the graph."""
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    assert (1, 2) in graph.physical_edges or (2, 1) in graph.physical_edges
    assert graph.num_physical_edges == 1


def test_add_duplicate_physical_edge(graph: GraphState) -> None:
    """Test adding a duplicate physical edge to the graph."""
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    with pytest.raises(ValueError, match="Edge already exists node1=1, node2=2"):
        graph.add_physical_edge(1, 2)


def test_add_edge_with_nonexistent_node(graph: GraphState) -> None:
    """Test adding an edge with a nonexistent node to the graph."""
    graph.add_physical_node(1)
    with pytest.raises(ValueError, match="Node does not exist node=2"):
        graph.add_physical_edge(1, 2)


def test_remove_physical_edge_with_nonexistent_nodes(graph: GraphState) -> None:
    """Test removing an edge with nonexistent nodes from the graph."""
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.remove_physical_edge(1, 2)


def test_remove_physical_edge_with_nonexistent_edge(graph: GraphState) -> None:
    """Test removing a nonexistent edge from the graph."""
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    with pytest.raises(ValueError, match="Edge does not exist"):
        graph.remove_physical_edge(1, 2)


def test_remove_physical_edge(graph: GraphState) -> None:
    """Test removing a physical edge from the graph."""
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    graph.remove_physical_edge(1, 2)
    assert (1, 2) not in graph.physical_edges
    assert (2, 1) not in graph.physical_edges
    assert graph.num_physical_edges == 0


def test_set_input(graph: GraphState) -> None:
    """Test setting a physical node as input."""
    graph.add_physical_node(1)
    graph.set_input(1)
    assert 1 in graph.input_nodes


def test_set_output(graph: GraphState) -> None:
    """Test setting a physical node as output."""
    graph.add_physical_node(1)
    graph.set_output(1)
    assert 1 in graph.output_nodes


def test_set_meas_basis(graph: GraphState) -> None:
    """Test setting the measurement basis of a physical node."""
    graph.add_physical_node(1)
    meas_basis = PlannerMeasBasis(Plane.XZ, 0.5 * np.pi)
    graph.set_meas_basis(1, meas_basis)
    assert graph.meas_bases[1].plane == Plane.XZ
    assert graph.meas_bases[1].angle == 0.5 * np.pi


def test_append_graph() -> None:
    """Test appending a graph to another graph."""
    graph1 = GraphState()
    graph1.add_physical_node(1, is_input=True)
    graph1.add_physical_node(2, is_output=True)
    graph1.add_physical_edge(1, 2)

    graph2 = GraphState()
    graph2.add_physical_node(2, is_input=True)
    graph2.add_physical_node(3, is_output=True)
    graph2.add_physical_edge(2, 3)

    graph1.append(graph2)

    assert graph1.num_physical_nodes == 3
    assert graph1.num_physical_edges == 2
    assert 1 in graph1.input_nodes
    assert 3 in graph1.output_nodes


def test_check_meas_raises_value_error(graph: GraphState) -> None:
    """Test if measurement planes and angles are set improperly."""
    graph.add_physical_node(1)
    with pytest.raises(ValueError, match="Measurement basis not set for node 1"):
        graph.check_meas_bases()


def test_check_meas_basis_success(graph: GraphState) -> None:
    """Test if measurement planes and angles are set properly."""
    graph.check_meas_bases()
    graph.add_physical_node(1)
    meas_basis = PlannerMeasBasis(Plane.XY, 0.5 * np.pi)
    graph.set_meas_basis(1, meas_basis)
    graph.check_meas_bases()

    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    graph.set_output(2)
    graph.check_meas_bases()


if __name__ == "__main__":
    pytest.main()
