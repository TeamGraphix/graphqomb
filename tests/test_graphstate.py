"""Tests for the GraphState class."""

from __future__ import annotations

import numpy as np
import pytest

from graphix_zx.common import Plane
from graphix_zx.graphstate import GraphState, bipartite_edges


@pytest.fixture
def graph() -> GraphState:
    """Generate an empty GraphState object.

    Returns
    -------
        GraphState: An empty GraphState object.
    """
    return GraphState()


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


def test_remove_physical_node_with_nonexistent_node(graph: GraphState) -> None:
    """Test removing a nonexistent physical node from the graph."""
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.remove_physical_node(1)


def test_remove_physical_node(graph: GraphState) -> None:
    """Test removing a physical node from the graph."""
    graph.add_physical_node(1)
    graph.remove_physical_node(1)
    assert 1 not in graph.physical_nodes
    assert graph.num_physical_nodes == 0


def test_remove_physical_node_from_minimal_graph(graph: GraphState) -> None:
    """Test removing a physical node from the graph with edges."""
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    graph.remove_physical_node(1)
    assert 1 not in graph.physical_nodes
    assert 2 in graph.physical_nodes
    assert (1, 2) not in graph.physical_edges
    assert (2, 1) not in graph.physical_edges
    assert graph.num_physical_nodes == 1
    assert graph.num_physical_edges == 0


def test_remove_physical_node_from_3_nodes_graph(graph: GraphState) -> None:
    """Test removing a physical node from the graph with 3 nodes and edges."""
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_node(3)
    graph.add_physical_edge(1, 2)
    graph.add_physical_edge(2, 3)
    graph.set_input(2)
    graph.set_output(2)
    graph.remove_physical_node(2)
    assert graph.physical_nodes == {1, 3}
    assert graph.physical_edges == set()
    assert graph.num_physical_nodes == 2
    assert graph.num_physical_edges == 0
    assert graph.input_nodes == set()
    assert graph.output_nodes == set()


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


def test_set_output_raises_1(graph: GraphState) -> None:
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.set_output(1)
    graph.add_physical_node(1)
    graph.set_meas_angle(1, 0.5 * np.pi)
    with pytest.raises(ValueError, match="Cannot set output node with measurement basis."):
        graph.set_output(1)


def test_set_output_raises_2(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.set_meas_plane(1, Plane.XY)
    with pytest.raises(ValueError, match="Cannot set output node with measurement basis."):
        graph.set_output(1)


def test_set_output(graph: GraphState) -> None:
    """Test setting a physical node as output."""
    graph.add_physical_node(1)
    graph.set_output(1)
    assert 1 in graph.output_nodes


def test_set_meas_plane_raises(graph: GraphState) -> None:
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.set_meas_plane(1, Plane.XY)
    graph.add_physical_node(1)
    graph.set_output(1)
    with pytest.raises(ValueError, match="Cannot set measurement plane for output node."):
        graph.set_meas_plane(1, Plane.XY)


def test_set_meas_plane(graph: GraphState) -> None:
    """Test setting the measurement plane of a physical node."""
    graph.add_physical_node(1)
    graph.set_meas_plane(1, Plane.XZ)
    assert graph.meas_planes[1] == Plane.XZ


def test_set_meas_angle_raises(graph: GraphState) -> None:
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.set_meas_angle(1, 0.5 * np.pi)
    graph.add_physical_node(1)
    graph.set_output(1)
    with pytest.raises(ValueError, match="Cannot set measurement angle for output node."):
        graph.set_meas_angle(1, 0.5 * np.pi)


def test_set_meas_angle(graph: GraphState) -> None:
    """Test setting the measurement angle of a physical node."""
    graph.add_physical_node(1)
    graph.set_meas_angle(1, 0.5 * np.pi)
    assert graph.meas_angles[1] == 0.5 * np.pi


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
        graph.check_meas_basis()
    graph.set_meas_angle(1, 0.5 * np.pi)
    graph.set_meas_plane(1, "invalid plane")  # type: ignore[reportArgumentType, arg-type, unused-ignore]
    with pytest.raises(ValueError, match="Invalid measurement plane 'invalid plane' for node 1"):
        graph.check_meas_basis()


def test_check_meas_basis_success(graph: GraphState) -> None:
    """Test if measurement planes and angles are set properly."""
    graph.check_meas_basis()
    graph.add_physical_node(1)
    graph.set_meas_plane(1, Plane.XZ)
    graph.set_meas_angle(1, 0.5 * np.pi)
    graph.check_meas_basis()

    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    graph.set_output(2)
    graph.check_meas_basis()


def test_bipartite_edges() -> None:
    """Test the function that generate complete bipartite edges"""
    assert bipartite_edges(set(), set()) == set()
    assert bipartite_edges({1, 2, 3}, {1, 2, 3}) == {(1, 2), (1, 3), (2, 3)}
    assert bipartite_edges({1, 2}, {3, 4}) == {(1, 3), (1, 4), (2, 3), (2, 4)}


if __name__ == "__main__":
    pytest.main()
