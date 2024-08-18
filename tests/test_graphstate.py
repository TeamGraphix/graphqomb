from __future__ import annotations

import pytest

from graphix_zx.common import Plane
from graphix_zx.graphstate import GraphState


@pytest.fixture
def graph() -> GraphState:
    return GraphState()


def test_add_physical_node(graph: GraphState) -> None:
    graph.add_physical_node(1)
    assert 1 in graph.get_physical_nodes()
    assert graph.num_physical_nodes == 1


def test_add_physical_node_input_output(graph: GraphState) -> None:
    graph.add_physical_node(1, is_input=True, is_output=True)
    assert 1 in graph.input_nodes
    assert 1 in graph.output_nodes


def test_add_duplicate_physical_node(graph: GraphState) -> None:
    graph.add_physical_node(1)
    with pytest.raises(Exception, match="Node already exists"):
        graph.add_physical_node(1)


def test_add_physical_edge(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    assert (1, 2) in graph.get_physical_edges() or (2, 1) in graph.get_physical_edges()
    assert graph.num_physical_edges == 1


def test_add_duplicate_physical_edge(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    with pytest.raises(ValueError, match="Edge already exists node1=1, node2=2"):
        graph.add_physical_edge(1, 2)


def test_add_edge_with_nonexistent_node(graph: GraphState) -> None:
    graph.add_physical_node(1)
    with pytest.raises(ValueError, match="Node does not exist node2=2"):
        graph.add_physical_edge(1, 2)


def test_set_input(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.set_input(1)
    assert 1 in graph.input_nodes


def test_set_output(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.set_output(1)
    assert 1 in graph.output_nodes


def test_set_meas_plane(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.set_meas_plane(1, Plane.XZ)
    assert graph.get_meas_planes()[1] == Plane.XZ


def test_set_meas_angle(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.set_meas_angle(1, 45.0)
    assert graph.get_meas_angles()[1] == 45.0


def test_append_graph() -> None:
    graph1 = GraphState()
    graph1.add_physical_node(1, is_input=True)
    graph1.add_physical_node(2, is_output=True)
    graph1.add_physical_edge(1, 2)

    graph2 = GraphState()
    graph2.add_physical_node(2, is_input=True)
    graph2.add_physical_node(3, is_output=True)
    graph2.add_physical_edge(2, 3)

    graph = graph1.append_graph(graph2)

    assert graph.num_physical_nodes == 3
    assert graph.num_physical_edges == 2
    assert 1 in graph.input_nodes
    assert 3 in graph.output_nodes


if __name__ == "__main__":
    pytest.main()
