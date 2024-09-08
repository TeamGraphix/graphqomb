from __future__ import annotations

import pytest

from graphix_zx.common import Plane
from graphix_zx.graphstate import GraphState, ZXGraphState


@pytest.fixture
def graph() -> GraphState:
    return GraphState()


@pytest.fixture
def zx_graph() -> ZXGraphState:
    return ZXGraphState()


def test_add_physical_node(graph: GraphState) -> None:
    graph.add_physical_node(1)
    assert 1 in graph.physical_nodes
    assert graph.num_physical_nodes == 1


def test_add_physical_node_input_output(graph: GraphState) -> None:
    graph.add_physical_node(1, is_input=True, is_output=True)
    assert 1 in graph.input_nodes
    assert 1 in graph.output_nodes


def test_add_duplicate_physical_node(graph: GraphState) -> None:
    graph.add_physical_node(1)
    with pytest.raises(Exception, match="Node already exists"):
        graph.add_physical_node(1)


def test_ensure_node_exists_raises(graph: GraphState) -> None:
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.ensure_node_exists(1)


def test_ensure_node_exists(graph: GraphState) -> None:
    graph.add_physical_node(1)
    assert graph.ensure_node_exists(1) is None


def test_adjacent_nodes(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_node(3)
    graph.add_physical_edge(1, 2)
    graph.add_physical_edge(2, 3)
    assert graph.adjacent_nodes(1) == {2}
    assert graph.adjacent_nodes(2) == {1, 3}
    assert graph.adjacent_nodes(3) == {2}


def test_add_physical_edge(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    assert (1, 2) in graph.physical_edges or (2, 1) in graph.physical_edges
    assert graph.num_physical_edges == 1


def test_add_duplicate_physical_edge(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    with pytest.raises(ValueError, match="Edge already exists node1=1, node2=2"):
        graph.add_physical_edge(1, 2)


def test_add_edge_with_nonexistent_node(graph: GraphState) -> None:
    graph.add_physical_node(1)
    with pytest.raises(ValueError, match="Node does not exist node=2"):
        graph.add_physical_edge(1, 2)


def test_remove_physical_edge_with_nonexistent_nodes(graph: GraphState) -> None:
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        graph.remove_physical_edge(1, 2)


def test_remove_physical_edge_with_nonexistent_edge(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    with pytest.raises(ValueError, match="Edge does not exist"):
        graph.remove_physical_edge(1, 2)


def test_remove_physical_edge(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.add_physical_node(2)
    graph.add_physical_edge(1, 2)
    graph.remove_physical_edge(1, 2)
    assert (1, 2) not in graph.physical_edges
    assert (2, 1) not in graph.physical_edges
    assert graph.num_physical_edges == 0


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
    assert graph.meas_planes[1] == Plane.XZ


def test_set_meas_angle(graph: GraphState) -> None:
    graph.add_physical_node(1)
    graph.set_meas_angle(1, 0.5)
    assert graph.meas_angles[1] == 0.5


def test_append_graph() -> None:
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


def test_is_zx_graph_returns_false(zx_graph: GraphState) -> None:
    zx_graph.add_physical_node(1)
    assert not zx_graph.is_zx_graph()
    zx_graph.set_meas_plane(1, "test plane")
    assert not zx_graph.is_zx_graph()


def test_is_zx_graph_returns_true(zx_graph: ZXGraphState) -> None:
    assert zx_graph.is_zx_graph()
    zx_graph.add_physical_node(1)
    zx_graph.set_meas_plane(1, Plane.XZ)
    zx_graph.set_meas_angle(1, 0.5)
    assert zx_graph.is_zx_graph()

    zx_graph.add_physical_node(2)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.set_output(2)
    assert zx_graph.is_zx_graph()


def test_local_complement_raises(zx_graph: ZXGraphState) -> None:
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        zx_graph.local_complement(1)
    zx_graph.add_physical_node(1)
    zx_graph.set_input(1)
    with pytest.raises(ValueError, match="Cannot apply local complement to input node"):
        zx_graph.local_complement(1)
    zx_graph.add_physical_node(2)
    with pytest.raises(ValueError, match="The graph is not a ZX-diagram. Set measurement planes and angles properly."):
        zx_graph.local_complement(2)


def test_local_complement_with_no_edge(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.local_complement(1)
    assert set() == zx_graph.physical_edges


def test_local_complement_with_two_nodes_graph(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_edge(1, 2)
    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(1)
    assert original_edges == zx_graph.physical_edges


def test_local_complement_with_minimal_graph(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_node(3)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.add_physical_edge(2, 3)
    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(2)
    assert {(1, 2), (2, 3), (1, 3)} == zx_graph.physical_edges

    zx_graph.local_complement(2)
    assert original_edges == zx_graph.physical_edges


def test_local_complement_with_h_shaped_graph(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_node(3)
    zx_graph.add_physical_node(4)
    zx_graph.add_physical_node(5)
    zx_graph.add_physical_node(6)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.add_physical_edge(2, 3)
    zx_graph.add_physical_edge(2, 5)
    zx_graph.add_physical_edge(4, 5)
    zx_graph.add_physical_edge(5, 6)
    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(2)
    assert {(1, 2), (1, 3), (1, 5), (2, 3), (2, 5), (3, 5), (4, 5), (5, 6)} == zx_graph.physical_edges

    zx_graph.local_complement(2)
    assert original_edges == zx_graph.physical_edges


def test_pivot_with_obvious_graph(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_node(3)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.add_physical_edge(2, 3)
    original_edges = zx_graph.physical_edges.copy()
    expected_edges = {(1, 3), (2, 3)}
    zx_graph.pivot(2, 3)
    assert expected_edges == zx_graph.physical_edges
    zx_graph.pivot(2, 3)
    assert original_edges == zx_graph.physical_edges

    zx_graph.pivot(3, 2)
    assert expected_edges == zx_graph.physical_edges
    zx_graph.pivot(3, 2)
    assert original_edges == zx_graph.physical_edges


# def test_pivot_with_minimal_graph(zx_graph: ZXGraphState) -> None:
#     zx_graph.add_physical_node(1)
#     zx_graph.add_physical_node(2)
#     zx_graph.add_physical_node(3)
#     zx_graph.add_physical_node(4)
#     zx_graph.add_physical_node(5)
#     zx_graph.add_physical_edge(1, 2)
#     zx_graph.add_physical_edge(2, 3)
#     zx_graph.add_physical_edge(2, 4)
#     zx_graph.add_physical_edge(3, 4)
#     zx_graph.add_physical_edge(3, 5)
#     original_edges = zx_graph.physical_edges.copy()
#     expected_edges = {(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)}
#     zx_graph.pivot(2, 3)
#     assert expected_edges == zx_graph.physical_edges
#     zx_graph.pivot(2, 3)
#     assert original_edges == zx_graph.physical_edges

#     zx_graph.pivot(3, 2)
#     assert expected_edges == zx_graph.physical_edges
#     zx_graph.pivot(3, 2)
#     assert original_edges == zx_graph.physical_edges


# def test_pivot_with_h_shaped_graph(zx_graph: ZXGraphState) -> None:
#     zx_graph.add_physical_node(1)
#     zx_graph.add_physical_node(2)
#     zx_graph.add_physical_node(3)
#     zx_graph.add_physical_node(4)
#     zx_graph.add_physical_node(5)
#     zx_graph.add_physical_node(6)
#     zx_graph.add_physical_edge(1, 2)
#     zx_graph.add_physical_edge(2, 3)
#     zx_graph.add_physical_edge(2, 5)
#     zx_graph.add_physical_edge(4, 5)
#     zx_graph.add_physical_edge(5, 6)
#     original_edges = zx_graph.physical_edges.copy()
#     expected_edges = {(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6)}
#     zx_graph.pivot(2, 5)
#     assert expected_edges == zx_graph.physical_edges
#     zx_graph.pivot(2, 5)
#     assert original_edges == zx_graph.physical_edges

#     zx_graph.pivot(5, 2)
#     assert expected_edges == zx_graph.physical_edges
#     zx_graph.pivot(5, 2)
#     assert original_edges == zx_graph.physical_edges


if __name__ == "__main__":
    pytest.main()
