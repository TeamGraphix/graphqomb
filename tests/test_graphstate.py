from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from graphix_zx.common import Plane
from graphix_zx.graphstate import GraphState, ZXGraphState, adj_pairs, meas_action


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
    graph.set_meas_angle(1, 0.5 * np.pi)
    assert graph.meas_angles[1] == 0.5 * np.pi


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
    zx_graph.set_meas_angle(1, 0.5 * np.pi)
    assert zx_graph.is_zx_graph()

    zx_graph.add_physical_node(2)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.set_output(2)
    assert zx_graph.is_zx_graph()


def test_meas_action_with_some_planes_missing() -> None:
    ma = meas_action({Plane.XY: (Plane.XY, 0.0)})
    assert ma[Plane.YX] == (Plane.XY, 0.0)
    assert ma[Plane.XZ] == (None, None)
    assert ma[Plane.YZ] == (None, None)
    assert ma[Plane.ZX] == (None, None)
    assert ma[Plane.ZY] == (None, None)


def test_meas_action() -> None:
    measurement_action = meas_action(
        {
            Plane.XY: (Plane.XY, 0.0),
            Plane.XZ: (Plane.XZ, 0.0),
            Plane.YZ: (Plane.YZ, 0.0),
        }
    )
    assert measurement_action[Plane.XY] == (Plane.XY, 0.0)
    assert measurement_action[Plane.XZ] == (Plane.XZ, 0.0)
    assert measurement_action[Plane.YZ] == (Plane.YZ, 0.0)
    assert measurement_action[Plane.YX] == measurement_action[Plane.XY]
    assert measurement_action[Plane.ZX] == measurement_action[Plane.XZ]
    assert measurement_action[Plane.ZY] == measurement_action[Plane.YZ]


def test_adj_pairs() -> None:
    assert adj_pairs(set(), set()) == set()
    assert adj_pairs({1, 2, 3}, {1, 2, 3}) == {(1, 2), (1, 3), (2, 3)}
    assert adj_pairs({1, 2}, {3, 4}) == {(1, 3), (1, 4), (2, 3), (2, 4)}


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
    zx_graph.set_meas_plane(1, Plane.XY)
    zx_graph.set_meas_angle(1, 1.1 * np.pi)
    zx_graph.local_complement(1)
    assert zx_graph.physical_edges == set()
    assert zx_graph.meas_planes[1] == Plane.XZ
    assert pytest.approx(zx_graph.meas_angles[1]) == 1.4 * np.pi

    zx_graph.set_meas_plane(1, Plane.XZ)
    zx_graph.set_meas_angle(1, 1.1 * np.pi)
    zx_graph.local_complement(1)
    assert zx_graph.meas_planes[1] == Plane.XY
    assert pytest.approx(zx_graph.meas_angles[1]) == 0.6 * np.pi

    zx_graph.set_meas_plane(1, Plane.YZ)
    zx_graph.set_meas_angle(1, 1.1 * np.pi)
    zx_graph.local_complement(1)
    assert zx_graph.meas_planes[1] == Plane.YZ
    assert pytest.approx(zx_graph.meas_angles[1]) == 1.6 * np.pi


def test_local_complement_with_two_nodes_graph(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.set_meas_plane(1, Plane.XZ)
    zx_graph.set_meas_angle(1, 1.1 * np.pi)
    zx_graph.set_meas_plane(2, Plane.XZ)
    zx_graph.set_meas_angle(2, 1.2 * np.pi)
    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(1)
    assert zx_graph.physical_edges == original_edges
    assert zx_graph.meas_planes == {1: Plane.XY, 2: Plane.YZ}
    assert pytest.approx(zx_graph.meas_angles[1]) == 0.6 * np.pi
    assert pytest.approx(zx_graph.meas_angles[2]) == 1.2 * np.pi


def test_local_complement_with_minimal_graph(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_node(3)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.add_physical_edge(2, 3)
    zx_graph.set_meas_plane(1, Plane.XY)
    zx_graph.set_meas_angle(1, 1.1 * np.pi)
    zx_graph.set_meas_plane(2, Plane.XZ)
    zx_graph.set_meas_angle(2, 1.2 * np.pi)
    zx_graph.set_meas_plane(3, Plane.YZ)
    zx_graph.set_meas_angle(3, 1.3 * np.pi)
    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == {(1, 2), (2, 3), (1, 3)}
    assert zx_graph.meas_planes == {1: Plane.XY, 2: Plane.XY, 3: Plane.XZ}
    assert pytest.approx(zx_graph.meas_angles[1]) == 0.6 * np.pi
    assert pytest.approx(zx_graph.meas_angles[2]) == 0.7 * np.pi
    assert pytest.approx(zx_graph.meas_angles[3]) == 0.7 * np.pi

    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == original_edges
    assert zx_graph.meas_planes == {1: Plane.XY, 2: Plane.XZ, 3: Plane.YZ}
    assert pytest.approx(zx_graph.meas_angles[1]) == 0.1 * np.pi
    assert pytest.approx(zx_graph.meas_angles[2]) == 1.8 * np.pi
    assert pytest.approx(zx_graph.meas_angles[3]) == 0.7 * np.pi


def test_local_complement_4_times(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_node(3)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.add_physical_edge(2, 3)
    zx_graph.set_meas_plane(1, Plane.XY)
    zx_graph.set_meas_angle(1, 1.1 * np.pi)
    zx_graph.set_meas_plane(2, Plane.XZ)
    zx_graph.set_meas_angle(2, 1.2 * np.pi)
    zx_graph.set_meas_plane(3, Plane.YZ)
    zx_graph.set_meas_angle(3, 1.3 * np.pi)
    original_edges = zx_graph.physical_edges.copy()
    for _ in range(4):
        zx_graph.local_complement(2)
    assert zx_graph.physical_edges == original_edges
    assert zx_graph.meas_planes == {1: Plane.XY, 2: Plane.XZ, 3: Plane.YZ}
    assert pytest.approx(zx_graph.meas_angles[1]) == 1.1 * np.pi
    assert pytest.approx(zx_graph.meas_angles[2]) == 1.2 * np.pi
    assert pytest.approx(zx_graph.meas_angles[3]) == 1.3 * np.pi


def test_local_complement_with_inversed_planes(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_node(3)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.add_physical_edge(2, 3)
    zx_graph.set_meas_plane(1, Plane.YX)
    zx_graph.set_meas_angle(1, 1.1 * np.pi)
    zx_graph.set_meas_plane(2, Plane.ZX)
    zx_graph.set_meas_angle(2, 1.2 * np.pi)
    zx_graph.set_meas_plane(3, Plane.ZY)
    zx_graph.set_meas_angle(3, 1.3 * np.pi)
    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == {(1, 2), (2, 3), (1, 3)}
    assert zx_graph.meas_planes == {1: Plane.XY, 2: Plane.XY, 3: Plane.XZ}
    assert pytest.approx(zx_graph.meas_angles[1]) == 0.6 * np.pi
    assert pytest.approx(zx_graph.meas_angles[2]) == 0.7 * np.pi
    assert pytest.approx(zx_graph.meas_angles[3]) == 0.7 * np.pi

    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == original_edges
    assert zx_graph.meas_planes == {1: Plane.XY, 2: Plane.XZ, 3: Plane.YZ}
    assert pytest.approx(zx_graph.meas_angles[1]) == 0.1 * np.pi
    assert pytest.approx(zx_graph.meas_angles[2]) == 1.8 * np.pi
    assert pytest.approx(zx_graph.meas_angles[3]) == 0.7 * np.pi


def test_local_complement_with_h_shaped_graph(zx_graph: ZXGraphState) -> None:
    for i in range(1, 7):
        zx_graph.add_physical_node(i)

    zx_graph.set_input(1)
    zx_graph.set_input(4)

    for i, j in [(1, 2), (2, 3), (2, 5), (4, 5), (5, 6)]:
        zx_graph.add_physical_edge(i, j)

    measurements = [
        (1, Plane.XY, 1.1 * np.pi),
        (2, Plane.XZ, 1.2 * np.pi),
        (3, Plane.YZ, 1.3 * np.pi),
        (4, Plane.XY, 1.4 * np.pi),
        (5, Plane.XZ, 1.5 * np.pi),
        (6, Plane.YZ, 1.6 * np.pi),
    ]
    for node_id, plane, angle in measurements:
        zx_graph.set_meas_plane(node_id, plane)
        zx_graph.set_meas_angle(node_id, angle)

    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == {(1, 2), (1, 3), (1, 5), (2, 3), (2, 5), (3, 5), (4, 5), (5, 6)}
    assert zx_graph.meas_planes == {1: Plane.XY, 2: Plane.XY, 3: Plane.XZ, 4: Plane.XY, 5: Plane.YZ, 6: Plane.YZ}
    assert pytest.approx(zx_graph.meas_angles[1]) == 0.6 * np.pi
    assert pytest.approx(zx_graph.meas_angles[2]) == 0.7 * np.pi
    assert pytest.approx(zx_graph.meas_angles[3]) == 0.7 * np.pi
    assert pytest.approx(zx_graph.meas_angles[4]) == 1.4 * np.pi
    assert pytest.approx(zx_graph.meas_angles[5]) == 1.5 * np.pi
    assert pytest.approx(zx_graph.meas_angles[6]) == 1.6 * np.pi

    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == original_edges
    assert zx_graph.meas_planes == {1: Plane.XY, 2: Plane.XZ, 3: Plane.YZ, 4: Plane.XY, 5: Plane.XZ, 6: Plane.YZ}
    assert pytest.approx(zx_graph.meas_angles[1]) == 0.1 * np.pi
    assert pytest.approx(zx_graph.meas_angles[2]) == 1.8 * np.pi
    assert pytest.approx(zx_graph.meas_angles[3]) == 0.7 * np.pi
    assert pytest.approx(zx_graph.meas_angles[4]) == 1.4 * np.pi
    assert pytest.approx(zx_graph.meas_angles[5]) == 0.5 * np.pi
    assert pytest.approx(zx_graph.meas_angles[6]) == 1.6 * np.pi


def test_pivot_fails_with_nonexistent_nodes(zx_graph: ZXGraphState) -> None:
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        zx_graph.pivot(1, 2)
    zx_graph.add_physical_node(1)
    with pytest.raises(ValueError, match="Node does not exist node=2"):
        zx_graph.pivot(1, 2)


def test_pivot_fails_with_input_node(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.set_input(1)
    with pytest.raises(ValueError, match="Cannot apply pivot to input node"):
        zx_graph.pivot(1, 2)


def test_pivot_with_obvious_graph(zx_graph: ZXGraphState) -> None:
    # 1---2---3
    for i in range(1, 4):
        zx_graph.add_physical_node(i)

    for i, j in [(1, 2), (2, 3)]:
        zx_graph.add_physical_edge(i, j)

    measurements = [
        (1, Plane.XY, 1.1 * np.pi),
        (2, Plane.XZ, 1.2 * np.pi),
        (3, Plane.YZ, 1.3 * np.pi),
    ]
    for node_id, plane, angle in measurements:
        zx_graph.set_meas_plane(node_id, plane)
        zx_graph.set_meas_angle(node_id, angle)

    original_zx_graph = deepcopy(zx_graph)
    zx_graph.pivot(2, 3)
    original_zx_graph.local_complement(2)
    original_zx_graph.local_complement(3)
    original_zx_graph.local_complement(2)
    assert zx_graph.physical_edges == original_zx_graph.physical_edges
    assert zx_graph.meas_planes == original_zx_graph.meas_planes


def test_pivot_with_minimal_graph(zx_graph: ZXGraphState) -> None:
    # 1---2---3---5
    #      \ /
    #       4
    for i in range(1, 6):
        zx_graph.add_physical_node(i)

    for i, j in [(1, 2), (2, 3), (2, 4), (3, 4), (3, 5)]:
        zx_graph.add_physical_edge(i, j)

    measurements = [
        (1, Plane.XY, 1.1 * np.pi),
        (2, Plane.XZ, 1.2 * np.pi),
        (3, Plane.YZ, 1.3 * np.pi),
        (4, Plane.XY, 1.4 * np.pi),
        (5, Plane.XZ, 1.5 * np.pi),
    ]
    for node_id, plane, angle in measurements:
        zx_graph.set_meas_plane(node_id, plane)
        zx_graph.set_meas_angle(node_id, angle)

    original_edges = zx_graph.physical_edges.copy()
    expected_edges = {(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)}
    zx_graph.pivot(2, 3)
    assert zx_graph.physical_edges == expected_edges
    zx_graph.pivot(2, 3)
    assert zx_graph.physical_edges == original_edges
    zx_graph.pivot(3, 2)
    assert zx_graph.physical_edges == expected_edges
    zx_graph.pivot(3, 2)
    assert zx_graph.physical_edges == original_edges


def test_pivot_with_h_shaped_graph(zx_graph: ZXGraphState) -> None:
    # 3   6
    # |   |
    # 2---5
    # |   |
    # 1   4
    for i in range(1, 7):
        zx_graph.add_physical_node(i)
    for i, j in [(1, 2), (2, 3), (2, 5), (4, 5), (5, 6)]:
        zx_graph.add_physical_edge(i, j)

    measurements = [
        (1, Plane.XY, 1.1 * np.pi),
        (2, Plane.XZ, 1.2 * np.pi),
        (3, Plane.YZ, 1.3 * np.pi),
        (4, Plane.XY, 1.4 * np.pi),
        (5, Plane.XZ, 1.5 * np.pi),
        (6, Plane.YZ, 1.6 * np.pi),
    ]

    for node_id, plane, angle in measurements:
        zx_graph.set_meas_plane(node_id, plane)
        zx_graph.set_meas_angle(node_id, angle)

    original_edges = zx_graph.physical_edges.copy()
    expected_edges = {(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6)}
    zx_graph.pivot(2, 5)
    assert zx_graph.physical_edges == expected_edges
    zx_graph.pivot(2, 5)
    assert zx_graph.physical_edges == original_edges
    zx_graph.pivot(5, 2)
    assert zx_graph.physical_edges == expected_edges
    zx_graph.pivot(5, 2)
    assert zx_graph.physical_edges == original_edges


def test_pivot_with_8_nodes_graph(zx_graph: ZXGraphState) -> None:
    # 1   4   7
    #  \ / \ /
    #   3 - 6
    #  / \ / \
    # 2   5   8
    for i in range(1, 9):
        zx_graph.add_physical_node(i)

    for i, j in [(1, 3), (2, 3), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6), (6, 7), (6, 8)]:
        zx_graph.add_physical_edge(i, j)

    measurements = [
        (1, Plane.XY, 1.1),
        (2, Plane.XZ, 1.2),
        (3, Plane.YZ, 1.3),
        (4, Plane.XY, 1.4),
        (5, Plane.XZ, 1.5),
        (6, Plane.YZ, 1.6),
        (7, Plane.XY, 1.7),
        (8, Plane.XZ, 1.8),
    ]
    for node_id, plane, angle in measurements:
        zx_graph.set_meas_plane(node_id, plane)
        zx_graph.set_meas_angle(node_id, angle)

    original_edges = zx_graph.physical_edges.copy()
    expected_edges = {
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 7),
        (3, 8),
        (4, 6),
        (4, 7),
        (4, 8),
        (5, 6),
        (5, 7),
        (5, 8),
    }
    zx_graph.pivot(3, 6)
    assert zx_graph.physical_edges == expected_edges
    zx_graph.pivot(3, 6)
    assert zx_graph.physical_edges == original_edges
    zx_graph.pivot(6, 3)
    assert zx_graph.physical_edges == expected_edges
    zx_graph.pivot(6, 3)
    assert zx_graph.physical_edges == original_edges


if __name__ == "__main__":
    pytest.main()
