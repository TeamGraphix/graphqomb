"""Tests for the GraphState class."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from graphix_zx.common import Plane
from graphix_zx.graphstate import GraphState, bipartite_edges
from graphix_zx.zxgraphstate import ZXGraphState
from graphix_zx.random_objects import get_random_flow_graph


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


def _initialize_graph(
    zx_graph: ZXGraphState,
    nodes: range,
    edges: list[tuple[int, int]],
    inputs: tuple[int, ...] = (),
    outputs: tuple[int, ...] = (),
) -> None:
    """Initialize a ZXGraphState object with the given nodes and edges.

    Parameters
    ----------
    zx_graph : ZXGraphState
        The ZXGraphState object to initialize.
    nodes : range
        nodes to add to the graph.
    edges : list[tuple[int, int]]
        edges to add to the graph.
    inputs : tuple[int, ...], optional
        input nodes, by default ().
    outputs : tuple[int, ...], optional
        output nodes, by default ().
    """
    for i in nodes:
        zx_graph.add_physical_node(i)
    for i, j in edges:
        zx_graph.add_physical_edge(i, j)
    input_nodes = () if inputs is None else inputs
    for i in input_nodes:
        zx_graph.set_input(i)
    for i in outputs:
        zx_graph.set_output(i)


def _apply_measurements(zx_graph: ZXGraphState, measurements: list[tuple[int, Plane, float]]) -> None:
    for node_id, plane, angle in measurements:
        if node_id in zx_graph.output_nodes:
            continue
        zx_graph.set_meas_plane(node_id, plane)
        zx_graph.set_meas_angle(node_id, angle)


def _test(
    zx_graph: ZXGraphState,
    exp_nodes: set[int],
    exp_edges: set[tuple[int, int]],
    exp_measurements: list[tuple[int, Plane, float]],
) -> None:
    assert zx_graph.physical_nodes == exp_nodes
    assert zx_graph.physical_edges == exp_edges
    for node_id, plane, angle in exp_measurements:
        assert zx_graph.meas_planes[node_id] == plane
        assert pytest.approx(zx_graph.meas_angles[node_id]) == angle


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


def test_local_complement_fails_if_nonexistent_node(zx_graph: ZXGraphState) -> None:
    """Test local complement raises an error if the node does not exist."""
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        zx_graph.local_complement(1)
    zx_graph.add_physical_node(1)
    with pytest.raises(ValueError, match="Node does not exist node=2"):
        zx_graph.local_complement(2)


def test_local_complement_fails_if_not_zx_graph(zx_graph: ZXGraphState) -> None:
    """Test local complement raises an error if the graph is not a ZX-diagram."""
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        zx_graph.local_complement(1)
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    with pytest.raises(ValueError, match="Measurement basis not set for node 1"):
        zx_graph.local_complement(2)


def test_local_complement_fails_with_input_node(zx_graph: ZXGraphState) -> None:
    """Test local complement fails with input node."""
    zx_graph.add_physical_node(1)
    zx_graph.set_input(1)
    with pytest.raises(ValueError, match="Cannot apply local complement to input node."):
        zx_graph.local_complement(1)


def test_local_complement_with_no_edge(zx_graph: ZXGraphState) -> None:
    """Test local complement with a graph with no edge."""
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
    # this might be a bug in mypy, as it's useful comparison
    assert zx_graph.meas_planes[1] == Plane.XY  # type: ignore[comparison-overlap]
    assert pytest.approx(zx_graph.meas_angles[1]) == 0.6 * np.pi

    zx_graph.set_meas_plane(1, Plane.YZ)
    zx_graph.set_meas_angle(1, 1.1 * np.pi)
    zx_graph.local_complement(1)
    assert zx_graph.meas_planes[1] == Plane.YZ
    assert pytest.approx(zx_graph.meas_angles[1]) == 1.6 * np.pi


def test_local_complement_on_output_node(zx_graph: ZXGraphState) -> None:
    """Test local complement on an output node."""
    _initialize_graph(zx_graph, range(1, 4), [(1, 2), (2, 3)], outputs=(2,))
    measurements = [
        (1, Plane.XY, 1.1 * np.pi),
        (3, Plane.YZ, 1.3 * np.pi),
    ]
    _apply_measurements(zx_graph, measurements)
    zx_graph.local_complement(2)

    exp_measurements = [
        (1, Plane.XY, 0.6 * np.pi),
        (3, Plane.XZ, 0.7 * np.pi),
    ]
    _test(zx_graph, exp_nodes={1, 2, 3}, exp_edges={(1, 2), (1, 3), (2, 3)}, exp_measurements=exp_measurements)


def test_local_complement_with_two_nodes_graph(zx_graph: ZXGraphState) -> None:
    """Test local complement with a graph with two nodes."""
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
    """Test local complement with a minimal graph."""
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
    """Test local complement is applied 4 times and the graph goes back to the original shape."""
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
    """Test local complement with inversed planes (e.g. YX would be equivalent to XY)."""
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
    """Test local complement with an H-shaped graph."""
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
    """Test pivot fails with nonexistent nodes."""
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        zx_graph.pivot(1, 2)
    zx_graph.add_physical_node(1)
    with pytest.raises(ValueError, match="Node does not exist node=2"):
        zx_graph.pivot(1, 2)


def test_pivot_fails_with_input_node(zx_graph: ZXGraphState) -> None:
    """Test pivot fails with input node."""
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.set_input(1)
    with pytest.raises(ValueError, match="Cannot apply pivot to input node"):
        zx_graph.pivot(1, 2)


def test_pivot_with_obvious_graph(zx_graph: ZXGraphState) -> None:
    """Test pivot with an obvious graph."""
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
    """Test pivot with a minimal graph."""
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
    """Test pivot with an H-shaped graph."""
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
    """Test pivot with a graph with 8 nodes."""
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


def test_remove_clifford_fails_if_nonexistent_node(zx_graph: ZXGraphState) -> None:
    """Test remove_clifford raises an error if the node does not exist."""
    with pytest.raises(ValueError, match="Node does not exist node=1"):
        zx_graph.remove_clifford(1)


def test_remove_clifford_fails_with_input_node(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.set_input(1)
    with pytest.raises(ValueError, match="Clifford vertex removal not allowed for input node"):
        zx_graph.remove_clifford(1)


def test_remove_clifford_fails_with_invalid_plane(zx_graph: ZXGraphState) -> None:
    """Test remove_clifford fails if the measurement plane is invalid."""
    zx_graph.add_physical_node(1)
    zx_graph.set_meas_plane(1, "test_plane")  # type: ignore[reportArgumentType, arg-type, unused-ignore]
    zx_graph.set_meas_angle(1, 0.5 * np.pi)
    with pytest.raises(ValueError, match="This node is not a Clifford vertex"):
        zx_graph.remove_clifford(1)


def test_remove_clifford_fails_for_non_clifford_vertex(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.set_meas_plane(1, Plane.XY)
    zx_graph.set_meas_angle(1, 0.1 * np.pi)
    with pytest.raises(ValueError, match="This node is not a Clifford vertex"):
        zx_graph.remove_clifford(1)


def graph_1(zx_graph: ZXGraphState) -> None:
    # _needs_nop
    # 4---1---2      4       2
    #     |      ->
    #     3              3
    _initialize_graph(zx_graph, nodes=range(1, 5), edges=[(1, 2), (1, 3), (1, 4)])


def graph_2(zx_graph: ZXGraphState) -> None:
    # _needs_lc
    # 1---2---3  ->  1---3
    _initialize_graph(zx_graph, nodes=range(1, 4), edges=[(1, 2), (2, 3)])


def graph_3(zx_graph: ZXGraphState) -> None:
    # _needs_pivot_1 on (2, 3)
    #         4(I)                4(I)
    #         / \                / | \
    # 1(I) - 2 - 3 - 6  ->  1(I) - 3  6 - 1(I)
    #         \ /                \ | /
    #         5(I)                5(I)
    _initialize_graph(
        zx_graph, nodes=range(1, 7), edges=[(1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6)], inputs=(1, 4, 5)
    )


def graph_4(zx_graph: ZXGraphState) -> None:
    # _needs_pivot_2 on (2, 4)
    # 1(I) 3(O) 5(O)      1(I)-3(O)-5(O)-1(I)
    #   \  / \ /      ->         \ /
    #   2(O)- 4                  2(O)
    _initialize_graph(
        zx_graph,
        nodes=range(1, 6),
        edges=[(1, 2), (2, 3), (2, 4), (3, 4), (4, 5)],
        inputs=(1,),
        outputs=(2, 3, 5),
    )


def _test_remove_clifford(
    zx_graph: ZXGraphState,
    node: int,
    measurements: list[tuple[int, Plane, float]],
    exp_graph: tuple[set[int], set[tuple[int, int]]],
    exp_measurements: list[tuple[int, Plane, float]],
) -> None:
    _apply_measurements(zx_graph, measurements)
    zx_graph.remove_clifford(node)
    exp_nodes = exp_graph[0]
    exp_edges = exp_graph[1]
    _test(zx_graph, exp_nodes, exp_edges, exp_measurements)


def test_remove_clifford_removable_with_xz_0(zx_graph: ZXGraphState) -> None:
    """Test removing a removable Clifford vertex with measurement plane XZ and angle 0."""
    graph_1(zx_graph)
    measurements = [
        (1, Plane.XZ, 0),
        (2, Plane.XY, 0.1 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
    ]
    exp_measurements = [
        (2, Plane.XY, 0.1 * np.pi),
        (3, Plane.YZ, 1.8 * np.pi),
        (4, Plane.XZ, 0.3 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph, node=1, measurements=measurements, exp_graph=({2, 3, 4}, set()), exp_measurements=exp_measurements
    )


def test_remove_clifford_removable_with_xz_pi(zx_graph: ZXGraphState) -> None:
    """Test removing a removable Clifford vertex with measurement plane XZ and angle pi."""
    graph_1(zx_graph)
    measurements = [
        (1, Plane.XZ, np.pi),
        (2, Plane.XY, 0.1 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
    ]
    exp_measurements = [
        (2, Plane.XY, 1.1 * np.pi),
        (3, Plane.YZ, 1.8 * np.pi),
        (4, Plane.XZ, 0.3 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=1,
        measurements=measurements,
        exp_graph=({2, 3, 4}, set()),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_removable_with_yz_0(zx_graph: ZXGraphState) -> None:
    """Test removing a removable Clifford vertex with measurement plane YZ and angle 0."""
    graph_1(zx_graph)
    measurements = [
        (1, Plane.YZ, 0),
        (2, Plane.XY, 0.1 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
    ]
    exp_measurements = [
        (2, Plane.XY, 0.1 * np.pi),
        (3, Plane.YZ, 1.8 * np.pi),
        (4, Plane.XZ, 0.3 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=1,
        measurements=measurements,
        exp_graph=({2, 3, 4}, set()),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_removable_with_yz_pi(zx_graph: ZXGraphState) -> None:
    """Test removing a removable Clifford vertex with measurement plane YZ and angle pi."""
    graph_1(zx_graph)
    measurements = [
        (1, Plane.YZ, np.pi),
        (2, Plane.XY, 0.1 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
    ]
    exp_measurements = [
        (2, Plane.XY, 1.1 * np.pi),
        (3, Plane.YZ, 1.8 * np.pi),
        (4, Plane.XZ, 0.3 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=1,
        measurements=measurements,
        exp_graph=({2, 3, 4}, set()),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_lc_with_xy_0p5_pi(zx_graph: ZXGraphState) -> None:
    graph_2(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.XY, 0.5 * np.pi),
        (3, Plane.YZ, 0.2 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 1.6 * np.pi),
        (3, Plane.YZ, 0.2 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph, node=2, measurements=measurements, exp_graph=({1, 3}, {(1, 3)}), exp_measurements=exp_measurements
    )


def test_remove_clifford_lc_with_xy_1p5_pi(zx_graph: ZXGraphState) -> None:
    graph_2(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.XY, 1.5 * np.pi),
        (3, Plane.YZ, 0.2 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 0.6 * np.pi),
        (3, Plane.YZ, 0.2 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph, node=2, measurements=measurements, exp_graph=({1, 3}, {(1, 3)}), exp_measurements=exp_measurements
    )


def test_remove_clifford_lc_with_yz_0p5_pi(zx_graph: ZXGraphState) -> None:
    graph_2(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.YZ, 0.5 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 0.6 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=2,
        measurements=measurements,
        exp_graph=({1, 3}, {(1, 3)}),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_lc_with_yz_1p5_pi(zx_graph: ZXGraphState) -> None:
    graph_2(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.YZ, 1.5 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 1.6 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=2,
        measurements=measurements,
        exp_graph=({1, 3}, {(1, 3)}),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_pivot1_with_xy_0(zx_graph: ZXGraphState) -> None:
    graph_3(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.XY, 0),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.XZ, 0.5 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (3, Plane.YZ, 1.7 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 1.4 * np.pi),
        (6, Plane.YZ, 1.5 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=2,
        measurements=measurements,
        exp_graph=({1, 3, 4, 5, 6}, {(1, 3), (1, 4), (1, 5), (1, 6), (3, 4), (3, 5), (4, 6), (5, 6)}),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_pivot1_with_xy_pi(zx_graph: ZXGraphState) -> None:
    graph_3(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.XY, np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.XZ, 0.5 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (3, Plane.YZ, 1.7 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.YZ, 1.5 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=2,
        measurements=measurements,
        exp_graph=({1, 3, 4, 5, 6}, {(1, 3), (1, 4), (1, 5), (1, 6), (3, 4), (3, 5), (4, 6), (5, 6)}),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_pivot1_with_xz_0p5_pi(zx_graph: ZXGraphState) -> None:
    graph_3(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.XZ, 0.5 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.XZ, 0.5 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (3, Plane.YZ, 1.7 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 1.4 * np.pi),
        (6, Plane.YZ, 1.5 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=2,
        measurements=measurements,
        exp_graph=({1, 3, 4, 5, 6}, {(1, 3), (1, 4), (1, 5), (1, 6), (3, 4), (3, 5), (4, 6), (5, 6)}),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_pivot1_with_xz_1p5_pi(zx_graph: ZXGraphState) -> None:
    graph_3(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.XZ, 1.5 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.XZ, 0.5 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (3, Plane.YZ, 1.7 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.YZ, 1.5 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=2,
        measurements=measurements,
        exp_graph=({1, 3, 4, 5, 6}, {(1, 3), (1, 4), (1, 5), (1, 6), (3, 4), (3, 5), (4, 6), (5, 6)}),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_pivot2_with_xy_0(zx_graph: ZXGraphState) -> None:
    graph_4(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (4, Plane.XY, 0),
    ]
    exp_measurements = [
        (1, Plane.XY, 0.1 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=4,
        measurements=measurements,
        exp_graph=({1, 2, 3, 5}, {(1, 3), (1, 5), (2, 3), (2, 5), (3, 5)}),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_pivot2_with_xy_pi(zx_graph: ZXGraphState) -> None:
    graph_4(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (4, Plane.XY, np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 1.1 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=4,
        measurements=measurements,
        exp_graph=({1, 2, 3, 5}, {(1, 3), (1, 5), (2, 3), (2, 5), (3, 5)}),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_pivot2_with_xz_0p5_pi(zx_graph: ZXGraphState) -> None:
    graph_4(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (4, Plane.XZ, 0.5 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 0.1 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=4,
        measurements=measurements,
        exp_graph=({1, 2, 3, 5}, {(1, 3), (1, 5), (2, 3), (2, 5), (3, 5)}),
        exp_measurements=exp_measurements,
    )


def test_remove_clifford_pivot2_with_xz_1p5_pi(zx_graph: ZXGraphState) -> None:
    graph_4(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (4, Plane.XZ, 1.5 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 1.1 * np.pi),
    ]
    _test_remove_clifford(
        zx_graph,
        node=4,
        measurements=measurements,
        exp_graph=({1, 2, 3, 5}, {(1, 3), (1, 5), (2, 3), (2, 5), (3, 5)}),
        exp_measurements=exp_measurements,
    )


def test_unremovable_clifford_vertex(zx_graph: ZXGraphState) -> None:
    _initialize_graph(zx_graph, nodes=range(1, 4), edges=[(1, 2), (2, 3)], inputs=(1, 3))
    measurements = [
        (1, Plane.XY, 0.5 * np.pi),
        (2, Plane.XY, np.pi),
        (3, Plane.XY, 0.5 * np.pi),
    ]
    _apply_measurements(zx_graph, measurements)
    with pytest.raises(ValueError, match="This Clifford vertex is unremovable."):
        zx_graph.remove_clifford(2)


def test_remove_cliffords(zx_graph: ZXGraphState) -> None:
    """Test removing multiple Clifford vertices."""
    _initialize_graph(zx_graph, nodes=range(1, 5), edges=[(1, 2), (1, 3), (1, 4)])
    measurements = [
        (1, Plane.XY, 0.5 * np.pi),
        (2, Plane.XY, 0.5 * np.pi),
        (3, Plane.XY, 0.5 * np.pi),
        (4, Plane.XY, 0.5 * np.pi),
    ]
    _apply_measurements(zx_graph, measurements)
    zx_graph.remove_cliffords()
    _test(zx_graph, set(), set(), [])


def test_remove_cliffords_graph1(zx_graph: ZXGraphState) -> None:
    """Test removing multiple Clifford vertices."""
    graph_1(zx_graph)
    measurements = [
        (1, Plane.YZ, np.pi),
        (2, Plane.XY, 0.1 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
    ]
    exp_measurements = [
        (2, Plane.XY, 1.1 * np.pi),
        (3, Plane.YZ, 1.8 * np.pi),
        (4, Plane.XZ, 0.3 * np.pi),
    ]
    _apply_measurements(zx_graph, measurements)
    zx_graph.remove_cliffords()
    _test(zx_graph, {2, 3, 4}, set(), exp_measurements=exp_measurements)


def test_remove_cliffords_graph2(zx_graph: ZXGraphState) -> None:
    graph_2(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.YZ, 1.5 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 1.6 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
    ]
    _apply_measurements(zx_graph, measurements)
    zx_graph.remove_cliffords()
    _test(zx_graph, {1, 3}, {(1, 3)}, exp_measurements=exp_measurements)


def test_remove_cliffords_graph3(zx_graph: ZXGraphState) -> None:
    graph_3(zx_graph)
    measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.XZ, 1.5 * np.pi),
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.XZ, 0.5 * np.pi),
    ]
    exp_measurements = [
        (1, Plane.XY, 1.6 * np.pi),
        (3, Plane.YZ, 1.7 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 1.9 * np.pi),
    ]
    _apply_measurements(zx_graph, measurements)
    zx_graph.remove_cliffords()
    _test(zx_graph, {1, 3, 4, 5}, {(1, 3), (3, 4), (3, 5), (4, 5)}, exp_measurements=exp_measurements)


def test_remove_cliffords_graph4(zx_graph: ZXGraphState) -> None:
    """Test removing multiple Clifford vertices."""
    graph_4(zx_graph)
    measurements = [
        (1, Plane.XY, np.pi),
        (4, Plane.XZ, 0.5 * np.pi),
    ]
    _apply_measurements(zx_graph, measurements)
    zx_graph.remove_cliffords()
    _test(zx_graph, {1, 2, 3, 5}, {(1, 3), (1, 5), (2, 3), (2, 5), (3, 5)}, [(1, Plane.XY, np.pi)])


def test_random_graph(zx_graph: ZXGraphState) -> None:
    """Test removing multiple Clifford vertices from a random graph."""
    random_graph, flow = get_random_flow_graph(5, 5)
    zx_graph.append(random_graph)

    for i in zx_graph.physical_nodes - zx_graph.output_nodes:
        rnd = np.random.rand()
        if 0 <= rnd < 0.33:
            pass
        elif 0.33 <= rnd < 0.66:
            zx_graph.set_meas_plane(i, Plane.XZ)
        else:
            zx_graph.set_meas_plane(i, Plane.YZ)

    try:
        zx_graph.remove_cliffords()
    except Exception as e:
        print(random_graph)
        print(flow)
        raise e


if __name__ == "__main__":
    pytest.main()
