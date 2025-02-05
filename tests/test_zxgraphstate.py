from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.euler import is_clifford_angle
from graphix_zx.random_objects import get_random_flow_graph
from graphix_zx.zxgraphstate import ZXGraphState


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
        zx_graph.set_meas_basis(node_id, PlannerMeasBasis(plane, angle))


def _test(
    zx_graph: ZXGraphState,
    exp_nodes: set[int],
    exp_edges: set[tuple[int, int]],
    exp_measurements: list[tuple[int, Plane, float]],
) -> None:
    assert zx_graph.physical_nodes == exp_nodes
    assert zx_graph.physical_edges == exp_edges
    for node_id, plane, angle in exp_measurements:
        assert zx_graph.meas_bases[node_id].plane == plane
        assert np.isclose(zx_graph.meas_bases[node_id].angle, angle)


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
    zx_graph.set_meas_basis(1, PlannerMeasBasis(Plane.XY, 1.1 * np.pi))
    zx_graph.local_complement(1)
    assert zx_graph.physical_edges == set()
    assert zx_graph.meas_bases[1].plane == Plane.XZ
    assert np.isclose(zx_graph.meas_bases[1].angle, 1.4 * np.pi)

    zx_graph.set_meas_basis(1, PlannerMeasBasis(Plane.XZ, 1.1 * np.pi))
    zx_graph.local_complement(1)
    # this might be a bug in mypy, as it's useful comparison
    assert zx_graph.meas_bases[1].plane == Plane.XY  # type: ignore[comparison-overlap]
    assert np.isclose(zx_graph.meas_bases[1].angle, 0.6 * np.pi)

    zx_graph.set_meas_basis(1, PlannerMeasBasis(Plane.YZ, 1.1 * np.pi))
    zx_graph.local_complement(1)
    assert zx_graph.meas_bases[1].plane == Plane.YZ
    assert np.isclose(zx_graph.meas_bases[1].angle, 1.6 * np.pi)


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
    zx_graph.set_meas_basis(1, PlannerMeasBasis(Plane.XZ, 1.1 * np.pi))
    zx_graph.set_meas_basis(2, PlannerMeasBasis(Plane.XZ, 1.2 * np.pi))
    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(1)
    assert zx_graph.physical_edges == original_edges
    for node_id, plane, angle in [(1, Plane.XY, 0.6 * np.pi), (2, Plane.YZ, 1.2 * np.pi)]:
        assert zx_graph.meas_bases[node_id].plane == plane
        assert is_clifford_angle(zx_graph.meas_bases[node_id].angle, angle)


def test_local_complement_with_minimal_graph(zx_graph: ZXGraphState) -> None:
    """Test local complement with a minimal graph."""
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_node(3)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.add_physical_edge(2, 3)
    zx_graph.set_meas_basis(1, PlannerMeasBasis(Plane.XY, 1.1 * np.pi))
    zx_graph.set_meas_basis(2, PlannerMeasBasis(Plane.XZ, 1.2 * np.pi))
    zx_graph.set_meas_basis(3, PlannerMeasBasis(Plane.YZ, 1.3 * np.pi))
    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == {(1, 2), (2, 3), (1, 3)}
    exp_measurements = [
        (1, Plane.XY, 0.6 * np.pi),
        (2, Plane.XY, 0.7 * np.pi),
        (3, Plane.XZ, 0.7 * np.pi),
    ]
    for node_id, plane, angle in exp_measurements:
        assert zx_graph.meas_bases[node_id].plane == plane
        assert is_clifford_angle(zx_graph.meas_bases[node_id].angle, angle)

    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == original_edges
    exp_measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.XZ, 1.8 * np.pi),
        (3, Plane.YZ, 0.7 * np.pi),
    ]
    for node_id, plane, angle in exp_measurements:
        assert zx_graph.meas_bases[node_id].plane == plane
        assert np.isclose(zx_graph.meas_bases[node_id].angle, angle)


def test_local_complement_4_times(zx_graph: ZXGraphState) -> None:
    """Test local complement is applied 4 times and the graph goes back to the original shape."""
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_node(3)
    zx_graph.add_physical_edge(1, 2)
    zx_graph.add_physical_edge(2, 3)
    zx_graph.set_meas_basis(1, PlannerMeasBasis(Plane.XY, 1.1 * np.pi))
    zx_graph.set_meas_basis(2, PlannerMeasBasis(Plane.XZ, 1.2 * np.pi))
    zx_graph.set_meas_basis(3, PlannerMeasBasis(Plane.YZ, 1.3 * np.pi))
    original_edges = zx_graph.physical_edges.copy()
    for _ in range(4):
        zx_graph.local_complement(2)
    assert zx_graph.physical_edges == original_edges
    exp_measurements = [
        (1, Plane.XY, 1.1 * np.pi),
        (2, Plane.XZ, 1.2 * np.pi),
        (3, Plane.YZ, 1.3 * np.pi),
    ]
    for node_id, plane, angle in exp_measurements:
        assert zx_graph.meas_bases[node_id].plane == plane
        assert np.isclose(zx_graph.meas_bases[node_id].angle, angle)


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
    _apply_measurements(zx_graph, measurements)

    original_edges = zx_graph.physical_edges.copy()
    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == {(1, 2), (1, 3), (1, 5), (2, 3), (2, 5), (3, 5), (4, 5), (5, 6)}
    exp_measurements = [
        (1, Plane.XY, 0.6 * np.pi),
        (2, Plane.XY, 0.7 * np.pi),
        (3, Plane.XZ, 0.7 * np.pi),
        (4, Plane.XY, 1.4 * np.pi),
        (5, Plane.YZ, 1.5 * np.pi),
        (6, Plane.YZ, 1.6 * np.pi),
    ]
    for node_id, plane, angle in exp_measurements:
        assert zx_graph.meas_bases[node_id].plane == plane
        assert np.isclose(zx_graph.meas_bases[node_id].angle, angle)

    zx_graph.local_complement(2)
    assert zx_graph.physical_edges == original_edges
    exp_measurements = [
        (1, Plane.XY, 0.1 * np.pi),
        (2, Plane.XZ, 1.8 * np.pi),
        (3, Plane.YZ, 0.7 * np.pi),
        (4, Plane.XY, 1.4 * np.pi),
        (5, Plane.XZ, 0.5 * np.pi),
        (6, Plane.YZ, 1.6 * np.pi),
    ]
    for node_id, plane, angle in exp_measurements:
        assert zx_graph.meas_bases[node_id].plane == plane
        assert np.isclose(zx_graph.meas_bases[node_id].angle, angle)


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
    _apply_measurements(zx_graph, measurements)

    original_zx_graph = deepcopy(zx_graph)
    zx_graph.pivot(2, 3)
    original_zx_graph.local_complement(2)
    original_zx_graph.local_complement(3)
    original_zx_graph.local_complement(2)
    assert zx_graph.physical_edges == original_zx_graph.physical_edges
    original_planes = [original_zx_graph.meas_bases[i].plane for i in range(1, 4)]
    planes = [zx_graph.meas_bases[i].plane for i in range(1, 4)]
    assert planes == original_planes


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
    _apply_measurements(zx_graph, measurements)

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
    _apply_measurements(zx_graph, measurements)

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
    _apply_measurements(zx_graph, measurements)

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
    zx_graph.set_meas_basis(
        1,
        PlannerMeasBasis("test_plane", 0.5 * np.pi),  # type: ignore[reportArgumentType, arg-type, unused-ignore]
    )
    with pytest.raises(ValueError, match="This node is not a Clifford vertex"):
        zx_graph.remove_clifford(1)


def test_remove_clifford_fails_for_non_clifford_vertex(zx_graph: ZXGraphState) -> None:
    zx_graph.add_physical_node(1)
    zx_graph.set_meas_basis(1, PlannerMeasBasis(Plane.XY, 0.1 * np.pi))
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
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
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
        (3, Plane.XZ, 1.8 * np.pi),
        (4, Plane.YZ, 1.7 * np.pi),
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
        (3, Plane.XZ, 0.2 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
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
        (3, Plane.XZ, 1.8 * np.pi),
        (4, Plane.YZ, 1.7 * np.pi),
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
        (3, Plane.XZ, 1.8 * np.pi),
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
        (3, Plane.XZ, 0.2 * np.pi),
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
        (3, Plane.YZ, 1.8 * np.pi),
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
        (3, Plane.YZ, 0.2 * np.pi),
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
        (3, Plane.XZ, 0.3 * np.pi),
        (4, Plane.YZ, 1.7 * np.pi),
        (5, Plane.XY, 1.4 * np.pi),
        (6, Plane.XZ, 0.5 * np.pi),
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
        (3, Plane.XZ, 1.7 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.XZ, 1.5 * np.pi),
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
        (3, Plane.XZ, 0.3 * np.pi),
        (4, Plane.YZ, 1.7 * np.pi),
        (5, Plane.XY, 1.4 * np.pi),
        (6, Plane.XZ, 0.5 * np.pi),
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
        (3, Plane.XZ, 1.7 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.XZ, 1.5 * np.pi),
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
    _test(zx_graph, {3}, set(), [])


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
        (3, Plane.XZ, 1.8 * np.pi),
        (4, Plane.YZ, 1.7 * np.pi),
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
        (3, Plane.YZ, 0.2 * np.pi),
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
        (1, Plane.XY, 0.1 * np.pi),
        (3, Plane.XZ, 1.7 * np.pi),
        (4, Plane.YZ, 0.3 * np.pi),
        (5, Plane.XY, 0.4 * np.pi),
        (6, Plane.XZ, 1.5 * np.pi),
    ]
    _apply_measurements(zx_graph, measurements)
    zx_graph.remove_cliffords()
    _test(
        zx_graph,
        {1, 3, 4, 5, 6},
        {(1, 3), (1, 4), (1, 5), (1, 6), (3, 4), (3, 5), (4, 6), (5, 6)},
        exp_measurements=exp_measurements,
    )


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
    random_graph, _ = get_random_flow_graph(5, 5)
    zx_graph.append(random_graph)

    for i in zx_graph.physical_nodes - zx_graph.output_nodes:
        rng = np.random.default_rng(seed=0)
        rnd = rng.random()
        if 0 <= rnd < 0.33:
            pass
        elif 0.33 <= rnd < 0.66:
            angle = zx_graph.meas_bases[i].angle
            zx_graph.set_meas_basis(i, PlannerMeasBasis(Plane.XZ, angle))
        else:
            angle = zx_graph.meas_bases[i].angle
            zx_graph.set_meas_basis(i, PlannerMeasBasis(Plane.YZ, angle))

    zx_graph.remove_cliffords()
    atol = 1e-9
    nodes = zx_graph.physical_nodes - zx_graph.input_nodes - zx_graph.output_nodes
    clifford_nodes = [
        node
        for node in nodes
        if is_clifford_angle(zx_graph.meas_bases[node].angle, atol) and zx_graph.is_removable_clifford(node, atol)
    ]
    assert clifford_nodes == []


@pytest.mark.parametrize(
    ("measurements", "exp_measurements", "exp_edges"),
    [
        # no pair of adjacent nodes with YZ measurements
        # and no node with XZ measurement
        (
            [
                (1, Plane.XY, 0.11 * np.pi),
                (2, Plane.XY, 0.22 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.XY, 0.44 * np.pi),
                (5, Plane.XY, 0.55 * np.pi),
                (6, Plane.XY, 0.66 * np.pi),
            ],
            [
                (1, Plane.XY, 0.11 * np.pi),
                (2, Plane.XY, 0.22 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.XY, 0.44 * np.pi),
                (5, Plane.XY, 0.55 * np.pi),
                (6, Plane.XY, 0.66 * np.pi),
            ],
            {(1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6)},
        ),
        # a pair of adjacent nodes with YZ measurements
        #             4(XY)                           4(XY)     4       4       4
        #            /     \                            |       |       |       |
        # 1(XY) - 2(YZ) - 3(YZ) - 6(XY)  ->  1(XY) - 3(XY) - 2(XY) - 6(XY) - 1
        #            \     /                            |       |       |       |
        #             5(XY)                            5(XY)    5       5       5
        (
            [
                (1, Plane.XY, 0.11 * np.pi),
                (2, Plane.YZ, 0.22 * np.pi),
                (3, Plane.YZ, 0.33 * np.pi),
                (4, Plane.XY, 0.44 * np.pi),
                (5, Plane.XY, 0.55 * np.pi),
                (6, Plane.XY, 0.66 * np.pi),
            ],
            [
                (1, Plane.XY, 0.11 * np.pi),
                (2, Plane.XY, 0.22 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.XY, 1.44 * np.pi),
                (5, Plane.XY, 1.55 * np.pi),
                (6, Plane.XY, 0.66 * np.pi),
            ],
            {(1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (4, 6), (5, 6)},
        ),
        # no pair of adjacent nodes with YZ measurements
        # but a node with XZ measurement
        #             4(XZ)                             4(XY)
        #            /     \                           /     \
        # 1(XY) - 2(XY) - 3(XY) - 6(XY)  ->  1(XY) - 2(XY)   3(XY) - 6(XY)
        #            \     /                           \     /
        #             5(XY)                             5(XY)
        (
            [
                (1, Plane.XY, 0.11 * np.pi),
                (2, Plane.XY, 0.22 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.XZ, 0.44 * np.pi),
                (5, Plane.XY, 0.55 * np.pi),
                (6, Plane.XY, 0.66 * np.pi),
            ],
            [
                (1, Plane.XY, 0.11 * np.pi),
                (2, Plane.XY, 1.72 * np.pi),
                (3, Plane.XY, 1.83 * np.pi),
                (4, Plane.XY, 1.94 * np.pi),
                (5, Plane.XY, 0.55 * np.pi),
                (6, Plane.XY, 0.66 * np.pi),
            ],
            {(1, 2), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6)},
        ),
        # a pair of adjacent nodes with YZ measurements
        # and a node with XZ measurement
        #             4(XZ)                  6(YZ) - 3(XY)
        #            /     \                   |   x    |
        # 1(XZ) - 2(YZ) - 3(YZ) - 6(XZ)  ->  1(XY)   2(XY)
        #            \     /                   |   x    |
        #             5(XZ)                  5(XY) - 4(XY)
        (
            [
                (1, Plane.XZ, 0.11 * np.pi),
                (2, Plane.YZ, 0.22 * np.pi),
                (3, Plane.YZ, 0.33 * np.pi),
                (4, Plane.XZ, 0.44 * np.pi),
                (5, Plane.XZ, 0.55 * np.pi),
                (6, Plane.XZ, 0.66 * np.pi),
            ],
            [
                (1, Plane.XY, 0.61 * np.pi),
                (2, Plane.XY, 1.22 * np.pi),
                (3, Plane.XY, 1.83 * np.pi),
                (4, Plane.XY, 1.56 * np.pi),
                (5, Plane.XY, 1.45 * np.pi),
                (6, Plane.YZ, 0.66 * np.pi),
            ],
            {(1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 6), (4, 5)},
        ),
    ],
)
def test_convert_to_phase_gadget(
    zx_graph: ZXGraphState,
    measurements: list[tuple[int, Plane, float]],
    exp_measurements: list[tuple[int, Plane, float]],
    exp_edges: set[tuple[int, int]],
) -> None:
    _initialize_graph(
        zx_graph,
        nodes=range(1, 7),
        edges=[(1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6)],
    )
    _apply_measurements(zx_graph, measurements)
    zx_graph.convert_to_phase_gadget()
    _test(
        zx_graph,
        exp_nodes={1, 2, 3, 4, 5, 6},
        exp_edges=exp_edges,
        exp_measurements=exp_measurements,
    )


if __name__ == "__main__":
    pytest.main()
