"""Tests for ZXGraphState

Measurement actions for the followings are used:
    - Local complement (LC): MEAS_ACTION_LC_*
    - Pivot (PV): MEAS_ACTION_PV_*
    - Remove Cliffords (RC): MEAS_ACTION_RC

Reference:
    M. Backens et al., Quantum 5, 421 (2021).
    https://doi.org/10.22331/q-2021-03-25-421
"""

from __future__ import annotations

import itertools
import operator
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.euler import _is_close_angle
from graphix_zx.random_objects import get_random_flow_graph
from graphix_zx.zxgraphstate import ZXGraphState

if TYPE_CHECKING:
    from typing import Callable

    Measurements = list[tuple[int, Plane, float]]

MEAS_ACTION_LC_TARGET: dict[Plane, tuple[Plane, Callable[[float], float]]] = {
    Plane.XY: (Plane.XZ, lambda angle: angle + np.pi / 2),
    Plane.XZ: (Plane.XY, lambda angle: -angle + np.pi / 2),
    Plane.YZ: (Plane.YZ, lambda angle: angle + np.pi / 2),
}
MEAS_ACTION_LC_NEIGHBORS: dict[Plane, tuple[Plane, Callable[[float], float]]] = {
    Plane.XY: (Plane.XY, lambda angle: angle + np.pi / 2),
    Plane.XZ: (Plane.YZ, lambda angle: angle),
    Plane.YZ: (Plane.XZ, operator.neg),
}
MEAS_ACTION_PV_TARGET: dict[Plane, tuple[Plane, Callable[[float], float]]] = {
    Plane.XY: (Plane.YZ, operator.neg),
    Plane.XZ: (Plane.XZ, lambda angle: (np.pi / 2 - angle)),
    Plane.YZ: (Plane.XY, operator.neg),
}
MEAS_ACTION_PV_NEIGHBORS: dict[Plane, tuple[Plane, Callable[[float], float]]] = {
    Plane.XY: (Plane.XY, lambda angle: (angle + np.pi) % (2.0 * np.pi)),
    Plane.XZ: (Plane.XZ, lambda angle: -angle % (2.0 * np.pi)),
    Plane.YZ: (Plane.YZ, lambda angle: -angle % (2.0 * np.pi)),
}
ATOL = 1e-9
MEAS_ACTION_RC: dict[Plane, tuple[Plane, Callable[[float, float], float]]] = {
    Plane.XY: (
        Plane.XY,
        lambda a_pi, alpha: (alpha if _is_close_angle(a_pi, 0, ATOL) else alpha + np.pi) % (2.0 * np.pi),
    ),
    Plane.XZ: (
        Plane.XZ,
        lambda a_pi, alpha: (alpha if _is_close_angle(a_pi, 0, ATOL) else -alpha) % (2.0 * np.pi),
    ),
    Plane.YZ: (
        Plane.YZ,
        lambda a_pi, alpha: (alpha if _is_close_angle(a_pi, 0, ATOL) else -alpha) % (2.0 * np.pi),
    ),
}


def plane_combinations(n: int) -> list[tuple[Plane, ...]]:
    """Generate all combinations of planes of length n.

    Parameters
    ----------
    n : int
        The length of the combinations. n > 1.

    Returns
    -------
    list[tuple[Plane, ...]]
        A list of tuples containing all combinations of planes of length n.
    """
    return list(itertools.product(Plane, repeat=n))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


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
    edges: set[tuple[int, int]],
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


def _apply_measurements(zx_graph: ZXGraphState, measurements: Measurements) -> None:
    for node_id, plane, angle in measurements:
        if node_id in zx_graph.output_nodes:
            continue
        zx_graph.set_meas_basis(node_id, PlannerMeasBasis(plane, angle))


def _test(
    zx_graph: ZXGraphState,
    exp_nodes: set[int],
    exp_edges: set[tuple[int, int]],
    exp_measurements: Measurements,
) -> None:
    assert zx_graph.physical_nodes == exp_nodes
    assert zx_graph.physical_edges == exp_edges
    for node_id, plane, angle in exp_measurements:
        assert zx_graph.meas_bases[node_id].plane == plane
        assert _is_close_angle(zx_graph.meas_bases[node_id].angle, angle)


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
    with pytest.raises(ValueError, match=r"Cannot apply local complement to input node."):
        zx_graph.local_complement(1)


@pytest.mark.parametrize("plane", list(Plane))
def test_local_complement_with_no_edge(zx_graph: ZXGraphState, plane: Plane, rng: np.random.Generator) -> None:
    angle = rng.random() * 2 * np.pi
    ref_plane, ref_angle_func = MEAS_ACTION_LC_TARGET[plane]
    ref_angle = ref_angle_func(angle)
    zx_graph.add_physical_node(1)
    zx_graph.set_meas_basis(1, PlannerMeasBasis(plane, angle))

    zx_graph.local_complement(1)
    assert zx_graph.physical_edges == set()
    assert zx_graph.meas_bases[1].plane == ref_plane
    assert _is_close_angle(zx_graph.meas_bases[1].angle, ref_angle)


@pytest.mark.parametrize(("plane1", "plane3"), plane_combinations(2))
def test_local_complement_on_output_node(
    zx_graph: ZXGraphState, plane1: Plane, plane3: Plane, rng: np.random.Generator
) -> None:
    """Test local complement on an output node."""
    _initialize_graph(zx_graph, range(1, 4), {(1, 2), (2, 3)}, outputs=(2,))
    angle1 = rng.random() * 2 * np.pi
    angle3 = rng.random() * 2 * np.pi
    measurements = [(1, plane1, angle1), (3, plane3, angle3)]
    _apply_measurements(zx_graph, measurements)
    zx_graph.local_complement(2)

    ref_plane1, ref_angle_func1 = MEAS_ACTION_LC_NEIGHBORS[plane1]
    ref_plane3, ref_angle_func3 = MEAS_ACTION_LC_NEIGHBORS[plane3]
    exp_measurements = [
        (1, ref_plane1, ref_angle_func1(measurements[0][2])),
        (3, ref_plane3, ref_angle_func3(measurements[1][2])),
    ]
    _test(zx_graph, exp_nodes={1, 2, 3}, exp_edges={(1, 2), (1, 3), (2, 3)}, exp_measurements=exp_measurements)
    assert zx_graph.meas_bases.get(2) is None


@pytest.mark.parametrize(("plane1", "plane2"), plane_combinations(2))
def test_local_complement_with_two_nodes_graph(
    zx_graph: ZXGraphState, plane1: Plane, plane2: Plane, rng: np.random.Generator
) -> None:
    """Test local complement with a graph with two nodes."""
    zx_graph.add_physical_node(1)
    zx_graph.add_physical_node(2)
    zx_graph.add_physical_edge(1, 2)
    angle1 = rng.random() * 2 * np.pi
    angle2 = rng.random() * 2 * np.pi
    zx_graph.set_meas_basis(1, PlannerMeasBasis(plane1, angle1))
    zx_graph.set_meas_basis(2, PlannerMeasBasis(plane2, angle2))
    zx_graph.local_complement(1)

    ref_plane1, ref_angle_func1 = MEAS_ACTION_LC_TARGET[plane1]
    ref_plane2, ref_angle_func2 = MEAS_ACTION_LC_NEIGHBORS[plane2]
    exp_measurements = [(1, ref_plane1, ref_angle_func1(angle1)), (2, ref_plane2, ref_angle_func2(angle2))]
    _test(zx_graph, exp_nodes={1, 2}, exp_edges={(1, 2)}, exp_measurements=exp_measurements)


@pytest.mark.parametrize("planes", plane_combinations(3))
def test_local_complement_with_minimal_graph(
    zx_graph: ZXGraphState, planes: tuple[Plane, Plane, Plane], rng: np.random.Generator
) -> None:
    """Test local complement with a minimal graph."""
    for i in range(1, 4):
        zx_graph.add_physical_node(i)
    for i, j in [(1, 2), (2, 3)]:
        zx_graph.add_physical_edge(i, j)
    angles = [rng.random() * 2 * np.pi for _ in range(3)]
    for i in range(1, 4):
        zx_graph.set_meas_basis(i, PlannerMeasBasis(planes[i - 1], angles[i - 1]))
    zx_graph.local_complement(2)
    ref_plane1, ref_angle_func1 = MEAS_ACTION_LC_NEIGHBORS[planes[0]]
    ref_plane2, ref_angle_func2 = MEAS_ACTION_LC_TARGET[planes[1]]
    ref_plane3, ref_angle_func3 = MEAS_ACTION_LC_NEIGHBORS[planes[2]]
    ref_angle1 = ref_angle_func1(angles[0])
    ref_angle2 = ref_angle_func2(angles[1])
    ref_angle3 = ref_angle_func3(angles[2])
    exp_measurements = [
        (1, ref_plane1, ref_angle1),
        (2, ref_plane2, ref_angle2),
        (3, ref_plane3, ref_angle3),
    ]
    _test(zx_graph, exp_nodes={1, 2, 3}, exp_edges={(1, 2), (2, 3), (1, 3)}, exp_measurements=exp_measurements)

    zx_graph.local_complement(2)
    ref_plane1, ref_angle_func1 = MEAS_ACTION_LC_NEIGHBORS[ref_plane1]
    ref_plane2, ref_angle_func2 = MEAS_ACTION_LC_TARGET[ref_plane2]
    ref_plane3, ref_angle_func3 = MEAS_ACTION_LC_NEIGHBORS[ref_plane3]
    exp_measurements = [
        (1, ref_plane1, ref_angle_func1(ref_angle1)),
        (2, ref_plane2, ref_angle_func2(ref_angle2)),
        (3, ref_plane3, ref_angle_func3(ref_angle3)),
    ]
    _test(zx_graph, exp_nodes={1, 2, 3}, exp_edges={(1, 2), (2, 3)}, exp_measurements=exp_measurements)


@pytest.mark.parametrize("planes", plane_combinations(3))
def test_local_complement_4_times(
    zx_graph: ZXGraphState, planes: tuple[Plane, Plane, Plane], rng: np.random.Generator
) -> None:
    """Test local complement is applied 4 times and the graph goes back to the original shape."""
    for i in range(1, 4):
        zx_graph.add_physical_node(i)
    for i, j in [(1, 2), (2, 3)]:
        zx_graph.add_physical_edge(i, j)
    angles = [rng.random() * 2 * np.pi for _ in range(3)]
    for i in range(1, 4):
        zx_graph.set_meas_basis(i, PlannerMeasBasis(planes[i - 1], angles[i - 1]))

    for _ in range(4):
        zx_graph.local_complement(2)

    exp_measurements = [(i, planes[i - 1], angles[i - 1]) for i in range(1, 4)]
    _test(zx_graph, exp_nodes={1, 2, 3}, exp_edges={(1, 2), (2, 3)}, exp_measurements=exp_measurements)


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


@pytest.mark.parametrize("planes", plane_combinations(5))
def test_pivot_with_minimal_graph(
    zx_graph: ZXGraphState, planes: tuple[Plane, Plane, Plane, Plane, Plane], rng: np.random.Generator
) -> None:
    """Test pivot with a minimal graph."""
    # 1---2---3---5
    #      \ /
    #       4
    for i in range(1, 6):
        zx_graph.add_physical_node(i)

    for i, j in [(1, 2), (2, 3), (2, 4), (3, 4), (3, 5)]:
        zx_graph.add_physical_edge(i, j)

    angles = [rng.random() * 2 * np.pi for _ in range(5)]
    measurements = [(i, planes[i - 1], angles[i - 1]) for i in range(1, 6)]
    _apply_measurements(zx_graph, measurements)
    zx_graph_cp = deepcopy(zx_graph)

    zx_graph.pivot(2, 3)
    zx_graph_cp.local_complement(2)
    zx_graph_cp.local_complement(3)
    zx_graph_cp.local_complement(2)
    assert zx_graph.physical_edges == zx_graph_cp.physical_edges
    assert zx_graph.meas_bases[2].plane == zx_graph_cp.meas_bases[2].plane
    assert zx_graph.meas_bases[3].plane == zx_graph_cp.meas_bases[3].plane

    _, ref_angle_func2 = MEAS_ACTION_PV_TARGET[planes[1]]
    _, ref_angle_func3 = MEAS_ACTION_PV_TARGET[planes[2]]
    _, ref_angle_func4 = MEAS_ACTION_PV_NEIGHBORS[planes[3]]
    ref_angle2 = ref_angle_func2(angles[1])
    ref_angle3 = ref_angle_func3(angles[2])
    ref_angle4 = ref_angle_func4(angles[3])
    assert _is_close_angle(zx_graph.meas_bases[2].angle, ref_angle2)
    assert _is_close_angle(zx_graph.meas_bases[3].angle, ref_angle3)
    assert _is_close_angle(zx_graph.meas_bases[4].angle, ref_angle4)


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
    _initialize_graph(zx_graph, nodes=range(1, 5), edges={(1, 2), (1, 3), (1, 4)})


def graph_2(zx_graph: ZXGraphState) -> None:
    # _needs_lc
    # 1---2---3  ->  1---3
    _initialize_graph(zx_graph, nodes=range(1, 4), edges={(1, 2), (2, 3)})


def graph_3(zx_graph: ZXGraphState) -> None:
    # _needs_pivot_1 on (2, 3)
    #         4(I)                4(I)
    #         / \                / | \
    # 1(I) - 2 - 3 - 6  ->  1(I) - 3  6 - 1(I)
    #         \ /                \ | /
    #         5(I)                5(I)
    _initialize_graph(
        zx_graph, nodes=range(1, 7), edges={(1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6)}, inputs=(1, 4, 5)
    )


def graph_4(zx_graph: ZXGraphState) -> None:
    # _needs_pivot_2 on (2, 4)
    # 1(I) 3(O) 5(O)      1(I)-3(O)-5(O)-1(I)
    #   \  / \ /      ->         \ /
    #   2(O)- 4                  2(O)
    _initialize_graph(
        zx_graph,
        nodes=range(1, 6),
        edges={(1, 2), (2, 3), (2, 4), (3, 4), (4, 5)},
        inputs=(1,),
        outputs=(2, 3, 5),
    )


def _test_remove_clifford(
    zx_graph: ZXGraphState,
    node: int,
    measurements: Measurements,
    exp_graph: tuple[set[int], set[tuple[int, int]]],
    exp_measurements: Measurements,
) -> None:
    _apply_measurements(zx_graph, measurements)
    zx_graph.remove_clifford(node)
    exp_nodes = exp_graph[0]
    exp_edges = exp_graph[1]
    _test(zx_graph, exp_nodes, exp_edges, exp_measurements)


@pytest.mark.parametrize(
    "planes",
    list(itertools.product(list(Plane), [Plane.XZ, Plane.YZ], list(Plane))),
)
def test_remove_clifford(
    zx_graph: ZXGraphState,
    planes: tuple[Plane, Plane, Plane],
    rng: np.random.Generator,
) -> None:
    graph_2(zx_graph)
    angles = [rng.random() * 2 * np.pi for _ in range(3)]
    angles[1] = rng.choice([0.0, np.pi])
    measurements = [(i, planes[i - 1], angles[i - 1]) for i in range(1, 4)]
    ref_plane1, ref_angle_func1 = MEAS_ACTION_RC[planes[0]]
    ref_plane3, ref_angle_func3 = MEAS_ACTION_RC[planes[2]]
    ref_angle1 = ref_angle_func1(angles[1], angles[0])
    ref_angle3 = ref_angle_func3(angles[1], angles[2])
    exp_measurements = [
        (1, ref_plane1, ref_angle1),
        (3, ref_plane3, ref_angle3),
    ]
    _test_remove_clifford(
        zx_graph, node=2, measurements=measurements, exp_graph=({1, 3}, set()), exp_measurements=exp_measurements
    )


def test_unremovable_clifford_vertex(zx_graph: ZXGraphState) -> None:
    _initialize_graph(zx_graph, nodes=range(1, 4), edges={(1, 2), (2, 3)}, inputs=(1, 3))
    measurements = [
        (1, Plane.XY, 0.5 * np.pi),
        (2, Plane.XY, np.pi),
        (3, Plane.XY, 0.5 * np.pi),
    ]
    _apply_measurements(zx_graph, measurements)
    with pytest.raises(ValueError, match=r"This Clifford vertex is unremovable."):
        zx_graph.remove_clifford(2)


def test_remove_cliffords(zx_graph: ZXGraphState) -> None:
    """Test removing multiple Clifford vertices."""
    _initialize_graph(zx_graph, nodes=range(1, 5), edges={(1, 2), (1, 3), (1, 4)})
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
        (1, Plane.XY, 0.6 * np.pi),
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
    clifford_nodes = [node for node in nodes if zx_graph.is_removable_clifford(node, atol)]
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
    ],
)
def test_convert_to_phase_gadget(
    zx_graph: ZXGraphState,
    measurements: Measurements,
    exp_measurements: Measurements,
    exp_edges: set[tuple[int, int]],
) -> None:
    initial_edges = {(1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6)}
    _initialize_graph(zx_graph, nodes=range(1, 7), edges=initial_edges)
    _apply_measurements(zx_graph, measurements)
    zx_graph.convert_to_phase_gadget()
    _test(zx_graph, exp_nodes={1, 2, 3, 4, 5, 6}, exp_edges=exp_edges, exp_measurements=exp_measurements)


@pytest.mark.parametrize(
    ("initial_edges", "measurements", "exp_measurements", "exp_edges"),
    [
        #         4(XY)              4(XY)
        #          |             ->   |
        # 1(YZ) - 2(XY) - 3(XY)      2(XY) - 3(XY)
        (
            {(1, 2), (2, 3), (2, 4)},
            [
                (1, Plane.YZ, 0.11 * np.pi),
                (2, Plane.XY, 0.22 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.XY, 0.44 * np.pi),
            ],
            [
                (2, Plane.XY, 0.33 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.XY, 0.44 * np.pi),
            ],
            {(2, 3), (2, 4)},
        ),
        #         4(YZ)              4(YZ)
        #          |    \        ->   |    \
        # 1(YZ) - 2(XY) - 3(XY)      2(XY) - 3(XY)
        (
            {(1, 2), (2, 3), (2, 4), (3, 4)},
            [
                (1, Plane.YZ, 0.11 * np.pi),
                (2, Plane.XY, 0.22 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.YZ, 0.44 * np.pi),
            ],
            [
                (2, Plane.XY, 0.33 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.YZ, 0.44 * np.pi),
            ],
            {(2, 3), (2, 4), (3, 4)},
        ),
    ],
)
def test_merge_yz_to_xy(
    zx_graph: ZXGraphState,
    initial_edges: set[tuple[int, int]],
    measurements: Measurements,
    exp_measurements: Measurements,
    exp_edges: set[tuple[int, int]],
) -> None:
    _initialize_graph(zx_graph, nodes=range(1, 5), edges=initial_edges)
    _apply_measurements(zx_graph, measurements)
    zx_graph.merge_yz_to_xy()
    _test(zx_graph, exp_nodes={2, 3, 4}, exp_edges=exp_edges, exp_measurements=exp_measurements)


@pytest.mark.parametrize(
    ("initial_edges", "measurements", "exp_zxgraph"),
    [
        #         4(YZ)                    4(YZ)
        #       /       \                /       \
        # 1(XY) - 2(XY) - 3(XY) -> 1(XY) - 2(XY) - 3(XY)
        #       \       /
        #         5(YZ)
        (
            {(1, 2), (1, 4), (1, 5), (2, 3), (3, 4), (3, 5)},
            [
                (1, Plane.XY, 0.11 * np.pi),
                (2, Plane.XY, 0.22 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.YZ, 0.44 * np.pi),
                (5, Plane.YZ, 0.55 * np.pi),
            ],
            (
                [
                    (1, Plane.XY, 0.11 * np.pi),
                    (2, Plane.XY, 0.22 * np.pi),
                    (3, Plane.XY, 0.33 * np.pi),
                    (4, Plane.YZ, 0.99 * np.pi),
                ],
                {(1, 2), (1, 4), (2, 3), (3, 4)},
                {1, 2, 3, 4},
            ),
        ),
        #         4(YZ)
        #       /       \
        # 1(XY) - 2(YZ) - 3(XY) -> 1(XY) - 2(YZ) - 3(XY)
        #       \       /
        #         5(YZ)
        (
            {(1, 2), (1, 4), (1, 5), (2, 3), (3, 4), (3, 5)},
            [
                (1, Plane.XY, 0.11 * np.pi),
                (2, Plane.YZ, 0.22 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.YZ, 0.44 * np.pi),
                (5, Plane.YZ, 0.55 * np.pi),
            ],
            (
                [
                    (1, Plane.XY, 0.11 * np.pi),
                    (2, Plane.YZ, 1.21 * np.pi),
                    (3, Plane.XY, 0.33 * np.pi),
                ],
                {(1, 2), (2, 3)},
                {1, 2, 3},
            ),
        ),
        #         4(YZ)
        #       /       \
        # 1(XY) - 2(YZ) - 3(XY) - 1(XY) -> 1(XY) - 2(YZ) - 3(XY) - 1(XY)
        #       \       /
        #         5(YZ)
        (
            {(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (3, 5)},
            [
                (1, Plane.XY, 0.11 * np.pi),
                (2, Plane.YZ, 0.22 * np.pi),
                (3, Plane.XY, 0.33 * np.pi),
                (4, Plane.YZ, 0.44 * np.pi),
                (5, Plane.YZ, 0.55 * np.pi),
            ],
            (
                [
                    (1, Plane.XY, 0.11 * np.pi),
                    (2, Plane.YZ, 1.21 * np.pi),
                    (3, Plane.XY, 0.33 * np.pi),
                ],
                {(1, 2), (1, 3), (2, 3)},
                {1, 2, 3},
            ),
        ),
    ],
)
def test_merge_yz_nodes(
    zx_graph: ZXGraphState,
    initial_edges: set[tuple[int, int]],
    measurements: Measurements,
    exp_zxgraph: tuple[Measurements, set[tuple[int, int]], set[int]],
) -> None:
    _initialize_graph(zx_graph, nodes=range(1, 6), edges=initial_edges)
    _apply_measurements(zx_graph, measurements)
    zx_graph.merge_yz_nodes()
    exp_measurements, exp_edges, exp_nodes = exp_zxgraph
    _test(zx_graph, exp_nodes, exp_edges, exp_measurements)


@pytest.mark.parametrize(
    ("initial_zxgraph", "measurements", "exp_zxgraph"),
    [
        # test for a phase gadget: apply merge_yz_to_xy then remove_cliffords
        (
            (range(1, 5), {(1, 2), (2, 3), (2, 4)}),
            [
                (1, Plane.YZ, 0.1 * np.pi),
                (2, Plane.XY, 0.4 * np.pi),
                (3, Plane.XY, 0.3 * np.pi),
                (4, Plane.XY, 0.4 * np.pi),
            ],
            (
                [
                    (3, Plane.XY, 1.8 * np.pi),
                    (4, Plane.XY, 1.9 * np.pi),
                ],
                {(3, 4)},
                {3, 4},
            ),
        ),
        # apply convert_to_phase_gadget, merge_yz_to_xy, then remove_cliffords
        (
            (range(1, 5), {(1, 2), (2, 3), (2, 4)}),
            [
                (1, Plane.YZ, 0.1 * np.pi),
                (2, Plane.XY, 0.9 * np.pi),
                (3, Plane.XZ, 0.8 * np.pi),
                (4, Plane.XY, 0.4 * np.pi),
            ],
            (
                [
                    (3, Plane.XY, 0.2 * np.pi),
                    (4, Plane.XY, 0.9 * np.pi),
                ],
                {(3, 4)},
                {3, 4},
            ),
        ),
        # apply remove_cliffords, convert_to_phase_gadget, merge_yz_to_xy, then remove_cliffords
        (
            (range(1, 7), {(1, 2), (2, 3), (2, 4), (3, 6), (4, 5)}),
            [
                (1, Plane.YZ, 0.1 * np.pi),
                (2, Plane.XY, 0.9 * np.pi),
                (3, Plane.YZ, 1.2 * np.pi),
                (4, Plane.XY, 1.4 * np.pi),
                (5, Plane.YZ, 1.0 * np.pi),
                (6, Plane.XY, 0.5 * np.pi),
            ],
            (
                [
                    (3, Plane.XY, 1.8 * np.pi),
                    (4, Plane.XY, 0.9 * np.pi),
                ],
                {(3, 4)},
                {3, 4},
            ),
        ),
    ],
)
def test_full_reduce(
    zx_graph: ZXGraphState,
    initial_zxgraph: tuple[range, set[tuple[int, int]]],
    measurements: Measurements,
    exp_zxgraph: tuple[Measurements, set[tuple[int, int]], set[int]],
) -> None:
    nodes, edges = initial_zxgraph
    _initialize_graph(zx_graph, nodes, edges)
    exp_measurements, exp_edges, exp_nodes = exp_zxgraph
    _apply_measurements(zx_graph, measurements)
    zx_graph.full_reduce()
    _test(zx_graph, exp_nodes, exp_edges, exp_measurements)


if __name__ == "__main__":
    pytest.main()
