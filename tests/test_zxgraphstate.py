from __future__ import annotations

from copy import deepcopy
import itertools
import operator
from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.euler import is_clifford_angle, _is_close_angle, LocalClifford, update_lc_basis
from graphix_zx.random_objects import get_random_flow_graph
from graphix_zx.zxgraphstate import ZXGraphState

if TYPE_CHECKING:
    from typing import Callable

    Func = Callable[[float], float]
    MeasurementAction = dict[Plane, tuple[Plane, Func]]
    Measurements = list[tuple[int, Plane, float]]

measurement_action_lc_target: MeasurementAction = {
    Plane.XY: (Plane.XZ, lambda angle: angle + np.pi / 2),
    Plane.XZ: (Plane.XY, lambda angle: -angle + np.pi / 2),
    Plane.YZ: (Plane.YZ, lambda angle: angle + np.pi / 2),
}
measurement_action_lc_neighbors: MeasurementAction = {
    Plane.XY: (Plane.XY, lambda angle: angle + np.pi / 2),
    Plane.XZ: (Plane.YZ, lambda angle: angle),
    Plane.YZ: (Plane.XZ, operator.neg),
}


def plane_combinations(n: int) -> list[tuple[Plane, ...]]:
    """Generate all combinations of planes of length n.

    Parameters
    ----------
    n : int
        The length of the combinations. n > 1.

    Returns
    -------
        list[tuple[Plane, ...]]: A list of tuples containing all combinations of planes of length n.
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


@pytest.mark.parametrize("plane", [Plane.XY, Plane.XZ, Plane.YZ])
def test_local_complement_with_no_edge(zx_graph: ZXGraphState, plane: Plane, rng: np.random.Generator) -> None:
    angle = rng.random() * 2 * np.pi
    ref_plane, ref_angle_func = measurement_action_lc_target[plane]
    ref_angle = ref_angle_func(angle)
    zx_graph.add_physical_node(1)
    zx_graph.set_meas_basis(1, PlannerMeasBasis(plane, angle))

    zx_graph.local_complement(1)
    assert zx_graph.physical_edges == set()
    assert zx_graph.meas_bases[1].plane == ref_plane
    assert _is_close_angle(zx_graph.meas_bases[1].angle, ref_angle)


@pytest.mark.parametrize("plane1, plane3", plane_combinations(2))
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

    ref_plane1, ref_angle_func1 = measurement_action_lc_neighbors[plane1]
    ref_plane3, ref_angle_func3 = measurement_action_lc_neighbors[plane3]
    exp_measurements = [
        (1, ref_plane1, ref_angle_func1(measurements[0][2])),
        (3, ref_plane3, ref_angle_func3(measurements[1][2])),
    ]
    _test(zx_graph, exp_nodes={1, 2, 3}, exp_edges={(1, 2), (1, 3), (2, 3)}, exp_measurements=exp_measurements)
    assert zx_graph.meas_bases.get(2) is None


@pytest.mark.parametrize("plane1, plane2", plane_combinations(2))
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

    ref_plane1, ref_angle_func1 = measurement_action_lc_target[plane1]
    ref_plane2, ref_angle_func2 = measurement_action_lc_neighbors[plane2]
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
    ref_plane1, ref_angle_func1 = measurement_action_lc_neighbors[planes[0]]
    ref_plane2, ref_angle_func2 = measurement_action_lc_target[planes[1]]
    ref_plane3, ref_angle_func3 = measurement_action_lc_neighbors[planes[2]]
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
    ref_plane1, ref_angle_func1 = measurement_action_lc_neighbors[ref_plane1]
    ref_plane2, ref_angle_func2 = measurement_action_lc_target[ref_plane2]
    ref_plane3, ref_angle_func3 = measurement_action_lc_neighbors[ref_plane3]
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

    exp_measurements = [
        (1, planes[0], angles[0]),
        (2, planes[1], angles[1]),
        (3, planes[2], angles[2]),
    ]
    _test(zx_graph, exp_nodes={1, 2, 3}, exp_edges={(1, 2), (2, 3)}, exp_measurements=exp_measurements)


def test_local_complement_with_h_shaped_graph(zx_graph: ZXGraphState) -> None:
    pass


if __name__ == "__main__":
    pytest.main()
