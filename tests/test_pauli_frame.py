"""Tests for pauli_frame module."""

from __future__ import annotations

import math

import pytest

from graphqomb.common import Axis, Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.pauli_frame import PauliFrame


@pytest.fixture
def simple_graph_with_flows() -> tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]:
    """Create a simple graph with X and Z flows for testing.

    Returns
    -------
    tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]
        GraphState, xflow, and zflow
    """
    graph = GraphState()
    n0 = graph.add_physical_node()
    n1 = graph.add_physical_node()
    n2 = graph.add_physical_node()

    q_idx = 0
    graph.register_input(n0, q_idx)
    graph.register_output(n2, q_idx)

    graph.add_physical_edge(n0, n1)
    graph.add_physical_edge(n1, n2)

    # Z measurement on n0, X measurement on n1
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow = {n0: {n0}}

    return graph, xflow, zflow


@pytest.fixture
def simple_pauli_frame(
    simple_graph_with_flows: tuple[GraphState, dict[int, set[int]], dict[int, set[int]]],
) -> PauliFrame:
    """Create a simple PauliFrame for testing.

    Parameters
    ----------
    simple_graph_with_flows : tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]
        Graph, xflow, and zflow from fixture

    Returns
    -------
    PauliFrame
        A simple PauliFrame instance
    """
    graph, xflow, zflow = simple_graph_with_flows
    return PauliFrame(graph, xflow, zflow)


@pytest.fixture
def simple_nodes(simple_graph_with_flows: tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]) -> list[int]:
    """Get list of physical nodes from simple graph.

    Parameters
    ----------
    simple_graph_with_flows : tuple[GraphState, dict[int, set[int]], dict[int, set[int]]]
        Graph, xflow, and zflow from fixture

    Returns
    -------
    list[int]
        List of physical node IDs
    """
    graph, _, _ = simple_graph_with_flows
    return list(graph.physical_nodes)


@pytest.fixture
def x_axis_pauli_frame() -> PauliFrame:
    """Create a PauliFrame with X axis measurements.

    Returns
    -------
    PauliFrame
        PauliFrame with X axis measurements
    """
    graph = GraphState()
    n0 = graph.add_physical_node()
    n1 = graph.add_physical_node()
    n2 = graph.add_physical_node()

    graph.register_input(n0, 0)
    graph.register_output(n2, 0)

    graph.add_physical_edge(n0, n1)
    graph.add_physical_edge(n1, n2)

    # X measurement (XY plane, angle 0)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow: dict[int, set[int]] = {}

    return PauliFrame(graph, xflow, zflow)


@pytest.fixture
def y_axis_pauli_frame() -> PauliFrame:
    """Create a PauliFrame with Y axis measurements.

    Returns
    -------
    PauliFrame
        PauliFrame with Y axis measurements
    """
    graph = GraphState()
    n0 = graph.add_physical_node()
    n1 = graph.add_physical_node()
    n2 = graph.add_physical_node()

    graph.register_input(n0, 0)
    graph.register_output(n2, 0)

    graph.add_physical_edge(n0, n1)
    graph.add_physical_edge(n1, n2)

    # Y measurement (XY plane, angle pi/2)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, math.pi / 2))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, math.pi / 2))

    xflow = {n0: {n1}, n1: {n2}}
    zflow = {n0: {n0}}

    return PauliFrame(graph, xflow, zflow)


@pytest.fixture
def z_axis_pauli_frame() -> PauliFrame:
    """Create a PauliFrame with Z axis measurements.

    Returns
    -------
    PauliFrame
        PauliFrame with Z axis measurements
    """
    graph = GraphState()
    n0 = graph.add_physical_node()
    n1 = graph.add_physical_node()
    n2 = graph.add_physical_node()

    graph.register_input(n0, 0)
    graph.register_output(n2, 0)

    graph.add_physical_edge(n0, n1)
    graph.add_physical_edge(n1, n2)

    # Z measurement (XZ plane, angle 0)
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XZ, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow: dict[int, set[int]] = {}

    return PauliFrame(graph, xflow, zflow)


def test_x_flip(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test X Pauli flip operation."""
    pframe = simple_pauli_frame
    node = simple_nodes[0]

    # Initially False
    assert pframe.x_pauli[node] is False

    # Flip once
    pframe.x_flip(node)
    assert pframe.x_pauli[node] is True

    # Flip again
    pframe.x_flip(node)
    assert pframe.x_pauli[node] is False


def test_z_flip(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test Z Pauli flip operation."""
    pframe = simple_pauli_frame
    node = simple_nodes[0]

    # Initially False
    assert pframe.z_pauli[node] is False

    # Flip once
    pframe.z_flip(node)
    assert pframe.z_pauli[node] is True

    # Flip again
    pframe.z_flip(node)
    assert pframe.z_pauli[node] is False


def test_meas_flip(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test measurement flip operation updates Pauli frame correctly."""
    pframe = simple_pauli_frame
    n0, n1, n2 = simple_nodes[0], simple_nodes[1], simple_nodes[2]

    # Initially all False
    assert pframe.x_pauli[n1] is False
    assert pframe.z_pauli[n0] is False

    # Flip n0: should affect xflow[n0] = {n1} and zflow[n0] = {n0}
    pframe.meas_flip(n0)
    assert pframe.x_pauli[n1] is True  # n1 is in xflow[n0]
    assert pframe.z_pauli[n0] is True  # n0 is in zflow[n0]

    # Flip n1: should affect xflow[n1] = {n2}
    pframe.meas_flip(n1)
    assert pframe.x_pauli[n2] is True  # n2 is in xflow[n1]


def test_children(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test getting children of a node in the Pauli frame."""
    pframe = simple_pauli_frame
    n0, n1, n2 = simple_nodes[0], simple_nodes[1], simple_nodes[2]

    # n0 has children from xflow and zflow (excluding itself)
    children_n0 = pframe.children(n0)
    assert n1 in children_n0  # from xflow[n0]
    assert n0 not in children_n0  # self is excluded

    # n1 has child n2 from xflow
    children_n1 = pframe.children(n1)
    assert n2 in children_n1


def test_parents(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test getting parents of a node in the Pauli frame."""
    pframe = simple_pauli_frame
    n0, n1, n2 = simple_nodes[0], simple_nodes[1], simple_nodes[2]

    # n1 has parent n0 from inv_xflow
    parents_n1 = pframe.parents(n1)
    assert n0 in parents_n1
    assert n1 not in parents_n1

    # n2 has parent n1 from inv_xflow
    parents_n2 = pframe.parents(n2)
    assert n1 in parents_n2


def test_pauli_axis_cache_initialization(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test that Pauli axis cache is correctly initialized."""
    pframe = simple_pauli_frame
    n0, n1 = simple_nodes[0], simple_nodes[1]

    # n0 has Z measurement (XZ plane, angle 0)
    assert n0 in pframe._pauli_axis_cache
    assert pframe._pauli_axis_cache[n0] == Axis.Z

    # n1 has X measurement (XY plane, angle 0)
    assert n1 in pframe._pauli_axis_cache
    assert pframe._pauli_axis_cache[n1] == Axis.X


def test_chain_cache_memoization(simple_pauli_frame: PauliFrame, simple_nodes: list[int]) -> None:
    """Test that chain cache memoization works correctly."""
    pframe = simple_pauli_frame
    n1 = simple_nodes[1]

    # First call should compute and cache
    chain1 = pframe._collect_dependent_chain(n1)
    assert n1 in pframe._chain_cache

    # Second call should return cached result
    chain2 = pframe._collect_dependent_chain(n1)
    assert chain1 == chain2

    # Verify cache hit by checking the cached value
    cached_value = pframe._chain_cache[n1]
    assert set(cached_value) == chain1


def test_collect_dependent_chain_x_axis(x_axis_pauli_frame: PauliFrame) -> None:
    """Test dependent chain collection for X axis measurement."""
    pframe = x_axis_pauli_frame
    # Node 1 is the second node in the graph
    nodes = list(pframe.graphstate.physical_nodes)
    n1 = nodes[1]

    # For X axis, parents come from inv_zflow
    chain = pframe._collect_dependent_chain(n1)
    assert isinstance(chain, set)
    assert n1 in chain


def test_collect_dependent_chain_y_axis(y_axis_pauli_frame: PauliFrame) -> None:
    """Test dependent chain collection for Y axis measurement."""
    pframe = y_axis_pauli_frame
    # Node 1 is the second node in the graph
    nodes = list(pframe.graphstate.physical_nodes)
    n1 = nodes[1]

    # For Y axis, parents come from symmetric difference of inv_xflow and inv_zflow
    chain = pframe._collect_dependent_chain(n1)
    assert isinstance(chain, set)
    assert n1 in chain


def test_collect_dependent_chain_z_axis(z_axis_pauli_frame: PauliFrame) -> None:
    """Test dependent chain collection for Z axis measurement."""
    pframe = z_axis_pauli_frame
    # Node 1 is the second node in the graph
    nodes = list(pframe.graphstate.physical_nodes)
    n1 = nodes[1]

    # For Z axis, parents come from inv_xflow
    chain = pframe._collect_dependent_chain(n1)
    assert isinstance(chain, set)
    assert n1 in chain


def test_detector_groups() -> None:
    """Test detector groups generation with parity check groups."""
    graph = GraphState()
    n0 = graph.add_physical_node()
    n1 = graph.add_physical_node()
    n2 = graph.add_physical_node()

    graph.register_input(n0, 0)
    graph.register_output(n2, 0)

    graph.add_physical_edge(n0, n1)
    graph.add_physical_edge(n1, n2)

    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n2, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow: dict[int, set[int]] = {n0: {n0, n2}}
    parity_check_group = [{n1}, {n1, n2}]

    pframe = PauliFrame(graph, xflow, zflow, parity_check_group)

    # Get detector groups
    groups = pframe.detector_groups()
    assert isinstance(groups, list)
    assert len(groups) == 2
    for group in groups:
        assert isinstance(group, set)

    assert groups[0] == {n1}
    assert groups[1] == {n0, n1, n2}  # n0 is included via dependent chain


def test_logical_observables_group() -> None:
    """Test logical observables group generation."""
    graph = GraphState()
    n0 = graph.add_physical_node()
    n1 = graph.add_physical_node()
    n2 = graph.add_physical_node()

    graph.register_input(n0, 0)
    graph.register_output(n2, 0)

    graph.add_physical_edge(n0, n1)
    graph.add_physical_edge(n1, n2)

    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(n2, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {n0: {n1}, n1: {n2}}
    zflow: dict[int, set[int]] = {n0: {n0, n2}}

    pframe = PauliFrame(graph, xflow, zflow)

    # Get logical observables group
    target_nodes = [n2]
    group = pframe.logical_observables_group(target_nodes)
    assert isinstance(group, set)
    assert group == {n0, n2}  # n0 is included via dependent chain


def test_collect_dependent_chain_cache_hit() -> None:
    """Test that cache is correctly used when same node is queried multiple times."""
    graph = GraphState()
    n0 = graph.add_physical_node()
    n1 = graph.add_physical_node()
    n2 = graph.add_physical_node()
    n3 = graph.add_physical_node()

    graph.register_input(n0, 0)
    graph.register_output(n3, 0)

    graph.add_physical_edge(n0, n1)
    graph.add_physical_edge(n1, n2)
    graph.add_physical_edge(n2, n3)

    # Mix of X and Z measurements
    graph.assign_meas_basis(n0, PlannerMeasBasis(Plane.XZ, 0.0))  # Z
    graph.assign_meas_basis(n1, PlannerMeasBasis(Plane.XY, 0.0))  # X
    graph.assign_meas_basis(n2, PlannerMeasBasis(Plane.XZ, 0.0))  # Z

    xflow = {n0: {n1}, n1: {n2}, n2: {n3}}
    zflow = {n0: {n0}}

    pframe = PauliFrame(graph, xflow, zflow)

    # First call to n2
    chain1 = pframe._collect_dependent_chain(n2)
    assert n2 in pframe._chain_cache

    # Clear internal cache to test that memoization returns correct result
    cached_result = pframe._chain_cache[n2]

    # Second call should use cache
    chain2 = pframe._collect_dependent_chain(n2)

    # Results should be identical
    assert chain1 == chain2
    assert set(cached_result) == chain1
