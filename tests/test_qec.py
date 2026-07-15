"""Tests for QEC graph-state builders."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
from scipy.sparse import csr_array

from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.qec.qeccode import StabilizerCode, YFoliation, build_graph_state


def _matrix(data: list[list[int]]) -> csr_array[Any, tuple[int, int]]:
    return cast("csr_array[Any, tuple[int, int]]", csr_array(data))


def _assert_axis_meas_basis(meas_basis: object, axis: Axis) -> None:
    assert isinstance(meas_basis, AxisMeasBasis)
    assert meas_basis.axis == axis
    assert meas_basis.sign == Sign.PLUS
    expected_angle = math.pi / 2 if axis == Axis.Y else 0.0
    assert math.isclose(meas_basis.angle, expected_angle)


def test_build_graph_state_connects_stabilizer_supports() -> None:
    matrix = _matrix(
        [
            [0, 0, 0, 1, 0, 2],  # Z support on q0 and q2.
            [0, 3, 0, 0, 0, 0],  # X support on q1.
            [1, 0, 0, 1, 0, 0],  # X and Z support on q0.
        ]
    )
    code = StabilizerCode(matrix)

    result = build_graph_state(code)
    graph = result.graph

    for qubit in range(3):
        assert graph.has_edge(result.data_nodes[qubit, 0], result.data_nodes[qubit, 1])

    ancilla0 = result.ancilla_nodes[0]
    assert graph.has_edge(ancilla0, result.data_nodes[0, 0])
    assert graph.has_edge(ancilla0, result.data_nodes[2, 0])

    ancilla1 = result.ancilla_nodes[1]
    assert graph.has_edge(ancilla1, result.data_nodes[1, 1])

    ancilla2 = result.ancilla_nodes[2]
    assert graph.has_edge(ancilla2, result.data_nodes[0, 0])
    assert graph.has_edge(ancilla2, result.data_nodes[0, 1])

    assert graph.number_of_nodes() == 9
    assert graph.number_of_edges() == 8


def test_build_graph_state_assigns_x_measurement_to_all_nodes() -> None:
    code = StabilizerCode(_matrix([[1, 0, 0, 1]]))

    result = build_graph_state(code)

    for node in result.graph.nodes:
        _assert_axis_meas_basis(result.graph.meas_bases[node], Axis.X)


def test_build_graph_state_returns_index_to_node_maps() -> None:
    code = StabilizerCode(_matrix([[1, 0, 0, 1]]))

    result = build_graph_state(code)

    assert set(result.data_nodes) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert set(result.ancilla_nodes) == {0}
    assert set(result.data_nodes.values()).isdisjoint(result.ancilla_nodes.values())


def test_build_graph_state_type_ii_routes_y_support_through_three_node_y_chain() -> None:
    code = StabilizerCode(
        _matrix(
            [
                [0, 1],  # Z support.
                [1, 1],  # Y support.
                [1, 0],  # X support.
            ]
        )
    )

    result = build_graph_state(code, y_foliation=YFoliation.TYPE_II)
    graph = result.graph

    assert set(result.data_nodes) == {(0, 0), (0, 1), (0, 2)}
    assert graph.has_edge(result.data_nodes[0, 0], result.data_nodes[0, 1])
    assert graph.has_edge(result.data_nodes[0, 1], result.data_nodes[0, 2])
    assert graph.has_edge(result.ancilla_nodes[0], result.data_nodes[0, 0])
    assert graph.has_edge(result.ancilla_nodes[1], result.data_nodes[0, 1])
    assert graph.has_edge(result.ancilla_nodes[2], result.data_nodes[0, 2])
    assert not graph.has_edge(result.ancilla_nodes[1], result.data_nodes[0, 0])
    assert not graph.has_edge(result.ancilla_nodes[1], result.data_nodes[0, 2])

    for data_node in result.data_nodes.values():
        _assert_axis_meas_basis(graph.meas_bases[data_node], Axis.Y)
    for ancilla_node in result.ancilla_nodes.values():
        _assert_axis_meas_basis(graph.meas_bases[ancilla_node], Axis.X)

    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 5


def test_build_graph_state_type_ii_keeps_two_node_x_chain_without_y_support() -> None:
    code = StabilizerCode(
        _matrix(
            [
                [0, 1],  # Z support.
                [1, 0],  # X support.
            ]
        )
    )

    result = build_graph_state(code, y_foliation=YFoliation.TYPE_II)
    graph = result.graph

    assert set(result.data_nodes) == {(0, 0), (0, 1)}
    assert graph.has_edge(result.data_nodes[0, 0], result.data_nodes[0, 1])
    assert graph.has_edge(result.ancilla_nodes[0], result.data_nodes[0, 0])
    assert graph.has_edge(result.ancilla_nodes[1], result.data_nodes[0, 1])
    for data_node in result.data_nodes.values():
        _assert_axis_meas_basis(graph.meas_bases[data_node], Axis.X)


def test_build_graph_state_type_ii_aligns_three_node_chain_output_z_with_two_node_chain() -> None:
    code = StabilizerCode(
        _matrix([[1, 1]]),
        qubit_coords={0: (10.0, 20.0)},
    )

    result = build_graph_state(code, z_base=5, y_foliation=YFoliation.TYPE_II)
    coords = result.graph.coordinates

    assert set(result.data_nodes) == {(0, 5), (0, 6), (0, 7)}
    assert coords[result.data_nodes[0, 5]] == (10.0, 20.0, 5.0)
    assert coords[result.data_nodes[0, 6]] == (10.0, 20.0, 5.5)
    assert coords[result.data_nodes[0, 7]] == (10.0, 20.0, 6.0)
    assert coords[result.ancilla_nodes[0]] == (10.0, 20.0, 5.5)


def test_build_graph_state_lifts_coordinates_to_shifted_3d_layers() -> None:
    code = StabilizerCode(
        _matrix([[0, 1, 1, 0]]),
        qubit_coords={
            0: (10.0, 20.0),
            1: (30.0, 40.0, 999.0),
        },
    )

    result = build_graph_state(code, z_base=5)
    coords = result.graph.coordinates

    assert set(result.data_nodes) == {(0, 5), (0, 6), (1, 5), (1, 6)}
    assert coords[result.data_nodes[0, 5]] == (10.0, 20.0, 5.0)
    assert coords[result.data_nodes[0, 6]] == (10.0, 20.0, 6.0)
    assert coords[result.data_nodes[1, 5]] == (30.0, 40.0, 5.0)
    assert coords[result.data_nodes[1, 6]] == (30.0, 40.0, 6.0)
    assert coords[result.ancilla_nodes[0]] == (20.0, 30.0, 5.5)


def test_build_graph_state_explicit_ancilla_coordinate_overrides_average() -> None:
    code = StabilizerCode(
        _matrix([[0, 1, 1, 0]]),
        stabilizer_coords={0: (1.0, 2.0, 3.0)},
        qubit_coords={
            0: (10.0, 20.0),
            1: (30.0, 40.0),
        },
    )

    result = build_graph_state(code)

    assert result.graph.coordinates[result.ancilla_nodes[0]] == (1.0, 2.0, 3.0)


def test_build_graph_state_allows_missing_coordinates() -> None:
    code = StabilizerCode(_matrix([[1, 0, 0, 1]]))

    result = build_graph_state(code)

    assert result.graph.coordinates == {}


def test_stabilizer_code_preserves_higher_dimensional_coordinates() -> None:
    code = StabilizerCode(
        _matrix([[1, 0]]),
        stabilizer_coords={0: (1.0, 2.0, 3.0, 4.0)},
        qubit_coords={0: (5.0, 6.0, 7.0, 8.0)},
    )

    assert code.stabilizer_coord == {0: (1.0, 2.0, 3.0, 4.0)}
    assert code.qubit_coord == {0: (5.0, 6.0, 7.0, 8.0)}


def test_build_graph_state_rejects_non_integer_z_base() -> None:
    code = StabilizerCode(_matrix([[1, 0]]))

    with pytest.raises(TypeError, match="z_base must be an integer"):
        build_graph_state(code, z_base=0.5)  # type: ignore[arg-type]
