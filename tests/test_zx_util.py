from __future__ import annotations

import math
from fractions import Fraction

import pyzx as zx

from graphqomb.common import Plane
from graphqomb.zx_util import (
    _collect_edge_map,
    _collect_node_map,
    _collect_phase_gadgets,
    _rewrite_input_boundary_maps,
    _rewrite_output_boundary_maps,
    from_pyzx,
)


def _build_graphlike_diagram() -> tuple[zx.graph_s.GraphS, int, int, int, int]:
    diagram = zx.Graph()
    input_boundary = diagram.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=0, row=0)
    first_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=0, row=1, phase=Fraction(1, 2))
    second_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=1, row=2, ground=True)
    output_boundary = diagram.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=1, row=3)

    diagram.add_edge((input_boundary, first_spider), edgetype=zx.EdgeType.SIMPLE)
    diagram.add_edge((first_spider, second_spider), edgetype=zx.EdgeType.HADAMARD)
    diagram.add_edge((second_spider, output_boundary), edgetype=zx.EdgeType.SIMPLE)
    diagram.set_inputs((input_boundary,))
    diagram.set_outputs((output_boundary,))

    return diagram, input_boundary, first_spider, second_spider, output_boundary


def _build_single_boundary_diagram(
    edge_type: zx.EdgeType,
    *,
    is_output: bool = False,
) -> tuple[zx.graph_s.GraphS, int, int]:
    diagram = zx.Graph()
    boundary = diagram.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=0, row=0, phase=Fraction(1, 2))
    spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=0, row=1)
    diagram.add_edge((boundary, spider), edgetype=edge_type)
    diagram.set_inputs(() if is_output else (boundary,))
    diagram.set_outputs((boundary,) if is_output else ())
    return diagram, boundary, spider


def _build_internal_phase_gadget_diagram() -> tuple[zx.graph_s.GraphS, int, int, int, int, int]:
    diagram = zx.Graph()
    input_boundary = diagram.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=0, row=0)
    left_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=0, row=1)
    hub_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=1, row=2)
    phase_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=1, row=3, phase=Fraction(1, 4))
    right_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=2, row=3)
    output_boundary = diagram.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=2, row=4)

    diagram.add_edge((input_boundary, left_spider), edgetype=zx.EdgeType.SIMPLE)
    diagram.add_edge((left_spider, hub_spider), edgetype=zx.EdgeType.HADAMARD)
    diagram.add_edge((hub_spider, phase_spider), edgetype=zx.EdgeType.HADAMARD)
    diagram.add_edge((hub_spider, right_spider), edgetype=zx.EdgeType.HADAMARD)
    diagram.add_edge((right_spider, output_boundary), edgetype=zx.EdgeType.SIMPLE)
    diagram.set_inputs((input_boundary,))
    diagram.set_outputs((output_boundary,))

    return diagram, input_boundary, left_spider, hub_spider, phase_spider, output_boundary


def _build_io_phase_gadget_diagram() -> tuple[zx.graph_s.GraphS, int, int, int]:
    diagram = zx.Graph()
    input_boundary = diagram.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=0, row=0)
    io_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=0, row=1)
    phase_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=0, row=2, phase=Fraction(1, 4))

    diagram.add_edge((input_boundary, io_spider), edgetype=zx.EdgeType.SIMPLE)
    diagram.add_edge((io_spider, phase_spider), edgetype=zx.EdgeType.HADAMARD)
    diagram.set_inputs((input_boundary,))
    diagram.set_outputs(())

    return diagram, input_boundary, io_spider, phase_spider


def test_collect_node_map_preserves_pyzx_vertex_metadata() -> None:
    diagram, input_boundary, first_spider, second_spider, output_boundary = _build_graphlike_diagram()

    node_map = _collect_node_map(diagram)

    assert set(node_map) == {input_boundary, first_spider, second_spider, output_boundary}
    assert node_map[input_boundary].vertex_type == zx.VertexType.BOUNDARY
    assert node_map[first_spider].vertex_type == zx.VertexType.Z
    assert node_map[first_spider].phase == Fraction(1, 2)
    assert node_map[first_spider].qubit == 0
    assert node_map[first_spider].row == 1
    assert node_map[second_spider].is_ground is True


def test_collect_edge_map_preserves_edge_types() -> None:
    diagram, input_boundary, first_spider, second_spider, output_boundary = _build_graphlike_diagram()

    edge_map = _collect_edge_map(diagram)

    assert set(edge_map) == {
        (input_boundary, first_spider),
        (first_spider, second_spider),
        (second_spider, output_boundary),
    }
    assert edge_map[input_boundary, first_spider].edge_type == zx.EdgeType.SIMPLE
    assert edge_map[first_spider, second_spider].edge_type == zx.EdgeType.HADAMARD
    assert edge_map[second_spider, output_boundary].edge_type == zx.EdgeType.SIMPLE


def test_rewrite_input_boundary_maps_removes_hadamard_boundary_vertices() -> None:
    diagram, boundary, spider = _build_single_boundary_diagram(zx.EdgeType.HADAMARD)
    node_map = _collect_node_map(diagram)
    edge_map = _collect_edge_map(diagram)

    rewritten_inputs = _rewrite_input_boundary_maps(diagram, node_map, edge_map)

    assert rewritten_inputs == (spider,)
    assert boundary not in node_map
    assert spider in node_map
    assert (boundary, spider) not in edge_map


def test_rewrite_input_boundary_maps_promotes_simple_boundaries_to_z_spiders() -> None:
    diagram, boundary, spider = _build_single_boundary_diagram(zx.EdgeType.SIMPLE)
    node_map = _collect_node_map(diagram)
    edge_map = _collect_edge_map(diagram)

    rewritten_inputs = _rewrite_input_boundary_maps(diagram, node_map, edge_map)

    assert rewritten_inputs == (boundary,)
    assert node_map[boundary].vertex_type == zx.VertexType.Z
    assert node_map[boundary].phase == 0
    assert edge_map[boundary, spider].edge_type == zx.EdgeType.HADAMARD


def test_rewrite_output_boundary_maps_keeps_hadamard_output_vertex() -> None:
    diagram, boundary, spider = _build_single_boundary_diagram(zx.EdgeType.HADAMARD, is_output=True)
    node_map = _collect_node_map(diagram)
    edge_map = _collect_edge_map(diagram)

    rewritten_outputs = _rewrite_output_boundary_maps(diagram, node_map, edge_map)

    assert rewritten_outputs == (boundary,)
    assert node_map[boundary].vertex_type == zx.VertexType.BOUNDARY
    assert edge_map[boundary, spider].edge_type == zx.EdgeType.HADAMARD


def test_rewrite_output_boundary_maps_adds_synthetic_output_for_simple_boundary() -> None:
    diagram, boundary, spider = _build_single_boundary_diagram(zx.EdgeType.SIMPLE, is_output=True)
    node_map = _collect_node_map(diagram)
    edge_map = _collect_edge_map(diagram)

    rewritten_outputs = _rewrite_output_boundary_maps(diagram, node_map, edge_map)

    (synthetic_output,) = rewritten_outputs
    assert synthetic_output != boundary
    assert node_map[boundary].vertex_type == zx.VertexType.Z
    assert node_map[boundary].phase == 0
    assert node_map[synthetic_output].vertex_type == zx.VertexType.BOUNDARY
    assert node_map[synthetic_output].qubit == node_map[boundary].qubit
    assert node_map[synthetic_output].row == node_map[boundary].row + 1
    assert edge_map[boundary, spider].edge_type == zx.EdgeType.HADAMARD
    assert edge_map[boundary, synthetic_output].edge_type == zx.EdgeType.HADAMARD


def test_collect_phase_gadgets_rewrites_internal_phase_gadget() -> None:
    (
        diagram,
        input_boundary,
        left_spider,
        hub_spider,
        phase_spider,
        output_boundary,
    ) = _build_internal_phase_gadget_diagram()

    rewritten = _collect_phase_gadgets(diagram)

    assert phase_spider in diagram.vertex_set()
    assert phase_spider not in rewritten.vertex_set()
    assert rewritten.type(hub_spider) == zx.VertexType.X
    assert rewritten.phase(hub_spider) == Fraction(1, 4)
    assert rewritten.inputs() == (input_boundary,)
    assert rewritten.outputs() == (output_boundary,)
    assert rewritten.connected(left_spider, hub_spider)


def test_collect_phase_gadgets_rewrites_boundary_adjacent_z_spider() -> None:
    diagram, input_boundary, io_spider, phase_spider = _build_io_phase_gadget_diagram()

    rewritten = _collect_phase_gadgets(diagram)

    assert rewritten.inputs() == (input_boundary,)
    assert phase_spider not in rewritten.vertex_set()
    assert rewritten.type(io_spider) == zx.VertexType.X
    assert rewritten.phase(io_spider) == Fraction(1, 4)


def test_from_pyzx_builds_graphstate_and_node_map() -> None:
    diagram, input_boundary, first_spider, second_spider, output_boundary = _build_graphlike_diagram()

    graph, node_map = from_pyzx(diagram)

    synthetic_output = max(node_map)

    assert set(node_map) == {input_boundary, first_spider, second_spider, output_boundary, synthetic_output}
    assert graph.input_node_indices == {node_map[input_boundary]: 0}
    assert graph.output_node_indices == {node_map[synthetic_output]: 0}
    assert graph.physical_edges == {
        tuple(sorted((node_map[input_boundary], node_map[first_spider]))),
        tuple(sorted((node_map[first_spider], node_map[second_spider]))),
        tuple(sorted((node_map[second_spider], node_map[output_boundary]))),
        tuple(sorted((node_map[output_boundary], node_map[synthetic_output]))),
    }

    assert graph.meas_bases[node_map[input_boundary]].plane == Plane.XY
    assert math.isclose(graph.meas_bases[node_map[input_boundary]].angle, 0.0)
    assert graph.meas_bases[node_map[first_spider]].plane == Plane.XY
    assert math.isclose(graph.meas_bases[node_map[first_spider]].angle, math.pi / 2)
    assert graph.meas_bases[node_map[second_spider]].plane == Plane.XY
    assert math.isclose(graph.meas_bases[node_map[second_spider]].angle, 0.0)
    assert graph.meas_bases[node_map[output_boundary]].plane == Plane.XY
    assert math.isclose(graph.meas_bases[node_map[output_boundary]].angle, 0.0)
    assert node_map[synthetic_output] not in graph.meas_bases

    assert graph.coordinates[node_map[input_boundary]] == (0.0, 0.0)
    assert graph.coordinates[node_map[first_spider]] == (0.0, 1.0)
    assert graph.coordinates[node_map[second_spider]] == (1.0, 2.0)
    assert graph.coordinates[node_map[output_boundary]] == (1.0, 3.0)
    assert graph.coordinates[node_map[synthetic_output]] == (1.0, 4.0)
