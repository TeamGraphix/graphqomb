from __future__ import annotations

import importlib
import math
from fractions import Fraction
from typing import Protocol, cast

from graphqomb.common import Plane
from graphqomb.zx_util import (
    EdgeType,
    FloatInt,
    FractionLike,
    PyZXDiagram,
    VertexType,
    _collect_edge_map,
    _collect_node_map,
    _collect_phase_gadget_meas_bases,
    _rewrite_input_boundary_maps,
    _rewrite_output_boundary_maps,
    from_pyzx,
)


class _PyTestMarkNamespace(Protocol):
    pyzx: object


class _PyTestModule(Protocol):
    mark: _PyTestMarkNamespace

    def importorskip(self, modname: str) -> object: ...


class _PyZXVertexTypeNamespace(Protocol):
    BOUNDARY: VertexType
    Z: VertexType
    X: VertexType


class _PyZXEdgeTypeNamespace(Protocol):
    SIMPLE: EdgeType
    HADAMARD: EdgeType


class _PyZXTestDiagram(PyZXDiagram, Protocol):
    def add_edge(self, edge_pair: tuple[int, int], edgetype: EdgeType = ...) -> object: ...

    def add_vertex(
        self,
        *,
        ty: VertexType,
        qubit: FloatInt = ...,
        row: FloatInt = ...,
        phase: FractionLike = ...,
        ground: bool = ...,
    ) -> int: ...

    def connected(self, vertex_1: int, vertex_2: int) -> bool: ...

    def set_inputs(self, inputs: tuple[int, ...]) -> None: ...

    def set_outputs(self, outputs: tuple[int, ...]) -> None: ...

    def vertex_set(self) -> set[int]: ...


class _PyZXModule(Protocol):
    EdgeType: _PyZXEdgeTypeNamespace
    VertexType: _PyZXVertexTypeNamespace

    def Graph(self) -> _PyZXTestDiagram: ...  # noqa: N802


pytest = cast("_PyTestModule", importlib.import_module("pytest"))
zx = cast("_PyZXModule", pytest.importorskip("pyzx"))
pytestmark = pytest.mark.pyzx


def _build_graphlike_diagram() -> tuple[_PyZXTestDiagram, int, int, int, int]:
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
    edge_type: EdgeType,
    *,
    is_output: bool = False,
) -> tuple[_PyZXTestDiagram, int, int]:
    diagram = zx.Graph()
    boundary = diagram.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=0, row=0, phase=Fraction(1, 2))
    spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=0, row=1)
    diagram.add_edge((boundary, spider), edgetype=edge_type)
    diagram.set_inputs(() if is_output else (boundary,))
    diagram.set_outputs((boundary,) if is_output else ())
    return diagram, boundary, spider


def _build_internal_phase_gadget_diagram() -> tuple[_PyZXTestDiagram, int, int, int, int, int, int]:
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

    return diagram, input_boundary, left_spider, hub_spider, phase_spider, right_spider, output_boundary


def _build_io_phase_gadget_diagram() -> tuple[_PyZXTestDiagram, int, int, int]:
    diagram = zx.Graph()
    input_boundary = diagram.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=0, row=0)
    io_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=0, row=1)
    phase_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=0, row=2, phase=Fraction(1, 4))

    diagram.add_edge((input_boundary, io_spider), edgetype=zx.EdgeType.SIMPLE)
    diagram.add_edge((io_spider, phase_spider), edgetype=zx.EdgeType.HADAMARD)
    diagram.set_inputs((input_boundary,))
    diagram.set_outputs(())

    return diagram, input_boundary, io_spider, phase_spider


def _build_ambiguous_phase_gadget_diagram() -> tuple[_PyZXTestDiagram, int, int, int]:
    diagram = zx.Graph()
    hub_spider = diagram.add_vertex(ty=zx.VertexType.Z, qubit=0, row=0)
    phase_spider_0 = diagram.add_vertex(ty=zx.VertexType.Z, qubit=1, row=1, phase=Fraction(1, 4))
    phase_spider_1 = diagram.add_vertex(ty=zx.VertexType.Z, qubit=2, row=1, phase=Fraction(1, 2))

    diagram.add_edge((hub_spider, phase_spider_0), edgetype=zx.EdgeType.HADAMARD)
    diagram.add_edge((hub_spider, phase_spider_1), edgetype=zx.EdgeType.HADAMARD)

    return diagram, hub_spider, phase_spider_0, phase_spider_1


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


def test_collect_phase_gadget_meas_bases_removes_internal_lone_spider() -> None:
    (
        diagram,
        input_boundary,
        left_spider,
        hub_spider,
        phase_spider,
        right_spider,
        output_boundary,
    ) = _build_internal_phase_gadget_diagram()
    node_map = _collect_node_map(diagram)
    edge_map = _collect_edge_map(diagram)

    meas_basis_overrides = _collect_phase_gadget_meas_bases(diagram, node_map, edge_map)

    assert set(node_map) == {input_boundary, left_spider, hub_spider, right_spider, output_boundary}
    assert phase_spider not in node_map
    assert (hub_spider, phase_spider) not in edge_map
    assert meas_basis_overrides[hub_spider].plane == Plane.YZ
    assert math.isclose(meas_basis_overrides[hub_spider].angle, math.pi / 4)


def test_collect_phase_gadget_meas_bases_removes_boundary_adjacent_lone_spider() -> None:
    diagram, input_boundary, io_spider, phase_spider = _build_io_phase_gadget_diagram()
    node_map = _collect_node_map(diagram)
    edge_map = _collect_edge_map(diagram)

    meas_basis_overrides = _collect_phase_gadget_meas_bases(diagram, node_map, edge_map)

    assert set(node_map) == {input_boundary, io_spider}
    assert phase_spider not in node_map
    assert (io_spider, phase_spider) not in edge_map
    assert meas_basis_overrides[io_spider].plane == Plane.YZ
    assert math.isclose(meas_basis_overrides[io_spider].angle, math.pi / 4)


def test_collect_phase_gadget_meas_bases_skips_ambiguous_lone_spiders() -> None:
    diagram, hub_spider, phase_spider_0, phase_spider_1 = _build_ambiguous_phase_gadget_diagram()
    node_map = _collect_node_map(diagram)
    edge_map = _collect_edge_map(diagram)

    meas_basis_overrides = _collect_phase_gadget_meas_bases(diagram, node_map, edge_map)

    assert meas_basis_overrides == {}
    assert set(node_map) == {hub_spider, phase_spider_0, phase_spider_1}
    assert set(edge_map) == {
        tuple(sorted((hub_spider, phase_spider_0))),
        tuple(sorted((hub_spider, phase_spider_1))),
    }


def test_from_pyzx_builds_graphstate() -> None:
    diagram, _, _, _, _ = _build_graphlike_diagram()

    graph = from_pyzx(diagram)
    coord_to_node = {coord: node for node, coord in graph.coordinates.items()}
    input_node = coord_to_node[0.0, 0.0]
    first_node = coord_to_node[1.0, 0.0]
    second_node = coord_to_node[2.0, 1.0]
    output_node = coord_to_node[3.0, 1.0]
    synthetic_output = coord_to_node[4.0, 1.0]

    assert graph.input_node_indices == {input_node: 0}
    assert graph.output_node_indices == {synthetic_output: 0}
    assert graph.physical_edges == {
        tuple(sorted((input_node, first_node))),
        tuple(sorted((first_node, second_node))),
        tuple(sorted((second_node, output_node))),
        tuple(sorted((output_node, synthetic_output))),
    }

    assert graph.meas_bases[input_node].plane == Plane.XY
    assert math.isclose(graph.meas_bases[input_node].angle, 0.0)
    assert graph.meas_bases[first_node].plane == Plane.XY
    assert math.isclose(graph.meas_bases[first_node].angle, math.pi / 2)
    assert graph.meas_bases[second_node].plane == Plane.XY
    assert math.isclose(graph.meas_bases[second_node].angle, 0.0)
    assert graph.meas_bases[output_node].plane == Plane.XY
    assert math.isclose(graph.meas_bases[output_node].angle, 0.0)
    assert synthetic_output not in graph.meas_bases

    assert graph.coordinates[input_node] == (0.0, 0.0)
    assert graph.coordinates[first_node] == (1.0, 0.0)
    assert graph.coordinates[second_node] == (2.0, 1.0)
    assert graph.coordinates[output_node] == (3.0, 1.0)
    assert graph.coordinates[synthetic_output] == (4.0, 1.0)


def test_from_pyzx_recognizes_phase_gadget_as_yz_measurement() -> None:
    diagram = _build_internal_phase_gadget_diagram()[0]

    graph = from_pyzx(diagram, recognize_pg=True)
    coord_to_node = {coord: node for node, coord in graph.coordinates.items()}
    hub_node = coord_to_node[2.0, 1.0]

    assert (3.0, 1.0) not in coord_to_node
    assert graph.meas_bases[hub_node].plane == Plane.YZ
    assert math.isclose(graph.meas_bases[hub_node].angle, math.pi / 4)
    assert len(graph.physical_nodes) == 6
