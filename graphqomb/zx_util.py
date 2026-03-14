"""PyZX utilities."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, TypeAlias

import pyzx as zx
from pyzx.graph.base import BaseGraph

from graphqomb.graphstate import GraphState

if TYPE_CHECKING:
    from pyzx.utils import EdgeType, FloatInt, FractionLike, VertexType

PyZXDiagram: TypeAlias = BaseGraph[int, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class VertexData:
    vertex_id: int
    vertex_type: VertexType
    phase: FractionLike
    qubit: FloatInt
    row: FloatInt
    is_ground: bool


@dataclasses.dataclass(frozen=True, slots=True)
class EdgeData:
    source: int
    target: int
    edge_type: EdgeType


def from_pyzx(diagram: PyZXDiagram, recognize_pg: bool = False) -> GraphState:

    # check if the diagram is in graph-like form or not
    if not zx.is_graph_like(diagram, strict=True):
        msg = "The input diagram is not in graph-like form. Please apply the graph-like transformation first."
        raise ValueError(msg)

    # if a vertex is a green spider and has only one phase-free Z-spider neighbor, then we can treat it as a phase gadget
    pg_vertices = _collect_phase_gadgets(diagram) if recognize_pg else set()

    # collect all the vertices and edges in the diagram
    node_map = _collect_node_map(diagram, excluded_vertices=pg_vertices)
    edge_map = _collect_edge_map(diagram, excluded_vertices=pg_vertices)

    # process input/output boundaries
    rewritten_inputs = _rewrite_input_boundary_maps(diagram, node_map, edge_map)
    rewritten_outputs = _rewrite_output_boundary_maps(diagram, node_map, edge_map)

    graph = GraphState()

    return graph


def _collect_node_map(
    diagram: PyZXDiagram,
    *,
    excluded_vertices: set[int] | None = None,
) -> dict[int, VertexData]:
    """Collect vertex metadata from a PyZX diagram."""
    excluded = excluded_vertices or set()
    node_map: dict[int, VertexData] = {}

    for vertex_id in diagram.vertices():
        if vertex_id in excluded:
            continue

        node_map[vertex_id] = VertexData(
            vertex_id=vertex_id,
            vertex_type=diagram.type(vertex_id),
            phase=diagram.phase(vertex_id),
            qubit=diagram.qubit(vertex_id),
            row=diagram.row(vertex_id),
            is_ground=diagram.is_ground(vertex_id),
        )

    return node_map


def _collect_edge_map(
    diagram: PyZXDiagram,
    *,
    excluded_vertices: set[int] | None = None,
) -> dict[tuple[int, int], EdgeData]:
    """Collect edge metadata from a PyZX diagram."""
    excluded = excluded_vertices or set()
    edge_map: dict[tuple[int, int], EdgeData] = {}

    for edge in diagram.edges():
        source, target = diagram.edge_st(edge)
        if source in excluded or target in excluded:
            continue

        edge_key = _edge_key(source, target)
        if edge_key in edge_map:
            msg = f"Parallel edges are not supported for PyZX import: {edge_key}"
            raise ValueError(msg)

        edge_map[edge_key] = EdgeData(
            source=edge_key[0],
            target=edge_key[1],
            edge_type=diagram.edge_type(edge),
        )

    return edge_map


def _edge_key(source: int, target: int) -> tuple[int, int]:
    """Return a canonical key for an undirected edge."""
    return (source, target) if source <= target else (target, source)


def _rewrite_input_boundary_maps(
    diagram: PyZXDiagram,
    node_map: dict[int, VertexData],
    edge_map: dict[tuple[int, int], EdgeData],
) -> tuple[int, ...]:
    """Rewrite PyZX input boundary vertices into GraphState-compatible node/edge maps."""
    rewritten_inputs: list[int] = []
    input_vertices = diagram.inputs()

    for input_vertex in input_vertices:
        if input_vertex not in node_map:
            msg = f"Missing input vertex in collected node map: {input_vertex}"
            raise ValueError(msg)

        vertex_data = node_map[input_vertex]
        if vertex_data.vertex_type != zx.VertexType.BOUNDARY:
            msg = f"Input vertex must be a boundary vertex: {input_vertex}"
            raise ValueError(msg)

        neighbors = list(diagram.neighbors(input_vertex))
        if len(neighbors) != 1:
            msg = f"Input boundary must have exactly one neighbor: {input_vertex}"
            raise ValueError(msg)

        neighbor = neighbors[0]
        edge_key = _edge_key(input_vertex, neighbor)
        edge_data = edge_map.get(edge_key)
        if edge_data is None:
            msg = f"Missing incident edge for input boundary: {input_vertex}"
            raise ValueError(msg)

        if edge_data.edge_type == zx.EdgeType.HADAMARD:
            del node_map[input_vertex]
            del edge_map[edge_key]
            rewritten_inputs.append(neighbor)
            continue

        if edge_data.edge_type == zx.EdgeType.SIMPLE:
            node_map[input_vertex] = dataclasses.replace(
                vertex_data,
                vertex_type=zx.VertexType.Z,
                phase=0,
                is_ground=False,
            )
            rewritten_inputs.append(input_vertex)
            continue

        msg = f"Unsupported edge type for input boundary: {edge_data.edge_type}"
        raise ValueError(msg)

    return tuple(rewritten_inputs)


def _rewrite_output_boundary_maps(
    diagram: PyZXDiagram,
    node_map: dict[int, VertexData],
    edge_map: dict[tuple[int, int], EdgeData],
) -> tuple[int, ...]:
    """Rewrite PyZX output boundary vertices into GraphState-compatible node/edge maps."""
    rewritten_outputs: list[int] = []
    output_vertices = diagram.outputs()

    for output_vertex in output_vertices:
        if output_vertex not in node_map:
            msg = f"Missing output vertex in collected node map: {output_vertex}"
            raise ValueError(msg)

        vertex_data = node_map[output_vertex]
        if vertex_data.vertex_type != zx.VertexType.BOUNDARY:
            msg = f"Output vertex must be a boundary vertex: {output_vertex}"
            raise ValueError(msg)

        neighbors = list(diagram.neighbors(output_vertex))
        if len(neighbors) != 1:
            msg = f"Output boundary must have exactly one neighbor: {output_vertex}"
            raise ValueError(msg)

        neighbor = neighbors[0]
        edge_key = _edge_key(output_vertex, neighbor)
        edge_data = edge_map.get(edge_key)
        if edge_data is None:
            msg = f"Missing incident edge for output boundary: {output_vertex}"
            raise ValueError(msg)

        if edge_data.edge_type == zx.EdgeType.HADAMARD:
            rewritten_outputs.append(output_vertex)
            continue

        if edge_data.edge_type == zx.EdgeType.SIMPLE:
            node_map[output_vertex] = dataclasses.replace(
                vertex_data,
                vertex_type=zx.VertexType.Z,
                phase=0,
                is_ground=False,
            )
            edge_map[edge_key] = dataclasses.replace(edge_data, edge_type=zx.EdgeType.HADAMARD)

            new_output_vertex = _next_vertex_id(node_map)
            node_map[new_output_vertex] = VertexData(
                vertex_id=new_output_vertex,
                vertex_type=zx.VertexType.BOUNDARY,
                phase=0,
                qubit=vertex_data.qubit,
                row=vertex_data.row + 1,
                is_ground=False,
            )
            new_output_edge_key = _edge_key(output_vertex, new_output_vertex)
            edge_map[new_output_edge_key] = EdgeData(
                source=new_output_edge_key[0],
                target=new_output_edge_key[1],
                edge_type=zx.EdgeType.HADAMARD,
            )
            rewritten_outputs.append(new_output_vertex)

            continue

        msg = f"Unsupported edge type for output boundary: {edge_data.edge_type}"
        raise ValueError(msg)

    return tuple(rewritten_outputs)


def _next_vertex_id(node_map: dict[int, VertexData]) -> int:
    """Return a fresh synthetic vertex id for import-time rewrites."""
    return max(node_map, default=-1) + 1


def _collect_phase_gadgets(diagram: PyZXDiagram) -> set[int]:
    raise NotImplementedError
