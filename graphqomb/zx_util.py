"""PyZX integration utilities.

This module provides:

- `PyZXDiagram`: Protocol describing the PyZX graph interface used for import.
- `VertexData`: Collected PyZX vertex metadata used during import.
- `EdgeData`: Collected PyZX edge metadata used during import.
- `from_pyzx`: Convert a graph-like PyZX diagram into a `GraphState`.
"""
# ruff: noqa: D102

from __future__ import annotations

import dataclasses
import importlib
import math
from typing import TYPE_CHECKING, Protocol, SupportsFloat, TypeAlias, cast

from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableMapping
    from collections.abc import Set as AbstractSet

FloatInt: TypeAlias = float | int
FractionLike: TypeAlias = SupportsFloat
VertexType: TypeAlias = int
EdgeType: TypeAlias = int


class _PyZXVertexTypeNamespace(Protocol):
    """Static view of the PyZX vertex-type enum namespace."""

    BOUNDARY: int
    Z: int
    X: int


class _PyZXEdgeTypeNamespace(Protocol):
    """Static view of the PyZX edge-type enum namespace."""

    HADAMARD: int
    SIMPLE: int


class PyZXDiagram(Protocol):
    """Protocol covering the PyZX graph surface used by this module."""

    def copy(self) -> PyZXDiagram: ...

    def edges(self) -> Iterable[object]: ...

    def edge_st(self, edge: object) -> tuple[int, int]: ...

    def edge_type(self, edge: object) -> EdgeType: ...

    def inputs(self) -> tuple[int, ...]: ...

    def is_ground(self, vertex: int) -> bool: ...

    def neighbors(self, vertex: int) -> Iterable[int]: ...

    def outputs(self) -> tuple[int, ...]: ...

    def phase(self, vertex: int) -> FractionLike: ...

    def qubit(self, vertex: int) -> FloatInt: ...

    def remove_vertex(self, vertex: int) -> None: ...

    def row(self, vertex: int) -> FloatInt: ...

    def set_phase(self, vertex: int, phase: FractionLike) -> None: ...

    def set_type(self, vertex: int, vertex_type: VertexType) -> None: ...

    def type(self, vertex: int) -> VertexType: ...

    def vertex_degree(self, vertex: int) -> int: ...

    def vertices(self) -> Iterable[int]: ...


class PyZXModule(Protocol):
    """Protocol for the runtime `pyzx` module attributes used here."""

    EdgeType: _PyZXEdgeTypeNamespace
    VertexType: _PyZXVertexTypeNamespace

    def is_graph_like(self, diagram: PyZXDiagram, *, strict: bool = ...) -> bool: ...


_PYZX_INSTALL_HINT = "PyZX support requires the optional dependency `graphqomb[pyzx]`."


def _require_pyzx() -> PyZXModule:
    """Import PyZX on demand for optional integration paths.

    Returns
    -------
    `PyZXModule`
        Imported `pyzx` module.

    Raises
    ------
    ModuleNotFoundError
        If the optional `pyzx` dependency is not installed.
    """
    try:
        zx = importlib.import_module("pyzx")
    except ModuleNotFoundError as exc:
        msg = f"{_PYZX_INSTALL_HINT} Install it with `pip install graphqomb[pyzx]`."
        raise ModuleNotFoundError(msg) from exc

    return cast("PyZXModule", zx)


@dataclasses.dataclass(frozen=True, slots=True)
class VertexData:
    """Collected PyZX vertex metadata used during import.

    Attributes
    ----------
    vertex_id : `int`
        Original PyZX vertex id.
    vertex_type : VertexType
        PyZX vertex type.
    phase : FractionLike
        PyZX vertex phase in multiples of pi.
    qubit : FloatInt
        PyZX qubit coordinate.
    row : FloatInt
        PyZX row coordinate.
    is_ground : `bool`
        Whether the vertex is marked as ground in PyZX.
    """

    vertex_id: int
    vertex_type: VertexType
    phase: FractionLike
    qubit: FloatInt
    row: FloatInt
    is_ground: bool


@dataclasses.dataclass(frozen=True, slots=True)
class EdgeData:
    """Collected PyZX edge metadata used during import.

    Attributes
    ----------
    source : `int`
        Smaller endpoint id of the undirected edge.
    target : `int`
        Larger endpoint id of the undirected edge.
    edge_type : EdgeType
        PyZX edge type.
    """

    source: int
    target: int
    edge_type: EdgeType


def from_pyzx(diagram: PyZXDiagram, *, recognize_pg: bool = False) -> tuple[GraphState, dict[int, int]]:
    r"""Convert a graph-like PyZX diagram into a graph state.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram in graph-like form.
    recognize_pg : `bool`, optional
        Whether to rewrite supported phase-gadget patterns before import.

    Returns
    -------
    `tuple`\[`GraphState`, `dict`\[`int`, `int`\]\]
        Imported graph state and a map from PyZX vertex ids to GraphState node ids.

    Raises
    ------
    ValueError
        If the input diagram is not in strict graph-like form.
    """
    pyzx = _require_pyzx()

    # Check whether the diagram is in graph-like form.
    if not pyzx.is_graph_like(diagram, strict=True):
        msg = "The input diagram is not in graph-like form. Please apply the graph-like transformation first."
        raise ValueError(msg)

    # Rewrite supported phase-gadget patterns before collecting graph data.
    diagram = _collect_phase_gadgets(diagram) if recognize_pg else diagram

    # Collect all vertices and edges in the diagram.
    node_map = _collect_node_map(diagram)
    edge_map = _collect_edge_map(diagram)

    # Process input and output boundaries.
    rewritten_inputs = _rewrite_input_boundary_maps(diagram, node_map, edge_map)
    rewritten_outputs = _rewrite_output_boundary_maps(diagram, node_map, edge_map)

    graph, graph_node_map = GraphState.from_graph(
        nodes=node_map,
        edges=edge_map,
        inputs=rewritten_inputs,
        outputs=rewritten_outputs,
        meas_bases=_build_meas_basis_map(node_map, output_nodes=set(rewritten_outputs)),
        coordinates=_build_coordinate_map(node_map),
    )

    return graph, graph_node_map


def _build_meas_basis_map(
    node_map: Mapping[int, VertexData],
    *,
    output_nodes: AbstractSet[int],
) -> dict[int, PlannerMeasBasis]:
    r"""Build GraphState measurement bases from PyZX vertex metadata.

    Parameters
    ----------
    node_map : `collections.abc.Mapping`\[`int`, `VertexData`\]
        Imported PyZX vertex metadata keyed by vertex id.
    output_nodes : `collections.abc.Set`\[`int`\]
        Node ids that should be treated as outputs and skipped.

    Returns
    -------
    dict[int, PlannerMeasBasis]
        Measurement-basis assignments for imported nodes.

    Raises
    ------
    ValueError
        If an imported vertex type cannot be represented as a measurement basis.
    """
    pyzx = _require_pyzx()
    meas_bases: dict[int, PlannerMeasBasis] = {}

    for vertex_id, vertex_data in node_map.items():
        if vertex_id in output_nodes or vertex_data.vertex_type == pyzx.VertexType.BOUNDARY:
            continue

        if vertex_data.vertex_type == pyzx.VertexType.Z:
            plane = Plane.XY
        elif vertex_data.vertex_type == pyzx.VertexType.X:
            plane = Plane.YZ
        else:
            msg = f"Unsupported PyZX vertex type for GraphState import: {vertex_data.vertex_type}"
            raise ValueError(msg)

        meas_bases[vertex_id] = PlannerMeasBasis(plane, _phase_to_angle(vertex_data.phase))

    return meas_bases


def _build_coordinate_map(node_map: Mapping[int, VertexData]) -> dict[int, tuple[float, float]]:
    r"""Build 2D coordinates from PyZX row and qubit placement.

    Parameters
    ----------
    node_map : `collections.abc.Mapping`\[`int`, `VertexData`\]
        Imported PyZX vertex metadata keyed by vertex id.

    Returns
    -------
    dict[int, tuple[float, float]]
        Coordinate map keyed by PyZX vertex id.
    """
    return {
        vertex_id: (float(vertex_data.row), float(vertex_data.qubit)) for vertex_id, vertex_data in node_map.items()
    }


def _phase_to_angle(phase: FractionLike) -> float:
    """Convert a PyZX phase expressed in multiples of pi into radians.

    Parameters
    ----------
    phase : FractionLike
        PyZX phase value expressed in multiples of pi.

    Returns
    -------
    float
        Phase angle in radians.

    Raises
    ------
    TypeError
        If the phase is symbolic and cannot be converted to a float.
    """
    try:
        return float(phase) * math.pi
    except TypeError as exc:
        msg = f"Unsupported symbolic PyZX phase for GraphState import: {phase!r}"
        raise TypeError(msg) from exc


def _collect_node_map(
    diagram: PyZXDiagram,
) -> dict[int, VertexData]:
    """Collect vertex metadata from a PyZX diagram.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.

    Returns
    -------
    dict[int, VertexData]
        Vertex metadata keyed by PyZX vertex id.
    """
    node_map: dict[int, VertexData] = {}

    for vertex_id in diagram.vertices():
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
) -> dict[tuple[int, int], EdgeData]:
    """Collect edge metadata from a PyZX diagram.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.

    Returns
    -------
    dict[tuple[int, int], EdgeData]
        Edge metadata keyed by canonical undirected endpoint pairs.

    Raises
    ------
    ValueError
        If the diagram contains parallel edges.
    """
    edge_map: dict[tuple[int, int], EdgeData] = {}

    for edge in diagram.edges():
        source, target = diagram.edge_st(edge)

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
    """Return a canonical key for an undirected edge.

    Parameters
    ----------
    source : `int`
        One endpoint of the edge.
    target : `int`
        The other endpoint of the edge.

    Returns
    -------
    tuple[int, int]
        Edge endpoints ordered increasingly.
    """
    return (source, target) if source <= target else (target, source)


def _rewrite_input_boundary_maps(
    diagram: PyZXDiagram,
    node_map: MutableMapping[int, VertexData],
    edge_map: MutableMapping[tuple[int, int], EdgeData],
) -> tuple[int, ...]:
    r"""Rewrite input boundaries into GraphState-compatible node and edge maps.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.
    node_map : `collections.abc.MutableMapping`\[`int`, `VertexData`\]
        Mutable imported vertex metadata keyed by vertex id.
    edge_map : `collections.abc.MutableMapping`\[`tuple`\[`int`, `int`\], `EdgeData`\]
        Mutable imported edge metadata keyed by canonical endpoint pairs.

    Returns
    -------
    tuple[int, ...]
        Imported input nodes in logical-qubit order.

    Raises
    ------
    ValueError
        If a boundary shape is unsupported or inconsistent with graph-like form.
    """
    pyzx = _require_pyzx()
    rewritten_inputs: list[int] = []
    input_vertices = diagram.inputs()

    for input_vertex in input_vertices:
        if input_vertex not in node_map:
            msg = f"Missing input vertex in collected node map: {input_vertex}"
            raise ValueError(msg)

        vertex_data = node_map[input_vertex]
        if vertex_data.vertex_type != pyzx.VertexType.BOUNDARY:
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

        if edge_data.edge_type == pyzx.EdgeType.HADAMARD:
            del node_map[input_vertex]
            del edge_map[edge_key]
            rewritten_inputs.append(neighbor)
            continue

        if edge_data.edge_type == pyzx.EdgeType.SIMPLE:
            node_map[input_vertex] = dataclasses.replace(
                vertex_data,
                vertex_type=pyzx.VertexType.Z,
                phase=0,
                is_ground=False,
            )
            edge_map[edge_key] = dataclasses.replace(edge_data, edge_type=pyzx.EdgeType.HADAMARD)
            rewritten_inputs.append(input_vertex)
            continue

        msg = f"Unsupported edge type for input boundary: {edge_data.edge_type}"
        raise ValueError(msg)

    return tuple(rewritten_inputs)


def _rewrite_output_boundary_maps(
    diagram: PyZXDiagram,
    node_map: MutableMapping[int, VertexData],
    edge_map: MutableMapping[tuple[int, int], EdgeData],
) -> tuple[int, ...]:
    r"""Rewrite output boundaries into GraphState-compatible node and edge maps.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.
    node_map : `collections.abc.MutableMapping`\[`int`, `VertexData`\]
        Mutable imported vertex metadata keyed by vertex id.
    edge_map : `collections.abc.MutableMapping`\[`tuple`\[`int`, `int`\], `EdgeData`\]
        Mutable imported edge metadata keyed by canonical endpoint pairs.

    Returns
    -------
    tuple[int, ...]
        Imported output nodes in logical-qubit order.

    Raises
    ------
    ValueError
        If a boundary shape is unsupported or inconsistent with graph-like form.
    """
    pyzx = _require_pyzx()
    rewritten_outputs: list[int] = []
    output_vertices = diagram.outputs()

    for output_vertex in output_vertices:
        if output_vertex not in node_map:
            msg = f"Missing output vertex in collected node map: {output_vertex}"
            raise ValueError(msg)

        vertex_data = node_map[output_vertex]
        if vertex_data.vertex_type != pyzx.VertexType.BOUNDARY:
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

        if edge_data.edge_type == pyzx.EdgeType.HADAMARD:
            rewritten_outputs.append(output_vertex)
            continue

        if edge_data.edge_type == pyzx.EdgeType.SIMPLE:
            node_map[output_vertex] = dataclasses.replace(
                vertex_data,
                vertex_type=pyzx.VertexType.Z,
                phase=0,
                is_ground=False,
            )
            edge_map[edge_key] = dataclasses.replace(edge_data, edge_type=pyzx.EdgeType.HADAMARD)

            new_output_vertex = _next_vertex_id(node_map)
            node_map[new_output_vertex] = VertexData(
                vertex_id=new_output_vertex,
                vertex_type=pyzx.VertexType.BOUNDARY,
                phase=0,
                qubit=vertex_data.qubit,
                row=vertex_data.row + 1,
                is_ground=False,
            )
            new_output_edge_key = _edge_key(output_vertex, new_output_vertex)
            edge_map[new_output_edge_key] = EdgeData(
                source=new_output_edge_key[0],
                target=new_output_edge_key[1],
                edge_type=pyzx.EdgeType.HADAMARD,
            )
            rewritten_outputs.append(new_output_vertex)

            continue

        msg = f"Unsupported edge type for output boundary: {edge_data.edge_type}"
        raise ValueError(msg)

    return tuple(rewritten_outputs)


def _next_vertex_id(node_map: Mapping[int, VertexData]) -> int:
    r"""Return a fresh synthetic vertex id for import-time rewrites.

    Parameters
    ----------
    node_map : `collections.abc.Mapping`\[`int`, `VertexData`\]
        Imported vertex metadata keyed by vertex id.

    Returns
    -------
    int
        Next available vertex id.
    """
    return max(node_map, default=-1) + 1


def _collect_phase_gadgets(diagram: PyZXDiagram) -> PyZXDiagram:
    r"""Rewrite supported phase-gadget patterns in a PyZX diagram.

    Parameters
    ----------
    diagram : `PyZXDiagram`
        Input PyZX diagram.

    Returns
    -------
    `PyZXDiagram`
        A copied diagram where each lone `Z` spider attached to a phase-free
        `Z` spider is absorbed into that neighbor, turning the neighbor into an
        `X` spider with the lone spider's phase.
    """
    pyzx = _require_pyzx()
    rewritten_diagram = diagram.copy()

    candidates: list[tuple[int, int]] = []
    for vertex in rewritten_diagram.vertices():
        if rewritten_diagram.type(vertex) != pyzx.VertexType.Z:
            continue
        if rewritten_diagram.vertex_degree(vertex) != 1:
            continue

        neighbor = next(iter(rewritten_diagram.neighbors(vertex)))
        if rewritten_diagram.type(neighbor) != pyzx.VertexType.Z:
            continue
        if rewritten_diagram.phase(neighbor) != 0:
            continue

        candidates.append((vertex, neighbor))

    for lone_z_spider, phase_free_neighbor in candidates:
        rewritten_diagram.set_type(phase_free_neighbor, pyzx.VertexType.X)
        rewritten_diagram.set_phase(phase_free_neighbor, rewritten_diagram.phase(lone_z_spider))
        rewritten_diagram.remove_vertex(lone_z_spider)

    return rewritten_diagram
