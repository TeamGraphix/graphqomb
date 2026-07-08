"""QEC Code object."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

from scipy.sparse import csr_array

from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState

if TYPE_CHECKING:
    from collections.abc import Mapping


_TYPE_II_CHAIN_LENGTH = 3


class YFoliation(Enum):
    """Y-foliation graph-state builder variant."""

    TYPE_I = auto()
    TYPE_II = auto()


class StabilizerCode:
    """A stabilizer code."""

    def __init__(
        self,
        stabilizer_matrix: csr_array,
        *,
        stabilizer_coords: Mapping[int, tuple[float, ...]] | None = None,
        qubit_coords: Mapping[int, tuple[float, ...]] | None = None,
    ) -> None:
        if stabilizer_matrix.shape[1] % 2 != 0:
            msg = "Stabilizer matrix must have an even number of columns."
            raise ValueError(msg)
        if stabilizer_coords is not None:
            _validate_coordinate_lengths(stabilizer_coords, expected_lengths={3}, label="stabilizer_coords")
        if qubit_coords is not None:
            _validate_coordinate_lengths(qubit_coords, expected_lengths={2, 3}, label="qubit_coords")

        self.hx = csr_array(stabilizer_matrix[:, : stabilizer_matrix.shape[1] // 2])
        self.hz = csr_array(stabilizer_matrix[:, stabilizer_matrix.shape[1] // 2 :])

        self.stabilizer_coord = stabilizer_coords
        self.qubit_coord = qubit_coords

    @property
    def num_stabilizers(self) -> int:
        """Return the number of stabilizers."""
        return int(self.hx.shape[0])

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return int(self.hx.shape[1])


class StabilizerGraphStateBuildResult(NamedTuple):
    """Result of building a graph state from a stabilizer code."""

    graph: GraphState
    data_nodes: dict[tuple[int, int], int]
    ancilla_nodes: dict[int, int]


class _DataLayerPlan(NamedTuple):
    """Data-node layer layout for graph-state construction."""

    data_layers: dict[int, tuple[int, ...]]
    coordinate_z_by_layer: dict[tuple[int, int], float]
    meas_basis_by_qubit: dict[int, AxisMeasBasis]
    y_foliation: YFoliation


class _StabilizerSupport(NamedTuple):
    """Sparse support sets for one stabilizer row."""

    hx: set[int]
    hz: set[int]


def build_graph_state(
    code: StabilizerCode,
    z_base: int = 0,
    *,
    y_foliation: YFoliation = YFoliation.TYPE_I,
) -> StabilizerGraphStateBuildResult:
    """Build a graph-state unit from a stabilizer code.

    Parameters
    ----------
    code : `StabilizerCode`
        Stabilizer code to convert. The X support is connected to the upper
        data layer and the Z support is connected to the lower data layer.
    z_base : `int`, optional
        Lower data-layer index. The builder creates layers `z_base` and
        `z_base + 1`, by default 0.
    y_foliation : `YFoliation`, optional
        Foliation variant. Type II uses a three-node Y-measured data chain only
        for qubits that have an Hx=Hz=1 support in at least one stabilizer row.

    Returns
    -------
    `StabilizerGraphStateBuildResult`
        Graph state and maps from stabilizer/data indices to graph nodes.

    Raises
    ------
    TypeError
        If z_base is not an integer.
    """
    if not isinstance(z_base, int):
        msg = "z_base must be an integer."
        raise TypeError(msg)

    graph = GraphState()
    x_meas_basis = AxisMeasBasis(Axis.X, Sign.PLUS)

    data_layer_plan = _data_layer_plan(
        code,
        z_base=z_base,
        y_foliation=y_foliation,
    )
    data_nodes = _add_layered_data_nodes(graph, code, data_layer_plan)
    ancilla_nodes = _add_ancilla_nodes(graph, code, data_nodes, data_layer_plan, x_meas_basis)

    return StabilizerGraphStateBuildResult(graph, data_nodes, ancilla_nodes)


def _data_layer_plan(
    code: StabilizerCode,
    *,
    z_base: int,
    y_foliation: YFoliation,
) -> _DataLayerPlan:
    data_layers: dict[int, tuple[int, ...]] = {}
    coordinate_z_by_layer: dict[tuple[int, int], float] = {}
    meas_basis_by_qubit: dict[int, AxisMeasBasis] = {}
    x_meas_basis = AxisMeasBasis(Axis.X, Sign.PLUS)
    y_meas_basis = AxisMeasBasis(Axis.Y, Sign.PLUS)
    y_chain_qubits: set[int] = _qubits_with_y_support(code) if y_foliation is YFoliation.TYPE_II else set()

    for qubit in range(code.num_qubits):
        if qubit in y_chain_qubits:
            layers = (z_base, z_base + 1, z_base + 2)
            coordinate_zs = (
                float(z_base),
                float(z_base) + 0.5,
                float(z_base + 1),
            )
            meas_basis = y_meas_basis
        else:
            layers = (z_base, z_base + 1)
            coordinate_zs = (float(z_base), float(z_base + 1))
            meas_basis = x_meas_basis

        data_layers[qubit] = layers
        meas_basis_by_qubit[qubit] = meas_basis
        for layer, coordinate_z in zip(layers, coordinate_zs, strict=True):
            coordinate_z_by_layer[qubit, layer] = coordinate_z

    return _DataLayerPlan(
        data_layers=data_layers,
        coordinate_z_by_layer=coordinate_z_by_layer,
        meas_basis_by_qubit=meas_basis_by_qubit,
        y_foliation=y_foliation,
    )


def _add_layered_data_nodes(
    graph: GraphState,
    code: StabilizerCode,
    data_layer_plan: _DataLayerPlan,
) -> dict[tuple[int, int], int]:
    """Add layered data nodes.

    Returns
    -------
    `dict`[`tuple`[`int`, `int`], `int`]
        Mapping from physical qubit and z layer to graph node.
    """
    data_nodes: dict[tuple[int, int], int] = {}
    for qubit in range(code.num_qubits):
        previous_node: int | None = None
        for layer in data_layer_plan.data_layers[qubit]:
            node = graph.add_node(
                coordinate=_data_coordinate(code, qubit, data_layer_plan.coordinate_z_by_layer[qubit, layer])
            )
            if previous_node is not None:
                graph.add_edge(previous_node, node)
            graph.assign_meas_basis(node, data_layer_plan.meas_basis_by_qubit[qubit])
            data_nodes[qubit, layer] = node
            previous_node = node

    return data_nodes


def _add_ancilla_nodes(
    graph: GraphState,
    code: StabilizerCode,
    data_nodes: dict[tuple[int, int], int],
    data_layer_plan: _DataLayerPlan,
    meas_basis: AxisMeasBasis,
) -> dict[int, int]:
    """Add ancilla nodes and stabilizer-support edges.

    Returns
    -------
    `dict`[`int`, `int`]
        Mapping from stabilizer row index to graph node.
    """
    ancilla_nodes: dict[int, int] = {}
    hx = code.hx.copy()
    hz = code.hz.copy()
    hx.eliminate_zeros()
    hz.eliminate_zeros()
    for stabilizer in range(code.num_stabilizers):
        explicit_ancilla_coord = _explicit_ancilla_coordinate(code, stabilizer)
        ancilla_node = graph.add_node(coordinate=explicit_ancilla_coord)
        graph.assign_meas_basis(ancilla_node, meas_basis)
        ancilla_nodes[stabilizer] = ancilla_node

        connected_data_nodes = _connect_stabilizer_support(
            graph,
            ancilla_node=ancilla_node,
            support=_StabilizerSupport(
                hx=set(_row_support(hx, stabilizer)),
                hz=set(_row_support(hz, stabilizer)),
            ),
            data_nodes=data_nodes,
            data_layer_plan=data_layer_plan,
        )

        if explicit_ancilla_coord is None:
            inferred_coord = _average_node_coordinates(graph, connected_data_nodes)
            if inferred_coord is not None:
                graph.set_coordinate(ancilla_node, inferred_coord)

    return ancilla_nodes


def _connect_stabilizer_support(
    graph: GraphState,
    *,
    ancilla_node: int,
    support: _StabilizerSupport,
    data_nodes: dict[tuple[int, int], int],
    data_layer_plan: _DataLayerPlan,
) -> list[int]:
    connected_data_nodes: list[int] = []
    for qubit in sorted(support.hx | support.hz):
        layers = data_layer_plan.data_layers[qubit]
        if data_layer_plan.y_foliation is YFoliation.TYPE_II and len(layers) == _TYPE_II_CHAIN_LENGTH:
            layer = _type_ii_support_layer(layers, has_x=qubit in support.hx, has_z=qubit in support.hz)
            data_node = data_nodes[qubit, layer]
            graph.add_edge(ancilla_node, data_node)
            connected_data_nodes.append(data_node)
            continue

        if qubit in support.hz:
            data_node = data_nodes[qubit, layers[0]]
            graph.add_edge(ancilla_node, data_node)
            connected_data_nodes.append(data_node)
        if qubit in support.hx:
            data_node = data_nodes[qubit, layers[-1]]
            graph.add_edge(ancilla_node, data_node)
            connected_data_nodes.append(data_node)

    return connected_data_nodes


def _type_ii_support_layer(layers: tuple[int, ...], *, has_x: bool, has_z: bool) -> int:
    if has_z and not has_x:
        return layers[0]
    if has_z and has_x:
        return layers[1]
    return layers[2]


def _qubits_with_y_support(code: StabilizerCode) -> set[int]:
    hx = code.hx.copy()
    hz = code.hz.copy()
    hx.eliminate_zeros()
    hz.eliminate_zeros()

    y_qubits: set[int] = set()
    for stabilizer in range(code.num_stabilizers):
        y_qubits.update(set(_row_support(hx, stabilizer)) & set(_row_support(hz, stabilizer)))
    return y_qubits


def _validate_coordinate_lengths(
    coordinates: Mapping[int, tuple[float, ...]],
    *,
    expected_lengths: set[int],
    label: str,
) -> None:
    """Validate coordinate tuple lengths.

    Raises
    ------
    ValueError
        If any coordinate has an invalid length.
    """
    for index, coord in coordinates.items():
        if len(coord) not in expected_lengths:
            expected = " or ".join(str(length) for length in sorted(expected_lengths))
            msg = f"{label}[{index}] must have length {expected}."
            raise ValueError(msg)


def _data_coordinate(code: StabilizerCode, qubit: int, z: float) -> tuple[float, float, float] | None:
    """Return the 3D coordinate of a layered data node.

    Returns
    -------
    `tuple`[`float`, `float`, `float`] | `None`
        Lifted 3D coordinate, or None when the qubit has no coordinate.
    """
    if code.qubit_coord is None or qubit not in code.qubit_coord:
        return None
    coord = code.qubit_coord[qubit]
    return (float(coord[0]), float(coord[1]), float(z))


def _explicit_ancilla_coordinate(code: StabilizerCode, stabilizer: int) -> tuple[float, float, float] | None:
    """Return an explicitly supplied ancilla coordinate, if present.

    Returns
    -------
    `tuple`[`float`, `float`, `float`] | `None`
        Explicit 3D coordinate, or None when no coordinate is supplied.
    """
    if code.stabilizer_coord is None or stabilizer not in code.stabilizer_coord:
        return None
    coord = code.stabilizer_coord[stabilizer]
    return (float(coord[0]), float(coord[1]), float(coord[2]))


def _row_support(matrix: csr_array, row: int) -> list[int]:
    """Return nonzero column indices in a CSR sparse row.

    Returns
    -------
    `list`[`int`]
        Nonzero column indices for the row.
    """
    start = int(matrix.indptr[row])
    end = int(matrix.indptr[row + 1])
    return [int(col) for col in matrix.indices[start:end]]


def _average_node_coordinates(graph: GraphState, nodes: list[int]) -> tuple[float, float, float] | None:
    """Return the componentwise average of node coordinates when all are available.

    Returns
    -------
    `tuple`[`float`, `float`, `float`] | `None`
        Average coordinate, or None when no average can be inferred.
    """
    if not nodes:
        return None
    coordinates = graph.coordinates
    if any(node not in coordinates for node in nodes):
        return None

    return (
        sum(coordinates[node][0] for node in nodes) / len(nodes),
        sum(coordinates[node][1] for node in nodes) / len(nodes),
        sum(coordinates[node][2] for node in nodes) / len(nodes),
    )
