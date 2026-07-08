"""QEC Code object."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

from scipy.sparse import csr_array

from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState

if TYPE_CHECKING:
    from collections.abc import Mapping


class YFoliation(Enum):
    """Y-foliation variant."""

    TypeI = auto()
    TypeII = auto()


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


class _DataLayerConfig(NamedTuple):
    lower_z: int
    upper_z: int
    meas_basis: AxisMeasBasis
    data_as_io: bool
    qubit_indices: Mapping[int, int] | None


def build_graph_state(
    code: StabilizerCode,
    z_base: int = 0,
    *,
    data_as_io: bool = False,
    qubit_indices: Mapping[int, int] | None = None,
) -> StabilizerGraphStateBuildResult:
    """Build a two-layer graph-state unit from a stabilizer code.

    Parameters
    ----------
    code : `StabilizerCode`
        Stabilizer code to convert. The X support is connected to the upper
        data layer and the Z support is connected to the lower data layer.
    z_base : `int`, optional
        Lower data-layer index. The builder creates layers `z_base` and
        `z_base + 1`, by default 0.
    data_as_io : `bool`, optional
        Whether to register lower data nodes as inputs and upper data nodes as
        outputs, by default False.
    qubit_indices : `collections.abc.Mapping`[`int`, `int`] | `None`, optional
        Mapping from stabilizer-code qubit columns to graph qindices when
        ``data_as_io`` is enabled. If omitted, code qubit columns are used.

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

    lower_z = z_base
    upper_z = z_base + 1
    data_nodes = _add_layered_data_nodes(
        graph,
        code,
        _DataLayerConfig(
            lower_z=lower_z,
            upper_z=upper_z,
            meas_basis=x_meas_basis,
            data_as_io=data_as_io,
            qubit_indices=qubit_indices,
        ),
    )
    ancilla_nodes = _add_ancilla_nodes(graph, code, data_nodes, (lower_z, upper_z), x_meas_basis)

    return StabilizerGraphStateBuildResult(graph, data_nodes, ancilla_nodes)


def _add_layered_data_nodes(
    graph: GraphState,
    code: StabilizerCode,
    config: _DataLayerConfig,
) -> dict[tuple[int, int], int]:
    """Add layered data nodes.

    Returns
    -------
    `dict`[`tuple`[`int`, `int`], `int`]
        Mapping from physical qubit and z layer to graph node.
    """
    data_nodes: dict[tuple[int, int], int] = {}
    for qubit in range(code.num_qubits):
        lower_node = graph.add_node(coordinate=_data_coordinate(code, qubit, config.lower_z))
        upper_node = graph.add_node(coordinate=_data_coordinate(code, qubit, config.upper_z))
        graph.add_edge(lower_node, upper_node)
        graph.assign_meas_basis(lower_node, config.meas_basis)
        if config.data_as_io:
            q_index = config.qubit_indices[qubit] if config.qubit_indices is not None else qubit
            graph.register_input(lower_node, q_index)
            graph.register_output(upper_node, q_index)
        else:
            graph.assign_meas_basis(upper_node, config.meas_basis)
        data_nodes[qubit, config.lower_z] = lower_node
        data_nodes[qubit, config.upper_z] = upper_node

    return data_nodes


def _add_ancilla_nodes(
    graph: GraphState,
    code: StabilizerCode,
    data_nodes: dict[tuple[int, int], int],
    z_layers: tuple[int, int],
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
    lower_z, upper_z = z_layers
    for stabilizer in range(code.num_stabilizers):
        explicit_ancilla_coord = _explicit_ancilla_coordinate(code, stabilizer)
        ancilla_node = graph.add_node(coordinate=explicit_ancilla_coord)
        graph.assign_meas_basis(ancilla_node, meas_basis)
        ancilla_nodes[stabilizer] = ancilla_node

        connected_data_nodes: list[int] = []
        for qubit in _row_support(hz, stabilizer):
            data_node = data_nodes[qubit, lower_z]
            graph.add_edge(ancilla_node, data_node)
            connected_data_nodes.append(data_node)
        for qubit in _row_support(hx, stabilizer):
            data_node = data_nodes[qubit, upper_z]
            graph.add_edge(ancilla_node, data_node)
            connected_data_nodes.append(data_node)

        if explicit_ancilla_coord is None:
            inferred_coord = _average_node_coordinates(graph, connected_data_nodes)
            if inferred_coord is not None:
                graph.set_coordinate(ancilla_node, inferred_coord)

    return ancilla_nodes


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


def _data_coordinate(code: StabilizerCode, qubit: int, z: int) -> tuple[float, float, float] | None:
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


def foliate(stabilizer: StabilizerCode, r: int, y_foliation: YFoliation = YFoliation.TypeI) -> None:
    """Foliate a stabilizer code.

    This placeholder keeps the existing no-op behavior until foliation is implemented.
    """
    del stabilizer, r, y_foliation
