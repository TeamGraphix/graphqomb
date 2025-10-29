"""Graph State classes for Measurement-based Quantum Computing.

This module provides:

- `BaseGraphState`: Abstract base class for Graph State.
- `GraphState`: Minimal implementation of Graph State.
- `LocalCliffordExpansion`: Local Clifford expansion.
- `ExpansionMaps`: Expansion maps for local clifford operators.
- `compose`: Function to compose two graph states sequentially.
- `bipartite_edges`: Function to create a complete bipartite graph between two sets of nodes.
- `odd_neighbors`: Function to get odd neighbors of a node.

"""

from __future__ import annotations

import abc
import functools
import itertools
import operator
from abc import ABC
from collections.abc import Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, NamedTuple, TypeVar

import typing_extensions

from graphqomb.common import MeasBasis, Plane, PlannerMeasBasis
from graphqomb.euler import update_lc_basis, update_lc_lc

if TYPE_CHECKING:
    from graphqomb.euler import LocalClifford

NodeT = TypeVar("NodeT", bound=Hashable)


class BaseGraphState(ABC):
    """Abstract base class for Graph State."""

    @property
    @abc.abstractmethod
    def input_node_indices(self) -> dict[int, int]:
        r"""Return map of input nodes to logical qubit indices.

        Returns
        -------
        `dict`\[`int`, `int`\]
            qubit indices map of input nodes.
        """

    @property
    @abc.abstractmethod
    def output_node_indices(self) -> dict[int, int]:
        r"""Return map of output nodes to logical qubit indices.

        Returns
        -------
        `dict`\[`int`, `int`\]
            qubit indices map of output nodes.
        """

    @property
    @abc.abstractmethod
    def physical_nodes(self) -> set[int]:
        r"""Return set of physical nodes.

        Returns
        -------
        `set`\[`int`\]
            set of physical nodes.
        """

    @property
    @abc.abstractmethod
    def physical_edges(self) -> set[tuple[int, int]]:
        r"""Return set of physical edges.

        Returns
        -------
        `set`\[`tuple`\[`int`, `int`\]`
            set of physical edges.
        """

    @property
    @abc.abstractmethod
    def meas_bases(self) -> dict[int, MeasBasis]:
        r"""Return measurement bases.

        Returns
        -------
        `dict`\[`int`, `MeasBasis`\]
            measurement bases of each physical node.
        """

    @abc.abstractmethod
    def add_physical_node(self) -> int:
        """Add a physical node to the graph state.

        Returns
        -------
        `int`
            The node index intenally generated
        """

    @abc.abstractmethod
    def add_physical_edge(self, node1: int, node2: int) -> None:
        """Add a physical edge to the graph state.

        Parameters
        ----------
        node1 : `int`
            node index
        node2 : `int`
            node index
        """

    @abc.abstractmethod
    def register_input(self, node: int, q_index: int) -> None:
        """Mark the node as an input node.

        Parameters
        ----------
        node : `int`
            node index
        q_index : `int`
            logical qubit index
        """

    @abc.abstractmethod
    def register_output(self, node: int, q_index: int) -> None:
        """Mark the node as an output node.

        Parameters
        ----------
        node : `int`
            node index
        q_index : `int`
            logical qubit index
        """

    @abc.abstractmethod
    def assign_meas_basis(self, node: int, meas_basis: MeasBasis) -> None:
        """Assign the measurement basis of the node.

        Parameters
        ----------
        node : `int`
            node index
        meas_basis : `MeasBasis`
            measurement basis
        """

    @abc.abstractmethod
    def neighbors(self, node: int) -> set[int]:
        r"""Return the neighbors of the node.

        Parameters
        ----------
        node : `int`
            node index

        Returns
        -------
        `set`\[`int`\]
            set of neighboring nodes
        """

    @abc.abstractmethod
    def check_canonical_form(self) -> None:
        r"""Check if the graph state is in canonical form."""


class GraphState(BaseGraphState):
    """Minimal implementation of GraphState."""

    __input_node_indices: dict[int, int]
    __output_node_indices: dict[int, int]
    __physical_nodes: set[int]
    __physical_edges: dict[int, set[int]]
    __meas_bases: dict[int, MeasBasis]
    __local_cliffords: dict[int, LocalClifford]

    __node_counter: int

    def __init__(self) -> None:
        self.__input_node_indices = {}
        self.__output_node_indices = {}
        self.__physical_nodes = set()
        self.__physical_edges = {}
        self.__meas_bases = {}
        self.__local_cliffords = {}

        self.__node_counter = 0

    @property
    @typing_extensions.override
    def input_node_indices(self) -> dict[int, int]:
        r"""Return map of input nodes to logical qubit indices.

        Returns
        -------
        `dict`\[`int`, `int`\]
            qubit indices map of input nodes.
        """
        return self.__input_node_indices.copy()

    @property
    @typing_extensions.override
    def output_node_indices(self) -> dict[int, int]:
        r"""Return map of output nodes to logical qubit indices.

        Returns
        -------
        `dict`\[`int`, `int`\]
            qubit indices map of output nodes.
        """
        return self.__output_node_indices.copy()

    @property
    @typing_extensions.override
    def physical_nodes(self) -> set[int]:
        r"""Return set of physical nodes.

        Returns
        -------
        `set`\[`int`\]
            set of physical nodes.
        """
        return self.__physical_nodes.copy()

    @property
    @typing_extensions.override
    def physical_edges(self) -> set[tuple[int, int]]:
        r"""Return set of physical edges.

        Returns
        -------
        `set`\[`tuple`\[`int`, `int`\]
            set of physical edges.
        """
        edges: set[tuple[int, int]] = set()
        for node1 in self.__physical_edges:
            for node2 in self.__physical_edges[node1]:
                if node1 < node2:
                    edges |= {(node1, node2)}
        return edges

    @property
    @typing_extensions.override
    def meas_bases(self) -> dict[int, MeasBasis]:
        r"""Return measurement bases.

        Returns
        -------
        `dict`\[`int`, `MeasBasis`\]
            measurement bases of each physical node.
        """
        return self.__meas_bases.copy()

    @property
    def local_cliffords(self) -> dict[int, LocalClifford]:
        r"""Return local clifford nodes.

        Returns
        -------
        `dict`\[`int`, `LocalClifford`\]
            local clifford nodes.
        """
        return self.__local_cliffords.copy()

    def _check_meas_basis(self) -> None:
        """Check if the measurement basis is set for all physical nodes except output nodes.

        Raises
        ------
        ValueError
            If the measurement basis is not set for a node or the measurement plane is invalid.
        """
        for v in self.physical_nodes - set(self.output_node_indices):
            if self.meas_bases.get(v) is None:
                msg = f"Measurement basis not set for node {v}"
                raise ValueError(msg)

    def _ensure_node_exists(self, node: int) -> None:
        """Ensure that the node exists in the graph state.

        Raises
        ------
        ValueError
            If the node does not exist in the graph state.
        """
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)

    @typing_extensions.override
    def add_physical_node(self) -> int:
        """Add a physical node to the graph state.

        Returns
        -------
        `int`
            The node index internally generated.
        """
        node = self.__node_counter
        self.__physical_nodes |= {node}
        self.__physical_edges[node] = set()
        self.__node_counter += 1

        return node

    @typing_extensions.override
    def add_physical_edge(self, node1: int, node2: int) -> None:
        """Add a physical edge to the graph state.

        Parameters
        ----------
        node1 : `int`
            node index
        node2 : `int`
            node index

        Raises
        ------
        ValueError
            1. If the node does not exist.
            2. If the edge already exists.
            3. If the edge is a self-loop.
        """
        self._ensure_node_exists(node1)
        self._ensure_node_exists(node2)
        if node1 in self.__physical_edges[node2] or node2 in self.__physical_edges[node1]:
            msg = f"Edge already exists {node1=}, {node2=}"
            raise ValueError(msg)
        if node1 == node2:
            msg = "Self-loops are not allowed"
            raise ValueError(msg)
        self.__physical_edges[node1] |= {node2}
        self.__physical_edges[node2] |= {node1}

    def remove_physical_node(self, node: int) -> None:
        """Remove a physical node from the graph state.

        Parameters
        ----------
        node : `int`
            node index to be removed

        Raises
        ------
        ValueError
            If the input node is specified
        """
        self._ensure_node_exists(node)
        if node in self.input_node_indices:
            msg = "The input node cannot be removed"
            raise ValueError(msg)
        self.__physical_nodes -= {node}
        for neighbor in self.__physical_edges[node]:
            self.__physical_edges[neighbor] -= {node}
        del self.__physical_edges[node]

        if node in self.output_node_indices:
            del self.__output_node_indices[node]
        self.__meas_bases.pop(node, None)
        self.__local_cliffords.pop(node, None)

    def remove_physical_edge(self, node1: int, node2: int) -> None:
        """Remove a physical edge from the graph state.

        Parameters
        ----------
        node1 : `int`
            node index
        node2 : `int`
            node index

        Raises
        ------
        ValueError
            If the edge does not exist.
        """
        self._ensure_node_exists(node1)
        self._ensure_node_exists(node2)
        if node1 not in self.__physical_edges[node2] or node2 not in self.__physical_edges[node1]:
            msg = "Edge does not exist"
            raise ValueError(msg)
        self.__physical_edges[node1] -= {node2}
        self.__physical_edges[node2] -= {node1}

    @typing_extensions.override
    def register_input(self, node: int, q_index: int) -> None:
        """Mark the node as an input node.

        Parameters
        ----------
        node : `int`
            node index
        q_index : `int`
            logical qubit index

        Raises
        ------
        ValueError
            If the node is already registered as an input node.
        """
        self._ensure_node_exists(node)
        if node in self.__input_node_indices:
            msg = "The node is already registered as an input node."
            raise ValueError(msg)
        if q_index in self.input_node_indices.values():
            msg = "The q_index already exists in input qubit indices"
            raise ValueError(msg)
        self.__input_node_indices[node] = q_index

    @typing_extensions.override
    def register_output(self, node: int, q_index: int) -> None:
        """Mark the node as an output node.

        Parameters
        ----------
        node : `int`
            node index
        q_index : `int`
            logical qubit index

        Raises
        ------
        ValueError
            1. If the node is already registered as an output node.
            2. If the invalid q_index specified.
            3. If the q_index already exists in output qubit indices.
        """
        self._ensure_node_exists(node)
        if node in self.__output_node_indices:
            msg = "The node is already registered as an output node."
            raise ValueError(msg)
        if q_index in self.output_node_indices.values():
            msg = "The q_index already exists in output qubit indices"
            raise ValueError(msg)
        self.__output_node_indices[node] = q_index

    @typing_extensions.override
    def assign_meas_basis(self, node: int, meas_basis: MeasBasis) -> None:
        """Set the measurement basis of the node.

        Parameters
        ----------
        node : `int`
            node index
        meas_basis : `MeasBasis`
            measurement basis
        """
        self._ensure_node_exists(node)
        self.__meas_bases[node] = meas_basis

    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
        """Apply a local clifford to the node.

        Parameters
        ----------
        node : `int`
            node index
        lc : `LocalClifford`
            local clifford operator
        """
        self._ensure_node_exists(node)
        if node in self.input_node_indices or node in self.output_node_indices:
            original_lc = self._pop_local_clifford(node)
            if original_lc is not None:
                new_lc = update_lc_lc(lc, original_lc)
                self.__local_cliffords[node] = new_lc
            else:
                self.__local_cliffords[node] = lc
        else:
            self._check_meas_basis()
            new_meas_basis = update_lc_basis(lc.conjugate(), self.meas_bases[node])
            self.assign_meas_basis(node, new_meas_basis)

    @typing_extensions.override
    def neighbors(self, node: int) -> set[int]:
        r"""Return the neighbors of the node.

        Parameters
        ----------
        node : `int`
            node index

        Returns
        -------
        `set`\[`int`\]
            set of neighboring nodes
        """
        self._ensure_node_exists(node)
        return self.__physical_edges[node].copy()

    @typing_extensions.override
    def check_canonical_form(self) -> None:
        r"""Check if the graph state is in canonical form.

        The definition of canonical form is:
        1. No Clifford operators applied.
        2. All non-output nodes have measurement basis

        Raises
        ------
        ValueError
            If the graph state is not in canonical form.
        """
        if self.__local_cliffords:
            msg = "Clifford operators are applied."
            raise ValueError(msg)
        for node in self.physical_nodes - set(self.output_node_indices):
            if self.meas_bases.get(node) is None:
                msg = "All non-output nodes must have measurement basis."
                raise ValueError(msg)

    def expand_local_cliffords(self) -> ExpansionMaps:
        r"""Expand local Clifford operators applied on the input and output nodes.

        Returns
        -------
        `ExpansionMaps`
            A tuple of dictionaries mapping input and output node indices to the new node indices created.
        """
        input_node_map = self._expand_input_local_cliffords()
        output_node_map = self._expand_output_local_cliffords()
        return ExpansionMaps(input_node_map, output_node_map)

    def _pop_local_clifford(self, node: int) -> LocalClifford | None:
        """Pop local clifford of the node.

        Parameters
        ----------
        node : `int`
            node index to remove local clifford.

        Returns
        -------
        `LocalClifford` | `None`
            removed local clifford
        """
        return self.__local_cliffords.pop(node, None)

    def _expand_input_local_cliffords(self) -> dict[int, LocalCliffordExpansion]:
        r"""Expand local Clifford operators applied on the input nodes.

        Returns
        -------
        `dict`\[`int`, `LocalCliffordExpansion`\]
            A dictionary mapping input node indices to the new node indices created.
        """
        node_index_addition_map: dict[int, LocalCliffordExpansion] = {}
        new_input_indices: dict[int, int] = {}
        for input_node, q_index in self.input_node_indices.items():
            lc = self._pop_local_clifford(input_node)
            if lc is None:
                new_input_indices[input_node] = q_index
                continue

            new_node_index0 = self.add_physical_node()
            new_input_indices[new_node_index0] = q_index
            new_node_index1 = self.add_physical_node()
            new_node_index2 = self.add_physical_node()

            self.add_physical_edge(new_node_index0, new_node_index1)
            self.add_physical_edge(new_node_index1, new_node_index2)
            self.add_physical_edge(new_node_index2, input_node)

            self.assign_meas_basis(new_node_index0, PlannerMeasBasis(Plane.XY, lc.alpha))
            self.assign_meas_basis(new_node_index1, PlannerMeasBasis(Plane.XY, lc.beta))
            self.assign_meas_basis(new_node_index2, PlannerMeasBasis(Plane.XY, lc.gamma))

            node_index_addition_map[input_node] = LocalCliffordExpansion(
                new_node_index0, new_node_index1, new_node_index2
            )

        self.__input_node_indices = {}
        for new_input_index, q_index in new_input_indices.items():
            self.register_input(new_input_index, q_index)

        return node_index_addition_map

    def _expand_output_local_cliffords(self) -> dict[int, LocalCliffordExpansion]:
        r"""Expand local Clifford operators applied on the output nodes.

        Returns
        -------
        `dict`\[`int`, `LocalCliffordExpansion`\]
            A dictionary mapping output node indices to the new node indices created.
        """
        node_index_addition_map: dict[int, LocalCliffordExpansion] = {}
        new_output_index_map: dict[int, int] = {}
        for output_node, q_index in self.output_node_indices.items():
            lc = self._pop_local_clifford(output_node)
            if lc is None:
                new_output_index_map[output_node] = q_index
                continue

            new_node_index0 = self.add_physical_node()
            new_node_index1 = self.add_physical_node()
            new_node_index2 = self.add_physical_node()
            new_output_index_map[new_node_index2] = q_index

            self.add_physical_edge(output_node, new_node_index0)
            self.add_physical_edge(new_node_index0, new_node_index1)
            self.add_physical_edge(new_node_index1, new_node_index2)

            self.assign_meas_basis(output_node, PlannerMeasBasis(Plane.XY, lc.alpha))
            self.assign_meas_basis(new_node_index0, PlannerMeasBasis(Plane.XY, lc.beta))
            self.assign_meas_basis(new_node_index1, PlannerMeasBasis(Plane.XY, lc.gamma))

            node_index_addition_map[output_node] = LocalCliffordExpansion(
                new_node_index0, new_node_index1, new_node_index2
            )

        self.__output_node_indices = {}
        for new_output_index, q_index in new_output_index_map.items():
            self.register_output(new_output_index, q_index)

        return node_index_addition_map

    @classmethod
    def from_graph(  # noqa: C901, PLR0912
        cls,
        nodes: Iterable[NodeT],
        edges: Iterable[tuple[NodeT, NodeT]],
        inputs: Sequence[NodeT] | None = None,
        outputs: Sequence[NodeT] | None = None,
        meas_bases: Mapping[NodeT, MeasBasis] | None = None,
    ) -> tuple[GraphState, dict[NodeT, int]]:
        r"""Create a graph state from nodes and edges with arbitrary node types.

        This factory method allows creating a graph state from any hashable node type
        (e.g., strings, tuples, custom objects). The method internally maps external
        node identifiers to integer indices used by GraphState.

        Parameters
        ----------
        nodes : `collections.abc.Iterable`\[NodeT\]
            Nodes to add to the graph. Can be any hashable type.
        edges : `collections.abc.Iterable`\[`tuple`\[NodeT, NodeT\]\]
            Edges as pairs of node identifiers.
        inputs : `collections.abc.Sequence`\[NodeT\] | `None`, optional
            Input nodes in order. Qubit indices are assigned sequentially (0, 1, 2, ...).
            Default is None (no inputs).
        outputs : `collections.abc.Sequence`\[NodeT\] | `None`, optional
            Output nodes in order. Qubit indices are assigned sequentially (0, 1, 2, ...).
            Default is None (no outputs).
        meas_bases : `collections.abc.Mapping`\[NodeT, `MeasBasis`\] | `None`, optional
            Measurement bases for nodes. Nodes not specified can be set later.
            Default is None (no bases assigned initially).

        Returns
        -------
        `tuple`\[`GraphState`, `dict`\[NodeT, `int`\]\]
            - Created GraphState instance
            - Mapping from external node IDs to internal integer indices

        Raises
        ------
        ValueError
            If duplicate nodes, invalid edges, or invalid input/output nodes.
        """
        # Convert nodes to list to preserve order
        nodes_list = list(nodes)

        # Check for duplicate nodes
        if len(nodes_list) != len(set(nodes_list)):
            msg = "Duplicate nodes in input"
            raise ValueError(msg)

        node_set = set(nodes_list)

        # Validate inputs
        if inputs is not None:
            for input_node in inputs:
                if input_node not in node_set:
                    msg = f"Input node {input_node} not in nodes collection"
                    raise ValueError(msg)

        # Validate outputs
        if outputs is not None:
            for output_node in outputs:
                if output_node not in node_set:
                    msg = f"Output node {output_node} not in nodes collection"
                    raise ValueError(msg)

        # Convert edges to list for validation
        edges_list = list(edges)

        # Validate edges
        for node1, node2 in edges_list:
            if node1 not in node_set:
                msg = f"Edge references non-existent node {node1}"
                raise ValueError(msg)
            if node2 not in node_set:
                msg = f"Edge references non-existent node {node2}"
                raise ValueError(msg)

        # Create GraphState instance
        graph_state = cls()

        # Add nodes and create node mapping
        node_map: dict[NodeT, int] = {}
        for node in nodes_list:
            new_node = graph_state.add_physical_node()
            node_map[node] = new_node

        # Add edges
        for node1, node2 in edges_list:
            idx1 = node_map[node1]
            idx2 = node_map[node2]
            graph_state.add_physical_edge(idx1, idx2)

        # Register inputs with sequential qubit indices
        if inputs is not None:
            for q_index, input_node in enumerate(inputs):
                graph_state.register_input(node_map[input_node], q_index)

        # Register outputs with sequential qubit indices
        if outputs is not None:
            for q_index, output_node in enumerate(outputs):
                graph_state.register_output(node_map[output_node], q_index)

        # Assign measurement bases
        if meas_bases is not None:
            for node, meas_basis in meas_bases.items():
                if node in node_set:
                    graph_state.assign_meas_basis(node_map[node], meas_basis)

        return graph_state, node_map

    @classmethod
    def from_base_graph_state(
        cls,
        base: BaseGraphState,
        copy_local_cliffords: bool = True,
    ) -> tuple[GraphState, dict[int, int]]:
        r"""Create a new GraphState from an existing BaseGraphState instance.

        This method creates a complete copy of the graph structure, including nodes,
        edges, input/output registrations, and measurement bases. Useful for creating
        mutable copies or converting between GraphState implementations.

        Parameters
        ----------
        base : `BaseGraphState`
            The source graph state to copy from.
        copy_local_cliffords : `bool`, optional
            Whether to copy local Clifford operators if the source is a GraphState.
            If True and the source has local Cliffords, they are copied.
            If False, local Cliffords are not copied (canonical form only).
            Default is True.

        Returns
        -------
        `tuple`\[`GraphState`, `dict`\[`int`, `int`\]\]
            - Created GraphState instance
            - Mapping from source node indices to new node indices
        """
        # Create new GraphState instance
        graph_state = cls()

        # Create node mapping
        node_map: dict[int, int] = {}
        for node in base.physical_nodes:
            new_node = graph_state.add_physical_node()
            node_map[node] = new_node

        # Add edges using node mapping
        for node1, node2 in base.physical_edges:
            graph_state.add_physical_edge(node_map[node1], node_map[node2])

        # Register inputs with same qubit indices
        for input_node, q_index in base.input_node_indices.items():
            graph_state.register_input(node_map[input_node], q_index)

        # Register outputs with same qubit indices
        for output_node, q_index in base.output_node_indices.items():
            graph_state.register_output(node_map[output_node], q_index)

        # Copy measurement bases
        for node, meas_basis in base.meas_bases.items():
            graph_state.assign_meas_basis(node_map[node], meas_basis)

        # Copy local Clifford operators if requested and source is GraphState
        if copy_local_cliffords and isinstance(base, GraphState):
            for node, lc in base.local_cliffords.items():
                # Access private attribute to copy local cliffords
                graph_state.apply_local_clifford(node_map[node], lc)

        return graph_state, node_map


class LocalCliffordExpansion(NamedTuple):
    """Local Clifford expansion map for each input/output node."""

    node1: int
    node2: int
    node3: int


class ExpansionMaps(NamedTuple):
    """Expansion maps for inputs and outputs with Local Clifford."""

    input_node_map: dict[int, LocalCliffordExpansion]
    output_node_map: dict[int, LocalCliffordExpansion]


def compose(  # noqa: C901
    graph1: BaseGraphState, graph2: BaseGraphState
) -> tuple[GraphState, dict[int, int], dict[int, int]]:
    r"""Compose two graph states sequentially.

    Qubits with matching indices are automatically connected. Graph2 is connected after graph1.
    All other qubit indices are preserved from their original graphs.

    Parameters
    ----------
    graph1 : `BaseGraphState`
        first graph state
    graph2 : `BaseGraphState`
        second graph state

    Returns
    -------
    `tuple`\[`GraphState`, `dict`\[`int`, `int`\], `dict`\[`int`, `int`\]\]
        composed graph state, node map for graph1, node map for graph2

    Raises
    ------
    ValueError
        1. If the graph states are not in canonical form.
        2. If there are qindex conflicts (same qindex used in both graphs but not for connection).
    """
    graph1.check_canonical_form()
    graph2.check_canonical_form()

    # Automatically detect connection targets: qindices that appear in both graphs
    output_q_indices1 = set(graph1.output_node_indices.values())
    input_q_indices2 = set(graph2.input_node_indices.values())
    target_q_indices = output_q_indices1 & input_q_indices2

    # Check for qindex conflicts: qindices used in both graphs but not for connection
    all_q_indices1 = set(graph1.input_node_indices.values()) | output_q_indices1
    all_q_indices2 = input_q_indices2 | set(graph2.output_node_indices.values())
    conflicting_q_indices = (all_q_indices1 & all_q_indices2) - target_q_indices

    if conflicting_q_indices:
        msg = (
            f"Qindex conflicts detected: {conflicting_q_indices}. "
            "These indices are used in both graphs but cannot be connected."
        )
        raise ValueError(msg)

    composed_graph = GraphState()

    # Copy nodes from graph1, excluding output nodes that will be connected
    node_map1 = _copy_nodes(
        src=graph1,
        dst=composed_graph,
        exclude_nodes={node for node, q_index in graph1.output_node_indices.items() if q_index in target_q_indices},
    )

    # Copy all nodes from graph2
    node_map2 = _copy_nodes(
        src=graph2,
        dst=composed_graph,
        exclude_nodes=set(),
    )

    # Connect output nodes from graph1 to input nodes from graph2 for target qindices
    q_index2output_node_index1 = {
        q_index: output_node_index1 for output_node_index1, q_index in graph1.output_node_indices.items()
    }
    for input_node_index2, q_index in graph2.input_node_indices.items():
        if q_index in target_q_indices:
            node_map1[q_index2output_node_index1[q_index]] = node_map2[input_node_index2]

    # Register input nodes with preserved qindices
    for input_node, q_index in graph1.input_node_indices.items():
        composed_graph.register_input(node_map1[input_node], q_index)

    for input_node, q_index in graph2.input_node_indices.items():
        if q_index not in target_q_indices:
            composed_graph.register_input(node_map2[input_node], q_index)

    # Register output nodes with preserved qindices
    for output_node, q_index in graph1.output_node_indices.items():
        if q_index not in target_q_indices:
            composed_graph.register_output(node_map1[output_node], q_index)

    for output_node, q_index in graph2.output_node_indices.items():
        composed_graph.register_output(node_map2[output_node], q_index)

    for u, v in graph1.physical_edges:
        composed_graph.add_physical_edge(node_map1[u], node_map1[v])
    for u, v in graph2.physical_edges:
        composed_graph.add_physical_edge(node_map2[u], node_map2[v])

    return composed_graph, node_map1, node_map2


def _copy_nodes(
    src: BaseGraphState,
    dst: BaseGraphState,
    exclude_nodes: AbstractSet[int],
) -> dict[int, int]:
    r"""Copy nodes from src to dst, excluding specified nodes.

    Parameters
    ----------
    src : `BaseGraphState`
        source graph state
    dst : `BaseGraphState`
        destination graph state
    exclude_nodes : `collections.abc.Set`\[`int`\]
        set of nodes to exclude from copying

    Returns
    -------
    `dict`\[`int`, `int`\]
        mapping from src node indices to dst node indices
    """
    node_map: dict[int, int] = {}
    for node in src.physical_nodes:
        if node in exclude_nodes:
            continue
        new_idx = dst.add_physical_node()
        meas = src.meas_bases.get(node)
        if meas is not None:
            dst.assign_meas_basis(new_idx, meas)
        node_map[node] = new_idx
    return node_map


def bipartite_edges(node_set1: AbstractSet[int], node_set2: AbstractSet[int]) -> set[tuple[int, int]]:
    r"""Return a set of edges for the complete bipartite graph between two sets of nodes.

    Parameters
    ----------
    node_set1 : `collections.abc.Set`\[`int`\]
        set of nodes
    node_set2 : `collections.abc.Set`\[`int`\]
        set of nodes

    Returns
    -------
    `set`\[`tuple`\[`int`, `int`\]
        set of edges for the complete bipartite graph

    Raises
    ------
    ValueError
        If the two sets of nodes are not disjoint.
    """
    if not node_set1.isdisjoint(node_set2):
        msg = "The two sets of nodes must be disjoint."
        raise ValueError(msg)
    return {(min(a, b), max(a, b)) for a, b in itertools.product(node_set1, node_set2)}


def odd_neighbors(nodes: AbstractSet[int], graphstate: BaseGraphState) -> set[int]:
    r"""Return the odd neighbors of a set of nodes in the graph state.

    Parameters
    ----------
    nodes : `collections.abc.Set`\[`int`\]
        set of nodes
    graphstate : `BaseGraphState`
        graph state

    Returns
    -------
    `set`\[`int`\]
        set of odd neighbors
    """
    return functools.reduce(operator.xor, (graphstate.neighbors(node) for node in nodes), set())  # pyright: ignore[reportUnknownArgumentType]
