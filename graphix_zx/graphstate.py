"""Graph State classes for Measurement-based Quantum Computing.

This module provides:

- `BaseGraphState`: Abstract base class for Graph State.
- `GraphState`: Minimal implementation of Graph State.
- `compose_sequentially`: Function to compose two graph states sequentially.
- `compose_in_parallel`: Function to compose two graph states in parallel.
- `bipartite_edges`: Function to create a complete bipartite graph between two sets of nodes.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING

import typing_extensions

from graphix_zx.common import MeasBasis, Plane, PlannerMeasBasis
from graphix_zx.euler import update_lc_basis, update_lc_lc

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from graphix_zx.euler import LocalClifford


class BaseGraphState(ABC):
    """Abstract base class for Graph State."""

    @property
    @abstractmethod
    def input_node_indices(self) -> MappingProxyType[int, int]:
        r"""Return map of input nodes to logical qubit indices.

        Returns
        -------
        `types.MappingProxyType`\[`int`, `int`\]
            qubit indices map of input nodes.
        """

    @property
    @abstractmethod
    def output_node_indices(self) -> MappingProxyType[int, int]:
        r"""Return map of output nodes to logical qubit indices.

        Returns
        -------
        `types.MappingProxyType`\[`int`, `int`\]
            qubit indices map of output nodes.
        """

    @property
    @abstractmethod
    def physical_nodes(self) -> set[int]:
        r"""Return set of physical nodes.

        Returns
        -------
        `set`\[`int`\]
            set of physical nodes.
        """

    @property
    @abstractmethod
    def physical_edges(self) -> set[tuple[int, int]]:
        r"""Return set of physical edges.

        Returns
        -------
        `set`\[`tuple`\[`int`, `int`\]`
            set of physical edges.
        """

    @property
    @abstractmethod
    def meas_bases(self) -> MappingProxyType[int, MeasBasis]:
        r"""Return measurement bases.

        Returns
        -------
        `types.MappingProxyType`\[`int`, `MeasBasis`\]
            measurement bases of each physical node.
        """

    @abstractmethod
    def add_physical_node(self) -> int:
        """Add a physical node to the graph state.

        Returns
        -------
        `int`
            The node index intenally generated
        """

    @abstractmethod
    def add_physical_edge(self, node1: int, node2: int) -> None:
        """Add a physical edge to the graph state.

        Parameters
        ----------
        node1 : `int`
            node index
        node2 : `int`
            node index
        """

    @abstractmethod
    def register_input(self, node: int) -> int:
        """Mark the node as an input node.

        Parameters
        ----------
        node : `int`
            node index

        Returns
        -------
        `int`
            logical qubit index
        """

    @abstractmethod
    def register_output(self, node: int, q_index: int) -> None:
        """Mark the node as an output node.

        Parameters
        ----------
        node : `int`
            node index
        q_index : `int`
            logical qubit index
        """

    @abstractmethod
    def assign_meas_basis(self, node: int, meas_basis: MeasBasis) -> None:
        """Assign the measurement basis of the node.

        Parameters
        ----------
        node : `int`
            node index
        meas_basis : `MeasBasis`
            measurement basis
        """

    @abstractmethod
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

    @abstractmethod
    def is_canonical_form(self) -> bool:
        r"""Check if the graph state is in canonical form.

        Returns
        -------
        `bool`
            `True` if the graph state is in canonical form, `False` otherwise.
        """


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
    def input_node_indices(self) -> MappingProxyType[int, int]:
        r"""Return map of input nodes to logical qubit indices.

        Returns
        -------
        `types.MappingProxyType`\[`int`, `int`\]
            qubit indices map of input nodes.
        """
        return MappingProxyType(self.__input_node_indices)

    @property
    @typing_extensions.override
    def output_node_indices(self) -> MappingProxyType[int, int]:
        r"""Return map of output nodes to logical qubit indices.

        Returns
        -------
        `types.MappingProxyType`\[`int`, `int`\]
            qubit indices map of output nodes.
        """
        return MappingProxyType(self.__output_node_indices)

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
        edges = set()
        for node1 in self.__physical_edges:
            for node2 in self.__physical_edges[node1]:
                if node1 < node2:
                    edges |= {(node1, node2)}
        return edges

    @property
    @typing_extensions.override
    def meas_bases(self) -> MappingProxyType[int, MeasBasis]:
        r"""Return measurement bases.

        Returns
        -------
        `types.MappingProxyType`\[`int`, `MeasBasis`\]
            measurement bases of each physical node.
        """
        return MappingProxyType(self.__meas_bases)

    @property
    def local_cliffords(self) -> MappingProxyType[int, LocalClifford]:
        r"""Return local clifford nodes.

        Returns
        -------
        `types.MappingProxyType`\[`int`, `LocalClifford`\]
            local clifford nodes.
        """
        return MappingProxyType(self.__local_cliffords)

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
    def register_input(self, node: int) -> int:
        """Mark the node as an input node.

        Parameters
        ----------
        node : `int`
            node index

        Returns
        -------
        `int`
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
        q_index = len(self.__input_node_indices)
        self.__input_node_indices[node] = q_index
        return q_index

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
            2. If the node has a measurement basis.
            3. If the invalid q_index specified.
            4. If the q_index already exists in output qubit indices.
        """
        self._ensure_node_exists(node)
        if node in self.__output_node_indices:
            msg = "The node is already registered as an output node."
            raise ValueError(msg)
        if self.meas_bases.get(node) is not None:
            msg = "Cannot set output node with measurement basis."
            raise ValueError(msg)
        if q_index >= len(self.input_node_indices):
            msg = "The q_index does not exist in input qubit indices"
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

        Raises
        ------
        ValueError
            If the node is an output node.
        """
        self._ensure_node_exists(node)
        if node in self.output_node_indices:
            msg = "Cannot set measurement basis for output node."
            raise ValueError(msg)
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
    def is_canonical_form(self) -> bool:
        r"""Check if the graph state is in canonical form.

        The definition of canonical form is:
        1. No Clifford operators applied.
        2. All non-output nodes have measurement basis.

        Returns
        -------
        `bool`
            `True` if the graph state is in canonical form, `False` otherwise.
        """
        if self.__local_cliffords:
            return False
        for node in self.physical_nodes - set(self.output_node_indices):
            if self.meas_bases.get(node) is None:
                return False
        return True

    def expand_local_cliffords(self) -> tuple[dict[int, tuple[int, int, int]], dict[int, tuple[int, int, int]]]:
        r"""Expand local Clifford operators applied on the input and output nodes.

        Returns
        -------
        `tuple`\[`dict`\[`int`, `tuple`\[`int`, `int`, `int`\]\], `dict`\[`int`, `tuple`\[`int`, `int`, `int`\]\]\]
            A tuple of dictionaries mapping input and output node indices to the new node indices created.
        """
        input_node_map = self._expand_input_local_cliffords()
        output_node_map = self._expand_output_local_cliffords()
        return input_node_map, output_node_map

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

    def _expand_input_local_cliffords(self) -> dict[int, tuple[int, int, int]]:
        r"""Expand local Clifford operators applied on the input nodes.

        Returns
        -------
        `dict`\[`int`, `tuple`\[`int`, `int`, `int`\]\]
            A dictionary mapping input node indices to the new node indices created.
        """
        node_index_addition_map = {}
        new_input_indices = []
        for input_node, _ in sorted(self.input_node_indices.items(), key=operator.itemgetter(1)):
            lc = self._pop_local_clifford(input_node)
            if lc is None:
                new_input_indices.append(input_node)
                continue

            new_node_index0 = self.add_physical_node()
            new_input_indices.append(new_node_index0)
            new_node_index1 = self.add_physical_node()
            new_node_index2 = self.add_physical_node()

            self.add_physical_edge(new_node_index0, new_node_index1)
            self.add_physical_edge(new_node_index1, new_node_index2)
            self.add_physical_edge(new_node_index2, input_node)

            self.assign_meas_basis(new_node_index0, PlannerMeasBasis(Plane.XY, lc.alpha))
            self.assign_meas_basis(new_node_index1, PlannerMeasBasis(Plane.XY, lc.beta))
            self.assign_meas_basis(new_node_index2, PlannerMeasBasis(Plane.XY, lc.gamma))

            node_index_addition_map[input_node] = (new_node_index0, new_node_index1, new_node_index2)

        self.__input_node_indices = {}
        for new_input_index in new_input_indices:
            self.register_input(new_input_index)

        return node_index_addition_map

    def _expand_output_local_cliffords(self) -> dict[int, tuple[int, int, int]]:
        r"""Expand local Clifford operators applied on the output nodes.

        Returns
        -------
        `dict`\[`int`, `tuple`\[`int`, `int`, `int`\]\]
            A dictionary mapping output node indices to the new node indices created.
        """
        node_index_addition_map = {}
        new_output_indices = []
        for output_node, _ in sorted(self.output_node_indices.items(), key=operator.itemgetter(1)):
            lc = self._pop_local_clifford(output_node)
            if lc is None:
                new_output_indices.append(output_node)
                continue

            new_node_index0 = self.add_physical_node()
            new_node_index1 = self.add_physical_node()
            new_node_index2 = self.add_physical_node()
            new_output_indices.append(new_node_index2)

            self.add_physical_edge(output_node, new_node_index0)
            self.add_physical_edge(new_node_index0, new_node_index1)
            self.add_physical_edge(new_node_index1, new_node_index2)

            self.assign_meas_basis(output_node, PlannerMeasBasis(Plane.XY, lc.alpha))
            self.assign_meas_basis(new_node_index0, PlannerMeasBasis(Plane.XY, lc.beta))
            self.assign_meas_basis(new_node_index1, PlannerMeasBasis(Plane.XY, lc.gamma))

            node_index_addition_map[output_node] = (new_node_index0, new_node_index1, new_node_index2)

        self.__output_node_indices = {}
        for new_output_index in new_output_indices:
            self.register_output(new_output_index, len(self.__output_node_indices))

        return node_index_addition_map


def compose_sequentially(  # noqa: C901
    graph1: BaseGraphState, graph2: BaseGraphState
) -> tuple[BaseGraphState, dict[int, int], dict[int, int]]:
    r"""Compose two graph states sequentially.

    Parameters
    ----------
    graph1 : `BaseGraphState`
        first graph state
    graph2 : `BaseGraphState`
        second graph state

    Returns
    -------
    `tuple`\[`BaseGraphState`, `dict`\[`int`, `int`\], `dict`\[`int`, `int`\]\]
        composed graph state, node map for graph1, node map for graph2

    Raises
    ------
    ValueError
        1. If graph1 or graph2 is not in canonical form.
        2. If the logical qubit indices of output nodes in graph1 do not match input nodes in graph2.
    """
    if not graph1.is_canonical_form():
        msg = "graph1 must be in canonical form."
        raise ValueError(msg)
    if not graph2.is_canonical_form():
        msg = "graph2 must be in canonical form."
        raise ValueError(msg)
    if set(graph1.output_node_indices.values()) != set(graph2.input_node_indices.values()):
        msg = "Logical qubit indices of output nodes in graph1 must match input nodes in graph2."
        raise ValueError(msg)
    node_map1 = {}
    node_map2 = {}
    composed_graph = GraphState()

    for node in graph1.physical_nodes - graph1.output_node_indices.keys():
        node_index = composed_graph.add_physical_node()
        meas_basis = graph1.meas_bases.get(node, None)
        if meas_basis is not None:
            composed_graph.assign_meas_basis(node_index, meas_basis)
        node_map1[node] = node_index

    for node in graph2.physical_nodes:
        node_index = composed_graph.add_physical_node()
        meas_basis = graph2.meas_bases.get(node, None)
        if meas_basis is not None:
            composed_graph.assign_meas_basis(node_index, meas_basis)
        node_map2[node] = node_index

    for input_node, _ in sorted(graph1.input_node_indices.items(), key=operator.itemgetter(1)):
        composed_graph.register_input(node_map1[input_node])

    for output_node, q_index in graph2.output_node_indices.items():
        composed_graph.register_output(node_map2[output_node], q_index)

    # overlapping node process
    q_index2output_node_index1 = {
        q_index: output_node_index1 for output_node_index1, q_index in graph1.output_node_indices.items()
    }
    for input_node_index2, q_index in graph2.input_node_indices.items():
        node_map1[q_index2output_node_index1[q_index]] = node_map2[input_node_index2]

    for u, v in graph1.physical_edges:
        composed_graph.add_physical_edge(node_map1[u], node_map1[v])
    for u, v in graph2.physical_edges:
        composed_graph.add_physical_edge(node_map2[u], node_map2[v])

    return composed_graph, node_map1, node_map2


def compose_in_parallel(  # noqa: C901
    graph1: BaseGraphState, graph2: BaseGraphState
) -> tuple[BaseGraphState, dict[int, int], dict[int, int]]:
    r"""Compose two graph states in parallel.

    Parameters
    ----------
    graph1 : `BaseGraphState`
        first graph state
    graph2 : `BaseGraphState`
        second graph state

    Returns
    -------
    `tuple`\[`BaseGraphState`, `dict`\[`int`, `int`\], `dict`\[`int`, `int`\]\]
        composed graph state, node map for graph1, node map for graph2

    Raises
    ------
    ValueError
        If graph1 or graph2 is not in canonical form.
    """
    if not graph1.is_canonical_form():
        msg = "graph1 must be in canonical form."
        raise ValueError(msg)
    if not graph2.is_canonical_form():
        msg = "graph2 must be in canonical form."
        raise ValueError(msg)
    node_map1 = {}
    node_map2 = {}
    composed_graph = GraphState()

    for node in graph1.physical_nodes:
        node_index = composed_graph.add_physical_node()
        meas_basis = graph1.meas_bases.get(node, None)
        if meas_basis is not None:
            composed_graph.assign_meas_basis(node_index, meas_basis)
        node_map1[node] = node_index

    for node in graph2.physical_nodes:
        node_index = composed_graph.add_physical_node()
        meas_basis = graph2.meas_bases.get(node, None)
        if meas_basis is not None:
            composed_graph.assign_meas_basis(node_index, meas_basis)
        node_map2[node] = node_index

    q_index_map1 = {}
    q_index_map2 = {}
    for input_node, old_q_index in sorted(graph1.input_node_indices.items(), key=operator.itemgetter(1)):
        new_q_index = composed_graph.register_input(node_map1[input_node])
        q_index_map1[old_q_index] = new_q_index

    for input_node, old_q_index in sorted(graph2.input_node_indices.items(), key=operator.itemgetter(1)):
        new_q_index = composed_graph.register_input(node_map2[input_node])
        q_index_map2[old_q_index] = new_q_index

    for output_node, q_index in graph1.output_node_indices.items():
        composed_graph.register_output(node_map1[output_node], q_index_map1[q_index])

    for output_node, q_index in graph2.output_node_indices.items():
        composed_graph.register_output(node_map2[output_node], q_index_map2[q_index])

    for u, v in graph1.physical_edges:
        composed_graph.add_physical_edge(node_map1[u], node_map1[v])
    for u, v in graph2.physical_edges:
        composed_graph.add_physical_edge(node_map2[u], node_map2[v])

    return composed_graph, node_map1, node_map2


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
    return {(min(a, b), max(a, b)) for a, b in product(node_set1, node_set2)}
