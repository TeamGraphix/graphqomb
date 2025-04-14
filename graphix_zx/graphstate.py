"""Graph State classes for Measurement-based Quantum Computing.

This module provides:

- `BaseGraphState`: Abstract base class for Graph State.
- `GraphState`: Minimal implementation of Graph State.
- `sequentialy_compose`: Function to compose two graph states sequentially.
- `bipartite_edges`: Function to create a complete bipartite graph between two sets of nodes.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from itertools import product
from typing import TYPE_CHECKING

from graphix_zx.common import MeasBasis, Plane, PlannerMeasBasis
from graphix_zx.euler import update_lc_basis

if TYPE_CHECKING:
    from graphix_zx.euler import LocalClifford


class BaseGraphState(ABC):
    """Abstract base class for Graph State."""

    @property
    @abstractmethod
    def input_node_indices(self) -> dict[int, int]:
        r"""Return set of input nodes.

        Returns
        -------
        `dict`\[`int`, `int`\]
            qubit indices map of input nodes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_node_indices(self) -> dict[int, int]:
        r"""Return set of output nodes.

        Returns
        -------
        `dict`\[`int`, `int`\]
            qubit indices map of output nodes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def physical_nodes(self) -> set[int]:
        r"""Return set of physical nodes.

        Returns
        -------
        `set`\[`int`\]
            set of physical nodes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def physical_edges(self) -> set[tuple[int, int]]:
        r"""Return set of physical edges.

        Returns
        -------
        `set`\[`tuple`\[`int`, `int`\]`
            set of physical edges.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def meas_bases(self) -> dict[int, MeasBasis]:
        r"""Return measurement bases.

        Returns
        -------
        `dict`\[`int`, `MeasBasis`\]
            measurement bases of each physical node.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def local_cliffords(self) -> dict[int, LocalClifford]:
        r"""Return local clifford nodes.

        Returns
        -------
        `dict`\[`int`, `LocalClifford`\]
            local clifford nodes.
        """
        raise NotImplementedError

    @abstractmethod
    def add_physical_node(
        self,
    ) -> int:
        """Add a physical node to the graph state.

        Returns
        -------
        `int`
            The node index intenally generated
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @abstractmethod
    def set_input(self, node: int) -> int:
        """Set the node as an input node.

        Parameters
        ----------
        node : `int`
            node index

        Returns
        -------
        `int`
            logical qubit index
        """
        raise NotImplementedError

    @abstractmethod
    def set_output(self, node: int, q_index: int) -> None:
        """Set the node as an output node.

        Parameters
        ----------
        node : `int`
            node index
        q_index : `int`
            logical qubit index
        """
        raise NotImplementedError

    @abstractmethod
    def set_meas_basis(self, node: int, meas_basis: MeasBasis) -> None:
        """Set the measurement basis of the node.

        Parameters
        ----------
        node : `int`
            node index
        meas_basis : `MeasBasis`
            measurement basis
        """
        raise NotImplementedError

    @abstractmethod
    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
        """Apply a local clifford to the node.

        Parameters
        ----------
        node : `int`
            node index
        lc : `LocalClifford`
            local clifford operator
        """
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self, node: int) -> set[int]:
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
        raise NotImplementedError


class GraphState(BaseGraphState):
    """Minimal implementation of GraphState."""

    __input_node_indices: dict[int, int]
    __output_node_indices: dict[int, int]
    __physical_nodes: set[int]
    __physical_edges: dict[int, set[int]]
    __meas_bases: dict[int, MeasBasis]
    __local_cliffords: dict[int, LocalClifford]

    __inner_index: int

    def __init__(self) -> None:
        self.__input_node_indices = {}
        self.__output_node_indices = {}
        self.__physical_nodes = set()
        self.__physical_edges = {}
        self.__meas_bases = {}
        self.__local_cliffords = {}

        self.__inner_index = 0

    @property
    def input_node_indices(self) -> dict[int, int]:
        r"""Return map of input nodes.

        Returns
        -------
        `dict`\[`int`, `int`\]
            qubit indices map of input nodes.
        """
        return self.__input_node_indices

    @property
    def output_node_indices(self) -> dict[int, int]:
        r"""Return map of output nodes.

        Returns
        -------
        `dict`\[`int`, `int`\]
            qubit indices map of output nodes.
        """
        return self.__output_node_indices

    @property
    def num_physical_nodes(self) -> int:
        """Return the number of physical nodes.

        Returns
        -------
        `int`
            number of physical nodes.
        """
        return len(self.__physical_nodes)

    @property
    def num_physical_edges(self) -> int:
        """Return the number of physical edges.

        Returns
        -------
        `int`
            number of physical edges.
        """
        return sum(len(edges) for edges in self.__physical_edges.values()) // 2

    @property
    def physical_nodes(self) -> set[int]:
        r"""Return set of physical nodes.

        Returns
        -------
        `set`\[`int`\]
            set of physical nodes.
        """
        return self.__physical_nodes

    @property
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
    def meas_bases(self) -> dict[int, MeasBasis]:
        r"""Return measurement bases.

        Returns
        -------
        `dict`\[`int`, `MeasBasis`\]
            measurement bases of each physical node.
        """
        return self.__meas_bases

    @property
    def local_cliffords(self) -> dict[int, LocalClifford]:
        r"""Return local clifford nodes.

        Returns
        -------
        `dict`\[`int`, `LocalClifford`\]
            local clifford nodes.
        """
        return self.__local_cliffords

    def check_meas_basis(self) -> None:
        """Check if the measurement basis is set for all physical nodes except output nodes.

        Raises
        ------
        ValueError
            If the measurement basis is not set for a node or the measurement plane is invalid.
        """
        for v in self.physical_nodes - set(self.output_node_indices.keys()):
            if self.meas_bases.get(v) is None:
                msg = f"Measurement basis not set for node {v}"
                raise ValueError(msg)

    def add_physical_node(
        self,
    ) -> int:
        """Add a physical node to the graph state.

        Returns
        -------
        `int`
            The node index internally generated.
        """
        node = self.__inner_index
        self.__physical_nodes |= {node}
        self.__physical_edges[node] = set()
        self.__inner_index += 1

        return node

    def ensure_node_exists(self, node: int) -> None:
        """Ensure that the node exists in the graph state.

        Raises
        ------
        ValueError
            If the node does not exist in the graph state.
        """
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)

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
            If the edge already exists.
        """
        self.ensure_node_exists(node1)
        self.ensure_node_exists(node2)
        if node1 in self.__physical_edges[node2] or node2 in self.__physical_edges[node1]:
            msg = f"Edge already exists {node1=}, {node2=}"
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
        if node in self.input_node_indices:
            msg = "The input node cannot be removed"
            raise ValueError(msg)
        self.ensure_node_exists(node)
        self.__physical_nodes -= {node}
        for neighbor in self.__physical_edges[node]:
            self.__physical_edges[neighbor] -= {node}
        del self.__physical_edges[node]
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
        self.ensure_node_exists(node1)
        self.ensure_node_exists(node2)
        if node1 not in self.__physical_edges[node2] or node2 not in self.__physical_edges[node1]:
            msg = "Edge does not exist"
            raise ValueError(msg)
        self.__physical_edges[node1] -= {node2}
        self.__physical_edges[node2] -= {node1}

    def set_input(self, node: int) -> int:
        """Set the node as an input node.

        Parameters
        ----------
        node : `int`
            node index

        Returns
        -------
        `int`
            logical qubit index
        """
        self.ensure_node_exists(node)
        q_index = len(self.__input_node_indices)
        self.__input_node_indices[node] = q_index
        return q_index

    def set_output(self, node: int, q_index: int) -> None:
        """Set the node as an output node.

        Parameters
        ----------
        node : `int`
            node index
        q_index : `int`
            logical qubit index

        Raises
        ------
        ValueError
            1. If the node does not exist.
            2. If the node has a measurement basis.
            3. If the invalid q_index specified.
        """
        self.ensure_node_exists(node)
        if self.meas_bases.get(node) is not None:
            msg = "Cannot set output node with measurement basis."
            raise ValueError(msg)
        if q_index >= len(self.input_node_indices):
            msg = "The q_index does not exist in input qubit indices"
            raise ValueError(msg)
        self.__output_node_indices[node] = q_index

    def set_meas_basis(self, node: int, meas_basis: MeasBasis) -> None:
        """Set the measurement basis of the node.

        Parameters
        ----------
        node : `int`
            node index
        meas_basis : `MeasBasis`
            measurement basis
        """
        self.ensure_node_exists(node)
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
        self.ensure_node_exists(node)
        if node in self.input_node_indices or node in self.output_node_indices:
            self.__local_cliffords[node] = lc
        else:
            new_meas_basis = update_lc_basis(lc.conjugate(), self.meas_bases[node])
            self.set_meas_basis(node, new_meas_basis)

    def pop_local_clifford(self, node: int) -> LocalClifford | None:
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

    def parse_input_local_cliffords(self) -> dict[int, tuple[int, int, int]]:
        r"""Parse local Clifford operators applied on the input nodes.

        Returns
        -------
        `dict`\[`int`, `tuple`\[`int`, `int`, `int`\]\]
            A dictionary mapping input node indices to the new node indices created.
        """
        node_index_addition_map = {}
        new_input_indices = []
        for input_node, _ in sorted(self.input_node_indices.items(), key=operator.itemgetter(1)):
            lc = self.pop_local_clifford(input_node)
            if lc is None:
                continue

            new_node_index0 = self.add_physical_node()
            new_input_indices.append(new_node_index0)
            new_node_index1 = self.add_physical_node()
            new_node_index2 = self.add_physical_node()

            self.add_physical_edge(new_node_index0, new_node_index1)
            self.add_physical_edge(new_node_index1, new_node_index2)
            self.add_physical_edge(new_node_index2, input_node)

            self.set_meas_basis(new_node_index0, PlannerMeasBasis(Plane.XY, lc.alpha))
            self.set_meas_basis(new_node_index1, PlannerMeasBasis(Plane.XY, lc.beta))
            self.set_meas_basis(new_node_index2, PlannerMeasBasis(Plane.XY, lc.gamma))

            node_index_addition_map[input_node] = (new_node_index0, new_node_index1, new_node_index2)

        self.__input_node_indices = {}
        for new_input_index in new_input_indices:
            self.set_input(new_input_index)

        return node_index_addition_map

    def get_neighbors(self, node: int) -> set[int]:
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
        self.ensure_node_exists(node)
        return self.__physical_edges[node]


def sequentialy_compose(  # noqa: C901
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
        If the logical qubit indices of output nodes in graph1 do not match input nodes in graph2.
    """
    if set(graph1.output_node_indices.values()) != set(graph2.input_node_indices.values()):
        msg = "Logical qubit indices of output nodes in graph1 must match input nodes in graph2."
        raise ValueError(msg)
    node_map1 = {}
    node_map2 = {}
    composed_graph = GraphState()

    for node in graph1.physical_nodes:
        node_index = composed_graph.add_physical_node()
        meas_basis = graph1.meas_bases.get(node, None)
        if meas_basis is not None:
            composed_graph.set_meas_basis(node_index, meas_basis)
        lc = graph1.local_cliffords.get(node, None)
        if lc is not None:
            composed_graph.apply_local_clifford(node_index, lc)
        node_map1[node] = node_index

    for node in graph2.physical_nodes:
        node_index = composed_graph.add_physical_node()
        meas_basis = graph2.meas_bases.get(node, None)
        if meas_basis is not None:
            composed_graph.set_meas_basis(node_index, meas_basis)
        lc = graph2.local_cliffords.get(node, None)
        if lc is not None:
            composed_graph.apply_local_clifford(node_index, lc)
        node_map2[node] = node_index

    for input_node, _ in sorted(graph1.input_node_indices.items(), key=operator.itemgetter(1)):
        composed_graph.set_input(node_map1[input_node])

    for output_node, q_index in graph2.output_node_indices.items():
        composed_graph.set_output(node_map2[output_node], q_index)

    for edge in graph1.physical_edges:
        composed_graph.add_physical_edge(node_map1[edge[0]], node_map1[edge[1]])
    for edge in graph2.physical_edges:
        composed_graph.add_physical_edge(node_map2[edge[0]], node_map2[edge[1]])

    return composed_graph, node_map1, node_map2


def bipartite_edges(node_set1: set[int], node_set2: set[int]) -> set[tuple[int, int]]:
    r"""Return a set of edges for the complete bipartite graph between two sets of nodes.

    Parameters
    ----------
    node_set1 : `set`\[`int`\]
        set of nodes
    node_set2 : `set`\[`int`\]
        set of nodes

    Returns
    -------
    `set`\[`tuple`\[`int`, `int`\]
        set of edges for the complete bipartite graph
    """
    return {(min(a, b), max(a, b)) for a, b in product(node_set1, node_set2) if a != b}
