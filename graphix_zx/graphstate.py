"""Graph State classes for Measurement-based Quantum Computing.

This module provides:
- `BaseGraphState`: Abstract base class for Graph State.
- `GraphState`: Minimal implementation of Graph State.
- `bipartite_edges`: Function to create a complete bipartite graph between two sets of nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from typing import TYPE_CHECKING

from graphix_zx.common import MeasBasis, Plane, PlannerMeasBasis, default_meas_basis
from graphix_zx.euler import update_lc_basis

if TYPE_CHECKING:
    from graphix_zx.euler import LocalClifford


class BaseGraphState(ABC):
    """Abstract base class for Graph State."""

    @property
    @abstractmethod
    def input_nodes(self) -> set[int]:
        """Return set of input nodes.

        Returns
        -------
        set[int]
            set of input nodes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_nodes(self) -> set[int]:
        """Return set of output nodes.

        Returns
        -------
        set[int]
            set of output nodes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def physical_nodes(self) -> set[int]:
        """Return set of physical nodes.

        Returns
        -------
        set[int]
            set of physical nodes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def physical_edges(self) -> set[tuple[int, int]]:
        """Return set of physical edges.

        Returns
        -------
        set[tuple[int, int]]
            set of physical edges.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    # Generics?
    def q_indices(self) -> dict[int, int]:
        """Return local qubit indices.

        Returns
        -------
        dict[int, int]
            logical qubit indices of each physical node.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def meas_bases(self) -> dict[int, MeasBasis]:
        """Return measurement bases.

        Returns
        -------
        dict[int, MeasBasis]
            measurement bases of each physical node.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def local_cliffords(self) -> dict[int, LocalClifford]:
        """Return local clifford nodes.

        Returns
        -------
        dict[int, LocalClifford]
            local clifford nodes.
        """
        raise NotImplementedError

    @abstractmethod
    def add_physical_node(
        self,
        node: int,
        q_index: int,
        *,
        is_input: bool = False,
        is_output: bool = False,
    ) -> None:
        """Add a physical node to the graph state.

        Parameters
        ----------
        node : int
            node index
        q_index : int
            logical qubit index
        is_input : bool
            True if node is input node
        is_output : bool
            True if node is output node
        """
        raise NotImplementedError

    @abstractmethod
    def add_physical_edge(self, node1: int, node2: int) -> None:
        """Add a physical edge to the graph state.

        Parameters
        ----------
        node1 : int
            node index
        node2 : int
            node index
        """
        raise NotImplementedError

    @abstractmethod
    def set_input(self, node: int) -> None:
        """Set the node as an input node.

        Parameters
        ----------
        node : int
            node index
        """
        raise NotImplementedError

    @abstractmethod
    def set_output(self, node: int) -> None:
        """Set the node as an output node.

        Parameters
        ----------
        node : int
            node index
        """
        raise NotImplementedError

    @abstractmethod
    def set_q_index(self, node: int, q_index: int) -> None:
        """Set the qubit index of the node.

        Parameters
        ----------
        node : int
            node index
        q_index:  int
            logical qubit index
        """
        raise NotImplementedError

    @abstractmethod
    def set_meas_basis(self, node: int, meas_basis: MeasBasis) -> None:
        """Set the measurement basis of the node.

        Parameters
        ----------
        node : int
            node index
        meas_basis : MeasBasis
            measurement basis
        """
        raise NotImplementedError

    @abstractmethod
    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
        """Apply a local clifford to the node.

        Parameters
        ----------
        node : int
            node index
        lc : LocalClifford
            local clifford operator
        """
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self, node: int) -> set[int]:
        """Return the neighbors of the node.

        Parameters
        ----------
        node : int
            node index

        Returns
        -------
        set[int]
            set of neighboring nodes
        """
        raise NotImplementedError


class GraphState(BaseGraphState):
    """Minimal implementation of GraphState.

    Attributes
    ----------
    input_nodes : set[int]
        set of input nodes
    output_nodes : set[int]
        set of output nodes
    physical_nodes : set[int]
        set of physical nodes
    physical_edges : dict[int, set[int]]
        physical edges
    meas_bases : dict[int, MeasBasis]
        measurement bases
    q_indices : dict[int, int]
        qubit indices
    local_cliffords : dict[int, LocalClifford]
        local clifford operators
    """

    __input_nodes: set[int]
    __output_nodes: set[int]
    __physical_nodes: set[int]
    __physical_edges: dict[int, set[int]]
    __meas_bases: dict[int, MeasBasis]
    __q_indices: dict[int, int]
    __local_cliffords: dict[int, LocalClifford]

    __inner_index: int
    __inner2nodes: dict[int, int]
    __nodes2inner: dict[int, int]

    def __init__(self) -> None:
        self.__input_nodes = set()
        self.__output_nodes = set()
        self.__physical_nodes = set()
        self.__physical_edges = {}
        self.__meas_bases = {}
        # NOTE: qubit index if allocated. -1 if not. used for simulation
        self.__q_indices = {}
        self.__local_cliffords = {}

        self.__inner_index = 0
        self.__inner2nodes = {}
        self.__nodes2inner = {}

    @property
    def input_nodes(self) -> set[int]:
        """Return set of input nodes.

        Returns
        -------
        set[int]
            set of input nodes.
        """
        return self.__input_nodes

    @property
    def output_nodes(self) -> set[int]:
        """Return set of output nodes.

        Returns
        -------
        set[int]
            set of output nodes.
        """
        return self.__output_nodes

    @property
    def num_physical_nodes(self) -> int:
        """Return the number of physical nodes.

        Returns
        -------
        int
            number of physical nodes.
        """
        return len(self.__physical_nodes)

    @property
    def num_physical_edges(self) -> int:
        """Return the number of physical edges.

        Returns
        -------
        int
            number of physical edges.
        """
        return sum(len(edges) for edges in self.__physical_edges.values()) // 2

    @property
    def physical_nodes(self) -> set[int]:
        """Return set of physical nodes.

        Returns
        -------
        set[int]
            set of physical nodes.
        """
        return self.__physical_nodes

    @property
    def physical_edges(self) -> set[tuple[int, int]]:
        """Return set of physical edges.

        Returns
        -------
        set[tuple[int, int]]
            set of physical edges.
        """
        edges = set()
        for node1 in self.__physical_edges:
            for node2 in self.__physical_edges[node1]:
                if node1 < node2:
                    edges |= {(node1, node2)}
        return edges

    @property
    def q_indices(self) -> dict[int, int]:
        """Return local qubit indices.

        Returns
        -------
        dict[int, int]
            logical qubit indices of each physical node.
        """
        return self.__q_indices

    @property
    def meas_bases(self) -> dict[int, MeasBasis]:
        """Return measurement bases.

        Returns
        -------
        dict[int, MeasBasis]
            measurement bases of each physical node.
        """
        return self.__meas_bases

    @property
    def local_cliffords(self) -> dict[int, LocalClifford]:
        """Return local clifford nodes.

        Returns
        -------
        dict[int, LocalClifford]
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
        for v in self.physical_nodes - self.output_nodes:
            if self.meas_bases.get(v) is None:
                msg = f"Measurement basis not set for node {v}"
                raise ValueError(msg)

    def add_physical_node(
        self,
        node: int,
        q_index: int = -1,
        *,
        is_input: bool = False,
        is_output: bool = False,
    ) -> None:
        """Add a physical node to the graph state.

        Parameters
        ----------
        node : int
            node index
        q_index : int
            logical qubit index
        is_input : bool
            True if node is input node
        is_output : bool
            True if node is output node

        Raises
        ------
        ValueError
            If the node already exists in the graph state.
        """
        if node in self.__physical_nodes:
            msg = f"Node already exists {node=}"
            raise ValueError(msg)
        self.__physical_nodes |= {node}
        self.__physical_edges[node] = set()
        self.set_q_index(node, q_index)
        if is_input:
            self.__input_nodes |= {node}
        if is_output:
            self.__output_nodes |= {node}

        self.__inner2nodes[self.__inner_index] = node
        self.__nodes2inner[node] = self.__inner_index
        self.__inner_index += 1

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
        node1 : int
            node index
        node2 : int
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
            node : int

        Raises
        ------
        ValueError
            If the node does not exist.
        """
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)
        self.ensure_node_exists(node)
        self.__physical_nodes -= {node}
        del self.__physical_edges[node]
        self.__input_nodes -= {node}
        self.__output_nodes -= {node}
        self.__meas_bases.pop(node, None)
        self.__q_indices.pop(node, None)
        self.__local_cliffords.pop(node, None)
        for neighbor in self.__physical_edges:
            self.__physical_edges[neighbor] -= {node}

        self.__inner2nodes.pop(self.__nodes2inner[node])
        self.__nodes2inner.pop(node)

    def remove_physical_edge(self, node1: int, node2: int) -> None:
        """Remove a physical edge from the graph state.

        Parameters
        ----------
        node1 : int
            node index
        node2 : int
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

    def set_input(self, node: int) -> None:
        """Set the node as an input node.

        Parameters
        ----------
        node : int
            node index
        """
        self.ensure_node_exists(node)
        self.__input_nodes |= {node}

    def set_output(self, node: int) -> None:
        """Set the node as an output node.

        Parameters
        ----------
        node : int
            node index

        Raises
        ------
        ValueError
            1. If the node does not exist.
            2. If the node has a measurement basis.
        """
        self.ensure_node_exists(node)
        if self.meas_bases.get(node) is not None:
            msg = "Cannot set output node with measurement basis."
            raise ValueError(msg)
        self.__output_nodes |= {node}

    def set_q_index(self, node: int, q_index: int = -1) -> None:
        """Set the qubit index of the node.

        Parameters
        ----------
        node : int
            node index
        q_index:  int, optional
            logical qubit index, by default -1

        Raises
        ------
        ValueError
            If the qubit index is invalid.
        """
        self.ensure_node_exists(node)
        if q_index < -1:
            msg = f"Invalid qubit index {q_index}. Must be -1 or greater"
            raise ValueError(msg)
        self.__q_indices[node] = q_index

    def set_meas_basis(self, node: int, meas_basis: MeasBasis) -> None:
        """Set the measurement basis of the node.

        Parameters
        ----------
        node : int
            node index
        meas_basis : MeasBasis
            measurement basis
        """
        self.ensure_node_exists(node)
        self.__meas_bases[node] = meas_basis

    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
        """Apply a local clifford to the node.

        Parameters
        ----------
        node : int
            node index
        lc : LocalClifford
            local clifford operator
        """
        self.ensure_node_exists(node)
        if node in self.input_nodes or node in self.output_nodes:
            self.__local_cliffords[node] = lc
        else:
            new_meas_basis = update_lc_basis(lc.conjugate(), self.meas_bases[node])
            self.set_meas_basis(node, new_meas_basis)

    def pop_local_clifford(self, node: int) -> LocalClifford | None:
        """Pop local clifford of the node.

        Parameters
        ----------
        node : int
            node index to remove local clifford.

        Returns
        -------
        LocalClifford | None
            removed local clifford
        """
        return self.__local_cliffords.pop(node, None)

    def parse_input_local_cliffords(self) -> None:
        """Parse local Clifford operators applied on the input nodes."""
        for input_node in self.input_nodes:
            lc = self.pop_local_clifford(input_node)
            if lc is None:
                continue
            node_indices = (self.__inner_index, self.__inner_index + 1, self.__inner_index + 2)

            self.add_physical_node(node_indices[0], q_index=self.q_indices[input_node], is_input=True)
            self.add_physical_node(node_indices[1], q_index=self.q_indices[input_node])
            self.add_physical_node(node_indices[2], q_index=self.q_indices[input_node])

            self.add_physical_edge(node_indices[0], node_indices[1])
            self.add_physical_edge(node_indices[1], node_indices[2])
            self.add_physical_edge(node_indices[2], input_node)

            self.set_meas_basis(node_indices[0], PlannerMeasBasis(Plane.XY, lc.alpha))
            self.set_meas_basis(node_indices[1], PlannerMeasBasis(Plane.XY, lc.beta))
            self.set_meas_basis(node_indices[2], PlannerMeasBasis(Plane.XY, lc.gamma))

            self._reset_input(input_node)

    def get_neighbors(self, node: int) -> set[int]:
        """Return the neighbors of the node.

        Parameters
        ----------
        node : int
            node index

        Returns
        -------
        set[int]
            set of neighboring nodes
        """
        self.ensure_node_exists(node)
        return self.__physical_edges[node]

    def _reset_input(self, node: int) -> None:
        """Reset the input status of the node.

        Parameters
        ----------
        node : int
            node index
        """
        if node in self.__input_nodes:
            self.__input_nodes.remove(node)
        lc = self.pop_local_clifford(node)
        if lc is not None:
            self.apply_local_clifford(node, lc)

    def _reset_output(self, node: int) -> None:
        """Reset the output status of the node.

        Parameters
        ----------
        node : int
            node index
        """
        if node in self.__output_nodes:
            self.__output_nodes.remove(node)

    def append(self, other: BaseGraphState) -> None:
        """Append another graph state to the current graph state.

        Parameters
        ----------
        other : BaseGraphState
            another graph state to append

        Raises
        ------
        ValueError
            If the qubit indices do not match.
        """
        common_nodes = self.physical_nodes & other.physical_nodes
        border_nodes = self.output_nodes & other.input_nodes

        if common_nodes != border_nodes:
            msg = "Qubit index mismatch"
            raise ValueError(msg)

        for node in other.physical_nodes:
            if node in border_nodes:
                self._reset_input(node)
                self._reset_output(node)
            else:
                self.add_physical_node(node)
                if node in other.input_nodes - self.output_nodes:
                    self.set_input(node)

            if node in other.output_nodes:
                self.set_output(node)
            else:
                meas_basis = other.meas_bases.get(node, default_meas_basis())
                self.set_meas_basis(node, meas_basis)

        for edge in other.physical_edges:
            self.add_physical_edge(edge[0], edge[1])

        # q_index update
        for node, q_index in other.q_indices.items():
            if (node in common_nodes) and (self.q_indices[node] != q_index):
                msg = "Qubit index mismatch."
                raise ValueError(msg)
            self.set_q_index(node, q_index)


def bipartite_edges(node_set1: set[int], node_set2: set[int]) -> set[tuple[int, int]]:
    """Return a set of edges for the complete bipartite graph between two sets of nodes.

    Parameters
    ----------
    node_set1 : set[int]
        set of nodes
    node_set2 : set[int]
        set of nodes

    Returns
    -------
    set[tuple[int, int]]
        set of edges for the complete bipartite graph
    """
    return {(min(a, b), max(a, b)) for a, b in product(node_set1, node_set2) if a != b}
