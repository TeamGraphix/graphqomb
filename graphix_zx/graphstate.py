"""Graph State classes for Measurement-based Quantum Computing.

This module provides:
- BaseGraphState: Abstract base class for Graph State.
- GraphState: Minimal implementation of Graph State.
- ZXGraphState: Graph State for the ZX-calculus.
- bipartite_edges: Return a set of edges for the complete bipartite graph between two sets of nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import MeasBasis, Plane
from graphix_zx.euler import update_lc_basis

if TYPE_CHECKING:
    from typing import Callable

    from graphix_zx.euler import LocalClifford


class BaseGraphState(ABC):
    """Abstract base class for Graph State."""

    @abstractmethod
    def __init__(self) -> None:
        pass

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
    def num_physical_nodes(self) -> int:
        """Return the number of physical nodes.

        Returns
        -------
        int
            number of physical nodes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_physical_edges(self) -> int:
        """Return the number of physical edges.

        Returns
        -------
        int
            number of physical edges.
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
    def meas_planes(self) -> dict[int, Plane]:
        """Return measurement planes.

        Returns
        -------
        dict[int, Plane]
            measurement planes of each physical node.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def meas_angles(self) -> dict[int, float]:
        """Return measurement angles.

        Returns
        -------
        dict[int, float]
            measurement angles of each physical node.
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
    def set_meas_plane(self, node: int, plane: Plane) -> None:
        """Set the measurement plane of the node.

        Parameters
        ----------
        node : int
            node index
        plane : Plane
            measurement plane
        """
        raise NotImplementedError

    @abstractmethod
    def set_meas_angle(self, node: int, angle: float) -> None:
        """Set the measurement angle of the node.

        Parameters
        ----------
        node : int
            node index
        angle : float
            measurement angle
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
    meas_planes : dict[int, Plane]
        measurement planes
    meas_angles : dict[int, float]
        measurement angles
    q_indices : dict[int, int]
        qubit indices
    local_cliffords : dict[int, LocalClifford]
        local clifford operators
    """

    __input_nodes: set[int]
    __output_nodes: set[int]
    __physical_nodes: set[int]
    __physical_edges: dict[int, set[int]]
    __meas_planes: dict[int, Plane]
    __meas_angles: dict[int, float]
    __q_indices: dict[int, int]
    __local_cliffords: dict[int, LocalClifford]

    def __init__(self) -> None:
        self.__input_nodes = set()
        self.__output_nodes = set()
        self.__physical_nodes = set()
        self.__physical_edges = {}
        self.__meas_planes = {}
        self.__meas_angles = {}
        # NOTE: qubit index if allocated. -1 if not. used for simulation
        self.__q_indices = {}
        self.__local_cliffords = {}

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
    def meas_planes(self) -> dict[int, Plane]:
        """Return measurement planes.

        Returns
        -------
        dict[int, Plane]
            measurement planes of each physical node.
        """
        return self.__meas_planes

    @property
    def meas_angles(self) -> dict[int, float]:
        """Return measurement angles.

        Returns
        -------
        dict[int, float]
            measurement angles of each physical node.
        """
        return self.__meas_angles

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
            if self.meas_planes.get(v) is None or self.meas_angles.get(v) is None:
                msg = f"Measurement basis not set for node {v}"
                raise ValueError(msg)
            if self.meas_planes[v] not in {Plane.XY, Plane.XZ, Plane.YZ, Plane.YX, Plane.ZX, Plane.ZY}:
                msg = f"Invalid measurement plane '{self.meas_planes[v]}' for node {v}"
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
        self.__meas_planes.pop(node, None)
        self.__meas_angles.pop(node, None)
        self.__q_indices.pop(node, None)
        self.__local_cliffords.pop(node, None)
        for neighbor in self.__physical_edges:
            self.__physical_edges[neighbor] -= {node}

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
        """
        self.ensure_node_exists(node)
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

    def set_meas_plane(self, node: int, plane: Plane) -> None:
        """Set the measurement plane of the node.

        Parameters
        ----------
        node : int
            node index
        plane : Plane
            measurement plane
        """
        self.ensure_node_exists(node)
        self.__meas_planes[node] = plane

    def set_meas_angle(self, node: int, angle: float) -> None:
        """Set the measurement angle of the node.

        Parameters
        ----------
        node : int
            node index
        angle : float
            measurement angle
        """
        self.ensure_node_exists(node)
        self.__meas_angles[node] = angle

    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
        """Apply a local clifford to the node.

        Parameters
        ----------
        node : int
            node index
        lc : LocalClifford
            local clifford operator

        Raises
        ------
        ValueError
            If the node does not exist.
        """
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)
        if node in self.input_nodes or node in self.output_nodes:
            self.__local_cliffords[node] = lc
        else:
            meas_plane, meas_angle = _update_meas_basis(lc, self.meas_planes[node], self.meas_angles[node])
            self.set_meas_plane(node, meas_plane)
            self.set_meas_angle(node, meas_angle)

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

    def _reset_input_output(self, node: int) -> None:
        """Reset the input/output status of the node.

        Parameters
        ----------
        node : int
            node index
        """
        if node in self.__input_nodes:
            self.__input_nodes.remove(node)
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
                self._reset_input_output(node)
            else:
                self.add_physical_node(node)
                if node in other.input_nodes - self.output_nodes:
                    self.set_input(node)

            if node in other.output_nodes:
                self.set_output(node)
            else:
                self.set_meas_plane(node, other.meas_planes.get(node, Plane.XY))
                self.set_meas_angle(node, other.meas_angles.get(node, 0.0))

        for edge in other.physical_edges:
            self.add_physical_edge(edge[0], edge[1])

        # q_index update
        for node, q_index in other.q_indices.items():
            if (node in common_nodes) and (self.q_indices[node] != q_index):
                msg = "Qubit index mismatch."
                raise ValueError(msg)
            self.set_q_index(node, q_index)

    def is_clifford(self, node: int) -> bool:
        """Check if the node is a Clifford vertex.

        Parameters
        ----------
        node : int
            node index

        Returns
        -------
        bool
            True if the node is a Clifford vertex.
        """
        atol = 1e-15
        angle = self.meas_angles[node] % (2 * np.pi)
        is_valid_plane = self.meas_planes[node] in {Plane.XY, Plane.XZ, Plane.YZ}
        return abs(angle % (0.5 * np.pi)) < atol and is_valid_plane


def _update_meas_basis(
    lc: LocalClifford,
    plane: Plane,
    angle: float,
) -> tuple[Plane, float]:
    """Update the measurement basis of the node.

    Returns
    -------
        tuple[Plane, float]: updated measurement basis
    """
    meas_basis = MeasBasis(plane, angle)
    new_basis = update_lc_basis(lc, meas_basis)
    return new_basis.plane, new_basis.angle


class ZXGraphState(GraphState):
    """Graph State for the ZX-calculus.

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
    meas_planes : dict[int, Plane]
        measurement planes
    meas_angles : dict[int, float]
        measurement angles
    q_indices : dict[int, int]
        qubit indices
    local_cliffords : dict[int, LocalClifford]
        local clifford operators
    """

    def __init__(self) -> None:
        super().__init__()

    def _update_connections(self, rmv_edges: set[tuple[int, int]], new_edges: set[tuple[int, int]]) -> None:
        for edge in rmv_edges:
            self.remove_physical_edge(edge[0], edge[1])
        for edge in new_edges:
            self.add_physical_edge(edge[0], edge[1])

    def _update_node_measurement(
        self, measurement_action: dict[Plane, tuple[Plane, Callable[[float], float]]], v: int
    ) -> None:
        new_plane, new_angle_func = measurement_action[self.meas_planes[v]]
        if new_plane:
            self.set_meas_plane(v, new_plane)
            self.set_meas_angle(v, new_angle_func(v) % (2.0 * np.pi))

    def local_complement(self, node: int) -> None:
        """Local complement operation on the graph state: G*u.

        Parameters
        ----------
        node : int
            node index

        Raises
        ------
        ValueError
            If the node is an input node, an output node, or the graph is not a ZX-diagram.
        """
        self.ensure_node_exists(node)
        if node in self.input_nodes or node in self.output_nodes:
            msg = "Cannot apply local complement to input node nor output node."
            raise ValueError(msg)
        self.check_meas_basis()

        nbrs: set[int] = self.get_neighbors(node)
        nbr_pairs = bipartite_edges(nbrs, nbrs)
        new_edges = nbr_pairs - self.physical_edges
        rmv_edges = self.physical_edges & nbr_pairs

        self._update_connections(rmv_edges, new_edges)

        # update node measurement
        measurement_action = {
            Plane.XY: (Plane.XZ, lambda v: (0.5 * np.pi - self.meas_angles[v]) % (2.0 * np.pi)),
            Plane.XZ: (Plane.XY, lambda v: (self.meas_angles[v] - 0.5 * np.pi) % (2.0 * np.pi)),
            Plane.YZ: (Plane.YZ, lambda v: (self.meas_angles[v] + 0.5 * np.pi) % (2.0 * np.pi)),
        }

        self._update_node_measurement(measurement_action, node)

        # update neighbors measurement
        measurement_action = {
            Plane.XY: (Plane.XY, lambda v: (self.meas_angles[v] - 0.5 * np.pi) % (2.0 * np.pi)),
            Plane.XZ: (Plane.YZ, lambda v: (self.meas_angles[v]) % (2.0 * np.pi)),
            Plane.YZ: (Plane.XZ, lambda v: (-self.meas_angles[v]) % (2.0 * np.pi)),
        }

        for v in nbrs:
            self._update_node_measurement(measurement_action, v)

    def _swap(self, node1: int, node2: int) -> None:
        """Swap nodes u and v in the graph state.

        Parameters
        ----------
        node1  : int
            node index
        node2 : int
            node index
        """
        node1_nbrs = self.get_neighbors(node1) - {node2}
        node2_nbrs = self.get_neighbors(node2) - {node1}
        nbr_b = node1_nbrs - node2_nbrs
        nbr_c = node2_nbrs - node1_nbrs
        for b in nbr_b:
            self.remove_physical_edge(node1, b)
            self.add_physical_edge(node2, b)
        for c in nbr_c:
            self.remove_physical_edge(node2, c)
            self.add_physical_edge(node1, c)

    def pivot(self, node1: int, node2: int) -> None:
        """Pivot operation on the graph state: G∧(uv) (= G*u*v*u = G*v*u*v) for neighboring nodes u and v.

        In order to maintain the ZX-diagram simple, pi-spiders are shifted properly.

        Parameters
        ----------
        node1 : int
            node index
        node2 : int
            node index

        Raises
        ------
        ValueError
            If the nodes are input nodes, output nodes, or the graph is not a ZX-diagram.
        """
        self.ensure_node_exists(node1)
        self.ensure_node_exists(node2)
        if node1 in self.input_nodes or node2 in self.input_nodes:
            msg = "Cannot apply pivot to input node"
            raise ValueError(msg)
        if node1 in self.output_nodes or node2 in self.output_nodes:
            msg = "Cannot apply pivot to output node"
            raise ValueError(msg)
        self.check_meas_basis()

        node1_nbrs = self.get_neighbors(node1) - {node2}
        node2_nbrs = self.get_neighbors(node2) - {node1}
        nbr_a = node1_nbrs & node2_nbrs
        nbr_b = node1_nbrs - node2_nbrs
        nbr_c = node2_nbrs - node1_nbrs
        nbr_pairs = [
            bipartite_edges(nbr_a, nbr_b),
            bipartite_edges(nbr_a, nbr_c),
            bipartite_edges(nbr_b, nbr_c),
        ]
        rmv_edges = set().union(*(p & self.physical_edges for p in nbr_pairs))
        add_edges = set().union(*(p - self.physical_edges for p in nbr_pairs))

        self._update_connections(rmv_edges, add_edges)
        self._swap(node1, node2)

        # update node1 and node2 measurement
        measurement_action = {
            Plane.XY: (Plane.YZ, lambda v: self.meas_angles[v]),
            Plane.XZ: (Plane.XZ, lambda v: (0.5 * np.pi - self.meas_angles[v]) % (2.0 * np.pi)),
            Plane.YZ: (Plane.XY, lambda v: self.meas_angles[v]),
        }

        for a in [node1, node2]:
            self._update_node_measurement(measurement_action, a)

        # update nodes measurement of nbr_a
        measurement_action = {
            Plane.XY: (Plane.XY, lambda v: (self.meas_angles[v] + np.pi) % (2.0 * np.pi)),
            Plane.XZ: (Plane.YZ, lambda v: -self.meas_angles[v] % (2.0 * np.pi)),
            Plane.YZ: (Plane.XZ, lambda v: -self.meas_angles[v] % (2.0 * np.pi)),
        }

        for w in nbr_a:
            self._update_node_measurement(measurement_action, w)

    def _needs_nop(self, node: int) -> bool:
        """Check if the node needs no operation in order to perform _remove_clifford.

        Parameters
        ----------
        node : int
            node index

        Returns
        -------
        bool
            True if the node needs no operation
        """
        atol = 1e-15
        alpha = self.meas_angles[node] % (2 * np.pi)
        return abs(alpha % np.pi) < atol and (self.meas_planes[node] in {Plane.YZ, Plane.XZ})

    def _needs_lc(self, node: int) -> bool:
        """Check if the node needs a local complementation in order to perform _remove_clifford.

        Parameters
        ----------
        node : int
            node index

        Returns
        -------
        bool
            True if the node needs a local complementation.
        """
        atol = 1e-15
        alpha = self.meas_angles[node] % (2 * np.pi)
        return abs((alpha + 0.5 * np.pi) % np.pi) < atol and self.meas_planes[node] in {Plane.YZ, Plane.XY}

    def _needs_pivot_1(self, node: int) -> bool:
        """Check if the nodes need a pivot operation in order to perform _remove_clifford.

        Parameters
        ----------
        node : int
            node index

        Returns
        -------
        bool
            True if the nodes need a pivot operation.
        """
        non_input_nbrs = self.get_neighbors(node) - self.input_nodes
        if len(non_input_nbrs) == 0:
            return False

        atol = 1e-15
        alpha = self.meas_angles[node] % (2 * np.pi)
        case_a = abs(alpha % np.pi) < atol and self.meas_planes[node] == Plane.XY
        case_b = abs((alpha + 0.5 * np.pi) % np.pi) < atol and self.meas_planes[node] == Plane.XZ
        return case_a or case_b

    def _needs_pivot_2(self, node: int) -> bool:
        """Check if the node needs a pivot operation on output nodes in order to perform _remove_clifford.

        Parameters
        ----------
        node : int
            node index

        Returns
        -------
        bool
            True if the node needs a pivot operation on output nodes.
        """
        nbrs = self.get_neighbors(node) - self.input_nodes
        if len(nbrs) == 0:
            return False

        atol = 1e-15
        alpha = self.meas_angles[node] % (2 * np.pi)
        case_a = abs(alpha % np.pi) < atol and self.meas_planes[node] == Plane.XY
        case_b = abs((alpha + 0.5 * np.pi) % np.pi) < atol and self.meas_planes[node] == Plane.XZ
        return case_a or case_b

    def _remove_clifford(self, node: int) -> None:
        """Perform the Clifford vertex removal.

        Parameters
        ----------
        node : int
            node index
        """
        atol = 1e-15
        alpha = self.meas_angles[node] % (2 * np.pi)
        measurement_action = {
            Plane.XY: (Plane.XY, lambda v: (alpha + self.meas_angles[v]) % (2.0 * np.pi)),
            Plane.XZ: (
                Plane.YZ,
                lambda v: -self.meas_angles[v] % (2 * np.pi) if abs(alpha % np.pi) < atol else self.meas_angles[v],
            ),
            Plane.YZ: (
                Plane.XZ,
                lambda v: self.meas_angles[v] if abs(alpha % np.pi) < atol else -self.meas_angles[v] % (2 * np.pi),
            ),
        }
        for v in self.get_neighbors(node):
            self._update_node_measurement(measurement_action, v)

        self.remove_physical_node(node)

    def remove_clifford(self, node: int) -> None:
        """Remove the local clifford node.

        Raises
        ------
        ValueError
            1. If the node is an input node.
            2. If the node is not a Clifford vertex.
            3. If all neighbors are input nodes
                in some special cases ((meas_plane, meas_angle) = (XY, a pi), (XZ, a pi/2) for a = 0, 1).
            4. If the node has no neighbors that are not connected only to output nodes.
        """
        self.ensure_node_exists(node)
        if node in self.input_nodes:
            msg = "Clifford vertex removal not allowed for input node"
            raise ValueError(msg)

        if not self.is_clifford(node):
            msg = "This node is not a Clifford vertex."
            raise ValueError(msg)

        if self._needs_nop(node):
            pass
        elif self._needs_lc(node):
            self.local_complement(node)
        elif self._needs_pivot_1(node):
            nbrs = self.get_neighbors(node) - self.input_nodes
            v = nbrs.pop()
            self.pivot(node, v)
        elif self._needs_pivot_2(node):
            nbrs = self.get_neighbors(node) - self.input_nodes
            v = nbrs.pop()
            self.pivot(node, v)
        else:
            msg = "Invalid case for Clifford vertex removal."
            raise ValueError(msg)

        self._remove_clifford(node)


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
