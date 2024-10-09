"""Graph State for the ZX-calculus"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import Plane

if TYPE_CHECKING:
    from typing import Callable

    from graphix_zx.euler import LocalClifford


def neighboring_pairs(nbr_nodes_a: set[int], nbr_nodes_b: set[int]) -> set[tuple[int, int]]:
    """Return all pairs of neighboring nodes between two sets of nodes.

    Args:
        nbr_nodes_a (set[int]): set of nodes
        nbr_nodes_b (set[int]): set of nodes

    Returns:
        set[tuple[int, int]]: set of neighboring pairs
    """
    return {(min(a, b), max(a, b)) for a, b in product(nbr_nodes_a, nbr_nodes_b) if a != b}


class BaseGraphState(ABC):
    """Abstract base class for Graph State"""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def input_nodes(self) -> set[int]:
        """Return set of input nodes"""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_nodes(self) -> set[int]:
        """Return set of output nodes"""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_physical_nodes(self) -> int:
        """Return the number of physical nodes"""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_physical_edges(self) -> int:
        """Return the number of physical edges"""
        raise NotImplementedError

    @property
    @abstractmethod
    def physical_nodes(self) -> set[int]:
        """Return set of physical nodes"""
        raise NotImplementedError

    @property
    @abstractmethod
    def physical_edges(self) -> set[tuple[int, int]]:
        """Return set of physical edges"""
        raise NotImplementedError

    @property
    @abstractmethod
    # Generics?
    def q_indices(self) -> dict[int, int]:
        """Return qubit indices"""
        raise NotImplementedError

    @property
    @abstractmethod
    def meas_planes(self) -> dict[int, Plane]:
        """Return measurement planes"""
        raise NotImplementedError

    @property
    @abstractmethod
    def meas_angles(self) -> dict[int, float]:
        """Return measurement angles"""
        raise NotImplementedError

    @property
    @abstractmethod
    def local_cliffords(self) -> dict[int, LocalClifford]:
        """Return local clifford nodes"""
        raise NotImplementedError

    @abstractmethod
    def add_physical_node(
        self,
        node: int,
        q_index: int,
        *,
        is_input: bool,
        is_output: bool,
    ) -> None:
        """Add a physical node to the graph state"""
        raise NotImplementedError

    @abstractmethod
    def add_physical_edge(self, node1: int, node2: int) -> None:
        """Add a physical edge to the graph state"""
        raise NotImplementedError

    @abstractmethod
    def set_input(self, node: int) -> None:
        """Set the node as an input node"""
        raise NotImplementedError

    @abstractmethod
    def set_output(self, node: int) -> None:
        """Set the node as an output node"""
        raise NotImplementedError

    @abstractmethod
    def set_meas_plane(self, node: int, plane: Plane) -> None:
        """Set the measurement plane of the node"""
        raise NotImplementedError

    @abstractmethod
    def set_meas_angle(self, node: int, angle: float) -> None:
        """Set the measurement angle of the node"""
        raise NotImplementedError

    # NOTE: on internal nodes -> update measurement basis, on input or output -> set local clifford object
    @abstractmethod
    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
        """Apply a local clifford to the node"""
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self, node: int) -> set[int]:
        """Return the neighbors of the node"""
        raise NotImplementedError


class GraphState(BaseGraphState):
    """Minimal implementation of GraphState"""

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
        """Return set of input nodes"""
        return self.__input_nodes

    @property
    def output_nodes(self) -> set[int]:
        """Return set of output nodes"""
        return self.__output_nodes

    @property
    def num_physical_nodes(self) -> int:
        """Return the number of physical nodes"""
        return len(self.__physical_nodes)

    @property
    def num_physical_edges(self) -> int:
        """Return the number of physical edges"""
        return sum(len(edges) for edges in self.__physical_edges.values()) // 2

    @property
    def physical_nodes(self) -> set[int]:
        """Return set of physical nodes"""
        return self.__physical_nodes

    @property
    def physical_edges(self) -> set[tuple[int, int]]:
        """Return set of physical edges"""
        edges = set()
        for node1 in self.__physical_edges:
            for node2 in self.__physical_edges[node1]:
                if node1 < node2:
                    edges |= {(node1, node2)}
        return edges

    @property
    def q_indices(self) -> dict[int, int]:
        """Return qubit indices"""
        return self.__q_indices

    @property
    def meas_planes(self) -> dict[int, Plane]:
        """Return measurement planes"""
        return self.__meas_planes

    @property
    def meas_angles(self) -> dict[int, float]:
        """Return measurement angles"""
        return self.__meas_angles

    @property
    def local_cliffords(self) -> dict[int, LocalClifford]:
        """Return local clifford nodes"""
        return self.__local_cliffords

    def check_meas_basis(self) -> bool:
        """Check if the measurement basis is set for all physical nodes except output nodes

        Raises:
            ValueError: If the measurement basis is not set for a node or the measurement plane is invalid.
        """
        for v in self.physical_nodes - self.output_nodes:
            if self.meas_planes.get(v) is None or self.meas_angles.get(v) is None:
                raise ValueError(f"Measurement basis not set for node {v}")
            if self.meas_planes[v] not in {Plane.XY, Plane.XZ, Plane.YZ, Plane.YX, Plane.ZX, Plane.ZY}:
                raise ValueError(f"Invalid measurement plane '{self.meas_planes[v]}' for node {v}")

    def add_physical_node(
        self,
        node: int,
        q_index: int = -1,
        *,
        is_input: bool = False,
        is_output: bool = False,
    ) -> None:
        """Add a physical node to the graph state

        Args:
            node (int): node index
            q_index (int, optional): qubit index. Defaults to -1.
            is_input (bool, optional): input node. Defaults to False.
            is_output (bool, optional): output node. Defaults to False.

        Raises:
            ValueError: If the node already exists in the graph state.
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
        """Ensure that the node exists in the graph state

        Raises:
            ValueError: If the node does not exist in the graph state.
        """
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)

    def add_physical_edge(self, node1: int, node2: int) -> None:
        """Add a physical edge to the graph state

        Args:
            node1 (int): node index
            node2 (int): node index

        Raises:
            ValueError: If the edge already exists.
        """
        self.ensure_node_exists(node1)
        self.ensure_node_exists(node2)
        if node1 in self.__physical_edges[node2] or node2 in self.__physical_edges[node1]:
            msg = f"Edge already exists {node1=}, {node2=}"
            raise ValueError(msg)
        self.__physical_edges[node1] |= {node2}
        self.__physical_edges[node2] |= {node1}

    def remove_physical_edge(self, node1: int, node2: int) -> None:
        """Remove a physical edge from the graph state

        Args:
            node1 (int): node index
            node2 (int): node index

        Raises:
            ValueError: If the edge does not exist.
        """
        self.ensure_node_exists(node1)
        self.ensure_node_exists(node2)
        if node1 not in self.__physical_edges[node2] or node2 not in self.__physical_edges[node1]:
            msg = "Edge does not exist"
            raise ValueError(msg)
        self.__physical_edges[node1] -= {node2}
        self.__physical_edges[node2] -= {node1}

    def set_input(self, node: int) -> None:
        """Set the node as an input node

        Args:
            node (int): node index
        """
        self.ensure_node_exists(node)
        self.__input_nodes |= {node}

    def set_output(self, node: int) -> None:
        """Set the node as an output node

        Args:
            node (int): node index
        """
        self.ensure_node_exists(node)
        self.__output_nodes |= {node}

    def set_q_index(self, node: int, q_index: int = -1) -> None:
        """Set the qubit index of the node

        Args:
            node (int): node index
            q_index (int, optional): qubit index. Defaults to -1.

        Raises:
            ValueError: If the qubit index is invalid.
        """
        self.ensure_node_exists(node)
        if q_index < -1:
            msg = f"Invalid qubit index {q_index}. Must be -1 or greater"
            raise ValueError(msg)
        self.__q_indices[node] = q_index

    def set_meas_plane(self, node: int, plane: Plane) -> None:
        """Set the measurement plane of the node

        Args:
            node (int): node index
            plane (Plane): measurement plane
        """
        self.ensure_node_exists(node)
        self.__meas_planes[node] = plane

    def set_meas_angle(self, node: int, angle: float) -> None:
        """Set the measurement angle of the node

        Args:
            node (int): node index
            angle (float): measurement angle
        """
        self.ensure_node_exists(node)
        self.__meas_angles[node] = angle

    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
        """Apply a local clifford to the node

        Args:
            node (int): node index
            lc (LocalClifford): local clifford node

        Raises:
            ValueError: If the node does not exist.
        """
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)
        if node in self.input_nodes or node in self.output_nodes:
            self.__local_cliffords[node] = lc
        else:
            meas_plane, meas_angle = update_meas_basis(lc, self.meas_planes[node], self.meas_angles[node])
            self.set_meas_plane(node, meas_plane)
            self.set_meas_angle(node, meas_angle)

    def get_neighbors(self, node: int) -> set[int]:
        """Return the neighbors of the node

        Args:
            node (int): node index

        Returns:
            set[int]: set of neighboring nodes
        """
        self.ensure_node_exists(node)
        return self.__physical_edges[node]

    def _reset_input_output(self, node: int) -> None:
        """Reset the input/output status of the node

        Args:
            node (int): node index
        """
        if node in self.__input_nodes:
            self.__input_nodes.remove(node)
        if node in self.__output_nodes:
            self.__output_nodes.remove(node)

    def append(self, other: BaseGraphState) -> None:
        """Append another graph state to the current graph state

        Args:
            other (BaseGraphState): another graph state

        Raises:
            ValueError: If the qubit indices do not match.
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


class ZXGraphState(GraphState):
    """Graph State for the ZX-calculus"""

    def __init__(self) -> None:
        super().__init__()

    def _update_connections(self, rmv_edges: set[tuple[int, int]], new_edges: set[tuple[int, int]]) -> None:
        for edge in rmv_edges:
            self.remove_physical_edge(edge[0], edge[1])
        for edge in new_edges:
            self.add_physical_edge(edge[0], edge[1])

    def _update_node_measurement(
        self, measurement_action: dict[Plane, tuple[Plane, float | Callable[[float], float]]], v: int
    ) -> None:
        new_plane, new_angle_func = measurement_action[self.meas_planes[v]]
        if new_plane:
            self.set_meas_plane(v, new_plane)
            self.set_meas_angle(v, new_angle_func(v) % (2.0 * np.pi))

    def local_complement(self, node: int) -> None:
        """Local complement operation on the graph state: G*u

        Args:
            node (int): node index

        Raises:
            ValueError: If the node is an input node, an output node, or the graph is not a ZX-diagram.
        """
        self.ensure_node_exists(node)
        if node in self.input_nodes or node in self.output_nodes:
            msg = "Cannot apply local complement to input node nor output node."
            raise ValueError(msg)
        self.check_meas_basis()

        nbrs: set[int] = self.get_neighbors(node)
        nbr_pairs = neighboring_pairs(nbrs, nbrs)
        new_edges = nbr_pairs - self.physical_edges
        rmv_edges = self.physical_edges & nbr_pairs

        self._update_connections(rmv_edges, new_edges)

        # update node measurement
        measurement_action = {
            Plane.XY: (Plane.XZ, lambda v: 0.5 * np.pi - self.meas_angles[v]),
            Plane.XZ: (Plane.XY, lambda v: self.meas_angles[v] - 0.5 * np.pi),
            Plane.YZ: (Plane.YZ, lambda v: self.meas_angles[v] + 0.5 * np.pi),
        }

        self._update_node_measurement(measurement_action, node)

        # update neighbors measurement
        measurement_action = {
            Plane.XY: (Plane.XY, lambda v: self.meas_angles[v] - 0.5 * np.pi),
            Plane.XZ: (Plane.YZ, lambda v: self.meas_angles[v]),
            Plane.YZ: (Plane.XZ, lambda v: -self.meas_angles[v]),
        }

        for v in nbrs:
            self._update_node_measurement(measurement_action, v)

    def _swap(self, node1: int, node2: int) -> None:
        """Swap nodes u and v in the graph state

        Args:
            node1 (int): node index
            node2 (int): node index
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

        Args:
            node1 (int): node index
            node2 (int): node index

        Raises:
            ValueError: If the nodes are input nodes, output nodes, or the graph is not a ZX-diagram.
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
            neighboring_pairs(nbr_a, nbr_b),
            neighboring_pairs(nbr_a, nbr_c),
            neighboring_pairs(nbr_b, nbr_c),
        ]
        rmv_edges = set().union(*(p & self.physical_edges for p in nbr_pairs))
        add_edges = set().union(*(p - self.physical_edges for p in nbr_pairs))

        self._update_connections(rmv_edges, add_edges)
        self._swap(node1, node2)

        # update node1 and node2 measurement
        measurement_action = {
            Plane.XY: (Plane.YZ, lambda v: self.meas_angles[v]),
            Plane.XZ: (Plane.XZ, lambda v: (0.5 * np.pi - self.meas_angles[v])),
            Plane.YZ: (Plane.XY, lambda v: self.meas_angles[v]),
        }

        for a in [node1, node2]:
            self._update_node_measurement(measurement_action, a)

        # update nodes measurement of nbr_a
        measurement_action = {
            Plane.XY: (Plane.XY, lambda v: (self.meas_angles[v] + np.pi)),
            Plane.XZ: (Plane.YZ, lambda v: -self.meas_angles[v]),
            Plane.YZ: (Plane.XZ, lambda v: -self.meas_angles[v]),
        }

        for w in nbr_a:
            self._update_node_measurement(measurement_action, w)


def update_meas_basis(
    lc: LocalClifford,
    plane: Plane,
    angle: float,
) -> tuple[Plane, float]:
    """Update the measurement basis of the node

    Args:
        lc (LocalClifford): local clifford node
        plane (Plane): measurement plane
        angle (float): measurement angle
    """
    raise NotImplementedError
