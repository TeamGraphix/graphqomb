from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import Plane

if TYPE_CHECKING:
    from typing import Callable

    from graphix_zx.euler import LocalClifford


def meas_action(
    base_action: dict[Plane, tuple[Plane, float | Callable[[float], float]]],
) -> dict[Plane, tuple[Plane, float | Callable[[float], float]]]:
    action = base_action.copy()
    action[Plane.YX] = action.get(Plane.XY, (None, None))
    action[Plane.ZX] = action.get(Plane.XZ, (None, None))
    action[Plane.ZY] = action.get(Plane.YZ, (None, None))
    return action


def adj_pairs(adj_nodes_a: set[int], adj_nodes_b: set[int]) -> set[tuple[int, int]]:
    return {(min(a, b), max(a, b)) for a, b in product(adj_nodes_a, adj_nodes_b) if a != b}


class BaseGraphState(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def input_nodes(self) -> set[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_nodes(self) -> set[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_physical_nodes(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_physical_edges(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def physical_nodes(self) -> set[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def physical_edges(self) -> set[tuple[int, int]]:
        raise NotImplementedError

    @property
    @abstractmethod
    # Generics?
    def q_indices(self) -> dict[int, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def meas_planes(self) -> dict[int, Plane]:
        raise NotImplementedError

    @property
    @abstractmethod
    def meas_angles(self) -> dict[int, float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def local_cliffords(self) -> dict[int, LocalClifford]:
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
        raise NotImplementedError

    @abstractmethod
    def add_physical_edge(self, node1: int, node2: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_input(self, node: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_output(self, node: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_meas_plane(self, node: int, plane: Plane) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_meas_angle(self, node: int, angle: float) -> None:
        raise NotImplementedError

    # NOTE: on internal nodes -> update measurement basis, on input or output -> set local clifford object
    @abstractmethod
    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self, node: int) -> set[int]:
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
        return self.__input_nodes

    @property
    def output_nodes(self) -> set[int]:
        return self.__output_nodes

    @property
    def num_physical_nodes(self) -> int:
        return len(self.__physical_nodes)

    @property
    def num_physical_edges(self) -> int:
        return sum(len(edges) for edges in self.__physical_edges.values()) // 2

    @property
    def physical_nodes(self) -> set[int]:
        return self.__physical_nodes

    @property
    def physical_edges(self) -> set[tuple[int, int]]:
        edges = set()
        for node1 in self.__physical_edges:
            for node2 in self.__physical_edges[node1]:
                if node1 < node2:
                    edges |= {(node1, node2)}
        return edges

    @property
    def q_indices(self) -> dict[int, int]:
        return self.__q_indices

    @property
    def meas_planes(self) -> dict[int, Plane]:
        return self.__meas_planes

    @property
    def meas_angles(self) -> dict[int, float]:
        return self.__meas_angles

    @property
    def local_cliffords(self) -> dict[int, LocalClifford]:
        return self.__local_cliffords

    def add_physical_node(
        self,
        node: int,
        q_index: int = -1,
        *,
        is_input: bool = False,
        is_output: bool = False,
    ) -> None:
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

    def adjacent_nodes(self, node: int) -> set[int]:
        return self.__physical_edges[node]

    def ensure_node_exists(self, node: int) -> None:
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)

    def add_physical_edge(self, node1: int, node2: int) -> None:
        self.ensure_node_exists(node1)
        self.ensure_node_exists(node2)
        if node1 in self.__physical_edges[node2] or node2 in self.__physical_edges[node1]:
            msg = f"Edge already exists {node1=}, {node2=}"
            raise ValueError(msg)
        self.__physical_edges[node1] |= {node2}
        self.__physical_edges[node2] |= {node1}

    def remove_physical_edge(self, node1: int, node2: int) -> None:
        self.ensure_node_exists(node1)
        self.ensure_node_exists(node2)
        if node1 not in self.__physical_edges[node2] or node2 not in self.__physical_edges[node1]:
            msg = "Edge does not exist"
            raise ValueError(msg)
        self.__physical_edges[node1] -= {node2}
        self.__physical_edges[node2] -= {node1}

    def set_input(self, node: int) -> None:
        self.ensure_node_exists(node)
        self.__input_nodes |= {node}

    def set_output(self, node: int) -> None:
        self.ensure_node_exists(node)
        self.__output_nodes |= {node}

    def set_q_index(self, node: int, q_index: int = -1) -> None:
        if q_index < -1:
            msg = f"Invalid qubit index {q_index}. Must be -1 or greater"
            raise ValueError(msg)
        self.__q_indices[node] = q_index

    def set_meas_plane(self, node: int, plane: Plane) -> None:
        self.ensure_node_exists(node)
        self.__meas_planes[node] = plane

    def set_meas_angle(self, node: int, angle: float) -> None:
        self.ensure_node_exists(node)
        self.__meas_angles[node] = angle

    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
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
        self.ensure_node_exists(node)
        return self.__physical_edges[node]

    def _reset_input_output(self, node: int) -> None:
        if node in self.__input_nodes:
            self.__input_nodes.remove(node)
        if node in self.__output_nodes:
            self.__output_nodes.remove(node)

    def append(self, other: BaseGraphState) -> None:
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

    def is_zx_graph(self) -> bool:
        for v in self.physical_nodes - self.output_nodes:
            if self.meas_planes.get(v) is None or self.meas_angles.get(v) is None:
                return False
            if self.meas_planes[v] not in {Plane.XY, Plane.XZ, Plane.YZ, Plane.YX, Plane.ZX, Plane.ZY}:
                return False
        return True

    def _local_complement(self, rmv_edges: set[tuple[int, int]], new_edges: set[tuple[int, int]]) -> None:
        for edge in rmv_edges:
            self.remove_physical_edge(edge[0], edge[1])
        for edge in new_edges:
            self.add_physical_edge(edge[0], edge[1])

    def local_complement(self, node: int) -> None:
        """Local complement operation on the graph state: G*u"""
        self.ensure_node_exists(node)
        if node in self.input_nodes:
            msg = "Cannot apply local complement to input node"
            raise ValueError(msg)
        if not self.is_zx_graph():
            msg = "The graph is not a ZX-diagram. Set measurement planes and angles properly."
            raise ValueError(msg)

        adjacent_nodes: set[int] = self.adjacent_nodes(node)
        adjacent_pairs = adj_pairs(adjacent_nodes, adjacent_nodes)
        new_edges = adjacent_pairs - self.physical_edges
        rmv_edges = self.physical_edges & adjacent_pairs

        self._local_complement(rmv_edges, new_edges)

        # update node measurement
        measurement_action = meas_action(
            {
                Plane.XY: (Plane.XZ, 0.5 * np.pi - self.meas_angles[node]),
                Plane.XZ: (Plane.XY, self.meas_angles[node] - 0.5 * np.pi),
                Plane.YZ: (Plane.YZ, self.meas_angles[node] + 0.5 * np.pi),
            }
        )
        new_plane, new_angle = measurement_action[self.meas_planes[node]]
        if new_plane:
            self.set_meas_plane(node, new_plane)
            self.set_meas_angle(node, new_angle % (2.0 * np.pi))

        # update adjacent nodes measurement
        measurement_action = meas_action(
            {
                Plane.XY: (Plane.XY, lambda v: self.meas_angles[v] - 0.5 * np.pi),
                Plane.XZ: (Plane.YZ, lambda v: self.meas_angles[v]),
                Plane.YZ: (Plane.XZ, lambda v: -self.meas_angles[v]),
            }
        )
        for v in adjacent_nodes:
            new_plane, new_angle_func = measurement_action[self.meas_planes[v]]
            if new_plane:
                self.set_meas_plane(v, new_plane)
                self.set_meas_angle(v, new_angle_func(v) % (2.0 * np.pi))

    def _swap(self, node1: int, node2: int) -> None:
        node1_adjs = self.adjacent_nodes(node1) - {node2}
        node2_adjs = self.adjacent_nodes(node2) - {node1}
        adj_b = node1_adjs - node2_adjs
        adj_c = node2_adjs - node1_adjs
        for b in adj_b:
            self.remove_physical_edge(node1, b)
            self.add_physical_edge(node2, b)
        for c in adj_c:
            self.remove_physical_edge(node2, c)
            self.add_physical_edge(node1, c)

    def pivot(self, node1: int, node2: int) -> None:
        """Pivot operation on the graph state: G∧(uv) (= G*u*v*u = G*v*u*v) for adjacent nodes u and v.

        In order to maintain the ZX-diagram simple, pi-spiders are shifted properly.
        """
        self.ensure_node_exists(node1)
        self.ensure_node_exists(node2)
        if node1 in self.input_nodes or node2 in self.input_nodes:
            msg = "Cannot apply pivot to input node"
            raise ValueError(msg)

        node1_adjs = self.adjacent_nodes(node1) - {node2}
        node2_adjs = self.adjacent_nodes(node2) - {node1}
        adj_a = node1_adjs & node2_adjs
        adj_b = node1_adjs - node2_adjs
        adj_c = node2_adjs - node1_adjs
        adjacent_pairs = [
            adj_pairs(adj_a, adj_b),
            adj_pairs(adj_a, adj_c),
            adj_pairs(adj_b, adj_c),
        ]
        rmv_edges = set().union(*(p & self.physical_edges for p in adjacent_pairs))
        add_edges = set().union(*(p - self.physical_edges for p in adjacent_pairs))

        self._local_complement(rmv_edges, add_edges)
        self._swap(node1, node2)

        # update node1 and node2 measurement
        measurement_action = meas_action(
            {
                Plane.XY: (Plane.YZ, lambda v: self.meas_angles[v]),
                Plane.XZ: (Plane.XZ, lambda v: (0.5 * np.pi - self.meas_angles[v])),
                Plane.YZ: (Plane.XY, lambda v: self.meas_angles[v]),
            }
        )
        for a in [node1, node2]:
            new_plane, new_angle_func = measurement_action[self.meas_planes[a]]
            if new_plane:
                self.set_meas_plane(a, new_plane)
                self.set_meas_angle(a, new_angle_func(a) % (2.0 * np.pi))

        # update nodes measurement of adj_a
        measurement_action = meas_action(
            {
                Plane.XY: (Plane.XY, lambda v: (self.meas_angles[v] + np.pi)),
                Plane.XZ: (Plane.YZ, lambda v: -self.meas_angles[v]),
                Plane.YZ: (Plane.XZ, lambda v: -self.meas_angles[v]),
            }
        )

        def _update_node_measurement(v: int) -> None:
            new_plane, new_angle_func = measurement_action[self.meas_planes[v]]
            if new_plane:
                self.set_meas_plane(v, new_plane)
                self.set_meas_angle(v, new_angle_func(v) % (2.0 * np.pi))

        for w in adj_a:
            _update_node_measurement(w)


def update_meas_basis(
    lc: LocalClifford,
    plane: Plane,
    angle: float,
) -> tuple[Plane, float]:
    raise NotImplementedError
