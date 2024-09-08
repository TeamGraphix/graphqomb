from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product

from graphix_zx.common import Plane


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

    def __init__(self) -> None:
        self.__input_nodes = set()
        self.__output_nodes = set()
        self.__physical_nodes = set()
        self.__physical_edges = {}
        self.__meas_planes = {}
        self.__meas_angles = {}
        # NOTE: qubit index if allocated. -1 if not. used for simulation
        self.__q_indices = {}

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

    def get_neighbors(self, node: int) -> set[int]:
        self.ensure_node_exists(node)
        return self.__physical_edges[node]

    def _reset_input_output(self, node: int) -> None:
        if node in self.__input_nodes:
            self.__input_nodes.remove(node)
        if node in self.__output_nodes:
            self.__output_nodes.remove(node)

    # TODO: overload with pattern
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
            if self.meas_planes[v] not in [Plane.XY, Plane.XZ, Plane.YZ, Plane.YX, Plane.ZX, Plane.ZY]:
                return False
        return True

    def local_complement(self, node: int) -> None:
        self.ensure_node_exists(node)
        if node in self.input_nodes:
            msg = "Cannot apply local complement to input node"
            raise ValueError(msg)
        if not self.is_zx_graph():
            msg = "The graph is not a ZX-diagram. Set measurement planes and angles properly."
            raise ValueError(msg)

        adjacent_nodes: set[int] = self.adjacent_nodes(node)
        adjacent_pairs = {(a, b) for a, b in product(adjacent_nodes, repeat=2) if a < b}
        new_edges = adjacent_pairs - self.physical_edges
        remove_target_edges = self.physical_edges.intersection(adjacent_pairs)

        for edge in new_edges:
            self.add_physical_edge(edge[0], edge[1])
        for edge in remove_target_edges:
            self.remove_physical_edge(edge[0], edge[1])

        # update node measurement
        measurement_action = {
            Plane.XY: (Plane.XZ, 0.5 - self.meas_angles[node]),
            Plane.XZ: (Plane.XY, self.meas_angles[node] - 0.5),
            Plane.YZ: (Plane.YZ, self.meas_angles[node] + 0.5),
        }
        new_plane, new_angle = measurement_action[self.meas_planes[node]]
        if new_plane:
            self.set_meas_plane(node, new_plane)
            self.set_meas_angle(node, new_angle % 2.0)

        # update adjacent nodes measurement
        measurement_action = {
            Plane.XY: (Plane.XY, lambda v: self.meas_angles[v] - 0.5),
            Plane.XZ: (Plane.YZ, lambda v: self.meas_angles[v]),
            Plane.YZ: (Plane.XZ, lambda v: -self.meas_angles[v]),
        }
        for v in adjacent_nodes:
            new_plane, new_angle_func = measurement_action[self.meas_planes[v]]
            if new_plane:
                self.set_meas_plane(v, new_plane)
                self.set_meas_angle(v, new_angle_func(v) % 2.0)

    def pivot(self, node1: int, node2: int) -> None:
        self.local_complement(node1)
        self.local_complement(node2)
        self.local_complement(node1)
