from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from graphix_zx.common import Plane


class BaseGraphState(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    # NOTE: input and output nodes are necessary because graph is open graph
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

    @abstractmethod
    def add_physical_node(
        self,
        node: int,
        q_index: int,
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
    def get_physical_nodes(self) -> set[int]:
        raise NotImplementedError

    @abstractmethod
    def get_physical_edges(self) -> set[tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def get_q_indices(self) -> dict[int, int]:
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self, node: int) -> set[int]:
        raise NotImplementedError

    @abstractmethod
    def get_meas_planes(self) -> dict[int, Plane]:
        raise NotImplementedError

    @abstractmethod
    def get_meas_angles(self) -> dict[int, float]:
        raise NotImplementedError

    @abstractmethod
    def append_graph(self, other: BaseGraphState) -> BaseGraphState:
        raise NotImplementedError


class GraphState(BaseGraphState):
    """Minimal implementation of GraphState"""

    def __init__(self) -> None:
        self.__input_nodes: set[int] = set()
        self.__output_nodes: set[int] = set()
        self.__physical_nodes: set[int] = set()
        self.__physical_edges: dict[int, set[int]] = {}
        self.__meas_planes: dict[int, Plane] = {}
        self.__meas_angles: dict[int, float] = {}
        # NOTE: qubit index if allocated. -1 if not. used for simulation
        self.__q_indices: dict[int, int] = {}

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
        return np.sum([len(edges) for edges in self.__physical_edges.values()]) // 2

    def add_physical_node(
        self,
        node: int,
        q_index: int = -1,
        is_input: bool = False,
        is_output: bool = False,
    ) -> None:
        if node in self.__physical_nodes:
            msg = f"Node already exists {node=}"
            raise Exception(msg)
        self.__physical_nodes |= {node}
        self.__physical_edges[node] = set()
        self.set_q_index(node, q_index)
        if is_input:
            self.__input_nodes |= {node}
        if is_output:
            self.__output_nodes |= {node}

    def add_physical_edge(self, node1: int, node2: int) -> None:
        if node1 not in self.__physical_nodes:
            msg = f"Node does not exist {node1=}"
            raise ValueError(msg)
        if node2 not in self.__physical_nodes:
            msg = f"Node does not exist {node2=}"
            raise ValueError(msg)
        if node1 in self.__physical_edges[node2] or node2 in self.__physical_edges[node1]:
            msg = f"Edge already exists {node1=}, {node2=}"
            raise ValueError(msg)
        self.__physical_edges[node1] |= {node2}
        self.__physical_edges[node2] |= {node1}

    def set_input(self, node: int) -> None:
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)
        self.__input_nodes |= {node}

    def set_output(self, node: int) -> None:
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)
        self.__output_nodes |= {node}

    def set_q_index(self, node: int, q_index: int = -1) -> None:
        if q_index < -1:
            msg = f"Invalid qubit index {q_index}. Must be -1 or greater"
            raise ValueError(msg)
        self.__q_indices[node] = q_index

    def set_meas_plane(self, node: int, plane: Plane) -> None:
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)
        self.__meas_planes[node] = plane

    def set_meas_angle(self, node: int, angle: float) -> None:
        if node not in self.__physical_nodes:
            msg = f"Node does not exist {node=}"
            raise ValueError(msg)
        self.__meas_angles[node] = angle

    def get_physical_nodes(self) -> set[int]:
        return self.__physical_nodes

    def get_physical_edges(self) -> set[tuple[int, int]]:
        edges = set()
        for node1 in self.__physical_edges:
            for node2 in self.__physical_edges[node1]:
                if node1 < node2:
                    edges |= {(node1, node2)}
        return edges

    def get_q_indices(self) -> dict[int, int]:
        return self.__q_indices

    def get_neighbors(self, node: int) -> set[int]:
        return self.__physical_edges[node]

    def get_meas_planes(self) -> dict[int, Plane]:
        return self.__meas_planes

    def get_meas_angles(self) -> dict[int, float]:
        return self.__meas_angles

    def append_graph(self, other: BaseGraphState) -> GraphState:
        common_nodes = self.get_physical_nodes() & other.get_physical_nodes()
        border_nodes = self.output_nodes & other.input_nodes

        if common_nodes != border_nodes:
            msg = "Qubit index mismatch"
            raise Exception(msg)

        new_graph = GraphState()
        for node in self.__physical_nodes:
            new_graph.add_physical_node(node)
            if node in set(self.__input_nodes):
                new_graph.set_input(node)

            if node in set(self.output_nodes) - set(other.input_nodes):
                new_graph.set_output(node)
            else:
                new_graph.set_meas_plane(node, self.__meas_planes.get(node, Plane.XY))
                new_graph.set_meas_angle(node, self.__meas_angles.get(node, 0.0))

        for edge in self.get_physical_edges():
            new_graph.add_physical_edge(edge[0], edge[1])

        for node in other.get_physical_nodes():
            if node in common_nodes:
                continue
            new_graph.add_physical_node(node)
            if node in set(other.input_nodes) - set(self.output_nodes):
                new_graph.set_input(node)

            if node in set(other.output_nodes):
                new_graph.set_output(node)
            else:
                new_graph.set_meas_plane(node, other.get_meas_planes().get(node, Plane.XY))
                new_graph.set_meas_angle(node, other.get_meas_angles().get(node, 0.0))

        for edge in other.get_physical_edges():
            new_graph.add_physical_edge(edge[0], edge[1])

        # q_index update
        for node, q_index in self.get_q_indices().items():
            new_graph.set_q_index(node, q_index)

        for node, q_index in other.get_q_indices().items():
            if (node in common_nodes) and (new_graph.get_q_indices()[node] != q_index):
                msg = "Qubit index mismatch."
                raise ValueError(msg)
            new_graph.set_q_index(node, q_index)

        return new_graph
