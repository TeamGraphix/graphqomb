from __future__ import annotations

from abc import ABC, abstractmethod


class GraphState(ABC):
    @abstractmethod
    def __init__(self):
        pass

    # NOTE: input and output nodes are necessary because graph is open graph
    @property
    @abstractmethod
    def input_nodes(self) -> list[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_nodes(self) -> list[int]:
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
    def add_physical_node(self, node: int, is_input: bool = False, is_output: bool = False):
        raise NotImplementedError

    @abstractmethod
    def add_physical_edge(self, node1: int, node2: int):
        raise NotImplementedError

    @abstractmethod
    def set_input(self, node: int):
        raise NotImplementedError

    @abstractmethod
    def set_output(self, node: int):
        raise NotImplementedError

    @abstractmethod
    def set_meas_plane(self, node: int, plane: str):
        raise

    @abstractmethod
    def set_meas_angle(self, node: int, angle: float):
        raise NotImplementedError

    @abstractmethod
    def get_physical_nodes(self) -> set[int]:
        raise NotImplementedError

    @abstractmethod
    def get_physical_edges(self) -> set[tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self, node: int) -> set[int]:
        raise NotImplementedError

    @abstractmethod
    def get_meas_planes(self) -> dict[int, str]:
        raise NotImplementedError

    @abstractmethod
    def get_meas_angles(self) -> dict[int, float]:
        raise NotImplementedError


class BasicGraphState(GraphState):
    """Minimal implementation of GraphState"""

    def __init__(self):
        self.__input_nodes: list[int] = []
        self.__output_nodes: list[int] = []
        self.__physical_nodes: set[int] = set()
        self.__physical_edges: dict[int, set[int]] = dict()
        self.__meas_planes: dict[int, str] = dict()
        self.__meas_angles: dict[int, float] = dict()

    @property
    def input_nodes(self) -> list[int]:
        return self.__input_nodes

    @property
    def output_nodes(self) -> list[int]:
        return self.__output_nodes

    @property
    def num_physical_nodes(self) -> int:
        return len(self.__physical_nodes)

    @property
    def num_physical_edges(self) -> int:
        num_edges = np.sum([len(edges) for edges in self.__physical_edges.values()]) // 2
        return num_edges

    def add_physical_node(self, node: int, is_input: bool = False, is_output: bool = False):
        if node in self.__physical_nodes:
            raise Exception("Node already exists")
        self.__physical_nodes |= {node}
        self.__physical_edges[node] = set()
        if is_input:
            self.__input_nodes.append(node)
        if is_output:
            self.__output_nodes.append(node)

    def add_physical_edge(self, node1: int, node2: int):
        if node1 not in self.__physical_nodes or node2 not in self.__physical_nodes:
            raise Exception("Node does not exist")
        if node1 in self.__physical_edges[node2] or node2 in self.__physical_edges[node1]:
            raise Exception("Edge already exists")
        self.__physical_edges[node1] |= {node2}
        self.__physical_edges[node2] |= {node1}

    def set_input(self, node: int):
        if node not in self.__physical_nodes:
            raise Exception("Node does not exist")
        self.__input_nodes.append(node)

    def set_output(self, node: int):
        if node not in self.__physical_nodes:
            raise Exception("Node does not exist")
        self.__output_nodes.append(node)

    def set_meas_plane(self, node: int, plane: str):
        if node not in self.__physical_nodes:
            raise Exception("Node does not exist")
        self.__meas_planes[node] = plane

    def set_meas_angle(self, node: int, angle: float):
        if node not in self.__physical_nodes:
            raise Exception("Node does not exist")
        self.__meas_angles[node] = angle

    def get_physical_nodes(self) -> set[int]:
        return self.__physical_nodes

    def get_physical_edges(self) -> set[tuple[int, int]]:
        edges = set()
        for node1 in self.__physical_edges.keys():
            for node2 in self.__physical_edges[node1]:
                if node1 < node2:
                    edges |= {(node1, node2)}
        return edges

    def get_neighbors(self, node: int) -> set[int]:
        return self.__physical_edges[node]

    def get_meas_planes(self) -> dict[int, str]:
        return self.__meas_planes

    def get_meas_angles(self) -> dict[int, float]:
        return self.__meas_angles
