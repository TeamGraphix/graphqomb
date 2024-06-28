from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray
import pyzx as zx
from fractions import Fraction
from pyzx.graph.base import BaseGraph, DocstringMeta

# NOTE: local rule(should be discussed)
# pyzx only supports Fraction | int | Polynomial types for phase
# so let us use `phase` in pi unit and `angle` in radian unit. like `phase = 1/2` and `angle = np.pi/2`


def angle2phase(angle: float) -> Fraction:
    return Fraction(angle / np.pi)


def phase2angle(phase: Fraction) -> float:
    return float(phase * np.pi)


class PhysicalNode(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def is_input(self):
        raise NotImplementedError

    @abstractmethod
    def is_output(self):
        raise NotImplementedError

    @abstractmethod
    def is_internal(self):
        raise NotImplementedError

    @abstractmethod
    def get_meas_plane(self):
        raise NotImplementedError

    @abstractmethod
    def get_meas_angle(self):
        raise NotImplementedError

    @abstractmethod
    def set_meas_plane(self, plane: str):
        raise NotImplementedError

    @abstractmethod
    def set_meas_angle(self, angle: float):
        raise NotImplementedError


# abstract class for graph state
# NOTE: this class just represents a graph state, not necessarily include optimization
class GraphState(ABC):
    @abstractmethod
    def __init__(self):
        pass

    # NOTE: input and output nodes are necessary because graph is open graph
    @property
    @abstractmethod
    def input_nodes(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def output_nodes(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_physical_nodes(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_physical_edges(self):
        raise NotImplementedError

    @abstractmethod
    def add_physical_node(self, node: int):
        raise NotImplementedError

    @abstractmethod
    def add_physical_edge(self, node1: int, node2: int):
        raise NotImplementedError

    @abstractmethod
    def set_meas_plane(self, node: int, plane: str):
        raise

    @abstractmethod
    def set_meas_angle(self, node: int, angle: float):
        raise NotImplementedError

    @abstractmethod
    def get_physical_nodes(self):
        raise NotImplementedError

    @abstractmethod
    def get_physical_edges(self):
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self, node: int):
        raise NotImplementedError

    @abstractmethod
    def get_meas_planes(self):
        raise NotImplementedError

    @abstractmethod
    def get_meas_angles(self):
        raise NotImplementedError


class BasicGraphState(GraphState):
    """Minimal implementation of GraphState"""

    def __init__(self):
        self.__input_nodes: list[int] = []
        self.__output_nodes: list[int] = []
        self.__physical_nodes: set[int] = {}
        self.__physical_edges: dict[int, set[int]] = {}
        self.__meas_planes: dict[int, str] = {}
        self.__meas_angles: dict[int, float] = {}

    @property
    def input_nodes(self) -> list[int]:
        return self.__input_nodes

    @property
    def output_nodes(self) -> list[int]:
        return self.__output_nodes

    @property
    def num_physical_nodes(self) -> int:
        return len(self.__physical_nodes)

    def add_physical_node(self, node: int):
        if node in self.__physical_nodes:
            raise Exception("Node already exists")
        self.__physical_nodes |= {node}
        self.__physical_edges[node] = set()

    def add_physical_edge(self, node1: int, node2: int):
        if node1 not in self.__physical_nodes or node2 not in self.__physical_nodes:
            raise Exception("Node does not exist")
        if node1 in self.__physical_edges[node2] or node2 in self.__physical_edges[node1]:
            raise Exception("Edge already exists")
        self.__physical_edges[node1] |= {node2}
        self.__physical_edges[node2] |= {node1}

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


class ZXPhysicalNode(BaseGraph, PhysicalNode, metaclass=DocstringMeta):
    def __init__(self, node_id: int | None = None, row: int = -1):
        super().__init__()

        self.__is_input: bool = False
        self.__is_output: bool = False
        self.__is_internal: bool = False

        if node_id is None:
            node_id = gen_new_index()

        self.node_id = node_id
        self.add_vertex(ty=zx.VertexType.Z, qubit=node_id, row=row, phase=0)

    def is_input(self):
        return self.__is_input

    def is_output(self):
        return self.__is_output

    def is_internal(self):
        return self.__is_internal

    def set_meas_plane(self, plane: str):
        # TODO: if plane is already set, raise error or remove the previous setting
        self.meas_id = gen_new_index()
        if plane == "XY":
            self.add_vertex(ty=zx.VertexType.Z, qubit=self.meas_id, phase=0)
            self.add_edge((self.node_id, self.meas_id), edgetype=zx.EdgeType.SIMPLE)
        elif plane == "YZ":
            self.add_vertex(ty=zx.VertexType.X, qubit=self.meas_id, phase=0)
            self.add_edge((self.node_id, self.meas_id), edgetype=zx.EdgeType.SIMPLE)
        elif plane == "XZ":
            self.add_vertex(ty=zx.VertexType.Z, qubit=gen_new_index(), phase=Fraction(1, 2))
            self.add_vertex(ty=zx.VertexType.X, qubit=self.meas_id, phase=0)

    def set_meas_angle(self, angle: float):
        self.set_phase(self.meas_id, angle2phase(angle))

    def get_meas_plane(self):
        v_type = zx.VertexType[self.type(self.meas_id)]
        if v_type == zx.VertexType.Z:
            return "XY"
        elif v_type == zx.VertexType.X:
            neighbors = self.neighbors(self.meas_id)
            if len(neighbors) > 1:
                raise Exception("Number of neighbors of the measurement node is not 1")
            if self.connected(self.node_id, self.meas_id):
                return "YZ"
            else:
                if neighbors[0] == zx.VertexType.Z and np.isclose(self.phase(neighbors[0]), Fraction(1, 2)):
                    return "XZ"
                else:
                    raise Exception("Invalid measurement node")
        else:
            raise Exception("Invalid measurement node")

    def get_meas_angle(self):
        return phase2angle(self.phase(self.meas_id))

    def get_zx_diagram(self):
        return self


# NOTE: for Arbitrary GraphState Construction permitted in MBQC
# TODO: ZXPhysicalNode is probably not recognized as a subgraph
class ZXGraphState(BaseGraph, GraphState, metaclass=DocstringMeta):
    def __init__(self):
        super().__init__()

        self.__input_nodes: list[int] = []
        self.__output_nodes: list[int] = []

        # NOTE: macro node is composed of XY, XZ, or YZ physical nodes
        self.__physical_nodes: dict[int, ZXPhysicalNode] = dict()
        self.__physical_edges: dict[int, set[int]] = dict()

    def __add__(self, other):
        raise NotImplementedError

    @property
    def input_nodes(self):
        return self.__input_nodes

    @property
    def output_nodes(self):
        return self.__output_nodes

    @property
    def num_physical_nodes(self):
        return len(self.__physical_nodes)

    @property
    def num_physical_edges(self):
        return len(self.__physical_edges)

    def add_physical_node(self, node: int | None = None, row: int = -1):
        # prepare |+> state(N)
        if node is None:
            node = gen_new_index()
        self.__physical_nodes[node] = ZXPhysicalNode(node, row)
        self.__physical_edges[node] = set()
        return node

    def add_physical_edge(self, node1: int, node2: int):
        # apply a Hadamard edge(CZ) between two nodes(E)
        physicalnode1 = self.__physical_nodes[node1]
        physicalnode2 = self.__physical_nodes[node2]

        self.add_edge(
            (physicalnode1.node_id, physicalnode2.node_id),
            edgetype=zx.EdgeType.HADAMARD,
        )
        self.__physical_edges[node1] |= {node2}
        self.__physical_edges[node2] |= {node1}

    def set_meas_plane(self, node: int, plane: str):
        self.__physical_nodes[node].set_meas_plane(plane)

    def set_meas_angle(self, node: int, angle: float):
        self.__physical_nodes[node].set_meas_angle(angle)

    def get_physical_nodes(self):
        return self.__physical_nodes.keys()

    def get_physical_edges(self):
        edges = set()
        for node1 in self.__physical_edges.keys():
            for node2 in self.__physical_edges[node1]:
                if node1 < node2:
                    edges |= {(node1, node2)}
        return edges

    def get_zx_nodes(self):
        return self.vertices()

    def get_zx_edges(self):
        return self.edges()

    def get_meas_planes(self):
        return [node.get_meas_plane() for node in self.__physical_nodes.values()]

    def get_meas_angles(self):
        return [node.get_meas_angle() for node in self.__physical_nodes.values()]


class GateKind(Enum):
    """Enum class for gate kind"""

    J = auto()
    CZ = auto()
    PhaseGadget = auto()


@dataclass(frozen=True)
class Gate:
    kind: GateKind

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


@dataclass(frozen=True)
class J(Gate):
    kind: GateKind = GateKind.J
    qubit: int | None = None
    angle: float = 0

    def get_matrix(self) -> NDArray:
        return np.array([[1, np.exp(1j * self.angle)], [1, -np.exp(1j * self.angle)]]) / np.sqrt(2)


@dataclass(frozen=True)
class CZ(Gate):
    kind: GateKind = GateKind.CZ
    qubits: tuple[int, int] | None = None

    def get_matrix(self) -> NDArray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


@dataclass(frozen=True)
class PhaseGadget(Gate):
    kind: GateKind = GateKind.PhaseGadget
    qubits: list[int] | None = None
    angle: float = 0

    def get_matrix(self) -> NDArray:
        raise NotImplementedError


class MBQCCircuit(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def input_nodes(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def num_qubits(self):
        raise NotImplementedError

    @abstractmethod
    def get_instructions(self):
        raise NotImplementedError

    @abstractmethod
    def j(self, qubit: int, angle: float):
        raise NotImplementedError

    @abstractmethod
    def cz(self, qubit1: int, qubit2: int):
        raise NotImplementedError

    @abstractmethod
    def phase_gadget(self, qubits: list[int], angle: float):
        raise NotImplementedError


# NOTE: for Unitary Construction
class ZXMBQCCircuit(ZXGraphState, MBQCCircuit, metaclass=DocstringMeta):
    def __init__(self, qubits: int):
        super().__init__()  # initialize ZXGraphState
        self.__input_nodes: list[int] = list(range(qubits))
        self.__output_nodes: list[int] = list(range(qubits))
        self.__num_qubits = qubits

        self.__gate_instructions: list[Gate] = []
        self.__gflow: dict[int, set[int]] = {input_node: set() for input_node in self.__input_nodes}

    # gate concatenation
    def __add__(self, other):
        raise NotImplementedError

    @property
    def input_nodes(self):
        return self.__input_nodes

    @property
    def output_nodes(self):
        return self.__output_nodes

    @property
    def num_qubits(self):
        return self.__num_qubits

    @property
    def gflow(self):
        return self.__gflow

    def get_instructions(self):
        return super().get_instructions()

    # unit gate of XY plane
    def j(self, qubit: int, angle: float):
        """
        Apply a J gate to a qubit.
        """
        old_node = self.__output_nodes[qubit]
        self.set_meas_angle(old_node, angle)
        new_node = gen_new_index()
        self.add_physical_node(new_node, row=qubit)
        self.add_physical_edge(old_node, new_node)
        self.__output_nodes[qubit] = new_node

        self.__gflow[old_node] |= {new_node}
        self.__gate_instructions.append(J(qubit=qubit, angle=angle))

    # TODO: unit gate of XZ and YZ planes
    def phase_gadget(self, qubits: list[int], angle: float):
        target_nodes = [self.__output_nodes[qubit] for qubit in qubits]
        new_node = gen_new_index()  # TODO: implement
        self.add_physical_node(new_node, row=-1)
        self.set_meas_angle(new_node, angle)
        self.set_meas_plane(new_node, "YZ")
        for node in target_nodes:
            self.add_physical_edge(node, new_node)

        # TODO: record gflow
        self.__gate_instructions.append(PhaseGadget(qubits=qubits, angle=angle))

    # vertical edge
    def cz(self, qubit1: int, qubit2: int):
        """
        Apply a CZ gate between two qubits.
        """
        node1 = self.__output_nodes[qubit1]
        node2 = self.__output_nodes[qubit2]
        self.add_physical_edge(node1, node2)

        self.__gate_instructions.append(CZ(qubits=(qubit1, qubit2)))

    def rx(self, qubit: int, angle: float):  # example
        self.j(qubit, 0)
        self.j(qubit, angle)

    def rz(self, qubit: int, angle: float):
        self.j(qubit, angle)
        self.j(qubit, 0)

    def cnot(self, qubit1: int, qubit2: int):
        self.j(qubit2, 0)
        self.cz(qubit1, qubit2)
        self.j(qubit2, 0)


def gen_new_index():
    return uuid.uuid4().int


def visualize(graph: GraphState):
    if isinstance(graph, MBQCCircuit):
        # visualize based on the logical qubit path
        raise NotImplementedError
    else:
        raise NotImplementedError
