from abc import ABC, abstractmethod

import pyzx as zx


# abstract class for graph state
# NOTE: want to avoid heavy dependency on pyzx
# NOTE: this class just represents the graph state, not necessarily include optimization
class GraphState(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    def input_nodes(self):
        return self.input_qubits

    def output_nodes(self):
        return self.output_qubits

    @abstractmethod
    def add_node(self, qubit: int):
        pass

    @abstractmethod
    def add_phase(self, node: int, plane: str, phase: float):
        pass

    @abstractmethod
    def add_edge(self, node1: int, node2: int):
        pass


# directly implement gates on the graph state
class MBQCCircuit(zx.BaseGraph):
    def __init__(self):
        super().__init__()
        self.output_qubits = []

    # gate concatenation
    def __add__(self, other):
        raise NotImplementedError

    # unit gate of XY plane
    def J(self, qubit: int, phase: float):
        """
        Apply a J gate to a qubit.
        """
        old_spider = self.output_qubits[qubit]
        self.set_phase(old_spider, phase)
        new_spider = self.add_vertex(ty=zx.VertexType.Z, qubit=qubit, row=qubit, phase=0)
        self.add_edge((old_spider, new_spider), ty=zx.EdgeType.HADAMARD)
        self.output_qubits[qubit] = new_spider

        # TODO: record gflow

    # TODO: unit gate of XZ and YZ planes
    def PhaseGadget(self, qubits: list[int], phase: float):
        raise NotImplementedError

    # vertical edge
    def CZ(self, qubit1: int, qubit2: int):
        """
        Apply a CZ gate between two qubits.
        """
        spider1 = self.output_qubits[qubit1]
        spider2 = self.output_qubits[qubit2]
        self.add_edge((spider1, spider2), ty=zx.EdgeType.HADAMARD)

    # NOTE: want to define other gates as subgraphs

    def Rx(self, qubit: int, phase: float):  # example
        self.J(qubit, 0)
        self.J(qubit, phase)

    def Rz(self, qubit: int, phase: float):
        self.J(qubit, phase)
        self.J(qubit, 0)

    def CNOT(self, qubit1: int, qubit2: int):
        self.J(qubit2, 0)
        self.CZ(qubit1, qubit2)
        self.J(qubit2, 0)
