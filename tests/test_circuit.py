"""Test circuit module."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from graphqomb.circuit import BaseCircuit, Circuit, MBQCCircuit, circuit2graph
from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.gates import (
    CNOT,
    CZ,
    Gate,
    H,
    J,
    PhaseGadget,
    S,
    T,
    UnitGate,
    X,
    Y,
    Z,
)

# MBQCCircuit tests


def test_mbqc_circuit_init() -> None:
    """Test MBQCCircuit initialization."""
    circuit = MBQCCircuit(num_qubits=3)
    assert circuit.num_qubits == 3
    assert circuit.instructions() == []


def test_mbqc_circuit_j_gate() -> None:
    """Test adding J gate."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.5)
    instructions = circuit.unit_instructions()
    assert len(instructions) == 1
    assert isinstance(instructions[0], J)
    assert instructions[0].qubit == 0
    assert instructions[0].angle == 0.5


def test_mbqc_circuit_cz_gate() -> None:
    """Test adding CZ gate."""
    circuit = MBQCCircuit(num_qubits=3)
    circuit.cz(qubit1=0, qubit2=2)
    instructions = circuit.unit_instructions()
    assert len(instructions) == 1
    assert isinstance(instructions[0], CZ)
    assert instructions[0].qubits == (0, 2)


def test_mbqc_circuit_phase_gadget() -> None:
    """Test adding phase gadget."""
    circuit = MBQCCircuit(num_qubits=4)
    circuit.phase_gadget(qubits=[0, 1, 3], angle=0.25)
    instructions = circuit.unit_instructions()
    assert len(instructions) == 1
    assert isinstance(instructions[0], PhaseGadget)
    assert instructions[0].qubits == [0, 1, 3]
    assert instructions[0].angle == 0.25


def test_mbqc_circuit_multiple_gates() -> None:
    """Test adding multiple gates."""
    circuit = MBQCCircuit(num_qubits=3)
    circuit.j(qubit=0, angle=0.5)
    circuit.cz(qubit1=0, qubit2=1)
    circuit.j(qubit=1, angle=-0.3)
    circuit.phase_gadget(qubits=[0, 1, 2], angle=0.25)
    circuit.cz(qubit1=1, qubit2=2)

    instructions = circuit.unit_instructions()
    assert len(instructions) == 5
    assert all(isinstance(inst, (J, CZ, PhaseGadget)) for inst in instructions)


def test_mbqc_circuit_instructions_returns_copy() -> None:
    """Test that instructions() returns a copy of the list."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.5)
    circuit.cz(qubit1=0, qubit2=1)

    instructions1 = circuit.unit_instructions()
    instructions2 = circuit.unit_instructions()

    # Should be different list objects
    assert instructions1 is not instructions2
    # But with same contents
    assert instructions1 == instructions2


def test_mbqc_circuit_abstract_base_class() -> None:
    """Test that MBQCCircuit is an instance of BaseCircuit."""
    circuit = MBQCCircuit(num_qubits=2)
    assert isinstance(circuit, BaseCircuit)


# Circuit tests


def test_circuit_init() -> None:
    """Test Circuit initialization."""
    circuit = Circuit(num_qubits=3)
    assert circuit.num_qubits == 3
    assert circuit.unit_instructions() == []
    assert circuit.instructions() == []


def test_circuit_single_qubit_gates() -> None:
    """Test single qubit gate applications."""
    circuit = Circuit(num_qubits=2)
    circuit.apply_macro_gate(H(qubit=0))
    circuit.apply_macro_gate(X(qubit=1))
    circuit.apply_macro_gate(Y(qubit=0))
    circuit.apply_macro_gate(Z(qubit=1))
    circuit.apply_macro_gate(S(qubit=0))
    circuit.apply_macro_gate(T(qubit=1))

    macro_gates = circuit.instructions()
    assert len(macro_gates) == 6

    # Test that each macro gate type is preserved
    assert isinstance(macro_gates[0], H)
    assert isinstance(macro_gates[1], X)
    assert isinstance(macro_gates[2], Y)
    assert isinstance(macro_gates[3], Z)
    assert isinstance(macro_gates[4], S)
    assert isinstance(macro_gates[5], T)


def test_circuit_two_qubit_gates() -> None:
    """Test two qubit gate applications."""
    circuit = Circuit(num_qubits=3)
    circuit.apply_macro_gate(CNOT(qubits=(0, 1)))
    circuit.apply_macro_gate(CNOT(qubits=(1, 2)))

    macro_gates = circuit.instructions()
    assert len(macro_gates) == 2
    assert all(isinstance(gate, CNOT) for gate in macro_gates)


def test_circuit_instructions_expansion() -> None:
    """Test that unit_instructions() correctly expands macro gates."""
    circuit = Circuit(num_qubits=3)

    # Add various macro gates
    circuit.apply_macro_gate(H(qubit=0))  # 1 unit gate
    circuit.apply_macro_gate(CNOT(qubits=(0, 1)))  # 3 unit gates
    circuit.apply_macro_gate(X(qubit=2))  # 2 unit gates
    circuit.apply_macro_gate(Y(qubit=1))  # 4 unit gates
    circuit.apply_macro_gate(Z(qubit=0))  # 2 unit gates

    # Get expanded instructions
    instructions = circuit.unit_instructions()

    # Calculate expected total
    expected_count = 1 + 3 + 2 + 4 + 2
    assert len(instructions) == expected_count

    # Verify all are UnitGate instances
    assert all(isinstance(inst, (J, CZ, PhaseGadget)) for inst in instructions)

    # Test that instructions() returns macro gates
    macro_instructions = circuit.instructions()
    assert len(macro_instructions) == 5  # 5 macro gates
    assert all(isinstance(inst, Gate) for inst in macro_instructions)


def test_circuit_instructions_matches_manual_expansion() -> None:
    """Test that instructions() matches manual expansion using chain."""
    circuit = Circuit(num_qubits=2)

    # Add some gates
    circuit.apply_macro_gate(H(qubit=0))
    circuit.apply_macro_gate(T(qubit=0))
    circuit.apply_macro_gate(CNOT(qubits=(0, 1)))
    circuit.apply_macro_gate(S(qubit=1))

    # Get unit instructions from method
    instructions = circuit.unit_instructions()

    # Manual expansion
    macro_gates = circuit.instructions()
    expected_instructions = list(itertools.chain.from_iterable(gate.unit_gates() for gate in macro_gates))

    assert len(instructions) == len(expected_instructions)
    # Compare each instruction
    for inst, expected in zip(instructions, expected_instructions):
        assert type(inst) is type(expected)
        if isinstance(inst, J) and isinstance(expected, J):
            assert inst.qubit == expected.qubit
            assert inst.angle == expected.angle
        elif isinstance(inst, CZ) and isinstance(expected, CZ):
            assert inst.qubits == expected.qubits

    # Test that instructions() returns macro gates
    macro_instructions = circuit.instructions()
    assert len(macro_instructions) == 4  # 4 macro gates
    assert macro_instructions == circuit.instructions()


def test_circuit_empty_circuit_instructions() -> None:
    """Test empty circuit returns empty instructions."""
    circuit = Circuit(num_qubits=3)
    assert circuit.instructions() == []
    assert circuit.instructions() == []


def test_circuit_instructions_returns_copy() -> None:
    """Test that instructions() returns a copy."""
    circuit = Circuit(num_qubits=2)
    circuit.apply_macro_gate(H(qubit=0))
    circuit.apply_macro_gate(X(qubit=1))

    macro1 = circuit.instructions()
    macro2 = circuit.instructions()

    # Should be different list objects
    assert macro1 is not macro2
    # But with same contents
    assert len(macro1) == len(macro2)
    for g1, g2 in zip(macro1, macro2):
        assert type(g1) is type(g2)


def test_circuit_abstract_base_class() -> None:
    """Test that Circuit is an instance of BaseCircuit."""
    circuit = Circuit(num_qubits=2)
    assert isinstance(circuit, BaseCircuit)


# circuit2graph tests


def test_circuit2graph_simple_circuit() -> None:
    """Test conversion of a simple circuit."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.5)
    circuit.cz(qubit1=0, qubit2=1)
    circuit.j(qubit=1, angle=-0.3)

    graph, gflow = circuit2graph(circuit)

    # Check graph properties
    assert len(graph.input_node_indices) == 2
    assert len(graph.output_node_indices) == 2
    # 2 input nodes + 2 J gate nodes = 4 total nodes
    assert len(graph.physical_nodes) == 4

    # Check gflow
    assert len(gflow) == 2  # Two J gates should have gflow


def test_circuit2graph_phase_gadget_circuit() -> None:
    """Test conversion with phase gadget."""
    circuit = MBQCCircuit(num_qubits=3)
    circuit.phase_gadget(qubits=[0, 1, 2], angle=0.25)

    graph, _ = circuit2graph(circuit)

    # Check graph properties
    assert len(graph.input_node_indices) == 3
    assert len(graph.output_node_indices) == 3
    # 3 input nodes + 1 phase gadget node = 4 total nodes
    assert len(graph.physical_nodes) == 4

    # Check phase gadget node has correct measurement basis
    pg_nodes = [n for n in graph.physical_nodes if n not in graph.input_node_indices]
    assert len(pg_nodes) == 1
    pg_node = pg_nodes[0]
    basis = graph.meas_bases[pg_node]
    assert isinstance(basis, PlannerMeasBasis)
    assert basis.plane == Plane.YZ
    assert basis.angle == 0.25


def test_circuit2graph_empty_circuit() -> None:
    """Test conversion of empty circuit."""
    circuit = MBQCCircuit(num_qubits=2)
    graph, gflow = circuit2graph(circuit)

    # Check graph properties
    assert len(graph.input_node_indices) == 2
    assert len(graph.output_node_indices) == 2
    assert len(graph.physical_nodes) == 2  # Only input/output nodes
    assert len(gflow) == 0  # No gflow for empty circuit


def test_circuit2graph_invalid_instruction() -> None:
    """Test that invalid instruction raises TypeError."""

    # Create a mock invalid circuit
    class MockCircuit(BaseCircuit):
        @property
        def num_qubits(self) -> int:
            return 1

        def instructions(self) -> list[Gate]:  # noqa: PLR6301
            return [X(qubit=0)]

        def unit_instructions(self) -> list[UnitGate]:  # noqa: PLR6301
            # Return a non-UnitGate object to trigger error
            return [X(qubit=0)]  # type: ignore[list-item]

    circuit = MockCircuit()
    with pytest.raises(TypeError, match="Invalid instruction"):
        circuit2graph(circuit)


def test_circuit2graph_complex_circuit() -> None:
    """Test conversion of a more complex circuit."""
    circuit = MBQCCircuit(num_qubits=4)

    # Build a circuit with multiple gate types
    circuit.j(qubit=0, angle=np.pi / 4)
    circuit.j(qubit=1, angle=np.pi / 2)
    circuit.cz(qubit1=0, qubit2=2)
    circuit.cz(qubit1=1, qubit2=3)
    circuit.phase_gadget(qubits=[2, 3], angle=0.5)
    circuit.j(qubit=2, angle=-np.pi / 4)
    circuit.j(qubit=3, angle=np.pi)

    graph, gflow = circuit2graph(circuit)

    # Check basic properties
    assert len(graph.input_node_indices) == 4
    assert len(graph.output_node_indices) == 4

    # Count nodes: 4 inputs + 4 J nodes + 1 phase gadget = 9
    assert len(graph.physical_nodes) == 9

    # Check gflow: 4 J gates + 1 phase gadget = 5 entries
    assert len(gflow) == 5


def test_circuit2graph_measurement_basis_assignment() -> None:
    """Test that measurement bases are correctly assigned."""
    circuit = MBQCCircuit(num_qubits=2)
    circuit.j(qubit=0, angle=0.7)
    circuit.j(qubit=1, angle=-1.2)

    graph, _ = circuit2graph(circuit)

    # Find non-output nodes with measurement basis (J gates are applied to input nodes)
    measured_nodes = [
        (n, basis)
        for n in graph.physical_nodes
        if n not in graph.output_node_indices and (basis := graph.meas_bases.get(n))
    ]

    assert len(measured_nodes) == 2
    for _node, basis in measured_nodes:
        assert isinstance(basis, PlannerMeasBasis)
        assert basis.plane == Plane.XY
        # Check angles match the J gate angles (negated)
        assert basis.angle in {-0.7, 1.2}


def test_circuit2graph_circuit_with_macro_gates() -> None:
    """Test conversion of Circuit with macro gates."""
    circuit = Circuit(num_qubits=2)
    circuit.apply_macro_gate(H(qubit=0))
    circuit.apply_macro_gate(CNOT(qubits=(0, 1)))

    graph, _ = circuit2graph(circuit)

    # Check that macro gates are properly expanded
    assert len(graph.input_node_indices) == 2
    assert len(graph.output_node_indices) == 2
    # H expands to 1 J, CNOT expands to 2 J + 1 CZ = 3 nodes total
    assert len(graph.physical_nodes) == 5  # 2 inputs + 3 new nodes
