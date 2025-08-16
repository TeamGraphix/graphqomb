"""Example of circuit simulation using graphix-zx.

This example demonstrates how to:
1. Create quantum circuits using different gate types
2. Simulate circuit execution
3. Extract the results and final state
"""

import math

import numpy as np

from graphix_zx.circuit import Circuit, MBQCCircuit
from graphix_zx.gates import CNOT, CZ, H, Rz, X
from graphix_zx.simulator import MBQCCircuitSimulator, SimulatorBackend

# Constants
TOLERANCE = 1e-10  # Tolerance for amplitude filtering


def create_simple_mbqc_circuit() -> MBQCCircuit:
    """Create a simple MBQC circuit with basic gates.

    Returns
    -------
    MBQCCircuit
        A simple circuit with J gates and CZ gates
    """
    circuit = MBQCCircuit(num_qubits=2)

    # Add some basic operations
    circuit.j(qubit=0, angle=math.pi / 4)  # RZ(π/4) on qubit 0
    circuit.j(qubit=1, angle=math.pi / 2)  # RZ(π/2) on qubit 1
    circuit.cz(qubit1=0, qubit2=1)  # CZ gate between qubits
    circuit.j(qubit=0, angle=-math.pi / 4)  # RZ(-π/4) on qubit 0

    return circuit


def simulate_mbqc_circuit_example() -> bool:
    """Example of simulating an MBQC circuit.

    Returns
    -------
    bool
        True if simulation succeeded, False otherwise.
    """
    print("Creating MBQC circuit...")
    circuit = create_simple_mbqc_circuit()

    print(f"Circuit has {circuit.num_qubits} qubits")
    instructions = circuit.instructions()
    print(f"Circuit has {len(instructions)} instructions:")
    for i, instr in enumerate(instructions):
        print(f"  {i}: {instr}")

    print("\nInitializing MBQC circuit simulator...")
    simulator = MBQCCircuitSimulator(mbqc_circuit=circuit, backend=SimulatorBackend.StateVector)

    print("Simulating circuit execution...")
    try:
        simulator.simulate()
        print("✓ MBQC circuit simulation completed successfully")

        # Get final state
        final_state = simulator.get_state()
        print(f"Final state shape: {final_state.state().shape}")
        print(f"Final state norm: {final_state.norm():.6f}")

        # Print final state vector
        state_vec = final_state.state()
        print("Final state vector:")
        flat_state = state_vec.ravel()
        for i, amp in enumerate(flat_state):
            if abs(amp) > TOLERANCE:
                print(f"  |{i:02b}⟩: {amp:.6f}")

    except (ValueError, RuntimeError) as e:
        print(f"✗ MBQC circuit simulation failed: {e}")
        return False

    return True


def create_macro_circuit() -> Circuit:
    """Create a circuit with macro gates.

    Returns
    -------
    Circuit
        A circuit with high-level gates
    """
    circuit = Circuit(num_qubits=3)

    # Add macro gates
    circuit.apply_macro_gate(H(qubit=0))  # Hadamard on qubit 0
    circuit.apply_macro_gate(X(qubit=1))  # Pauli-X on qubit 1
    circuit.apply_macro_gate(CNOT(qubits=(0, 1)))  # CNOT gate
    circuit.apply_macro_gate(Rz(qubit=2, angle=math.pi / 3))  # Rz rotation
    circuit.apply_macro_gate(CZ(qubits=(1, 2)))  # CZ gate

    return circuit


def simulate_macro_circuit_example() -> bool:
    """Example of simulating a circuit with macro gates.

    Returns
    -------
    bool
        True if simulation succeeded, False otherwise.
    """
    print("\nCreating circuit with macro gates...")
    circuit = create_macro_circuit()

    print(f"Circuit has {circuit.num_qubits} qubits")
    macro_instructions = circuit.macro_gate_instructions
    print(f"Circuit has {len(macro_instructions)} macro instructions:")
    for i, instr in enumerate(macro_instructions):
        print(f"  {i}: {instr}")

    # Convert to unit gates for simulation
    unit_instructions = circuit.instructions()
    print(f"Expanded to {len(unit_instructions)} unit instructions:")
    for i, instr in enumerate(unit_instructions):
        print(f"  {i}: {instr}")

    print("\nInitializing macro circuit simulator...")
    simulator = MBQCCircuitSimulator(mbqc_circuit=circuit, backend=SimulatorBackend.StateVector)

    print("Simulating macro circuit execution...")
    try:
        simulator.simulate()
        print("✓ Macro circuit simulation completed successfully")

        # Get final state
        final_state = simulator.get_state()
        print(f"Final state shape: {final_state.state().shape}")
        print(f"Final state norm: {final_state.norm():.6f}")

        # Print final state vector
        state_vec = final_state.state()
        print("Final state vector:")
        flat_state = state_vec.ravel()
        for i, amp in enumerate(flat_state):
            if abs(amp) > TOLERANCE:
                print(f"  |{i:03b}⟩: {amp:.6f}")

    except (ValueError, RuntimeError) as e:
        print(f"✗ Macro circuit simulation failed: {e}")
        return False

    return True


def create_phase_gadget_circuit() -> MBQCCircuit:
    """Create a circuit with phase gadgets.

    Returns
    -------
    MBQCCircuit
        A circuit demonstrating phase gadgets
    """
    circuit = MBQCCircuit(num_qubits=3)

    # Prepare initial state
    circuit.j(qubit=0, angle=math.pi / 2)  # RZ(π/2)
    circuit.j(qubit=1, angle=math.pi / 4)  # RZ(π/4)

    # Add phase gadget
    circuit.phase_gadget(qubits=[0, 1, 2], angle=math.pi / 6)

    # Final rotations
    circuit.j(qubit=2, angle=-math.pi / 3)  # RZ(-π/3)

    return circuit


def simulate_phase_gadget_example() -> bool:
    """Example of simulating a circuit with phase gadgets.

    Returns
    -------
    bool
        True if simulation succeeded, False otherwise.
    """
    print("\nCreating circuit with phase gadgets...")
    circuit = create_phase_gadget_circuit()

    print(f"Circuit has {circuit.num_qubits} qubits")
    instructions = circuit.instructions()
    print(f"Circuit has {len(instructions)} instructions:")
    for i, instr in enumerate(instructions):
        print(f"  {i}: {instr}")

    print("\nInitializing phase gadget circuit simulator...")
    simulator = MBQCCircuitSimulator(mbqc_circuit=circuit, backend=SimulatorBackend.StateVector)

    print("Simulating phase gadget circuit execution...")
    try:
        simulator.simulate()
        print("✓ Phase gadget circuit simulation completed successfully")

        # Get final state
        final_state = simulator.get_state()
        print(f"Final state shape: {final_state.state().shape}")
        print(f"Final state norm: {final_state.norm():.6f}")

        # Calculate some expectation values
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        expectation_z0 = final_state.expectation(pauli_z, 0)
        expectation_z1 = final_state.expectation(pauli_z, 1)
        expectation_z2 = final_state.expectation(pauli_z, 2)

        print(f"⟨Z₀⟩ = {expectation_z0:.6f}")
        print(f"⟨Z₁⟩ = {expectation_z1:.6f}")
        print(f"⟨Z₂⟩ = {expectation_z2:.6f}")

    except (ValueError, RuntimeError) as e:
        print(f"✗ Phase gadget circuit simulation failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("=== Circuit Simulation Examples ===")

    # Run MBQC circuit example
    success1 = simulate_mbqc_circuit_example()

    # Run macro circuit example
    success2 = simulate_macro_circuit_example()

    # Run phase gadget example
    success3 = simulate_phase_gadget_example()

    print("\n=== Summary ===")
    print(f"MBQC circuit: {'✓ Success' if success1 else '✗ Failed'}")
    print(f"Macro circuit: {'✓ Success' if success2 else '✗ Failed'}")
    print(f"Phase gadget circuit: {'✓ Success' if success3 else '✗ Failed'}")
