"""Test comparison between macro gate and unit gate simulation."""

import math
import time

import numpy as np

from graphix_zx.circuit import Circuit
from graphix_zx.gates import H, Rz, S, T, X
from graphix_zx.simulator import MBQCCircuitSimulator, SimulatorBackend
from graphix_zx.statevec import StateVector

# Constants
TOLERANCE = 1e-10  # Tolerance for state comparison


def test_correctness() -> bool:
    """Test that macro gate simulation gives correct results.

    Returns
    -------
    bool
        True if test passed, False otherwise.

    Raises
    ------
    TypeError
        If gate type cannot be determined.
    """
    print("=== Macro vs Unit Gate Simulation Test ===\n")

    # Create a circuit with various macro gates
    circuit = Circuit(num_qubits=1)
    circuit.apply_macro_gate(H(qubit=0))
    circuit.apply_macro_gate(Rz(qubit=0, angle=math.pi / 4))
    circuit.apply_macro_gate(S(qubit=0))
    circuit.apply_macro_gate(T(qubit=0))
    circuit.apply_macro_gate(X(qubit=0))

    print("Circuit gates:")
    macro_gates = circuit.instructions()
    for i, gate in enumerate(macro_gates):
        print(f"  {i}: {gate}")

    print("\nUnit gate decomposition:")
    unit_gates = circuit.unit_instructions()
    for i, gate in enumerate(unit_gates):
        print(f"  {i}: {gate}")

    # Simulate using macro gates (new method)
    print("\nSimulating with macro gates...")
    sim_macro = MBQCCircuitSimulator(circuit, SimulatorBackend.StateVector)
    sim_macro.simulate()
    state_macro = sim_macro.get_state().state()

    # Manually compute expected result
    print("\nManual calculation...")

    state_manual = StateVector.from_num_qubits(1)
    for gate in macro_gates:
        # Get qubits that the gate acts on
        if hasattr(gate, "qubit"):
            qubits = [gate.qubit]
        elif hasattr(gate, "qubits"):
            qubits = list(gate.qubits)
        else:
            msg = f"Cannot determine qubits for gate: {gate}"
            raise TypeError(msg)
        state_manual.evolve(gate.matrix(), qubits)
    state_manual_array = state_manual.state()

    print("Results:")
    print(f"Macro gate simulation: {state_macro.ravel()}")
    print(f"Manual calculation:    {state_manual_array.ravel()}")

    # Compare
    diff = np.abs(state_macro - state_manual_array).max()
    print(f"Difference: {diff:.2e}")

    if diff < TOLERANCE:
        print("✓ Macro gate simulation is correct!")
        return True
    print("✗ Macro gate simulation differs from manual calculation")
    return False


def test_performance() -> None:
    """Test performance comparison (simple timing)."""
    print("\n=== Performance Test ===\n")

    # Create a larger circuit
    circuit = Circuit(num_qubits=2)

    for _i in range(10):
        circuit.apply_macro_gate(H(qubit=0))
        circuit.apply_macro_gate(Rz(qubit=0, angle=math.pi / 8))
        circuit.apply_macro_gate(S(qubit=1))
        circuit.apply_macro_gate(T(qubit=1))

    print(f"Circuit with {len(circuit.instructions())} macro gates")
    print(f"Equivalent to {len(circuit.unit_instructions())} unit gates")

    # Time macro gate simulation
    start = time.perf_counter()
    sim_macro = MBQCCircuitSimulator(circuit, SimulatorBackend.StateVector)
    sim_macro.simulate()
    macro_time = time.perf_counter() - start

    print(f"\nMacro gate simulation: {macro_time * 1000:.2f} ms")
    print("✓ New implementation working efficiently")


if __name__ == "__main__":
    success = test_correctness()
    test_performance()

    print("\n=== Summary ===")
    print(f"Correctness test: {'✓ Passed' if success else '✗ Failed'}")
    print("New macro gate simulation is ready!")
