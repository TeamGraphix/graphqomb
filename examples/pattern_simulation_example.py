"""Example of pattern simulation using graphix-zx.

This example demonstrates how to:
1. Create a simple measurement pattern
2. Simulate the pattern execution
3. Extract the results and final state
"""

from typing import TYPE_CHECKING

from graphix_zx.pattern import Pattern
from graphix_zx.pauli_frame import PauliFrame
from graphix_zx.simulator import PatternSimulator, SimulatorBackend

if TYPE_CHECKING:
    from graphix_zx.statevec import StateVector

# Constants
TOLERANCE = 1e-10  # Tolerance for amplitude filtering
MAX_STATE_SIZE = 8  # Maximum state size to print


def create_simple_pattern() -> Pattern:
    """Create a simple measurement pattern for demonstrating simulation.

    This pattern implements identity operation on a single qubit:
    - Input qubit 0 -> output qubit 0 (no auxiliary nodes)

    Returns
    -------
    Pattern
        A simple measurement pattern
    """
    # Define input and output mappings
    input_node_indices = {0: 0}  # Node 0 is input qubit 0
    output_node_indices = {0: 0}  # Node 0 is also output qubit 0

    # Create commands: just identity (no measurements)
    commands = ()  # Empty tuple for identity operation

    # Create pauli frame to track corrections
    pauli_frame = PauliFrame(
        nodes={0},
        xflow={},  # No corrections needed for identity
        zflow={},  # No corrections needed for identity
    )

    return Pattern(
        input_node_indices=input_node_indices,
        output_node_indices=output_node_indices,
        commands=commands,
        pauli_frame=pauli_frame,
    )


def simulate_pattern_example() -> bool:
    """Example of simulating a measurement pattern.

    Returns
    -------
    bool
        True if simulation succeeded, False otherwise.
    """
    print("Creating measurement pattern...")
    pattern = create_simple_pattern()

    print(f"Pattern has {len(pattern)} commands")
    print("Commands:")
    for i, cmd in enumerate(pattern):
        print(f"  {i}: {cmd}")

    print("\nInitializing simulator...")
    simulator = PatternSimulator(
        pattern=pattern,
        backend=SimulatorBackend.StateVector,
        calc_prob=False,  # Use deterministic simulation
    )

    print("Simulating pattern execution...")
    try:
        simulator.simulate()
        print("✓ Pattern simulation completed successfully")

        # Get measurement results
        results = simulator.results
        print(f"Measurement results: {results}")

        # Get final state
        final_state: StateVector = simulator.get_state()  # type: ignore[assignment]
        print(f"Final state shape: {final_state.state().shape}")
        print(f"Final state norm: {final_state.norm():.6f}")

        # Print final state vector (if small enough)
        state_vec = final_state.state()
        if state_vec.size <= MAX_STATE_SIZE:
            print("Final state vector:")
            flat_state = state_vec.ravel()
            for i, amp in enumerate(flat_state):
                if abs(amp) > TOLERANCE:
                    print(f"  |{i:b}⟩: {amp:.6f}")

    except (ValueError, RuntimeError) as e:
        print(f"✗ Pattern simulation failed: {e}")
        return False

    return True


def create_two_qubit_pattern() -> Pattern:
    """Create a more complex two-qubit pattern.

    This pattern implements identity on two qubits (no measurements):
    - Input qubits 0, 1 -> output qubits 0, 1

    Returns
    -------
    Pattern
        A two-qubit measurement pattern
    """
    # Define input and output mappings
    input_node_indices = {0: 0, 1: 1}  # Nodes 0,1 are input qubits 0,1
    output_node_indices = {0: 0, 1: 1}  # Same nodes are output qubits 0,1

    # Create commands for simple pattern (identity)
    commands = ()  # No operations needed for identity

    # Create pauli frame for two-qubit pattern
    pauli_frame = PauliFrame(
        nodes={0, 1},
        xflow={},  # No measurements, no flow
        zflow={},  # No measurements, no flow
    )

    return Pattern(
        input_node_indices=input_node_indices,
        output_node_indices=output_node_indices,
        commands=commands,
        pauli_frame=pauli_frame,
    )


def simulate_two_qubit_example() -> bool:
    """Example of simulating a two-qubit measurement pattern.

    Returns
    -------
    bool
        True if simulation succeeded, False otherwise.
    """
    print("\nCreating two-qubit measurement pattern...")
    pattern = create_two_qubit_pattern()

    print(f"Pattern has {len(pattern)} commands")
    print("Commands:")
    for i, cmd in enumerate(pattern):
        print(f"  {i}: {cmd}")

    print("\nInitializing two-qubit simulator...")
    simulator = PatternSimulator(pattern=pattern, backend=SimulatorBackend.StateVector, calc_prob=False)

    print("Simulating two-qubit pattern execution...")
    try:
        simulator.simulate()
        print("✓ Two-qubit pattern simulation completed successfully")

        # Get measurement results
        results = simulator.results
        print(f"Measurement results: {results}")

        # Get final state
        final_state: StateVector = simulator.get_state()  # type: ignore[assignment]
        print(f"Final state shape: {final_state.state().shape}")
        print(f"Final state norm: {final_state.norm():.6f}")

    except (ValueError, RuntimeError) as e:
        print(f"✗ Two-qubit pattern simulation failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("=== Pattern Simulation Examples ===")

    # Run simple pattern example
    success1 = simulate_pattern_example()

    # Run two-qubit pattern example
    success2 = simulate_two_qubit_example()

    print("\n=== Summary ===")
    print(f"Simple pattern: {'✓ Success' if success1 else '✗ Failed'}")
    print(f"Two-qubit pattern: {'✓ Success' if success2 else '✗ Failed'}")
