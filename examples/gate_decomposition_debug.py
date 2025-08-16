"""Debug gate decomposition to unit gates."""

import math

import numpy as np

from graphix_zx.gates import H, Rz, S, T, X

# Constants
TOLERANCE = 1e-10  # Tolerance for matrix comparison


def check_gate_decompositions() -> None:
    """Check how macro gates decompose to unit gates."""
    print("=== Gate Decomposition Analysis ===")

    gates = [H(qubit=0), Rz(qubit=0, angle=math.pi / 4), S(qubit=0), T(qubit=0), X(qubit=0)]

    for gate in gates:
        print(f"\nGate: {gate}")
        print(f"Matrix:\n{gate.matrix()}")

        unit_gates = gate.unit_gates()
        print(f"Unit gates: {unit_gates}")

        # Manually combine unit gate matrices
        combined_matrix = np.eye(2, dtype=complex)
        for unit_gate in unit_gates:
            combined_matrix = unit_gate.matrix() @ combined_matrix

        print(f"Combined unit gate matrix:\n{combined_matrix}")

        # Check if they match
        diff = np.abs(gate.matrix() - combined_matrix).max()
        print(f"Difference: {diff:.2e}")
        if diff > TOLERANCE:
            print("⚠️  MISMATCH!")
        else:
            print("✓ Match")


if __name__ == "__main__":
    check_gate_decompositions()
