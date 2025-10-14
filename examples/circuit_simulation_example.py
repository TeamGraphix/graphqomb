"""
Circuit Simulation Examples
===========================

This example demonstrates how to create and simulate quantum circuits using graphix-zx,
including basic MBQC circuits, macro gate circuits, and phase gadget circuits.
"""

# %%
import math

import numpy as np

from graphqomb.circuit import Circuit, MBQCCircuit
from graphqomb.gates import CNOT, CZ, H, Rz, X
from graphqomb.simulator import CircuitSimulator, SimulatorBackend

# %%
# Simple MBQC Circuit Example
# ---------------------------
# Create a simple MBQC circuit with basic J gates and CZ gates

circuit = MBQCCircuit(num_qubits=2)

# Add some basic operations
circuit.j(qubit=0, angle=math.pi / 4)
circuit.j(qubit=1, angle=math.pi / 2)
circuit.cz(qubit1=0, qubit2=1)
circuit.j(qubit=0, angle=-math.pi / 4)

print(f"Circuit has {circuit.num_qubits} qubits")
instructions = circuit.instructions()
print(f"Circuit has {len(instructions)} instructions:")
for i, instr in enumerate(instructions):
    print(f"  {i}: {instr}")

# %%
# Simulate the MBQC circuit
simulator = CircuitSimulator(mbqc_circuit=circuit, backend=SimulatorBackend.StateVector)
simulator.simulate()

# Get final state
final_state = simulator.state
print(f"Final state shape: {final_state.state().shape}")
print(f"Final state norm: {final_state.norm():.6f}")

# Print final state vector
state_vec = final_state.state()
print("Final state vector:")
flat_state = state_vec.ravel()
TOLERANCE = 1e-10
for i, amp in enumerate(flat_state):
    if abs(amp) > TOLERANCE:
        print(f"  |{i:02b}⟩: {amp:.6f}")

# %%
# Macro Gate Circuit Example
# --------------------------
# Create a circuit with high-level macro gates

macro_circuit = Circuit(num_qubits=3)

# Add macro gates
macro_circuit.apply_macro_gate(H(qubit=0))  # Hadamard on qubit 0
macro_circuit.apply_macro_gate(X(qubit=1))  # Pauli-X on qubit 1
macro_circuit.apply_macro_gate(CNOT(qubits=(0, 1)))  # CNOT gate
macro_circuit.apply_macro_gate(Rz(qubit=2, angle=math.pi / 3))  # Rz rotation
macro_circuit.apply_macro_gate(CZ(qubits=(1, 2)))  # CZ gate

print(f"Circuit has {macro_circuit.num_qubits} qubits")
macro_instructions = macro_circuit.instructions()
print(f"Circuit has {len(macro_instructions)} macro instructions:")
for i, instr in enumerate(macro_instructions):
    print(f"  {i}: {instr}")

# %%
# Convert to unit gates for simulation
unit_instructions = macro_circuit.unit_instructions()
print(f"Expanded to {len(unit_instructions)} unit instructions:")
for i, instr in enumerate(unit_instructions):
    print(f"  {i}: {instr}")

# %%
# Simulate the macro circuit
simulator = CircuitSimulator(mbqc_circuit=macro_circuit, backend=SimulatorBackend.StateVector)
simulator.simulate()

# Get final state
final_state = simulator.state
print(f"Final state shape: {final_state.state().shape}")
print(f"Final state norm: {final_state.norm():.6f}")

# Print final state vector
state_vec = final_state.state()
print("Final state vector:")
flat_state = state_vec.ravel()
for i, amp in enumerate(flat_state):
    if abs(amp) > TOLERANCE:
        print(f"  |{i:03b}⟩: {amp:.6f}")

# %%
# Phase Gadget Circuit Example
# ----------------------------
# Create a circuit with phase gadgets

circuit = MBQCCircuit(num_qubits=3)

# Prepare initial state
circuit.j(qubit=0, angle=math.pi / 2)
circuit.j(qubit=1, angle=math.pi / 4)

# Add phase gadget
circuit.phase_gadget(qubits=[0, 1, 2], angle=math.pi / 6)

# Final rotations
circuit.j(qubit=2, angle=-math.pi / 3)

print(f"Circuit has {circuit.num_qubits} qubits")
instructions = circuit.instructions()
print(f"Circuit has {len(instructions)} instructions:")
for i, instr in enumerate(instructions):
    print(f"  {i}: {instr}")

# %%
# Simulate the phase gadget circuit
simulator = CircuitSimulator(mbqc_circuit=circuit, backend=SimulatorBackend.StateVector)
simulator.simulate()

# Get final state
final_state = simulator.state
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
