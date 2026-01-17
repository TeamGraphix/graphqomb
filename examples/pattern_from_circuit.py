"""
Pattern Generation from Circuit
===============================

Basic example of generating a pattern from a circuit and simulating it.
"""

# %%
import numpy as np

from graphqomb.circuit import MBQCCircuit, circuit2graph
from graphqomb.pattern import print_pattern
from graphqomb.qompiler import qompile
from graphqomb.simulator import (
    CircuitSimulator,
    PatternSimulator,
    SimulatorBackend,
)

# %%
# generate circuit
circuit = MBQCCircuit(3)
circuit.j(0, 0.5 * np.pi)
circuit.cz(0, 1)
circuit.cz(0, 2)
circuit.j(1, 0.75 * np.pi)
circuit.j(2, 0.25 * np.pi)
circuit.cz(0, 2)
circuit.cz(1, 2)

# %%
# convert circuit to graph, flow, and scheduler
graphstate, gflow, scheduler = circuit2graph(circuit)

# first, qompile it to standardized pattern
pattern = qompile(graphstate, gflow)
print("get max space of pattern:", pattern.max_space)
print_pattern(pattern)

# %%
# simulate the pattern
simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
simulator.simulate()
state = simulator.state
statevec = state.state()

# check by comparing the circuit simulator
circ_simulator = CircuitSimulator(circuit, SimulatorBackend.StateVector)
circ_simulator.simulate()
circ_state = circ_simulator.state.state()
inner_product = np.vdot(statevec, circ_state)
print("inner product:", np.abs(inner_product))  # should be 1

# %%
