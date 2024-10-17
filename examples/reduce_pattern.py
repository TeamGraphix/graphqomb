"""Demonstrates how to schedule a pattern within limited qubit resources.

We optimize qubit usage by dividing the entire graph into a sequence of subgraphs at each measurement step.

A simple algorithm is employed to determine a measurement order that reduces the pattern space incrementally.
"""

# %%
import numpy as np

from graphix_zx.circuit import MBQCCircuit, circuit2graph
from graphix_zx.pattern import is_standardized, print_pattern
from graphix_zx.qompiler import qompile_from_flow, qompile_from_subgraphs
from graphix_zx.resource_opt import get_reduced_space_meas_order, get_subgraph_sequences
from graphix_zx.simulator import (
    MBQCCircuitSimulator,
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
# convert circuit to graph and flow
graphstate, gflow = circuit2graph(circuit)

# first, naively qompile it to standardized pattern
pattern = qompile_from_flow(graphstate, gflow)
print("pattern is standardized:", is_standardized(pattern))
print("get max space of pattern:", pattern.max_space)
print_pattern(pattern)

# %%
# max space of the standardized pattern is 6, but we can reduce it to 4 by different qompiler strategy
reduced_order = get_reduced_space_meas_order(graphstate, gflow)

graph_seq = get_subgraph_sequences(graphstate, reduced_order)
reduced_pattern = qompile_from_subgraphs(graphstate, graph_seq, gflow)
print("get max space of reduced pattern:", reduced_pattern.max_space)
print_pattern(reduced_pattern)

# %%
# compare the result with the circuit simulator
simulator = PatternSimulator(reduced_pattern, SimulatorBackend.StateVector)
simulator.simulate()
state = simulator.get_state()
statevec = state.get_array()

# check by comparing the circuit simulator
circ_simulator = MBQCCircuitSimulator(circuit, SimulatorBackend.StateVector)
circ_simulator.simulate()
circ_state = circ_simulator.get_state().get_array()
inner_product = np.vdot(statevec, circ_state)
print("inner product:", np.abs(inner_product))  # should be 1
