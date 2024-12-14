"""Demonstration of feedforward optimization."""

# %%
import numpy as np

from graphix_zx.circuit import MBQCCircuit, circuit2graph
from graphix_zx.feedforward import gen_delayed_set, map2set
from graphix_zx.flow import oddneighbors
from graphix_zx.pattern import is_standardized, print_pattern
from graphix_zx.qompiler import qompile
from graphix_zx.simulator import (
    MBQCCircuitSimulator,
    PatternSimulator,
    SimulatorBackend,
)
from graphix_zx.visualizer import visualize

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
visualize(graphstate)

# %%
xflow = gflow
zflow = {node: oddneighbors(gflow[node], graphstate) for node in gflow}

print("xflow:", xflow)
print("-" * 20)
print("zflow:", zflow)

# %%
print("The original correction sets:")
x_corrections = map2set(graphstate, xflow)
z_corrections = map2set(graphstate, zflow)

print("x_corrections:", x_corrections)
print("-" * 20)
print("z_corrections:", z_corrections)

# %%

print("The delayed correction sets:")
x_corrections_d, z_corrections_d = gen_delayed_set(graphstate, xflow, zflow)

print("x_corrections:", x_corrections_d)
print("-" * 20)
print("z_corrections:", z_corrections_d)

# %%

dag = {node: xflow[node] | zflow[node] - {node} for node in graphstate.physical_nodes - graphstate.output_nodes}
for output in graphstate.output_nodes:
    dag[output] = set()

# qompile into standardized and signal shifted pattern
pattern = qompile(graphstate, x_corrections_d, z_corrections_d, dag)
print("pattern is standardized:", is_standardized(pattern))
print("get max space of pattern:", pattern.max_space)
# we can see focus property, aka signal shifting in measurement calculus
print_pattern(pattern)

pattern.mark_runnable()
pattern.mark_deterministic()
fixed_pattern = pattern.freeze()

# %%
# simulate the pattern
simulator = PatternSimulator(fixed_pattern, SimulatorBackend.StateVector)
simulator.simulate()
state = simulator.get_state()
statevec = state.get_array()

# check by comparing the circuit simulator
circ_simulator = MBQCCircuitSimulator(circuit, SimulatorBackend.StateVector)
circ_simulator.simulate()
circ_state = circ_simulator.get_state().get_array()
inner_product = np.vdot(statevec, circ_state)
print("inner product:", np.abs(inner_product))  # should be 1
