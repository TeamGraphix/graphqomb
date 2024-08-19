# %%
import numpy as np

from graphix_zx.circuit import MBQCCircuit, circuit2graph
from graphix_zx.focus_flow import focus_gflow
from graphix_zx.pattern import is_standardized, print_pattern
from graphix_zx.simulator import (
    MBQCCircuitSimulator,
    PatternSimulator,
    SimulatorBackend,
)
from graphix_zx.transpiler import transpile_from_flow

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

# in this example, we generate a pattern from focused flow
focused_flow = focus_gflow(gflow, graphstate)

# %%

# transpile into standardized and signal shifted pattern
pattern = transpile_from_flow(graphstate, focused_flow)
print("pattern is standardized:", is_standardized(pattern))
print("get max space of pattern:", pattern.calc_max_space())
# we can see focus property, aka signal shifting in measurement calculus
print_pattern(pattern)

# %%
# simulate the pattern
simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
simulator.simulate()
state = simulator.get_state()
statevec = state.get_array()

# check by comparing the circuit simulator
circ_simulator = MBQCCircuitSimulator(circuit, SimulatorBackend.StateVector)
circ_simulator.simulate()
circ_state = circ_simulator.get_state().get_array()
inner_product = np.vdot(statevec, circ_state)
print("inner product:", np.abs(inner_product))  # should be 1
