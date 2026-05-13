"""From Circuit to Executable Pattern
=====================================

This example transpiles an MBQC-native circuit into GraphQOMB's compiler IRs,
then lowers them to an executable pattern and validates it by simulation.
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
# Convert the circuit into GraphQOMB's compiler IRs.
graphstate, xflow, scheduler = circuit2graph(circuit)
print("graph nodes:", len(graphstate.physical_nodes))
print("graph edges:", len(graphstate.physical_edges))
print("feedforward entries:", len(xflow))
print("scheduled slices:", scheduler.num_slices())

# Lower the IRs into an executable pattern using the derived schedule.
pattern = qompile(graphstate, xflow, scheduler=scheduler)
print("pattern depth:", pattern.depth)
print("pattern max space:", pattern.max_space)
print("pattern active volume:", pattern.active_volume)
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
