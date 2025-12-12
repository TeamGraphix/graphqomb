"""
Basic example of simplifying a ZX-diagram.
==========================================

By using the full_reduce method,
we can remove all the internal Clifford nodes and some non-Clifford nodes from the graph state,
which generates a simpler ZX-diagram.
This example is a simple demonstration of the simplification process.

Note that as a result of the simplification, local Clifford operations are applied to the input/output nodes.
"""

# %%

from __future__ import annotations

from copy import deepcopy

import numpy as np

from graphqomb.gflow_utils import gflow_wrapper
from graphqomb.qompiler import qompile
from graphqomb.random_objects import generate_random_flow_graph
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.visualizer import visualize
from graphqomb.zxgraphstate import ZXGraphState

# %%
# Prepare an initial random graph state with flow
graph, flow = generate_random_flow_graph(width=3, depth=4, edge_p=0.5)
zx_graph, _ = ZXGraphState.from_base_graph_state(graph)
visualize(zx_graph)

# %%
# We can compile the graph state into a measurement pattern, simulate it, and get the resulting statevector.
pattern = qompile(zx_graph, flow)
sim = PatternSimulator(pattern, backend=SimulatorBackend.StateVector)
sim.simulate()
statevec_original = sim.state


# %%
def print_boundary_lcs(zxgraph: ZXGraphState) -> None:
    lc_map = zxgraph.local_cliffords
    for node in zxgraph.input_node_indices | zxgraph.output_node_indices:
        # check lc on input and output nodes
        lc = lc_map.get(node, None)
        if lc is not None:
            if node in zxgraph.input_node_indices:
                print(f"Input node {node} has local Clifford: alpha={lc.alpha}, beta={lc.beta}, gamma={lc.gamma}")
            else:
                print(f"Output node {node} has local Clifford: alpha={lc.alpha}, beta={lc.beta}, gamma={lc.gamma}")
        else:
            print(f"Node {node} has no local Clifford.")


def print_meas_bses(graph: ZXGraphState) -> None:
    print("node | plane | angle (/pi)")
    for node in graph.input_node_indices:
        print(f"{node} (input)", graph.meas_bases[node].plane, graph.meas_bases[node].angle / np.pi)
    for node in graph.physical_nodes - set(graph.input_node_indices) - set(graph.output_node_indices):
        print(node, graph.meas_bases[node].plane, graph.meas_bases[node].angle / np.pi)
    for node in graph.output_node_indices:
        print(f"{node} (output)", "-", "-")


# %%
print_boundary_lcs(zx_graph)

# %%
# Initial graph state before simplification
print_meas_bses(zx_graph)


# %%
# Simplify the graph state by full_reduce method
zx_graph_smp = deepcopy(zx_graph)
zx_graph_smp.full_reduce()

# %%
# Simplified graph state after full_reduce.
visualize(zx_graph_smp)
print_meas_bses(zx_graph_smp)
print_boundary_lcs(zx_graph_smp)


# %%
# Let us compare the graph state before and after simplification.
# We simulate the pattern obtained from the simplified graph state.
# Note that we need to call the `expand_local_cliffords` method before generating the pattern to get the gflow.

zx_graph_smp.expand_local_cliffords()
print("input_node_indices: ", set(zx_graph_smp.input_node_indices))
print("output_node_indices: ", set(zx_graph_smp.output_node_indices))
print("local_cliffords: ", zx_graph_smp.local_cliffords)

print_meas_bses(zx_graph_smp)
visualize(zx_graph_smp)
print_boundary_lcs(zx_graph_smp)

# %%
# Now we can obtain the gflow for the simplified graph state.
# Then, we compile the simplified graph state into a measurement pattern,
# simulate it, and get the resulting statevector.

# NOTE:
# gflow_wrapper does not support graph states with multiple subgraph structures in the gflow search wrapper below.
# Hence, in case you fail, ensure that the simplified graph state consists of a single connected component.
# To calculate the graph states with multiple subgraph structures,
# you need to calculate gflow for each connected component separately.
gflow_smp = gflow_wrapper(zx_graph_smp)
pattern_smp = qompile(zx_graph_smp, gflow_smp)
sim_smp = PatternSimulator(pattern_smp, backend=SimulatorBackend.StateVector)
sim_smp.simulate()

# %%
statevec_smp = sim_smp.state
# %%
# normalization check
print("norm of original statevector:", np.linalg.norm(statevec_original.state()))
print("norm of simplified statevector:", np.linalg.norm(statevec_smp.state()))

# %%
# Finally, we compare the expectation values of random observables before and after simplification.
rng = np.random.default_rng()
for i in range(len(zx_graph.input_node_indices)):
    rand_mat = rng.random((2, 2)) + 1j * rng.random((2, 2))
    rand_mat += rand_mat.T.conj()
    exp = statevec_original.expectation(rand_mat, [i])
    exp_cr = statevec_smp.expectation(rand_mat, [i])
    print("Expectation values for rand_mat\n===============================")
    print("rand_mat: \n", rand_mat)
    print("Original: \t\t", exp)
    print("After simplification: \t", exp_cr)

print("norm: ", np.linalg.norm(statevec_original.state()), np.linalg.norm(statevec_smp.state()))
print("data shape: ", statevec_original.state().shape, statevec_smp.state().shape)
psi_org = statevec_original.state()
psi_smp = statevec_smp.state()
print("inner product: ", np.abs(np.vdot(psi_org, psi_smp)))

# %%
