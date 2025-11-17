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
from copy import deepcopy

import networkx as nx
import numpy as np
import swiflow as sf
from swiflow import gflow

from graphqomb.common import Plane
from graphqomb.graphstate import BaseGraphState
from graphqomb.qompiler import qompile
from graphqomb.random_objects import generate_random_flow_graph
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.visualizer import visualize
from graphqomb.zxgraphstate import ZXGraphState, to_zx_graphstate

FlowLike = dict[int, set[int]]

# %%
# Create a random graph state with flow
graph, flow = generate_random_flow_graph(width=3, depth=4, edge_p=0.5)
zx_graph, _ = to_zx_graphstate(graph)
visualize(zx_graph)


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


# %%
print_boundary_lcs(zx_graph)

# %%
# Initial graph state before simplification
print("node | plane | angle (/pi)")
for node in zx_graph.input_node_indices:
    print(f"{node} (input)", zx_graph.meas_bases[node].plane, zx_graph.meas_bases[node].angle / np.pi)
for node in zx_graph.physical_nodes - set(zx_graph.input_node_indices) - set(zx_graph.output_node_indices):
    print(node, zx_graph.meas_bases[node].plane, zx_graph.meas_bases[node].angle / np.pi)
for node in zx_graph.output_node_indices:
    print(f"{node} (output)", "-", "-")


# %%
# Simplify the graph state by full_reduce method
zx_graph_smp = deepcopy(zx_graph)
# zx_graph_smp.full_reduce()
zx_graph_smp.remove_cliffords()

# %%
# Simplified graph state after full_reduce.
visualize(zx_graph_smp)
print("node | plane | angle (/pi)")
for node in zx_graph_smp.input_node_indices:
    print(f"{node} (input)", zx_graph_smp.meas_bases[node].plane, zx_graph_smp.meas_bases[node].angle / np.pi)
for node in zx_graph_smp.physical_nodes - set(zx_graph_smp.input_node_indices) - set(zx_graph_smp.output_node_indices):
    print(node, zx_graph_smp.meas_bases[node].plane, zx_graph_smp.meas_bases[node].angle / np.pi)
for node in zx_graph_smp.output_node_indices:
    print(f"{node} (output)", "-", "-")

# %%
print(zx_graph_smp.input_node_indices, "\n", zx_graph_smp.output_node_indices, "\n", zx_graph_smp.physical_edges)
print_boundary_lcs(zx_graph_smp)

# %%
# Supplementary Note:
# At first glance, the input/output nodes appear to remain unaffected.
# However, note that a local Clifford operation is actually applied as a result of the action of the full_reduce method.

# If you visualize the graph state after executing the `expand_local_cliffords` method,
# you will see additional nodes connected to the former input/output nodes,
# indicating that local Clifford operations on the input/output nodes have been expanded into the graph.


# %%
# Let us compare the graph state before and after simplification.
# First, we simulate the original graph state and get the resulting statevector.
pattern = qompile(zx_graph, flow)
sim = PatternSimulator(pattern, backend=SimulatorBackend.StateVector)
sim.simulate()
statevec_original = sim.state

# %%
# Next, we simulate the pattern obtained from the simplified graph state.
# Here is a wrapper function to find gflow using swiflow.gflow.


def gflow_wrapper(graphstate: BaseGraphState) -> FlowLike:
    """Wrapper function for swiflow.gflow

    Parameters
    ----------
    graphstate : `BaseGraphState`
        graph state to find gflow

    Returns
    -------
    `FlowLike`
        gflow object

    Raises
    ------
    ValueError
        If no gflow is found
    """
    graph = nx.Graph()
    graph.add_nodes_from(graphstate.physical_nodes)
    graph.add_edges_from(graphstate.physical_edges)

    bases = graphstate.meas_bases
    planes = {node: bases[node].plane for node in bases}
    swiflow_planes = {}
    for node, plane in planes.items():
        if plane == Plane.XY:
            swiflow_planes[node] = sf.common.Plane.XY
        elif plane == Plane.YZ:
            swiflow_planes[node] = sf.common.Plane.YZ
        elif plane == Plane.XZ:
            swiflow_planes[node] = sf.common.Plane.XZ
        else:
            msg = f"No match {plane}"
            raise ValueError(msg)

    gflow_object = gflow.find(
        graph, set(graphstate.input_node_indices), set(graphstate.output_node_indices), swiflow_planes
    )
    print(gflow_object)
    if gflow_object is None:
        msg = "No flow found"
        raise ValueError(msg)

    gflow_obj = gflow_object.f

    return {node: {child for child in children if child != node} for node, children in gflow_obj.items()}


# %%
# Note that we need to call the `expand_local_cliffords` method before generating the pattern to get the gflow.

zx_graph_smp.expand_local_cliffords()
print("input_node_indices: ", set(zx_graph_smp.input_node_indices))
print("output_node_indices: ", set(zx_graph_smp.output_node_indices))
print("local_cliffords: ", zx_graph_smp.local_cliffords)
print("node | plane | angle (/pi)")
for node in zx_graph_smp.input_node_indices:
    print(f"{node} (input)", zx_graph_smp.meas_bases[node].plane, zx_graph_smp.meas_bases[node].angle / np.pi)

for node in zx_graph_smp.physical_nodes - set(zx_graph_smp.input_node_indices) - set(zx_graph_smp.output_node_indices):
    print(node, zx_graph_smp.meas_bases[node].plane, zx_graph_smp.meas_bases[node].angle / np.pi)

for node in zx_graph_smp.output_node_indices:
    print(f"{node} (output)", "-", "-")
gflow_smp = gflow_wrapper(zx_graph_smp)
visualize(zx_graph_smp)
print_boundary_lcs(zx_graph_smp)

# %%
# Now we can compile the simplified graph state into a measurement pattern and simulate it.
pattern_smp = qompile(zx_graph_smp, gflow_smp)
sim_smp = PatternSimulator(pattern_smp, backend=SimulatorBackend.StateVector)
sim_smp.simulate()
print(pattern_smp)

# %%
statevec_smp = sim_smp.state
# %%
# normalization check
print("norm of original statevector:", np.linalg.norm(statevec_original.state()))
print("norm of simplified statevector:", np.linalg.norm(statevec_smp.state()))

# %%
# Finally, we compare the expectation values of random observables before and after simplification.
rng = np.random.default_rng()
rand_mat = rng.random((2, 2)) + 1j * rng.random((2, 2))
rand_mat += rand_mat.T.conj()
exp = statevec_original.expectation(rand_mat, [2])
exp_cr = statevec_smp.expectation(rand_mat, [2])
print("Expectation values for rand_mat\n===============================")
print("rand_mat: \n", rand_mat)

print("Original: ", exp, "\nAfter simplification: ", exp_cr)
print("norm: ", np.linalg.norm(statevec_original.state()), np.linalg.norm(statevec_smp.state()))
print("data shape: ", statevec_original.state().shape, statevec_smp.state().shape)
psi_org = statevec_original.state()
psi_smp = statevec_smp.state()
print("inner product: ", np.sqrt(np.abs(np.vdot(psi_org, psi_smp))))

# %%
