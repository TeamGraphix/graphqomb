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

import numpy as np

from graphix_zx.circuit import circuit2graph
from graphix_zx.random_objects import random_circ
from graphix_zx.visualizer import visualize
from graphix_zx.zxgraphstate import to_zx_graphstate

# %%
# Create a random circuit and convert it to a ZXGraphState
circ = random_circ(4, 4)
graph, flow = circuit2graph(circ)
zx_graph, _ = to_zx_graphstate(graph)
visualize(zx_graph)

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
zx_graph_smp.full_reduce()

# %%
# Simplified graph state after full_reduce.
visualize(zx_graph_smp)
print("node | plane | angle (/pi)")
for node in zx_graph.input_node_indices:
    print(f"{node} (input)", zx_graph.meas_bases[node].plane, zx_graph.meas_bases[node].angle / np.pi)
for node in zx_graph_smp.physical_nodes - set(zx_graph.input_node_indices) - set(zx_graph_smp.output_node_indices):
    print(node, zx_graph_smp.meas_bases[node].plane, zx_graph_smp.meas_bases[node].angle / np.pi)
for node in zx_graph_smp.output_node_indices:
    print(f"{node} (output)", "-", "-")

# %%
# Supplementary Note:
# At first glance, the input/output nodes appear to remain unaffected.
# However, note that a local Clifford operation is actually applied as a result of the action of the full_reduce method.

# If you visualize the circuit after executing the `expand_local_cliffords` method,
# you will see that the additional nodes are now visible on the input/output qubits.
