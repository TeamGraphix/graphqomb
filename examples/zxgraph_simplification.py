"""Basic example of simplifying a ZX-diagram.

By using the `full_reduce` method,
we can remove all the internal Clifford nodes and some non-Clifford nodes from the graph state,
which generates a simpler ZX-diagram.
This example is a simple demonstration of the simplification process.
"""

# %%
from copy import deepcopy

import numpy as np

from graphix_zx.circuit import circuit2graph
from graphix_zx.random_objects import random_circ
from graphix_zx.visualizer import visualize
from graphix_zx.zxgraphstate import ZXGraphState

# %%
circ = random_circ(4, 4)
graph, flow = circuit2graph(circ)
zx_graph = ZXGraphState()
zx_graph.append(graph)

visualize(zx_graph)
print("node | plane | angle (/pi)")
for node in zx_graph.input_nodes:
    print(f"{node} (input)", zx_graph.meas_bases[node].plane, zx_graph.meas_bases[node].angle / np.pi)
for node in zx_graph.physical_nodes - zx_graph.input_nodes - zx_graph.output_nodes:
    print(node, zx_graph.meas_bases[node].plane, zx_graph.meas_bases[node].angle / np.pi)

# %%
zx_graph_smp = deepcopy(zx_graph)
zx_graph_smp.full_reduce()

visualize(zx_graph_smp)
print("node | plane | angle (/pi)")
for node in zx_graph.input_nodes:
    print(f"{node} (input)", zx_graph.meas_bases[node].plane, zx_graph.meas_bases[node].angle / np.pi)
for node in zx_graph_smp.physical_nodes - zx_graph.input_nodes - zx_graph_smp.output_nodes:
    print(node, zx_graph_smp.meas_bases[node].plane, zx_graph_smp.meas_bases[node].angle / np.pi)

# %%
