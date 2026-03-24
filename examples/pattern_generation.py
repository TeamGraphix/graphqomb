"""Lowering a Labelled Graph State
==================================

This example starts from a labelled graph state and an explicit feedforward map,
then lowers them to an executable MBQC pattern.
"""

# %%
from graphqomb.pattern import print_pattern
from graphqomb.qompiler import qompile
from graphqomb.random_objects import generate_random_flow_graph

# %%
# Generate a labelled graph state and its X-correction map.
graphstate, xflow = generate_random_flow_graph(width=3, depth=5)

print("graph nodes:", len(graphstate.physical_nodes))
print("graph edges:", len(graphstate.physical_edges))

# Lower the graph/feedforward IRs to an executable pattern.
pattern = qompile(graphstate, xflow)
print("pattern depth:", pattern.depth)
print("pattern max space:", pattern.max_space)
print_pattern(pattern)
