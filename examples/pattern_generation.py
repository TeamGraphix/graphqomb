"""
Basic Pattern Generation Example
================================

This example demonstrates how to generate a pattern from a graph state
using the graphix-zx library.
"""

# %%
from graphix_zx.pattern import print_pattern
from graphix_zx.qompiler import qompile
from graphix_zx.random_objects import generate_random_flow_graph

# %%
# convert circuit to graph and flow
graphstate, gflow = generate_random_flow_graph(width=3, depth=5)
# first, qompile it to standardized pattern
pattern = qompile(graphstate, gflow)
print("get max space of pattern:", pattern.max_space)
print_pattern(pattern)
