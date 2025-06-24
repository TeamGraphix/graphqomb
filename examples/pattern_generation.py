"""Basic example of generating a pattern from a circuit and simulating it."""

# %%
from graphix_zx.pattern import print_pattern
from graphix_zx.qompiler import qompile_from_flow
from graphix_zx.random_objects import generate_random_flow_graph

# %%
# convert circuit to graph and flow
graphstate, gflow = generate_random_flow_graph(width=3, depth=5)
# first, qompile it to standardized pattern
pattern = qompile_from_flow(graphstate, gflow)
print("get max space of pattern:", pattern.max_space)
print_pattern(pattern)
