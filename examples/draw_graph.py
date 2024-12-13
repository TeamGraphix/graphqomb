"""Simple example to draw a GraphState in graphix-zx."""

# %%
from graphix_zx.flow import construct_dag
from graphix_zx.random_objects import get_random_flow_graph
from graphix_zx.visualizer import visualize

# Create a random flow graph
random_graph, flow = get_random_flow_graph(5, 5)

dag = construct_dag(flow, random_graph)


# Visualize the flow graph
visualize(random_graph, save=False, filename="flow_graph.png")
