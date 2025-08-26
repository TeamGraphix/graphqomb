"""Simple example to draw a GraphState in graphix-zx."""

# %%
from graphix_zx.feedforward import dag_from_flow
from graphix_zx.random_objects import generate_random_flow_graph
from graphix_zx.visualizer import visualize

# Create a random flow graph
random_graph, flow = generate_random_flow_graph(5, 5)

dag = dag_from_flow(random_graph, flow)


# Visualize the flow graph
visualize(random_graph, save=False, filename="flow_graph.png")
