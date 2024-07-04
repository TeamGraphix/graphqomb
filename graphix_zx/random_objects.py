from __future__ import annotations

import numpy as np
from numpy.random import Generator

from graphix_zx.interface import BasicGraphState


def get_random_flow_graph(
    width: int,
    depth: int,
    edge_p: float = 0.5,
    rng: Generator = np.random.default_rng(),
) -> tuple[BasicGraphState, dict[int, set[int]]]:
    """Generate a random flow graph."""
    graph = BasicGraphState()
    flow: dict[int, set[int]] = dict()
    num_nodes = 0

    # input nodes
    for w in range(width):
        graph.add_physical_node(num_nodes, is_input=True)
        graph.set_meas_plane(num_nodes, "XY")
        graph.set_meas_angle(num_nodes, 0.0)
        num_nodes += 1

    # internal nodes
    for _ in range(depth - 2):
        for w in range(width):
            graph.add_physical_node(num_nodes)
            graph.set_meas_plane(num_nodes, "XY")
            graph.set_meas_angle(num_nodes, 0.0)
            graph.add_physical_edge(num_nodes - width, num_nodes)
            flow[num_nodes - width] = {num_nodes}
            num_nodes += 1

        for w in range(width - 1):
            if rng.random() < edge_p:
                graph.add_physical_edge(num_nodes - width + w, num_nodes - width + w + 1)

    # output nodes
    for w in range(width):
        graph.add_physical_node(num_nodes, is_output=True)
        graph.add_physical_edge(num_nodes - width, num_nodes)
        flow[num_nodes - width] = {num_nodes}
        num_nodes += 1

    return graph, flow
