"""Random object generator.

This module provides:
- get_random_flow_graph: Generate a random flow graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import Plane
from graphix_zx.graphstate import GraphState

if TYPE_CHECKING:
    from numpy.random import Generator


def get_random_flow_graph(
    width: int,
    depth: int,
    edge_p: float = 0.5,
    rng: Generator | None = None,
) -> tuple[GraphState, dict[int, set[int]]]:
    """Generate a random flow graph.

    Parameters
    ----------
    width : int
        The width of the graph.
    depth : int
        The depth of the graph.
    edge_p : float, optional
        The probability of adding an edge between two adjacent nodes.
        Default is 0.5.
    rng : Generator, optional
        The random number generator.
        Default is None.

    Returns
    -------
    GraphState
        The generated graph.
    dict[int, set[int]]
        The flow of the graph.
    """
    graph = GraphState()
    flow: dict[int, set[int]] = {}
    num_nodes = 0

    if rng is None:
        rng = np.random.default_rng()

    # input nodes
    for _ in range(width):
        graph.add_physical_node(num_nodes, is_input=True)
        graph.set_meas_plane(num_nodes, Plane.XY)
        graph.set_meas_angle(num_nodes, 0.0)
        num_nodes += 1

    # internal nodes
    for _ in range(depth - 2):
        for _ in range(width):
            graph.add_physical_node(num_nodes)
            graph.set_meas_plane(num_nodes, Plane.XY)
            graph.set_meas_angle(num_nodes, 0.0)
            graph.add_physical_edge(num_nodes - width, num_nodes)
            flow[num_nodes - width] = {num_nodes}
            num_nodes += 1

        for w in range(width - 1):
            if rng.random() < edge_p:
                graph.add_physical_edge(num_nodes - width + w, num_nodes - width + w + 1)

    # output nodes
    for _ in range(width):
        graph.add_physical_node(num_nodes, is_output=True)
        graph.add_physical_edge(num_nodes - width, num_nodes)
        flow[num_nodes - width] = {num_nodes}
        num_nodes += 1

    return graph, flow
