"""Visualization tool.

This module provides:

- `visualize` : Visualize the GraphState.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

from graphix_zx.common import Plane

if TYPE_CHECKING:
    from graphix_zx.graphstate import BaseGraphState


if sys.version_info >= (3, 11):
    from enum import StrEnum

    class ColorMap(StrEnum):
        """Color map for the nodes."""

        XY = "tab:green"
        YZ = "tab:red"
        XZ = "tab:blue"
        OUTPUT = "tab:grey"

else:
    from enum import Enum

    class ColorMap(str, Enum):
        """Color map for the nodes."""

        XY = "tab:green"
        YZ = "tab:red"
        XZ = "tab:blue"
        OUTPUT = "tab:grey"


def visualize(
    graph: BaseGraphState,
    *,
    save: bool = False,
    filename: str | None = None,
) -> None:
    """Visualize the GraphState.

    note: This is the alpha version of the visualization tool. The visualization tool is still under development.

    Parameters
    ----------
    graph : `BaseGraphState`
        GraphState to visualize.
    save : `bool`, optional
        To save as a file or not, by default False
    filename : `str` | None, optional
        filename of the image, by default None
    """
    node_pos = _get_node_positions(graph)

    node_colors = _get_node_colors(graph)

    plt.figure()
    for node in graph.physical_nodes:
        plt.scatter(*node_pos[node], color=node_colors[node], s=350, zorder=2)

    for edge in graph.physical_edges:
        plt.plot(
            [node_pos[edge[0]][0], node_pos[edge[1]][0]],
            [node_pos[edge[0]][1], node_pos[edge[1]][1]],
            color="black",
            zorder=1,
        )

    nx_graph: nx.Graph = nx.Graph()
    nx_graph.add_nodes_from(graph.physical_nodes)
    nx_graph.add_edges_from(graph.physical_edges)
    nx.draw_networkx_labels(nx_graph, pos=node_pos, font_size=12)

    if save:
        if filename is None:
            filename = "graph.png"
        plt.savefig(filename)
    plt.show()


def _get_node_positions(graph: BaseGraphState) -> dict[int, tuple[float, float]]:
    r"""Calculate node positions for visualization with input/output nodes arranged vertically.

    Parameters
    ----------
    graph : `BaseGraphState`
        GraphState to visualize.

    Returns
    -------
    `dict`\[`int`, `tuple`\[`float`, `float`\]\]
        Dictionary mapping node indices to (x, y) positions.
    """
    input_nodes = set(graph.input_node_indices.keys())
    output_nodes = set(graph.output_node_indices.keys())
    internal_nodes = graph.physical_nodes - input_nodes - output_nodes

    pos: dict[int, tuple[float, float]] = {}

    # Arrange input nodes vertically on the left
    for node in sorted(input_nodes, key=lambda n: graph.input_node_indices[n]):
        pos[node] = (0.0, float(-graph.input_node_indices[node]))

    # Arrange output nodes vertically on the right
    max_x = 2.0
    for node in output_nodes:
        pos[node] = (max_x, float(-graph.output_node_indices[node]))

    # For internal nodes, use networkx layout to minimize crossings
    if internal_nodes:
        # Create subgraph of internal nodes and their connections
        internal_edges = [
            edge for edge in graph.physical_edges if edge[0] in internal_nodes and edge[1] in internal_nodes
        ]

        if internal_edges:
            # Use spring layout for internal nodes
            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(internal_nodes)
            nx_graph.add_edges_from(internal_edges)
            internal_pos = nx.spring_layout(nx_graph, k=1, iterations=50)

            # Scale and position internal nodes in the middle
            for node, (x, y) in internal_pos.items():
                pos[node] = (1.0 + x * 0.8, y * 2.0)  # Center between input and output
        else:
            # If no internal edges, arrange internal nodes in a column
            for i, node in enumerate(sorted(internal_nodes)):
                pos[node] = (1.0, float(-i))

    return pos


def _get_node_colors(graph: BaseGraphState) -> dict[int, ColorMap]:
    node_colors = {}
    for node, meas_bases in graph.meas_bases.items():
        if meas_bases.plane == Plane.XY:
            node_colors[node] = ColorMap.XY
        elif meas_bases.plane == Plane.YZ:
            node_colors[node] = ColorMap.YZ
        elif meas_bases.plane == Plane.XZ:
            node_colors[node] = ColorMap.XZ

    output_nodes = set(graph.output_node_indices.keys())
    for output_node in output_nodes:
        node_colors[output_node] = ColorMap.OUTPUT

    return node_colors
