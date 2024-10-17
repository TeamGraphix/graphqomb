"""Visualization tool(alpha).

This module provides:
- visualize: Visualize the GraphState.
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

        XY = "green"
        YZ = "red"
        XZ = "blue"
        OUTPUT = "grey"
else:
    from enum import Enum

    class ColorMap(str, Enum):
        """Color map for the nodes."""

        XY = "green"
        YZ = "red"
        XZ = "blue"
        OUTPUT = "grey"


def visualize(
    graph_state: BaseGraphState,
    topo_order: list[int],
    *,
    save: bool = False,
    filename: str | None = None,
) -> None:
    """Visualize the GraphState.

    note: This is the alpha version of the visualization tool. The visualization tool is still under development.

    Parameters
    ----------
    graph_state : BaseGraphState
        GraphState to visualize.
    topo_order : list[int]
        Topological order of the nodes
    save : bool, optional
        To save as a file or not, by default False
    filename : str | None, optional
        filename of the image, by default None

    Raises
    ------
    ValueError
        If the topological order is not consistent with the physical nodes.
    """
    if graph_state.physical_nodes != set(topo_order):
        msg = "The topological order is not consistent with the physical nodes."
        raise ValueError(msg)

    node_pos = _get_node_positions(graph_state, topo_order)

    node_colors = _get_node_colors(graph_state)

    plt.figure()
    for node in graph_state.physical_nodes:
        plt.scatter(*node_pos[node], color=node_colors[node], s=350)

    for edge in graph_state.physical_edges:
        plt.plot(
            [node_pos[edge[0]][0], node_pos[edge[1]][0]],
            [node_pos[edge[0]][1], node_pos[edge[1]][1]],
            color="black",
        )

    graph = nx.Graph()
    graph.add_nodes_from(graph_state.physical_nodes)
    graph.add_edges_from(graph_state.physical_edges)
    nx.draw_networkx_labels(graph, pos=node_pos, font_size=12)

    if save:
        if filename is None:
            filename = "graph.png"
        plt.savefig(filename)
    plt.show()


def _get_node_positions(graph_state: BaseGraphState, topo_order: list[int]) -> dict[int, tuple[int, int]]:
    pos = {}
    depth_qindecies = {graph_state.q_indices[node]: 0 for node in graph_state.input_nodes}
    for node in topo_order:
        q_index = graph_state.q_indices[node]
        pos[node] = (depth_qindecies[q_index], -q_index)  # (depth, q_index)
        depth_qindecies[q_index] += 1

    return pos


def _get_node_colors(graph_state: BaseGraphState) -> dict[int, ColorMap]:
    node_colors = {}
    for node, plane in graph_state.meas_planes.items():
        if plane == Plane.XY:
            node_colors[node] = ColorMap.XY
        elif plane == Plane.YZ:
            node_colors[node] = ColorMap.YZ
        elif plane == Plane.XZ:
            node_colors[node] = ColorMap.XZ

    for output_node in graph_state.output_nodes:
        node_colors[output_node] = ColorMap.OUTPUT

    return node_colors
