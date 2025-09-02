"""Visualization tool.

This module provides:

- `visualize` : Visualize the GraphState.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches

from graphix_zx.common import Plane

if TYPE_CHECKING:
    from graphix_zx.graphstate import BaseGraphState


if sys.version_info >= (3, 11):
    from enum import StrEnum

    class ColorMap(StrEnum):
        """Color map for the nodes."""

        XY = "#2ECC71"  # Emerald green
        YZ = "#E74C3C"  # Vibrant red
        XZ = "#3498DB"  # Bright blue
        OUTPUT = "#95A5A6"  # Cool grey

else:
    from enum import Enum

    class ColorMap(str, Enum):
        """Color map for the nodes."""

        XY = "#2ECC71"  # Emerald green
        YZ = "#E74C3C"  # Vibrant red
        XZ = "#3498DB"  # Bright blue
        OUTPUT = "#95A5A6"  # Cool grey


def _setup_figure(node_pos: dict[int, tuple[float, float]]) -> tuple[float, float, float, float, float]:
    """Set up matplotlib figure with proper aspect ratio based on node positions.

    Parameters
    ----------
    node_pos : dict[int, tuple[float, float]]
        Dictionary mapping node indices to (x, y) positions

    Returns
    -------
    tuple[float, float, float, float, float]
        x_min, x_max, y_min, y_max, padding values for plot limits
    """
    if node_pos:
        x_coords = [pos[0] for pos in node_pos.values()]
        y_coords = [pos[1] for pos in node_pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add padding around the graph
        padding = 0.5
        x_range = max(x_max - x_min, 0.5) + 2 * padding  # Minimum range to avoid too narrow plots
        y_range = max(y_max - y_min, 0.5) + 2 * padding

        # Calculate figure size to maintain reasonable aspect ratio
        # Base size of 8 inches, adjust based on content ratio
        base_size = 8.0
        if x_range > y_range:
            fig_width = base_size
            fig_height = base_size * (y_range / x_range)
        else:
            fig_width = base_size * (x_range / y_range)
            fig_height = base_size

        # Ensure minimum figure size for readability
        fig_width = max(fig_width, 4.0)
        fig_height = max(fig_height, 4.0)
    else:
        # Default size if no nodes
        fig_width = fig_height = 8.0
        x_min = x_max = y_min = y_max = 0
        padding = 0.5

    plt.figure(figsize=(fig_width, fig_height))  # pyright: ignore[reportUnknownMemberType]

    # Set equal aspect ratio to ensure circles appear circular, but let the plot adjust limits
    plt.gca().set_aspect("equal")  # pyright: ignore[reportUnknownMemberType]

    return x_min, x_max, y_min, y_max, padding


def visualize(
    graph: BaseGraphState,
    *,
    save: bool = False,
    filename: str | None = None,
    show_node_labels: bool = True,
    node_size: float = 300,
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
    show_node_labels : `bool`, optional
        Whether to show node index labels, by default True
    node_size : `float`, optional
        Size of nodes (scatter size), by default 300
    """
    node_pos = _get_node_positions(graph)

    node_colors = _get_node_colors(graph)

    # Setup figure with proper aspect ratio
    x_min, x_max, y_min, y_max, padding = _setup_figure(node_pos)

    # Calculate radius from scatter size for consistency across all node types
    node_radius = _scatter_size_to_radius(node_size)

    # Draw nodes with special handling for Pauli measurements
    pauli_nodes = _get_pauli_nodes(graph)
    # Calculate adjusted node size for scatter plots to match visual size of Pauli nodes
    # Pauli nodes have outer radius = node_radius, so we need scatter size that visually matches
    adjusted_node_size = node_size * 1.7  # Empirically determined multiplier for visual consistency

    for node in graph.physical_nodes:
        if node in pauli_nodes:
            _draw_pauli_node(node_pos[node], pauli_nodes[node], node_radius)
        else:
            plt.scatter(*node_pos[node], color=node_colors[node], s=adjusted_node_size, zorder=2)  # pyright: ignore[reportUnknownMemberType]

    for edge in graph.physical_edges:
        plt.plot(  # pyright: ignore[reportUnknownMemberType]
            [node_pos[edge[0]][0], node_pos[edge[1]][0]],
            [node_pos[edge[0]][1], node_pos[edge[1]][1]],
            color="black",
            zorder=1,
        )

    # Draw node labels if requested
    if show_node_labels:
        pauli_nodes = _get_pauli_nodes(graph)

        # Draw labels manually for better center alignment
        for node in graph.physical_nodes:
            x, y = node_pos[node]

            # Adjust font size based on node type
            if node in pauli_nodes:
                # For Pauli nodes, use smaller font to fit within inner circle
                # Inner radius is 0.7 * node_radius, so effective area is smaller
                effective_size = adjusted_node_size * 0.5  # Account for inner circle size
                font_size = _calculate_font_size(effective_size)
            else:
                # For regular nodes, use normal calculation
                font_size = _calculate_font_size(adjusted_node_size)

            plt.text(  # pyright: ignore[reportUnknownMemberType]
                x,
                y,
                str(node),
                fontsize=font_size,
                ha="center",  # horizontal alignment: center
                va="center",  # vertical alignment: center
                fontweight="bold",
                color="black",
                zorder=4,  # Above all node patches
            )

    # Set plot limits with proper padding
    if node_pos:
        plt.xlim(x_min - padding, x_max + padding)  # pyright: ignore[reportUnknownMemberType]
        plt.ylim(y_min - padding, y_max + padding)  # pyright: ignore[reportUnknownMemberType]

    if save:
        if filename is None:
            filename = "graph.png"
        plt.savefig(filename)  # pyright: ignore[reportUnknownMemberType]
    plt.show()  # pyright: ignore[reportUnknownMemberType]


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
            nx_graph: nx.Graph[int] = nx.Graph()
            nx_graph.add_nodes_from(internal_nodes)  # pyright: ignore[reportUnknownMemberType]
            nx_graph.add_edges_from(internal_edges)  # pyright: ignore[reportUnknownMemberType]
            internal_pos: dict[int, Any] = nx.spring_layout(nx_graph, k=1, iterations=50)  # pyright: ignore[reportUnknownMemberType]

            # Scale and position internal nodes in the middle
            for node, (x, y) in internal_pos.items():
                pos[node] = (1.0 + x * 0.8, y * 2.0)  # Center between input and output
        else:
            # If no internal edges, arrange internal nodes in a column
            for i, node in enumerate(sorted(internal_nodes)):
                pos[node] = (1.0, float(-i))

    return pos


def _get_node_colors(graph: BaseGraphState) -> dict[int, ColorMap]:
    node_colors: dict[int, ColorMap] = {}
    pauli_nodes = _get_pauli_nodes(graph)

    for node, meas_bases in graph.meas_bases.items():
        # Skip Pauli measurements as they will be handled separately
        if node in pauli_nodes:
            continue

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


def _get_pauli_nodes(graph: BaseGraphState) -> dict[int, str]:
    """Identify nodes with Pauli measurements (θ=0 or π).

    Returns
    -------
    dict[int, str]
        Dictionary mapping node indices to Pauli axis ('X', 'Y', 'Z')
    """
    pauli_nodes: dict[int, str] = {}

    for node, meas_bases in graph.meas_bases.items():
        angle = meas_bases.angle
        # Check if angle is 0, π, π/2, or 3π/2 (within tolerance for floating point)
        tolerance = 1e-10

        if abs(angle) < tolerance or abs(angle - math.pi) < tolerance:
            # X measurement (XY plane, θ=0 or π) or Z measurement (XZ plane, θ=0 or π)
            if meas_bases.plane == Plane.XY:
                pauli_nodes[node] = "X"
            elif meas_bases.plane == Plane.XZ:
                pauli_nodes[node] = "Z"
        elif (
            abs(angle - math.pi / 2) < tolerance or abs(angle - 3 * math.pi / 2) < tolerance
        ) and meas_bases.plane == Plane.YZ:
            # Y measurement (YZ plane, θ=π/2 or 3π/2)
            pauli_nodes[node] = "Y"

    return pauli_nodes


def _scatter_size_to_radius(scatter_size: float) -> float:
    """Convert matplotlib scatter size to radius for consistent patch sizing.

    Parameters
    ----------
    scatter_size : float
        Scatter plot size parameter (area in points^2)

    Returns
    -------
    float
        Equivalent radius for patches
    """
    # Empirically determined conversion factor for visual consistency
    # This gives the base radius that matches scatter plot visual size
    return math.sqrt(scatter_size) * 0.008


def _calculate_font_size(node_size: float) -> int:
    """Calculate appropriate font size based on node size.

    Parameters
    ----------
    node_size : float
        Node size parameter

    Returns
    -------
    int
        Font size for node labels
    """
    # Scale font size with node size, ensuring minimum readability
    base_size = math.sqrt(node_size) * 0.6  # Balanced scaling factor
    return max(6, min(14, int(base_size)))  # Reasonable range for various node sizes


def _draw_pauli_node(pos: tuple[float, float], pauli_axis: str, node_radius: float) -> None:
    """Draw a Pauli measurement node with better visibility.

    Parameters
    ----------
    pos : tuple[float, float]
        Node position (x, y)
    pauli_axis : str
        Pauli axis ('X', 'Y', 'Z')
    node_radius : float
        Radius for the node patches
    """
    x, y = pos

    # Define colors based on Pauli axis
    if pauli_axis == "X":
        # X measurement: XY (green) main with XZ (blue) border
        main_color, border_color = ColorMap.XY, ColorMap.XZ
    elif pauli_axis == "Y":
        # Y measurement: YZ (red) main with XY (green) border
        main_color, border_color = ColorMap.YZ, ColorMap.XY
    elif pauli_axis == "Z":
        # Z measurement: XZ (blue) main with YZ (red) border
        main_color, border_color = ColorMap.XZ, ColorMap.YZ
    else:
        # Fallback to solid color (use calculated scatter size)
        scatter_size = (node_radius / 0.008) ** 2  # Reverse conversion
        plt.scatter(x, y, color="black", s=scatter_size, zorder=2)  # pyright: ignore[reportUnknownMemberType]
        return

    # Use bordered node approach (default - most readable)
    _draw_bordered_node(x, y, main_color, border_color, node_radius)

    # Alternative approaches (uncomment one to try):
    # _draw_shaped_node(x, y, pauli_axis, node_radius)  # Different shapes for X, Y, Z
    # _draw_double_circle_node(x, y, main_color, border_color, node_radius)  # Concentric circles


def _draw_bordered_node(x: float, y: float, main_color: str, border_color: str, node_radius: float) -> None:
    """Draw a node with thick colored border.

    Parameters
    ----------
    x, y : float
        Node center position
    main_color : str
        Main node color
    border_color : str
        Border color
    node_radius : float
        Base radius for the node (should match scatter plot visual size)
    """
    # Use the base radius as the outer radius to match scatter plot size
    outer_radius = node_radius  # Same size as scatter plot
    inner_radius = node_radius * 0.7  # Smaller inner for border effect

    # Draw larger outer circle for border
    outer_circle = patches.Circle(
        (x, y), outer_radius, facecolor=border_color, edgecolor="black", linewidth=1.5, zorder=2
    )
    plt.gca().add_patch(outer_circle)  # pyright: ignore[reportUnknownMemberType]

    # Draw smaller inner circle for main color
    inner_circle = patches.Circle((x, y), inner_radius, facecolor=main_color, edgecolor="black", linewidth=1, zorder=3)
    plt.gca().add_patch(inner_circle)  # pyright: ignore[reportUnknownMemberType]
