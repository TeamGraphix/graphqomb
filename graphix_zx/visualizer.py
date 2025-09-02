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
from matplotlib.lines import Line2D

from graphix_zx.common import Plane

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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
    show_legend: bool = True,
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
    show_legend : `bool`, optional
        Whether to show color legend, by default True
    """
    node_pos = _get_node_positions(graph)

    node_colors = _get_node_colors(graph)

    # Setup figure with proper aspect ratio
    x_min, x_max, y_min, y_max, padding = _setup_figure(node_pos)

    # Get current axes for accurate size conversion
    ax = plt.gca()

    # Set plot limits before drawing nodes so coordinate transformation works correctly
    if node_pos:
        plt.xlim(x_min - padding, x_max + padding)  # pyright: ignore[reportUnknownMemberType]
        plt.ylim(y_min - padding, y_max + padding)  # pyright: ignore[reportUnknownMemberType]

    # All nodes use the same base size for consistency

    # Draw nodes with special handling for Pauli measurements
    pauli_nodes = _get_pauli_nodes(graph)

    for node in graph.physical_nodes:
        if node in pauli_nodes:
            # Calculate accurate patch radius for this specific position
            x, y = node_pos[node]
            patch_radius = _scatter_size_to_patch_radius(ax, x, y, node_size)
            _draw_pauli_node(ax, node_pos[node], pauli_nodes[node], patch_radius)
        else:
            # Ensure all nodes have a color, fallback to default if missing
            node_color = node_colors.get(node, ColorMap.OUTPUT)  # Default to output color
            plt.scatter(*node_pos[node], color=node_color, s=node_size, zorder=2)  # pyright: ignore[reportUnknownMemberType]

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

            # All nodes now have the same size, so use same font size calculation
            font_size = _calculate_font_size(node_size)

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

    # Add color legend if requested
    if show_legend:
        _add_legend(graph)
        plt.tight_layout()  # pyright: ignore[reportUnknownMemberType]

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

    # Set colors for all nodes with measurement bases
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

    # Set colors for output nodes (may override measurement colors)
    output_nodes = set(graph.output_node_indices.keys())
    for output_node in output_nodes:
        node_colors[output_node] = ColorMap.OUTPUT

    # Set colors for input nodes (if not already set by measurement bases)
    input_nodes = set(graph.input_node_indices.keys())
    for input_node in input_nodes:
        if input_node not in node_colors:
            # If input node has no measurement basis, use XY as default
            node_colors[input_node] = ColorMap.XY

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


def _scatter_size_to_patch_radius(ax: Axes, x: float, y: float, scatter_size: float) -> float:
    """Convert scatter size to patch radius for precise size matching.

    This function converts matplotlib scatter size (points²) to the equivalent
    radius in data coordinates for patches, ensuring patches appear the same
    size as scatter points regardless of axis scale or DPI.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    x : float
        X position of the node in data coordinates
    y : float
        Y position of the node in data coordinates
    scatter_size : float
        Scatter plot size parameter (area in points²)

    Returns
    -------
    float
        Equivalent radius in data coordinates for patches
    """
    # Convert scatter size (points²) to radius in points
    radius_pt = math.sqrt(scatter_size / math.pi)

    # Convert points to pixels
    radius_px = radius_pt * ax.figure.dpi / 72.0

    # Get transformation from data to display coordinates
    trans = ax.transData
    inv = trans.inverted()

    # Find display coordinates of the node position
    x_disp, y_disp = trans.transform((x, y))

    # Calculate data coordinate offset that corresponds to the pixel radius
    # Use x-direction for radius calculation (assumes roughly circular in display)
    x_offset_data = inv.transform((x_disp + radius_px, y_disp))[0] - x

    return float(abs(x_offset_data))


def _calculate_font_size(node_size: float) -> int:
    """Calculate appropriate font size based on node size that fits within the node.

    Parameters
    ----------
    node_size : float
        Node size parameter (scatter size in points^2)

    Returns
    -------
    int
        Font size for node labels that fit within the node
    """
    # Calculate the diameter of the node in points
    # scatter size is area in points^2, so diameter = 2 * sqrt(area / π)
    node_diameter_points = 2 * math.sqrt(node_size / math.pi)

    # Font size should be roughly 60% of the node diameter to fit comfortably
    # Empirically determined factor for good readability within circular nodes
    font_size = node_diameter_points * 0.4

    # Clamp to reasonable range (minimum for readability, maximum to avoid overflow)
    return max(6, min(16, int(font_size)))


def _draw_pauli_node(ax: Axes, pos: tuple[float, float], pauli_axis: str, node_radius: float) -> None:
    """Draw a Pauli measurement node with hatch patterns.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    pos : tuple[float, float]
        Node position (x, y)
    pauli_axis : str
        Pauli axis ('X', 'Y', 'Z')
    node_radius : float
        Radius for the node patches
    """
    x, y = pos

    # Use unified design for all Pauli measurements
    # Base color depends on the measurement plane, stripe color is contrasting
    if pauli_axis == "X":
        # X measurement: XY plane
        face_color = ColorMap.XY
        edge_color = ColorMap.XZ  # Contrasting color
    elif pauli_axis == "Y":
        # Y measurement: YZ plane
        face_color = ColorMap.YZ
        edge_color = ColorMap.XY  # Contrasting color
    elif pauli_axis == "Z":
        # Z measurement: XZ plane
        face_color = ColorMap.XZ
        edge_color = ColorMap.YZ  # Contrasting color
    else:
        # Fallback to solid color
        circle = patches.Circle((x, y), node_radius, facecolor="black", edgecolor="none", linewidth=0, zorder=2)
        ax.add_patch(circle)
        return

    # Unified hatch pattern for all Pauli measurements
    hatch_pattern = "////////"  # Diagonal stripes for all Pauli nodes

    # Create circle patch with hatch pattern - same size as regular scatter nodes
    circle = patches.Circle(
        (x, y),
        node_radius,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=0,  # No boundary, only hatch pattern
        hatch=hatch_pattern,
        zorder=2,
    )
    ax.add_patch(circle)


def _add_legend(graph: BaseGraphState) -> None:
    """Add color legend to the plot.

    Parameters
    ----------
    graph : BaseGraphState
        GraphState to analyze for legend items
    """
    planes_present, pauli_measurements = _analyze_graph_measurements(graph)
    legend_elements = _create_legend_elements(graph, planes_present, pauli_measurements)

    # Add legend to the plot if there are elements to show
    if legend_elements:
        plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.05, 0.5))  # pyright: ignore[reportUnknownMemberType]


def _analyze_graph_measurements(graph: BaseGraphState) -> tuple[set[Plane], set[str]]:
    """Analyze graph measurements to determine legend content.

    Parameters
    ----------
    graph : BaseGraphState
        GraphState to analyze

    Returns
    -------
    tuple[set[Plane], set[str]]
        Tuple of (planes_present, pauli_measurements)
    """
    planes_present = set()
    pauli_measurements = set()

    for meas_bases in graph.meas_bases.values():
        planes_present.add(meas_bases.plane)

        # Check for Pauli measurements
        angle = meas_bases.angle
        tolerance = 1e-10

        if abs(angle) < tolerance or abs(angle - math.pi) < tolerance:
            if meas_bases.plane == Plane.XY:
                pauli_measurements.add("X")
            elif meas_bases.plane == Plane.XZ:
                pauli_measurements.add("Z")
        elif (
            abs(angle - math.pi / 2) < tolerance or abs(angle - 3 * math.pi / 2) < tolerance
        ) and meas_bases.plane == Plane.YZ:
            pauli_measurements.add("Y")

    return planes_present, pauli_measurements


def _create_legend_elements(
    graph: BaseGraphState, planes_present: set[Plane], pauli_measurements: set[str]
) -> list[Line2D | patches.Circle]:
    """Create legend elements for the plot.

    Parameters
    ----------
    graph : BaseGraphState
        GraphState object
    planes_present : set[Plane]
        Set of measurement planes present in graph
    pauli_measurements : set[str]
        Set of Pauli measurement axes present in graph

    Returns
    -------
    list[Line2D | patches.Circle]
        List of matplotlib legend elements (Line2D and Circle patches)
    """
    legend_elements: list[Line2D | patches.Circle] = []

    # Add legend entries for measurement planes
    if Plane.XY in planes_present:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ColorMap.XY, markersize=8, label="XY measurement")
        )

    if Plane.YZ in planes_present:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ColorMap.YZ, markersize=8, label="YZ measurement")
        )

    if Plane.XZ in planes_present:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ColorMap.XZ, markersize=8, label="XZ measurement")
        )

    # Add legend entry for output nodes if present
    if graph.output_node_indices:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ColorMap.OUTPUT, markersize=8, label="Output node")
        )

    # Add legend entries for Pauli measurements if present
    pauli_entries = []
    for pauli_axis in sorted(pauli_measurements):
        # Create hatch pattern legend entry using Circle patch with same pattern as nodes
        if pauli_axis == "X":
            face_color = ColorMap.XY
            edge_color = ColorMap.XZ
            hatch_pattern = "////////"  # Dense stripes for 50/50 coverage
        elif pauli_axis == "Y":
            face_color = ColorMap.YZ
            edge_color = ColorMap.XY
            hatch_pattern = "////////"  # Dense stripes for 50/50 coverage
        elif pauli_axis == "Z":
            face_color = ColorMap.XZ
            edge_color = ColorMap.YZ
            hatch_pattern = "////////"  # Dense stripes for 50/50 coverage
        else:
            continue

        # Create a circle patch for the legend with same pattern as actual nodes
        circle_patch = patches.Circle(
            (0, 0),
            0.15,  # Small radius for legend
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=0,  # No boundary, only hatch pattern
            hatch=hatch_pattern,
            label=f"Pauli {pauli_axis}",
        )
        pauli_entries.append(circle_patch)

    legend_elements.extend(pauli_entries)

    return legend_elements
