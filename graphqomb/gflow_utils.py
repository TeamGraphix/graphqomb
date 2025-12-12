"""Utilities for generalized flow (gflow) computation.

This module provides:

- `gflow_wrapper`: Thin adapter around ``swiflow.gflow`` so that gflow can be computed directly
    from a `BaseGraphState` instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from swiflow import gflow
from swiflow.common import Plane as SfPlane
from typing_extensions import assert_never

from graphqomb.common import Plane

if TYPE_CHECKING:
    from networkx import Graph as NxGraph

    from graphqomb.graphstate import BaseGraphState


def gflow_wrapper(graphstate: BaseGraphState) -> dict[int, set[int]]:
    """Utilize ``swiflow.gflow`` to search gflow.

    Parameters
    ----------
    graphstate : `BaseGraphState`
        graph state to find gflow

    Returns
    -------
    ``dict[int, set[int]]``
        gflow object

    Raises
    ------
    ValueError
        If no gflow is found

    Notes
    -----
    This wrapper does not support graph states with multiple subgraph structures.
    """
    graph: NxGraph[int] = nx.Graph()
    graph.add_nodes_from(graphstate.physical_nodes)
    graph.add_edges_from(graphstate.physical_edges)

    bases = graphstate.meas_bases
    planes = {node: bases[node].plane for node in bases}
    swiflow_planes: dict[int, SfPlane] = {}
    for node, plane in planes.items():
        if plane == Plane.XY:
            swiflow_planes[node] = SfPlane.XY
        elif plane == Plane.YZ:
            swiflow_planes[node] = SfPlane.YZ
        elif plane == Plane.XZ:
            swiflow_planes[node] = SfPlane.XZ
        else:
            assert_never(plane)

    gflow_object = gflow.find(
        graph, graphstate.input_node_indices.keys(), graphstate.output_node_indices.keys(), swiflow_planes
    )
    if gflow_object is None:
        msg = "No flow found"
        raise ValueError(msg)

    return gflow_object.f
