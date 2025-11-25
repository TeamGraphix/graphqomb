"""Utilities for generalized flow (gflow) computation.

This module provides:

- `gflow_wrapper`: Thin adapter around `swiflow.gflow` so that gflow can be computed directly
from a `BaseGraphState` instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
from swiflow import gflow
from swiflow.common import Plane as SfPlane

from graphqomb.common import Plane

if TYPE_CHECKING:
    from typing import Any

    from networkx import Graph as NxGraph

    from graphqomb.graphstate import BaseGraphState

    FlowLike = dict[int, set[int]]


def gflow_wrapper(graphstate: BaseGraphState) -> FlowLike:
    """Utilize `swiflow.gflow` to search gflow.

    Parameters
    ----------
    graphstate : `BaseGraphState`
        graph state to find gflow

    Returns
    -------
    `FlowLike`
        gflow object

    Raises
    ------
    ValueError
        If no gflow is found
    """
    graph: NxGraph[Any] = nx.Graph()
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
            msg = f"No match {plane}"
            raise ValueError(msg)

    gflow_object = gflow.find(
        graph, set(graphstate.input_node_indices), set(graphstate.output_node_indices), swiflow_planes
    )
    if gflow_object is None:
        msg = "No flow found"
        raise ValueError(msg)

    gflow_obj = gflow_object.f

    return {node: {child for child in children if child != node} for node, children in gflow_obj.items()}
