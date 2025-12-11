"""Utilities for generalized flow (gflow) computation.

This module provides:

- `gflow_wrapper`: Thin adapter around ``swiflow.gflow`` so that gflow can be computed directly
    from a `BaseGraphState` instance.
- `_EQUIV_MEAS_BASIS_MAP`: A mapping between equivalent measurement bases used to improve gflow finding performance.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import networkx as nx
from swiflow import gflow
from swiflow.common import Plane as SfPlane
from typing_extensions import assert_never

from graphqomb.common import Plane, PlannerMeasBasis

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


#: Mapping between equivalent measurement bases.
#:
#: This map is used to replace a measurement basis by an equivalent one
#: to improve gflow search performance.
#:
#: Key:
#:   ``(Plane, angle)`` where angle is in radians.
#: Value:
#:   :class:`~graphqomb.common.PlannerMeasBasis`.
_EQUIV_MEAS_BASIS_MAP: dict[tuple[Plane, float], PlannerMeasBasis] = {
    # (XY, 0) <-> (XZ, pi/2)
    (Plane.XY, 0.0): PlannerMeasBasis(Plane.XZ, 0.5 * math.pi),
    (Plane.XZ, 0.5 * math.pi): PlannerMeasBasis(Plane.XY, 0.0),
    # (XY, pi/2) <-> (YZ, pi/2)
    (Plane.XY, 0.5 * math.pi): PlannerMeasBasis(Plane.YZ, 0.5 * math.pi),
    (Plane.YZ, 0.5 * math.pi): PlannerMeasBasis(Plane.XY, 0.5 * math.pi),
    # (XY, -pi/2) == (XY, 3pi/2) <-> (YZ, 3pi/2)
    (Plane.XY, 1.5 * math.pi): PlannerMeasBasis(Plane.YZ, 1.5 * math.pi),
    (Plane.YZ, 1.5 * math.pi): PlannerMeasBasis(Plane.XY, 1.5 * math.pi),
    # (XY, pi) <-> (XZ, -pi/2) == (XZ, 3pi/2)
    (Plane.XY, math.pi): PlannerMeasBasis(Plane.XZ, 1.5 * math.pi),
    (Plane.XZ, 1.5 * math.pi): PlannerMeasBasis(Plane.XY, math.pi),
    # (XZ, 0) <-> (YZ, 0)
    (Plane.XZ, 0.0): PlannerMeasBasis(Plane.YZ, 0.0),
    (Plane.YZ, 0.0): PlannerMeasBasis(Plane.XZ, 0.0),
    # (XZ, pi) <-> (YZ, pi)
    (Plane.XZ, math.pi): PlannerMeasBasis(Plane.YZ, math.pi),
    (Plane.YZ, math.pi): PlannerMeasBasis(Plane.XZ, math.pi),
}
