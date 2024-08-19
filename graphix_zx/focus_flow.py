"""focusing gflow is equivalent to shifting signals on a pattern"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.common import Plane
from graphix_zx.flow import construct_dag, oddneighbors

if TYPE_CHECKING:
    from graphix_zx.flow import FlowLike
    from graphix_zx.graphstate import BaseGraphState


def topological_sort_kahn(dag: dict[int, set[int]]) -> list[int]:
    in_degree = {node: 0 for node in dag}
    for node in dag:
        for child in dag[node]:
            in_degree[child] += 1

    queue = [node for node in in_degree if in_degree[node] == 0]

    topo_order = []

    while queue:
        node = queue.pop(0)
        topo_order.append(node)

        for child in dag[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(topo_order) == len(dag):
        return topo_order
    msg = "Cycle detected in the graph"
    raise ValueError(msg)


def focus_gflow(gflow: FlowLike, graph: BaseGraphState) -> FlowLike:
    # TODO: check the validity of the gflow if possible in fast way
    outputs = set(graph.physical_nodes) - set(gflow)
    topo_order = topological_sort_kahn(construct_dag(gflow, graph))

    for output in outputs:
        topo_order.remove(output)

    for target in topo_order:
        gflow = focus(target, gflow, graph, topo_order)

    return gflow


def focus(
    target: int,
    gflow: FlowLike,
    graph: BaseGraphState,
    topo_order: list[int],
) -> FlowLike:
    k = 0
    s_k = find_non_focused_signals(target, gflow, graph)
    while s_k:
        gflow = update_gflow(target, gflow, s_k, topo_order)
        s_k = find_non_focused_signals(target, gflow, graph)

        k += 1

    return gflow


def find_non_focused_signals(target: int, gflow: FlowLike, graph: BaseGraphState) -> set[int]:
    meas_planes = graph.meas_planes
    non_outputs = set(gflow)

    s_xy_candidate = oddneighbors(gflow[target], graph) & non_outputs - {target}
    s_xz_candidate = gflow[target] & non_outputs - {target}
    s_yz_candidate = gflow[target] & non_outputs - {target}

    s_xy = {node for node in s_xy_candidate if meas_planes[node] == Plane.XY}
    s_xz = {node for node in s_xz_candidate if meas_planes[node] == Plane.ZX}
    s_yz = {node for node in s_yz_candidate if meas_planes[node] == Plane.YZ}

    return s_xy | s_xz | s_yz


def update_gflow(target: int, gflow: FlowLike, s_k: set[int], topo_order: list[int]) -> FlowLike:
    minimal_in_s_k = min(s_k, key=lambda node: topo_order.index(node))  # TODO: check
    gflow[target] = gflow[target] ^ gflow[minimal_in_s_k]

    return gflow


def is_focused(gflow: FlowLike, graph: BaseGraphState) -> bool:
    meas_planes = graph.meas_planes
    outputs = set(graph.physical_nodes) - set(gflow)

    focused = True
    for node in gflow:
        for child in gflow[node]:
            if child in outputs:
                continue
            focused &= (meas_planes[child] == Plane.XY) or (node == child)

        for child in oddneighbors(gflow[node], graph):
            if child in outputs:
                continue
            focused &= (meas_planes[child] != Plane.XY) or (node == child)

    return focused
