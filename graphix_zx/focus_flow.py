from __future__ import annotations

from typing import Dict, Set

from graphix_zx.interface import GraphState

GFlow = Dict[int, Set[int]]


def oddneighbors(nodes: set[int], graph: GraphState) -> set[int]:
    odd_neighbors: set[int] = set()
    for node in nodes:
        odd_neighbors ^= set(graph.get_neighbors(node))
    return odd_neighbors


def construct_dag(gflow: GFlow, graph: GraphState, check: bool = False) -> dict[int, set[int]]:
    dag = dict()
    outputs = set(graph.get_physical_nodes()) - set(gflow.keys())
    for node in gflow.keys():
        dag[node] = (gflow[node] | oddneighbors(gflow[node], graph)) - {node}
    for output in outputs:
        dag[output] = set()

    if check:
        if not check_dag(dag):
            raise ValueError("Cycle detected in the graph")

    return dag


def check_dag(dag: dict[int, set[int]]) -> bool:
    for node in dag.keys():
        for child in dag[node]:
            if node in dag[child]:
                return False
    return True


def topological_sort_kahn(dag: dict[int, set[int]]) -> list[int]:
    in_degree = {node: 0 for node in dag.keys()}
    for node in dag.keys():
        for child in dag[node]:
            in_degree[child] += 1

    queue = [node for node in in_degree.keys() if in_degree[node] == 0]

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
    else:
        raise ValueError("Cycle detected in the graph")


def focus_gflow(gflow: GFlow, graph: GraphState, meas_planes: dict[int, str]) -> GFlow:
    # TODO: check the validity of the gflow if possible in fast way
    outputs = set(graph.get_physical_nodes()) - set(gflow.keys())

    topo_order = topological_sort_kahn(construct_dag(gflow, graph))

    for output in outputs:
        topo_order.remove(output)

    for target in topo_order:
        gflow = focus(target, gflow, graph, meas_planes, topo_order)

    return gflow


def focus(
    target: int,
    gflow: GFlow,
    graph: GraphState,
    meas_planes: dict[int, str],
    topo_order: list[int],
) -> GFlow:
    k = 0

    s_k = find_non_focused_signals(target, gflow, graph, meas_planes)
    while s_k:
        gflow = update_gflow(target, gflow, s_k, topo_order)
        s_k = find_non_focused_signals(target, gflow, graph, meas_planes)

        k += 1

    return gflow


def find_non_focused_signals(target: int, gflow: GFlow, graph: GraphState, meas_planes: dict[int, str]) -> set[int]:
    non_outputs = {node for node in gflow.keys()}

    s_xy_candidate = oddneighbors(gflow[target], graph) & non_outputs - {target}
    s_xz_candidate = gflow[target] & non_outputs - {target}
    s_yz_candidate = gflow[target] & non_outputs - {target}

    s_xy = {node for node in s_xy_candidate if meas_planes[node] == "XY"}
    s_xz = {node for node in s_xz_candidate if meas_planes[node] == "XZ"}
    s_yz = {node for node in s_yz_candidate if meas_planes[node] == "YZ"}

    return s_xy | s_xz | s_yz


def update_gflow(target: int, gflow: GFlow, s_k: set[int], topo_order: list[int]) -> GFlow:
    minimal_in_s_k = min(s_k, key=lambda node: topo_order.index(node))  # TODO: check
    gflow[target] = gflow[target] ^ gflow[minimal_in_s_k]

    return gflow


def is_focused(gflow: GFlow, graph: GraphState, meas_planes: dict[int, str]):
    focused = True
    outputs = set(graph.get_physical_nodes()) - set(gflow.keys())
    for node in gflow.keys():
        for child in gflow[node]:
            if child in outputs:
                continue
            focused &= (meas_planes[child] == "XY") or (node == child)

        for child in oddneighbors(gflow[node], graph):
            if child in outputs:
                continue
            focused &= (meas_planes[child] != "XY") or (node == child)

    return focused
