from __future__ import annotations

from typing import NewType

import networkx as nx

GFlow = NewType("GFlow", dict[int, set[int]])


def oddneighbors(nodes: set[int], graph: nx.Graph) -> set[int]:
    odd_neighbors = set()
    for node in nodes:
        odd_neighbors ^= set(graph.neighbors(node))
    return odd_neighbors


def construct_DAG(gflow: GFlow, graph: nx.Graph) -> dict[int, set[int]]:
    dag = dict()
    outputs = set(graph.nodes) - set(gflow.keys())
    for node in gflow.keys():
        dag[node] = (gflow[node] | oddneighbors(gflow[node], graph)) - {node}
    for output in outputs:
        dag[output] = set()

    return dag


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


def focus_gflow(gflow: GFlow, graph: nx.Graph, meas_planes: dict[int, str]) -> GFlow:
    # TODO: check the validity of the gflow if possible in fast way
    outputs = set(graph.nodes) - set(gflow.keys())

    topo_order = topological_sort_kahn(construct_DAG(gflow, graph))

    for output in outputs:
        topo_order.remove(output)

    for target in topo_order:
        gflow = focus(target, gflow, graph, meas_planes, topo_order)

    return gflow


def focus(
    target: int,
    gflow: GFlow,
    graph: nx.Graph,
    meas_planes: dict[int, str],
    topo_order: list[int],
) -> GFlow:
    k = 0

    Sk = find_non_focused_signals(target, gflow, graph, meas_planes)
    while Sk:
        gflow = update_gflow(target, gflow, Sk, topo_order)
        Sk = find_non_focused_signals(target, gflow, graph, meas_planes)

        k += 1

    return gflow


def find_non_focused_signals(target: int, gflow: GFlow, graph: nx.Graph, meas_planes: dict[int, str]) -> set[int]:
    non_outputs = {node for node in gflow.keys()}

    S_xy_candidate = oddneighbors(gflow[target], graph) & non_outputs - {target}
    S_xz_candidate = gflow[target] & non_outputs - {target}
    S_yz_candidate = gflow[target] & non_outputs - {target}

    S_xy = {node for node in S_xy_candidate if meas_planes[node] == "XY"}
    S_xz = {node for node in S_xz_candidate if meas_planes[node] == "XZ"}
    S_yz = {node for node in S_yz_candidate if meas_planes[node] == "YZ"}

    return S_xy | S_xz | S_yz


def update_gflow(target: int, gflow: GFlow, Sk: set[int], topo_order: list[int]) -> GFlow:
    minimal_in_Sk = min(Sk, key=lambda node: topo_order.index(node))  # TODO: check
    gflow[target] = gflow[target] ^ gflow[minimal_in_Sk]

    return gflow


def is_focused(gflow: GFlow, graph: nx.Graph, meas_planes: dict[int, str]):
    focused = True
    outputs = set(graph.nodes) - set(gflow.keys())
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
