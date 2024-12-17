"""Focus flow algorithm.

This module provides:
- topological_sort_kahn: Topological sort using Kahn's algorithm.
- is_focused: Check if a flowlike object is focused.
- focus_gflow: Focus a flowlike object.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from graphix_zx.common import Plane
from graphix_zx.flow import check_causality, construct_dag, oddneighbors

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphix_zx.flow import FlowLike
    from graphix_zx.graphstate import BaseGraphState


def topological_sort_kahn(dag: Mapping[int, AbstractSet[int]]) -> list[int]:
    """Topological sort using Kahn's algorithm.

    Parameters
    ----------
    dag : Mapping[int, AbstractSet[int]]
        directed acyclic graph

    Returns
    -------
    list[int]
        topological order of the dag

    Raises
    ------
    ValueError
        If a cycle is detected
    """
    in_degree = dict.fromkeys(dag, 0)
    for children in dag.values():
        for child in children:
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


def is_focused(gflow: FlowLike, graph: BaseGraphState) -> bool:
    """Check if a flowlike object is focused.

    Parameters
    ----------
    gflow : FlowLike
        flowlike object
    graph : BaseGraphState
        graph state

    Returns
    -------
    bool
        True if the flowlike object is focused, False otherwise
    """
    meas_bases = graph.meas_bases
    outputs = graph.physical_nodes - gflow.keys()

    focused = True
    for node in gflow:
        for child in gflow[node]:
            if child in outputs:
                continue
            focused &= (meas_bases[child].plane == Plane.XY) or (node == child)

        for child in oddneighbors(gflow[node], graph):
            if child in outputs:
                continue
            focused &= (meas_bases[child].plane != Plane.XY) or (node == child)

    return focused


def focus_gflow(gflow: FlowLike, graph: BaseGraphState) -> FlowLike:
    """Focus a flowlike object.

    Parameters
    ----------
    gflow : FlowLike
        flowlike object
    graph : BaseGraphState
        graph state

    Returns
    -------
    FlowLike
        focused flowlike object

    Raises
    ------
    ValueError
        if the flowlike object is not causal with respect to the graph state
    """
    gflow = deepcopy(gflow)
    if not check_causality(graph, gflow):
        msg = "The flowlike object is not causal with respect to the graph state"
        raise ValueError(msg)
    outputs = graph.physical_nodes - gflow.keys()
    topo_order = topological_sort_kahn(construct_dag(gflow, graph))

    for output in outputs:
        topo_order.remove(output)

    for target in topo_order:
        gflow = _focus(target, gflow, graph, topo_order)

    return gflow


def _focus(target: int, gflow: FlowLike, graph: BaseGraphState, topo_order: Sequence[int]) -> FlowLike:
    """Subroutine of the focus_gflow function.

    Parameters
    ----------
    target : int
        target node to be focused
    gflow : FlowLike
        flowlike object
    graph : BaseGraphState
        graph state
    topo_order : Sequence[int]
        topological order of the graph state

    Returns
    -------
    FlowLike
        flowlike object after focusing the target node
    """
    k = 0
    s_k = _find_unfocused_corrections(target, gflow, graph)
    while s_k:
        gflow = _update_gflow(target, gflow, s_k, topo_order)
        s_k = _find_unfocused_corrections(target, gflow, graph)

        k += 1

    return gflow


def _find_unfocused_corrections(target: int, gflow: FlowLike, graph: BaseGraphState) -> set[int]:
    """Subroutine of the _focus function.

    Parameters
    ----------
    target : int
        target node
    gflow : FlowLike
        flowlike object
    graph : BaseGraphState
        graph state

    Returns
    -------
    set[int]
        set of unfocused corrections
    """
    meas_bases = graph.meas_bases
    non_outputs = gflow.keys()

    s_xy_candidate = oddneighbors(gflow[target], graph) & non_outputs - {target}
    s_xz_candidate = gflow[target] & non_outputs - {target}
    s_yz_candidate = gflow[target] & non_outputs - {target}

    s_xy = {node for node in s_xy_candidate if meas_bases[node].plane == Plane.XY}
    s_xz = {node for node in s_xz_candidate if meas_bases[node].plane == Plane.XZ}
    s_yz = {node for node in s_yz_candidate if meas_bases[node].plane == Plane.YZ}

    return s_xy | s_xz | s_yz


def _update_gflow(target: int, gflow: FlowLike, s_k: Iterable[int], topo_order: Sequence[int]) -> FlowLike:
    """Subroutine of the _focus function.

    Parameters
    ----------
    target : int
        target node
    gflow : FlowLike
        flowlike object
    s_k : Iterable[int]
        unfocused correction
    topo_order : Sequence[int]
        topological order of the graph state

    Returns
    -------
    FlowLike
        flowlike object after updating the target node
    """
    minimal_in_s_k = min(s_k, key=topo_order.index)  # TODO: check if this is correct
    gflow[target] ^= gflow[minimal_in_s_k]

    return gflow
