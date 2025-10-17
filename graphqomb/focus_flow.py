"""Focus flow algorithm.

This module provides:

- `is_focused`: Check if a flowlike object is focused.
- `focus_gflow`: Focus a flowlike object.
"""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

from graphqomb.common import Plane
from graphqomb.feedforward import _is_flow, _is_gflow, check_flow, dag_from_flow
from graphqomb.graphstate import odd_neighbors

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


def is_focused(flowlike: Mapping[int, int] | Mapping[int, AbstractSet[int]], graph: BaseGraphState) -> bool:
    r"""Check if a flowlike object is focused.

    Parameters
    ----------
    flowlike : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]`
        flowlike object
    graph : `BaseGraphState`
        graph state

    Returns
    -------
    `bool`
        True if the flowlike object is focused, False otherwise

    Raises
    ------
    TypeError
        If the flowlike object is not a Flow or GFlow
    """  # noqa: E501
    meas_bases = graph.meas_bases
    outputs = set(graph.output_node_indices)

    focused = True
    for node in set(flowlike) - outputs:
        if _is_flow(flowlike):
            for child in graph.neighbors(flowlike[node]) - outputs:
                focused &= node == child
        elif _is_gflow(flowlike):
            for child in flowlike[node]:
                if child in outputs:
                    continue
                focused &= (meas_bases[child].plane == Plane.XY) or (node == child)

            for child in odd_neighbors(flowlike[node], graph):
                if child in outputs:
                    continue
                focused &= (meas_bases[child].plane != Plane.XY) or (node == child)
        else:
            msg = "Invalid flowlike object"
            raise TypeError(msg)

    return focused


def focus_gflow(
    flowlike: Mapping[int, int] | Mapping[int, AbstractSet[int]], graph: BaseGraphState
) -> dict[int, set[int]]:
    r"""Focus a flowlike object.

    Parameters
    ----------
    flowlike : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        flowlike object
    graph : `BaseGraphState`
        graph state

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        focused flowlike object

    Raises
    ------
    TypeError
        If the flowlike object is not a Flow or GFlow
    """  # noqa: E501
    if _is_flow(flowlike):
        flowlike = {key: {value} for key, value in flowlike.items()}
    elif _is_gflow(flowlike):
        flowlike = {key: set(value) for key, value in flowlike.items()}
    else:
        msg = "Invalid flowlike object"
        raise TypeError(msg)
    check_flow(graph, flowlike)
    outputs = graph.physical_nodes - set(flowlike)
    dag = dag_from_flow(graph, flowlike)
    topo_order = list(TopologicalSorter(dag).static_order())
    topo_order.reverse()  # children first

    for output in outputs:
        topo_order.remove(output)

    for target in topo_order:
        flowlike = _focus(target, flowlike, graph, topo_order)

    return flowlike


def _focus(
    target: int, gflow: dict[int, set[int]], graph: BaseGraphState, topo_order: Sequence[int]
) -> dict[int, set[int]]:
    r"""Subroutine of the focus_gflow function.

    Parameters
    ----------
    target : `int`
        target node to be focused
    gflow : `dict`\[`int`, `set`\[`int`\]\]
        gflow object
    graph : `BaseGraphState`
        graph state
    topo_order : `collections.abc.Sequence`\[`int`\]
        topological order of the graph state

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]
        flowlike object after focusing the target node
    """
    k = 0
    s_k = _find_unfocused_corrections(target, gflow, graph)
    while s_k:
        gflow = _update_gflow(target, gflow, s_k, topo_order)
        s_k = _find_unfocused_corrections(target, gflow, graph)

        k += 1

    return gflow


def _find_unfocused_corrections(target: int, gflow: dict[int, set[int]], graph: BaseGraphState) -> set[int]:
    r"""Subroutine of the _focus function.

    Parameters
    ----------
    target : `int`
        target node
    gflow : `dict`\[`int`, `set`\[`int`\]
        flowlike object
    graph : `BaseGraphState`
        graph state

    Returns
    -------
    `set`\[`int`\]
        set of unfocused corrections
    """
    meas_bases = graph.meas_bases
    non_outputs = set(gflow) - set(graph.output_node_indices)

    s_xy_candidate = odd_neighbors(gflow[target], graph) & non_outputs - {target}
    s_xz_candidate = gflow[target] & non_outputs - {target}
    s_yz_candidate = gflow[target] & non_outputs - {target}

    s_xy = {node for node in s_xy_candidate if meas_bases[node].plane == Plane.XY}
    s_xz = {node for node in s_xz_candidate if meas_bases[node].plane == Plane.XZ}
    s_yz = {node for node in s_yz_candidate if meas_bases[node].plane == Plane.YZ}

    return s_xy | s_xz | s_yz


def _update_gflow(
    target: int, gflow: dict[int, set[int]], s_k: Iterable[int], topo_order: Sequence[int]
) -> dict[int, set[int]]:
    r"""Subroutine of the _focus function.

    Parameters
    ----------
    target : `int`
        target node
    gflow : `dict`\[`int`, `set`\[`int`\]
        flowlike object
    s_k : `Iterable`\[`int`\]
        unfocused correction
    topo_order : `collections.abc.Sequence`\[`int`\]
        topological order of the graph state

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]
        gflow object after updating the target node
    """
    minimal_in_s_k = min(s_k, key=topo_order.index)
    gflow[target] ^= gflow[minimal_in_s_k]

    return gflow
