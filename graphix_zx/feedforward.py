"""Feedforward correction functions.

This module provides:

- `dag_from_flow`: Construct a directed acyclic graph (DAG) from a flowlike object.
- `check_dag`: Check if a directed acyclic graph (DAG) does not contain a cycle.
- `check_flow`: Check if the flowlike object is causal with respect to the graph state.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from typing import Any

from graphix_zx.graphstate import BaseGraphState, odd_neighbors

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard


def _is_flow(flowlike: Mapping[int, Any]) -> TypeGuard[Mapping[int, int]]:
    r"""Check if the flowlike object is a flow.

    Parameters
    ----------
    flowlike : `collections.abc.Mapping`\[`int`, `typing.Any`\]
        A flowlike object to check

    Returns
    -------
    `bool`
        True if the flowlike object is a flow, False otherwise
    """
    return all(isinstance(v, int) for v in flowlike.values())


def _is_gflow(flowlike: Mapping[int, Any]) -> TypeGuard[Mapping[int, AbstractSet[int]]]:
    r"""Check if the flowlike object is a GFlow.

    Parameters
    ----------
    flowlike : `collections.abc.Mapping`\[`int`, `typing.Any`\]
        A flowlike object to check

    Returns
    -------
    `bool`
        True if the flowlike object is a GFlow, False otherwise
    """
    return all(isinstance(v, AbstractSet) for v in flowlike.values())


def dag_from_flow(
    graph: BaseGraphState,
    xflow: Mapping[int, int] | Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, int] | Mapping[int, AbstractSet[int]] | None = None,
) -> dict[int, set[int]]:
    r"""Construct a directed acyclic graph (DAG) from a flowlike object.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state
    xflow : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The X correction flow (flow and gflow are included)
    zflow : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        The Z correction flow. If `None`, it is generated from xflow by odd neighbors.

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        The directed acyclic graph

    Raises
    ------
    TypeError
        If the flowlike object is not a Flow or GFlow
    """  # noqa: E501
    dag: dict[int, set[int]] = {}
    output_nodes = set(graph.output_node_indices)
    non_output_nodes = graph.physical_nodes - output_nodes
    if _is_flow(xflow):
        xflow = {node: {xflow[node]} for node in xflow}
    elif _is_gflow(xflow):
        xflow = {node: set(xflow[node]) for node in xflow}
    else:
        msg = "Invalid flowlike object"
        raise TypeError(msg)

    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) for node in xflow}
    elif _is_flow(zflow):
        zflow = {node: {zflow[node]} for node in zflow}
    elif _is_gflow(zflow):
        zflow = {node: set(zflow[node]) for node in zflow}
    else:
        msg = "Invalid zflow object"
        raise TypeError(msg)
    for node in non_output_nodes:
        target_nodes = xflow.get(node, set()) | zflow.get(node, set()) - {node}  # remove self-loops
        dag[node] = target_nodes
    for output in output_nodes:
        dag[output] = set()

    return dag


def check_dag(dag: Mapping[int, Iterable[int]]) -> None:
    r"""Check if a directed acyclic graph (DAG) does not contain a cycle.

    Parameters
    ----------
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Iterable`\[`int`\]\]
        directed acyclic graph

    Raises
    ------
    ValueError
        If the flowlike object is not causal with respect to the graph state
    """
    for node, children in dag.items():
        for child in children:
            if node in dag[child]:
                msg = f"Cycle detected in the graph: {node} -> {child}"
                raise ValueError(msg)


def check_flow(
    graph: BaseGraphState,
    xflow: Mapping[int, int] | Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, int] | Mapping[int, AbstractSet[int]] | None = None,
) -> None:
    r"""Check if the flowlike object is causal with respect to the graph state.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state
    xflow : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The  X correction flow (flow and gflow are included)
    zflow : `collections.abc.Mapping`\[`int`, `int`\] | `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        The  Z correction flow. If `None`, it is generated from xflow by odd neighbors.
    """  # noqa: E501
    dag = dag_from_flow(graph, xflow, zflow)
    check_dag(dag)
