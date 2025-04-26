"""Feedforward correction functions.

This module provides:

- `Flow`: Flow object.
- `GFlow`: GFlow object.
- `FlowLike`: Flowlike object.
- `is_flow`: Check if the flowlike object is a flow.
- `is_gflow`: Check if the flowlike object is a GFlow.
- `dag_from_flow`: Construct a directed acyclic graph (DAG) from a flowlike object.
- `check_causality`: Check if the flowlike object is causal with respect to the graph state.
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

Flow = Mapping[int, int]
GFlow = Mapping[int, AbstractSet[int]]

if sys.version_info >= (3, 10):
    FlowLike = Flow | GFlow
else:
    from typing import Union

    FlowLike = Union[Flow, GFlow]


def is_flow(flowlike: Mapping[int, Any]) -> TypeGuard[Flow]:
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


def is_gflow(flowlike: Mapping[int, Any]) -> TypeGuard[GFlow]:
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


def dag_from_flow(flowlike: FlowLike, graph: BaseGraphState, *, check: bool = True) -> dict[int, set[int]]:
    r"""Construct a directed acyclic graph (DAG) from a flowlike object.

    Parameters
    ----------
    flowlike : `FlowLike`
        A flowlike object
    graph : `BaseGraphState`
        The graph state
    check : `bool`, optional
        Raise an error if a cycle is detected, by default True

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        The directed acyclic graph

    Raises
    ------
    TypeError
        If the flowlike object is not a Flow or GFlow
    ValueError
        If a cycle is detected in the graph
    """
    dag = {}
    outputs = graph.physical_nodes - set(flowlike)
    for node in flowlike:
        if is_flow(flowlike):
            target_nodes = {flowlike[node]} | graph.neighbors(node) - {node}
        elif is_gflow(flowlike):
            target_nodes = set(flowlike[node] | odd_neighbors(flowlike[node], graph) - {node})
        else:
            msg = "Invalid flowlike object"
            raise TypeError(msg)
        dag[node] = target_nodes
    for output in outputs:
        dag[output] = set()

    if check and not _check_dag(dag):
        msg = "Cycle detected in the graph"
        raise ValueError(msg)

    return dag


def _check_dag(dag: Mapping[int, Iterable[int]]) -> bool:
    r"""Check if a directed acyclic graph (DAG) does not contain a cycle.

    Parameters
    ----------
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Iterable`\[`int`\]\]
        directed acyclic graph

    Returns
    -------
    `bool`
        True if the graph is valid, False otherwise
    """
    for node, children in dag.items():
        for child in children:
            if node in dag[child]:
                return False
    return True


def check_causality(
    graph: BaseGraphState,
    flowlike: FlowLike,
) -> bool:
    """Check if the flowlike object is causal with respect to the graph state.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state
    flowlike : `FlowLike`
        The flowlike object

    Returns
    -------
    `bool`
        True if the flowlike object is causal, False otherwise
    """
    dag = dag_from_flow(flowlike, graph, check=False)
    return _check_dag(dag)
