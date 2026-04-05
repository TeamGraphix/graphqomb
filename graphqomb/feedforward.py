"""Feedforward correction functions.

This module provides:

- `dag_from_flow`: Construct a directed acyclic graph (DAG) from a flowlike object.
- `inverse_dag_from_dag`: Construct an inverse DAG (node -> dependencies).
- `topo_order_from_inv_dag`: Construct a topological order from an inverse DAG.
- `check_dag`: Check if a directed acyclic graph (DAG) does not contain a cycle.
- `check_flow`: Check if the flowlike object is causal with respect to the graph state.
- `signal_shifting`: Convert the correction maps into more parallel-friendly forms using signal shifting.
- `propagate_correction_map`: Propagate the correction map through a measurement at the target node.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any, TypeGuard

import typing_extensions

from graphqomb.common import Axis, MeasBasis, Plane, determine_pauli_axis
from graphqomb.graphstate import BaseGraphState, odd_neighbors

TOPO_ORDER_CYCLE_ERROR_MSG = "No nodes can be measured; possible cyclic dependency or incomplete preparation."


@dataclass(slots=True)
class _SignalShiftState:
    """Mutable state for signal shifting propagation."""

    output_nodes: AbstractSet[int]
    meas_bases: Mapping[int, MeasBasis]
    xflow: dict[int, set[int]]
    zflow: dict[int, set[int]]
    inv_xflow: dict[int, set[int]]
    inv_zflow: dict[int, set[int]]


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
        target_nodes = (xflow.get(node, set()) | zflow.get(node, set())) - {node}  # remove self-loops
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


def inverse_dag_from_dag(
    dag: Mapping[int, Iterable[int]],
    all_nodes: Iterable[int] | None = None,
) -> dict[int, set[int]]:
    r"""Build inverse DAG (node -> dependencies) from parent->children DAG.

    Parameters
    ----------
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Iterable`\[`int`\]\]
        DAG represented as parent node -> children.
    all_nodes : `collections.abc.Iterable`\[`int`\] | `None`, optional
        Optional full node set to include isolated nodes.

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        Inverse DAG represented as node -> dependencies.
    """
    nodes = set(all_nodes) if all_nodes is not None else set(dag)
    for children in dag.values():
        nodes.update(children)

    inv_dag: dict[int, set[int]] = {node: set() for node in nodes}
    for parent, children in dag.items():
        for child in children:
            inv_dag[child].add(parent)

    return inv_dag


def topo_order_from_inv_dag(inv_dag: Mapping[int, Iterable[int]]) -> list[int]:
    r"""Build topological order from an inverse DAG (node -> dependencies).

    Parameters
    ----------
    inv_dag : `collections.abc.Mapping`\[`int`, `collections.abc.Iterable`\[`int`\]\]
        Inverse DAG where each node maps to the nodes it depends on.

    Returns
    -------
    `list`\[`int`\]
        Topological order from dependencies to dependents.

    Raises
    ------
    RuntimeError
        If topological ordering is not possible due to a cycle.
    """
    try:
        return list(TopologicalSorter(inv_dag).static_order())
    except CycleError as exc:
        raise RuntimeError(TOPO_ORDER_CYCLE_ERROR_MSG) from exc


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


def signal_shifting(
    graph: BaseGraphState, xflow: Mapping[int, AbstractSet[int]], zflow: Mapping[int, AbstractSet[int]] | None = None
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    r"""Convert the correction maps into more parallel-friendly forms using signal shifting.

    Parameters
    ----------
    graph : `BaseGraphState`
        Underlying graph state.
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        Correction map for X.
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Correction map for Z. If `None`, it is generated from xflow by odd neighbors.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `set`\[`int`\]\], `dict`\[`int`, `set`\[`int`\]\]]
        Updated correction maps for X and Z after signal shifting.
    """
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) - {node} for node in xflow}

    dag = dag_from_flow(graph, xflow, zflow)
    output_nodes = set(graph.output_node_indices)
    topo_order = [node for node in reversed(list(TopologicalSorter(dag).static_order())) if node not in output_nodes]

    new_xflow = {k: set(vs) for k, vs in xflow.items()}
    new_zflow = {k: set(vs) for k, vs in zflow.items()}
    state = _SignalShiftState(
        output_nodes=output_nodes,
        meas_bases=graph.meas_bases,
        xflow=new_xflow,
        zflow=new_zflow,
        inv_xflow=_inverse_flow_map(new_xflow),
        inv_zflow=_inverse_flow_map(new_zflow),
    )

    for target_node in topo_order:
        _propagate_correction_map_inplace(target_node=target_node, state=state)

    return new_xflow, new_zflow


def _inverse_flow_map(flow: Mapping[int, AbstractSet[int]]) -> dict[int, set[int]]:
    r"""Build an inverse correction map keyed by child node.

    Returns
    -------
    `dict`\[`int`, `set`\[`int`\]\]
        Inverse correction map.
    """
    inv_flow: dict[int, set[int]] = {}
    for parent, children in flow.items():
        for child in children:
            inv_flow.setdefault(child, set()).add(parent)
    return inv_flow


def _remove_flow_edge(
    flow: MutableMapping[int, set[int]],
    inv_flow: MutableMapping[int, set[int]],
    parent: int,
    child: int,
) -> None:
    """Remove a correction edge from both forward and inverse maps."""
    children = flow.get(parent)
    if children is None or child not in children:
        return

    children.remove(child)
    parents = inv_flow.get(child)
    if parents is None:
        return

    parents.discard(parent)
    if not parents:
        inv_flow.pop(child, None)


def _toggle_flow_edge(
    flow: MutableMapping[int, set[int]],
    inv_flow: MutableMapping[int, set[int]],
    parent: int,
    child: int,
) -> None:
    """Toggle a correction edge in both forward and inverse maps."""
    children = flow.get(parent)
    if children is None:
        children = set()
        flow[parent] = children

    if child in children:
        children.remove(child)
        parents = inv_flow.get(child)
        if parents is None:
            return
        parents.discard(parent)
        if not parents:
            inv_flow.pop(child, None)
        return

    children.add(child)
    inv_flow.setdefault(child, set()).add(parent)


def _shared_parents(
    inv_xflow: Mapping[int, AbstractSet[int]],
    inv_zflow: Mapping[int, AbstractSet[int]],
    target_node: int,
) -> tuple[int, ...]:
    r"""Return the parents appearing in both inverse correction maps.

    Returns
    -------
    `tuple`\[`int`, ...\]
        Parents shared by the two inverse correction maps.
    """
    x_parents = inv_xflow.get(target_node, set())
    z_parents = inv_zflow.get(target_node, set())
    if len(x_parents) > len(z_parents):
        x_parents, z_parents = z_parents, x_parents
    return tuple(parent for parent in x_parents if parent in z_parents)


def _target_parents_for_plane(target_node: int, state: _SignalShiftState) -> tuple[int, ...]:
    r"""Return and detach the relevant parents for a target node.

    Returns
    -------
    `tuple`\[`int`, ...\]
        Parents whose correction edges should propagate through the target node.

    Raises
    ------
    ValueError
        If the measurement plane is unsupported.
    """
    meas_basis = state.meas_bases[target_node]

    if meas_basis.plane == Plane.XY:
        target_parents = tuple(state.inv_zflow.get(target_node, ()))
        for parent in target_parents:
            _remove_flow_edge(state.zflow, state.inv_zflow, parent, target_node)
        return target_parents

    if meas_basis.plane == Plane.YZ:
        target_parents = tuple(state.inv_xflow.get(target_node, ()))
        for parent in target_parents:
            _remove_flow_edge(state.xflow, state.inv_xflow, parent, target_node)
        return target_parents

    if meas_basis.plane == Plane.XZ:
        target_parents = _shared_parents(state.inv_xflow, state.inv_zflow, target_node)
        for parent in target_parents:
            _remove_flow_edge(state.xflow, state.inv_xflow, parent, target_node)
            _remove_flow_edge(state.zflow, state.inv_zflow, parent, target_node)
        return target_parents

    typing_extensions.assert_never(meas_basis.plane)
    msg = f"Unsupported measurement plane: {meas_basis.plane}"
    raise ValueError(msg)


def _propagate_correction_map_inplace(target_node: int, state: _SignalShiftState) -> None:
    r"""Propagate one target node through mutable correction maps.

    Raises
    ------
    ValueError
        If the target node is an output node or the measurement plane is unsupported.
    """
    if target_node in state.output_nodes:
        msg = "Cannot propagate flow for output nodes."
        raise ValueError(msg)

    # Snapshot the target adjacency because propagation mutates the same maps.
    child_xs = tuple(state.xflow.get(target_node, ()))
    child_zs = tuple(state.zflow.get(target_node, ()))
    target_parents = _target_parents_for_plane(target_node, state)

    for parent in target_parents:
        for child_x in child_xs:
            _toggle_flow_edge(state.xflow, state.inv_xflow, parent, child_x)
        for child_z in child_zs:
            _toggle_flow_edge(state.zflow, state.inv_zflow, parent, child_z)


def propagate_correction_map(
    target_node: int,
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    r"""Propagate the correction map through a measurement at the target node.

    Parameters
    ----------
    target_node : `int`
        Node at which the measurement is performed.
    graph : `BaseGraphState`
        Underlying graph state.
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        Correction map for X.
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Correction map for Z. If `None`, it is generated from xflow by odd neighbors.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `set`\[`int`\]\], `dict`\[`int`, `set`\[`int`\]\]]
        Updated correction maps for X and Z after measurement at the target node.


    Notes
    -----
    This function converts the correction maps into more parallel-friendly forms.
    It is equivalent to the signal shifting technique in the measurement calculus.
    """
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) - {node} for node in xflow}

    new_xflow = {k: set(vs) for k, vs in xflow.items()}
    new_zflow = {k: set(vs) for k, vs in zflow.items()}
    state = _SignalShiftState(
        output_nodes=set(graph.output_node_indices),
        meas_bases=graph.meas_bases,
        xflow=new_xflow,
        zflow=new_zflow,
        inv_xflow=_inverse_flow_map(new_xflow),
        inv_zflow=_inverse_flow_map(new_zflow),
    )
    _propagate_correction_map_inplace(target_node=target_node, state=state)

    return new_xflow, new_zflow


def pauli_simplification(  # noqa: C901, PLR0912
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    r"""Simplify the correction maps by removing redundant Pauli corrections.

    Parameters
    ----------
    graph : `BaseGraphState`
        Underlying graph state.
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        Correction map for X.
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        Correction map for Z. If `None`, it is generated from xflow by odd neighbors.

    Returns
    -------
    `tuple`\[`dict`\[`int`, `set`\[`int`\]\], `dict`\[`int`, `set`\[`int`\]\]]
        Updated correction maps for X and Z after simplification.
    """
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) - {node} for node in xflow}

    new_xflow = {k: set(vs) for k, vs in xflow.items()}
    new_zflow = {k: set(vs) for k, vs in zflow.items()}

    inv_xflow: dict[int, set[int]] = {}
    inv_zflow: dict[int, set[int]] = {}
    for k, vs in xflow.items():
        for v in vs:
            inv_xflow.setdefault(v, set()).add(k)
    for k, vs in zflow.items():
        for v in vs:
            inv_zflow.setdefault(v, set()).add(k)

    for node in graph.physical_nodes - graph.output_node_indices.keys():
        meas_basis = graph.meas_bases.get(node)
        if meas_basis is None:
            continue
        meas_axis = determine_pauli_axis(meas_basis)
        if meas_axis is None:
            continue

        if meas_axis == Axis.X:
            for parent in inv_xflow.get(node, set()):
                new_xflow[parent] -= {node}
        elif meas_axis == Axis.Z:
            for parent in inv_zflow.get(node, set()):
                new_zflow[parent] -= {node}
        elif meas_axis == Axis.Y:
            for parent in inv_xflow.get(node, set()) & inv_zflow.get(node, set()):
                new_xflow[parent] -= {node}
                new_zflow[parent] -= {node}

    return new_xflow, new_zflow
