"""Feedforward correction functions."""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from graphix_zx.common import Plane
from graphix_zx.graphstate import BaseGraphState

CorrectionMap = Mapping[int, AbstractSet[int]]
CorrectionSet = Mapping[int, AbstractSet[int]]


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


def map2set(graph: BaseGraphState, correction_map: CorrectionMap) -> dict[int, set[int]]:
    """Convert a correction map to a correction set.

    Parameters
    ----------
    graph : BaseGraphState
        A graph state
    correction_map : CorrectionMap
        A correction map

    Returns
    -------
    dict[int, set[int]]
        A correction set
    """
    correction_set: dict[int, set[int]] = {node: set() for node in graph.physical_nodes}
    for node, map_nodes in correction_map.items():
        for map_node in map_nodes:
            correction_set[map_node] |= {node}

    # remove self-correction
    for node in graph.physical_nodes:
        correction_set[node] -= {node}
    return correction_set


def check_causality(
    correction_map: CorrectionMap,
) -> bool:
    """Check if the CorrectionMap object is causal.

    Parameters
    ----------
    correction_map : CorrectionMap
        A correction map

    Returns
    -------
    bool
        True if the CorrectionMap object is causal, False otherwise
    """
    try:
        topological_sort_kahn(correction_map)
    except ValueError:
        return False
    else:
        return True


def gen_delayed_set(
    graph: BaseGraphState, x_cm: CorrectionMap, z_cm: CorrectionMap
) -> tuple[CorrectionSet, CorrectionSet]:
    """Generate delayed correction sets.

    Parameters
    ----------
    graph : BaseGraphState
        A graph state
    x_cm : CorrectionMap
        Pauli X correction map
    z_cm : CorrectionMap
        Pauli Z correction map

    Returns
    -------
    tuple[CorrectionSet, CorrectionSet]
        delayed correction sets, (X, Z)

    Raises
    ------
    ValueError
        If a measurement plane is invalid
    """
    dag = {node: x_cm[node] | z_cm[node] - {node} for node in graph.physical_nodes - graph.output_nodes}
    for output in graph.output_nodes:
        dag[output] = set()
    topo_order = topological_sort_kahn(dag)

    x_cs = map2set(graph, x_cm)
    z_cs = map2set(graph, z_cm)

    for node in topo_order:
        if node in graph.output_nodes:
            continue
        if graph.meas_bases[node].plane == Plane.XY:
            not_dependency = z_cs[node]
        elif graph.meas_bases[node].plane in {Plane.XZ, Plane.YZ}:
            not_dependency = x_cs[node]
        else:
            msg = "Invalid measurement plane"
            raise ValueError(msg)
        for node_map in x_cm[node]:
            x_cs[node_map] ^= not_dependency
        for node_map in z_cm[node]:
            z_cs[node_map] ^= not_dependency

        if graph.meas_bases[node].plane == Plane.XY:
            z_cs[node] = set()
        elif graph.meas_bases[node].plane == Plane.YZ:
            x_cs[node] = set()
        else:
            z_cs[node] ^= x_cs[node]
            x_cs[node] = set()

    return x_cs, z_cs
