"""not refactored yet"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

    from graphix_zx.graphstate import BaseGraphState


FlowLike = dict[int, set[int]]
Layer = dict[int, int]


def oddneighbors(nodes: AbstractSet[int], graph: BaseGraphState) -> set[int]:
    odd_neighbors: set[int] = set()
    for node in nodes:
        odd_neighbors ^= graph.get_neighbors(node)
    return odd_neighbors


def construct_dag(gflow: FlowLike, graph: BaseGraphState, *, check: bool = False) -> dict[int, set[int]]:
    dag = {}
    outputs = graph.physical_nodes - gflow.keys()
    for node in gflow:
        dag[node] = (gflow[node] | oddneighbors(gflow[node], graph)) - {node}
    for output in outputs:
        dag[output] = set()

    if check and not check_dag(dag):
        msg = "Cycle detected in the graph"
        raise ValueError(msg)

    return dag


def check_dag(dag: Mapping[int, Iterable[int]]) -> bool:
    for node, children in dag.items():
        for child in children:
            if node in dag[child]:
                return False
    return True


# def find_flow(
#     graph: BaseGraphState,
# ) -> tuple[FlowLike, Layer]:
#     raise NotImplementedError


# def find_gflow(
#     graph: BaseGraphState,
# ) -> tuple[FlowLike, Layer]:
#     l_k: Layer = {}
#     g: FlowLike = {}
#     for node in graph.physical_nodes:
#         l_k[node] = 0
#     return gflowaux(graph, 1, l_k, g)


# def find_pflow(
#     graph: BaseGraphState,
# ) -> tuple[FlowLike, Layer]:
#     raise NotImplementedError


# def gflowaux(
#     graph: BaseGraphState,
#     k: int,
#     l_k: Layer,
#     g: FlowLike,
# ) -> tuple[FlowLike, Layer]:
#     raise NotImplementedError


def check_causality(
    graph: BaseGraphState,
    gflow: FlowLike,
) -> bool:
    dag = construct_dag(gflow, graph)
    return check_dag(dag)


# # NOTE: want to include Pauli simplification effect
# def check_stablizers(
#     graph: BaseGraphState,
#     gflow: dict[int, set[int]],
# ) -> bool:
#     raise NotImplementedError
