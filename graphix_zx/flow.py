"""not refactored yet"""

from __future__ import annotations

from typing import Dict, Set

from graphix_zx.common import Plane
from graphix_zx.graphstate import BaseGraphState

FlowLike = Dict[int, Set[int]]
Layer = Dict[int, int]


def find_flow(
    graph: BaseGraphState,
) -> tuple[FlowLike, Layer]:
    raise NotImplementedError


def find_gflow(
    graph: BaseGraphState,
) -> tuple[FlowLike, Layer]:
    l_k: Layer = dict()
    g: FlowLike = dict()
    for node in graph.get_physical_nodes():
        l_k[node] = 0
    return gflowaux(graph, 1, l_k, g)


def find_pflow(
    graph: BaseGraphState,
) -> tuple[FlowLike, Layer]:
    raise NotImplementedError


def gflowaux(
    graph: BaseGraphState,
    k: int,
    l_k: Layer,
    g: FlowLike,
) -> tuple[FlowLike, Layer]:
    raise NotImplementedError


def check_causality(
    graph: BaseGraphState,
    gflow: FlowLike,
) -> bool:
    raise NotImplementedError


# NOTE: want to include Pauli simplification effect
def check_stablizers(
    graph: BaseGraphState,
    gflow: dict[int, set[int]],
) -> bool:
    raise NotImplementedError
