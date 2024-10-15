from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from graphix_zx.pattern import is_standardized
from graphix_zx.qompiler import qompile_from_flow
from graphix_zx.random_objects import get_random_flow_graph

if TYPE_CHECKING:
    from graphix_zx.flow import FlowLike
    from graphix_zx.graphstate import GraphState


@pytest.fixture
def random_graph() -> tuple[GraphState, FlowLike]:
    graph, flow = get_random_flow_graph(10, 10)
    return graph, flow


def test_transpile_from_graph(random_graph: tuple[GraphState, FlowLike]) -> None:
    graph, flow = random_graph
    pattern = qompile_from_flow(graph, flow)
    assert is_standardized(pattern)


def test_transpile_from_subgraphs() -> None:
    pass


if __name__ == "__main__":
    pytest.main()
