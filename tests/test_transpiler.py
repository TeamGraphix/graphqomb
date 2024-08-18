import pytest

from graphix_zx.graphstate import GraphState
from graphix_zx.pattern import is_standardized
from graphix_zx.random_objects import get_random_flow_graph
from graphix_zx.transpiler import transpile_from_flow


@pytest.fixture()
def random_graph():
    graph, flow = get_random_flow_graph(10, 10)
    return graph, flow


def test_transpile_from_graph(random_graph: GraphState):
    graph, flow = random_graph
    pattern = transpile_from_flow(graph, flow)
    assert is_standardized(pattern)


def test_transpile_from_subgraphs():
    pass


if __name__ == "__main__":
    pytest.main()
