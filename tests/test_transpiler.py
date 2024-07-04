import pytest

from graphix_zx.transpiler import transpile, transpile_from_subgraphs
from graphix_zx.random_objects import get_random_flow_graph
from graphix_zx.command import is_standardized


@pytest.fixture
def random_graph():
    graph, flow = get_random_flow_graph(10, 10)
    return graph, flow


def test_transpile(random_graph):
    graph, flow = random_graph
    pattern = transpile(graph, flow)
    assert is_standardized(pattern)


if __name__ == "__main__":
    pytest.main()
