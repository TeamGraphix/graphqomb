import pytest

from graphix_zx.transpiler import transpile_from_flow, transpile_from_subgraphs
from graphix_zx.simulator import MBQCCircuitSimulator, PatternSimulator
from graphix_zx.random_objects import get_random_flow_graph
from graphix_zx.command import is_standardized


@pytest.fixture
def random_graph():
    graph, flow = get_random_flow_graph(10, 10)
    return graph, flow


@pytest.fixture
def random_circ():
    raise NotImplementedError


def test_transpile_from_graph(random_graph):
    graph, flow = random_graph
    pattern = transpile_from_flow(graph, flow)
    assert is_standardized(pattern)


def test_transpile_from_subgraphs():
    pass


if __name__ == "__main__":
    pytest.main()
