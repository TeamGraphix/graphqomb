import pytest

from graphix_zx.graphstate import GraphState
from graphix_zx.feedforward import is_flow, is_gflow, dag_from_flow, check_causality


def two_node_graph() -> tuple[GraphState, int, int]:
    graphstate = GraphState()
    node1 = graphstate.add_physical_node()
    node2 = graphstate.add_physical_node()
    graphstate.add_physical_edge(node1, node2)
    return graphstate, node1, node2


def test_is_flow_true() -> None:
    """All values are int → should be recognised as Flow."""
    flow = {0: 1, 2: 3}
    assert is_flow(flow)
    assert not is_gflow(flow)


def test_is_flow_false_if_mixed_types() -> None:
    """A mixture of int and set values is neither Flow nor GFlow."""
    mixed = {0: 1, 1: {2}}
    assert not is_flow(mixed)
    assert not is_gflow(mixed)


def test_is_gflow_true() -> None:
    """All values are sets → should be recognised as GFlow."""
    gflow = {0: {1}, 1: set()}
    assert is_gflow(gflow)
    assert not is_flow(gflow)


def test_dag_from_flow_basic_flow() -> None:
    """For a simple Flow the DAG should contain neighbour + target minus self."""
    graphstate, node1, node2 = two_node_graph()
    flow = {node1: node2}

    dag = dag_from_flow(flow, graphstate, check=True)

    assert dag[0] == {1}


def test_dag_from_flow_basic_gflow() -> None:
    """
    For a GFlow the target set is:

        flow[node]  ∪  odd_neighbors(flow[node])  − {node}
    """
    graphstate, node1, node2 = two_node_graph()
    gflow = {node1: {node2}, node2: set()}  # node-0 corrects by {1}, node-1 corrects by ∅

    dag = dag_from_flow(gflow, graphstate, check=True)

    assert dag[node1] == {node2}
    # Node-1 has no corrections, so empty
    assert dag[node2] == set()


def test_dag_from_flow_invalid_type_raises() -> None:
    """An object that is neither Flow nor GFlow triggers TypeError."""
    graphstate, node1, node2 = two_node_graph()
    invalid = {node1: node2, node2: {node2}}  # mixed types
    with pytest.raises(TypeError):
        dag_from_flow(invalid, graphstate)  # type: ignore[arg-type]


def test_dag_from_flow_cycle_detection() -> None:
    graphstate, node1, node2 = two_node_graph()
    cyclic_flow = {node1: node2, node2: node1}

    with pytest.raises(ValueError, match="Cycle detected in the graph"):
        dag_from_flow(cyclic_flow, graphstate, check=True)


def test_check_causality_false_for_cycle() -> None:
    """A cyclic flow should be reported as *not* causal."""
    graphstate, node1, node2 = two_node_graph()
    cyclic_flow = {node1: node2, node2: node1}
    assert not check_causality(graphstate, cyclic_flow)


def test_check_causality_true_for_acyclic() -> None:
    """An acyclic flow should be reported as causal."""
    graphstate, node1, node2 = two_node_graph()
    flow = {node1: node2}
    assert check_causality(graphstate, flow)
