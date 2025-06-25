from __future__ import annotations

import pytest

from graphix_zx.feedforward import _is_flow, _is_gflow, check_flow, dag_from_flow, check_dag
from graphix_zx.graphstate import GraphState


def two_node_graph() -> tuple[GraphState, int, int]:
    graphstate = GraphState()
    node1 = graphstate.add_physical_node()
    node2 = graphstate.add_physical_node()
    graphstate.add_physical_edge(node1, node2)
    return graphstate, node1, node2


def test_is_flow_true() -> None:
    flow = {0: 1, 2: 3}
    assert _is_flow(flow)
    assert not _is_gflow(flow)


def test_is_flow_false_if_mixed_types() -> None:
    mixed = {0: 1, 1: {2}}
    assert not _is_flow(mixed)
    assert not _is_gflow(mixed)


def test_is_gflow_true() -> None:
    gflow = {0: {1}, 1: set()}
    assert _is_gflow(gflow)
    assert not _is_flow(gflow)


def test_dag_from_flow_basic_flow() -> None:
    graphstate, node1, node2 = two_node_graph()
    flow = {node1: node2}

    dag = dag_from_flow(graphstate, flow)
    check_dag(dag)

    assert dag[0] == {1}


def test_dag_from_flow_basic_gflow() -> None:
    graphstate, node1, node2 = two_node_graph()
    gflow = {node1: {node2}, node2: set()}

    dag = dag_from_flow(graphstate, gflow)
    check_dag(dag)

    assert dag[node1] == {node2}
    assert dag[node2] == set()


def test_dag_from_flow_invalid_type_raises() -> None:
    graphstate, node1, node2 = two_node_graph()
    invalid = {node1: node2, node2: {node2}}  # mixed types
    with pytest.raises(TypeError):
        dag_from_flow(graphstate, invalid)  # type: ignore[arg-type]


def test_dag_from_flow_cycle_detection() -> None:
    graphstate, node1, node2 = two_node_graph()
    cyclic_flow = {node1: node2, node2: node1}

    dag = dag_from_flow(graphstate, cyclic_flow)
    with pytest.raises(ValueError, match="Cycle detected in the graph:"):
        check_dag(dag)


def test_check_flow_false_for_cycle() -> None:
    graphstate, node1, node2 = two_node_graph()
    cyclic_flow = {node1: node2, node2: node1}
    with pytest.raises(ValueError, match="Cycle detected in the graph:"):
        check_flow(graphstate, cyclic_flow)


def test_check_flow_true_for_acyclic() -> None:
    graphstate, node1, node2 = two_node_graph()
    flow = {node1: node2}
    check_flow(graphstate, flow)
