"""Tests for pattern module."""

from __future__ import annotations

import pytest

from graphqomb.command import TICK, E, M, N, X, Z
from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import PauliFrame


def test_n_command_str_with_coordinate() -> None:
    """Test N command string representation with coordinate."""
    cmd = N(node=5, coordinate=(1.0, 2.0))
    assert str(cmd) == "N: node=5, coord=(1.0, 2.0)"


def test_n_command_str_without_coordinate() -> None:
    """Test N command string representation without coordinate."""
    cmd = N(node=3)
    assert str(cmd) == "N: node=3"


@pytest.fixture
def pattern_components() -> tuple[dict[int, int], dict[int, int], PauliFrame, list[int]]:
    """Create shared components for building Pattern instances.

    Returns
    -------
    tuple[dict[int, int], dict[int, int], PauliFrame, list[int]]
        A tuple containing input node indices, output node indices,
        Pauli frame, and list of nodes.
    """
    graph = GraphState()
    nodes = [graph.add_physical_node() for _ in range(3)]

    graph.register_input(nodes[0], 0)
    graph.register_output(nodes[2], 0)

    pauli_frame = PauliFrame(graph, xflow={}, zflow={})

    return graph.input_node_indices, graph.output_node_indices, pauli_frame, nodes


def test_pattern_depth_counts_tick_commands(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that depth counts the number of TICK commands."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.0)
    commands = (
        N(node=nodes[0]),
        TICK(),
        E(nodes=(nodes[0], nodes[1])),
        M(node=nodes[1], meas_basis=meas_basis),
        TICK(),
        X(node=nodes[2]),
        Z(node=nodes[2]),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        pauli_frame=pauli_frame,
    )

    assert pattern.depth == 2


def test_pattern_depth_is_zero_without_ticks(
    pattern_components: tuple[dict[int, int], dict[int, int], PauliFrame, list[int]],
) -> None:
    """Test that depth returns zero when pattern has no TICK commands."""
    input_nodes, output_nodes, pauli_frame, nodes = pattern_components
    meas_basis = PlannerMeasBasis(Plane.XY, 0.5)
    commands = (
        N(node=nodes[0]),
        E(nodes=(nodes[0], nodes[1])),
        M(node=nodes[1], meas_basis=meas_basis),
        X(node=nodes[2]),
        Z(node=nodes[2]),
    )

    pattern = Pattern(
        input_node_indices=input_nodes,
        output_node_indices=output_nodes,
        commands=commands,
        pauli_frame=pauli_frame,
    )

    assert pattern.depth == 0
