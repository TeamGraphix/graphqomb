"""Tests for pattern simulation behavior."""

from __future__ import annotations

import numpy as np

from graphqomb.command import M
from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import PauliFrame
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector


def _single_output_pattern(*, measured: bool, axis: Axis = Axis.Z) -> tuple[Pattern, int]:
    graph = GraphState()
    node = graph.add_node()
    graph.register_input(node, 0)
    graph.register_output(node, 0)

    pauli_frame = PauliFrame(graph, xflow={}, zflow={})
    commands = (M(node, AxisMeasBasis(axis, Sign.PLUS)),) if measured else ()
    pattern = Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=commands,
        pauli_frame=pauli_frame,
    )
    return pattern, node


def test_pattern_simulator_applies_output_x_frame_to_statevector() -> None:
    """An unmeasured output statevector should include pending X frame corrections."""
    pattern, node = _single_output_pattern(measured=False)
    pattern.pauli_frame.x_flip(node)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector([1.0, 0.0])

    simulator.simulate()

    np.testing.assert_allclose(simulator.state.state(), np.asarray([0.0, 1.0]))


def test_pattern_simulator_applies_output_z_frame_to_statevector() -> None:
    """An unmeasured output statevector should include pending Z frame corrections."""
    pattern, node = _single_output_pattern(measured=False)
    pattern.pauli_frame.z_flip(node)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector([1 / np.sqrt(2), 1 / np.sqrt(2)])

    simulator.simulate()

    np.testing.assert_allclose(simulator.state.state(), np.asarray([1 / np.sqrt(2), -1 / np.sqrt(2)]))


def test_pattern_simulator_measures_output_after_pauli_frame() -> None:
    """Output measurement results should be reported after applying the output Pauli frame."""
    pattern, node = _single_output_pattern(measured=True)
    pattern.pauli_frame.x_flip(node)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector([0.0, 1.0])

    simulator.simulate(rng=np.random.default_rng(123))

    assert simulator.results == {node: False}
    assert simulator.output_results == {0: False}


def test_pattern_simulator_measures_output_z_frame_in_x_basis() -> None:
    """A pending output Z frame should flip an X-basis output measurement result."""
    pattern, node = _single_output_pattern(measured=True, axis=Axis.X)
    pattern.pauli_frame.z_flip(node)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector([1 / np.sqrt(2), 1 / np.sqrt(2)])

    simulator.simulate(rng=np.random.default_rng(123))

    assert simulator.results == {node: True}
    assert simulator.output_results == {0: True}
