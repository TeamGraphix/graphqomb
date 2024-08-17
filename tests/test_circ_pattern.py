"""Integrated test with real simulator"""

import numpy as np
import pytest

from graphix_zx.circuit import MBQCCircuit, circuit2graph
from graphix_zx.simulator import (
    MBQCCircuitSimulator,
    PatternSimulator,
    SimulatorBackend,
)
from graphix_zx.transpiler import transpile_from_flow


@pytest.fixture
def random_circ():
    circ = MBQCCircuit(3)
    circ.j(0, 0.5 * np.pi)
    circ.cz(0, 1)
    circ.cz(1, 2)
    circ.j(2, 0.25 * np.pi)
    circ.j(1, 0.75 * np.pi)
    return circ


@pytest.fixture
def random_circ_with_phase_gadget():
    circ = MBQCCircuit(3)
    circ.j(0, 0.5 * np.pi)
    circ.cz(0, 1)
    circ.cz(1, 2)
    circ.j(2, 0.25 * np.pi)
    circ.j(1, 0.75 * np.pi)
    circ.phase_gadget([0, 2], 0.5 * np.pi)
    circ.cz(0, 2)
    circ.j(0, 0.25 * np.pi)
    circ.cz(0, 1)
    circ.cz(1, 2)
    circ.j(2, 0.25 * np.pi)
    return circ


def test_circuit_sim(random_circ: MBQCCircuit):
    simulator = MBQCCircuitSimulator(random_circ, SimulatorBackend.StateVector)
    simulator.simulate()
    state = simulator.get_state()
    assert np.isclose(state.get_norm(), 1.0)


def test_pattern_sim(random_circ: MBQCCircuit):
    graph, gflow = circuit2graph(random_circ)
    pattern = transpile_from_flow(graph, gflow, correct_output=True)
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.simulate()
    state = simulator.get_state()
    assert np.isclose(state.get_norm(), 1.0)
    assert set(simulator.node_indices) == set(pattern.get_output_nodes())


def test_minimum_circ_pattern():
    circuit = MBQCCircuit(1)
    circuit.j(0, 0.3 * np.pi)
    graph, gflow = circuit2graph(circuit)
    pattern = transpile_from_flow(graph, gflow, correct_output=True)
    circ_simulator = MBQCCircuitSimulator(circuit, SimulatorBackend.StateVector)
    circ_simulator.simulate()
    pattern_sim = PatternSimulator(pattern, SimulatorBackend.StateVector)
    pattern_sim.simulate()

    circ_state = circ_simulator.get_state().get_state_vector()
    pattern_state = pattern_sim.get_state().get_state_vector()

    inner_prod = np.vdot(circ_state, pattern_state)
    assert np.isclose(np.abs(inner_prod), 1.0)


def test_match_circ_pattern(random_circ: MBQCCircuit):
    graph, gflow = circuit2graph(random_circ)
    pattern = transpile_from_flow(graph, gflow, correct_output=True)

    circ_sim = MBQCCircuitSimulator(random_circ, SimulatorBackend.StateVector)
    circ_sim.simulate()
    pattern_sim = PatternSimulator(pattern, SimulatorBackend.StateVector)
    pattern_sim.simulate()

    circ_state = circ_sim.get_state().get_state_vector()
    pattern_state = pattern_sim.get_state().get_state_vector()
    inner_prod = np.vdot(circ_state, pattern_state)
    assert np.isclose(np.abs(inner_prod), 1.0)


def test_match_circ_pattern_with_phase_gadget(random_circ_with_phase_gadget: MBQCCircuit):
    graph, gflow = circuit2graph(random_circ_with_phase_gadget)
    pattern = transpile_from_flow(graph, gflow, correct_output=True)

    circ_sim = MBQCCircuitSimulator(random_circ_with_phase_gadget, SimulatorBackend.StateVector)
    circ_sim.simulate()
    pattern_sim = PatternSimulator(pattern, SimulatorBackend.StateVector)
    pattern_sim.simulate()

    circ_state = circ_sim.get_state().get_state_vector()
    pattern_state = pattern_sim.get_state().get_state_vector()
    inner_prod = np.vdot(circ_state, pattern_state)
    assert np.isclose(np.abs(inner_prod), 1.0)
