import pytest
import numpy as np

# Assuming the following imports exist based on your code context
from graphix_zx.common import Plane
from graphix_zx.command import N, E, M, X, Z
from graphix_zx.pattern import MutablePattern
from graphix_zx.simulator import PatternSimulator, SimulatorBackend


# Mock StateVector class to test without actual quantum simulation
class MockStateVector:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def add_node(self, num):
        self.nodes.extend([None] * num)

    def entangle(self, nodes):
        pass

    def measure(self, node_id, plane, angle, result):
        pass

    def evolve(self, operation, nodes):
        pass

    def __eq__(self, other):
        return self.num_qubits == other.num_qubits


@pytest.fixture
def setup_pattern():
    pattern = MutablePattern([0])
    cmds = [
        N(node=1),
        E(nodes=(0, 1)),
        M(node=1, plane=Plane.XY, angle=0.5, s_domain=[], t_domain=[]),
        X(node=0, domain=[1]),
        Z(node=0, domain=[1]),
    ]
    pattern.extend(cmds)
    pattern.mark_runnable()
    pattern.mark_deterministic()
    return pattern


@pytest.fixture
def mock_statevector(monkeypatch):
    monkeypatch.setattr("graphix_zx.statevec.StateVector", MockStateVector)


def test_simulator_initialization(setup_pattern):
    pattern = setup_pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    assert simulator.get_state() == MockStateVector(len(pattern.get_input_nodes()))


def test_apply_command_add_node(setup_pattern):
    pattern = setup_pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    cmd = N(node=2)
    simulator.apply_cmd(cmd)
    assert len(simulator.node_indices) == len(pattern.get_input_nodes()) + 1


def test_apply_command_entangle(setup_pattern):
    pattern = setup_pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    cmd = N(node=1)
    simulator.apply_cmd(cmd)
    cmd = E(nodes=(0, 1))
    simulator.apply_cmd(cmd)
    # No assertion here as MockStateVector doesn't store entanglement info


def test_apply_command_measure(setup_pattern):
    pattern = setup_pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    cmd = M(node=0, plane=Plane.XY, angle=0.5, s_domain=[], t_domain=[])
    simulator.apply_cmd(cmd)
    assert 1 not in simulator.node_indices


def test_apply_command_byproduct_x(setup_pattern):
    pattern = setup_pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    cmd = N(node=1)
    simulator.apply_cmd(cmd)
    cmd = M(node=1, plane=Plane.XY, angle=0.5, s_domain=[], t_domain=[])
    simulator.apply_cmd(cmd)
    cmd = X(node=0, domain=[1])
    simulator.apply_cmd(cmd)
    # No specific assertion as MockStateVector doesn't evolve


def test_apply_command_byproduct_z(setup_pattern):
    pattern = setup_pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    cmd = N(node=1)
    simulator.apply_cmd(cmd)
    cmd = M(node=1, plane=Plane.XY, angle=0.5, s_domain=[], t_domain=[])
    simulator.apply_cmd(cmd)
    cmd = Z(node=0, domain=[1])
    simulator.apply_cmd(cmd)
    # No specific assertion as MockStateVector doesn't evolve


def test_simulate(setup_pattern):
    pattern = setup_pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.simulate()
    # Assertions depend on expected state after simulation
    # This is a placeholder, adjust according to actual expected state
    expected_state = MockStateVector(len(pattern.get_input_nodes()))
    assert simulator.get_state() == expected_state
