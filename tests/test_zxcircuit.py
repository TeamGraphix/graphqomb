import pytest
import numpy as np
from graphix_zx.statevec import StateVector
from graphix_zx.interface import ZXMBQCCircuit
from graphix_zx.simulator import CircuitSimulatorBackend, MBQCCircuitSimulator


def test_zxmbqcircuit_initialization():
    qubits = 3
    circuit = ZXMBQCCircuit(qubits)
    assert circuit.num_qubits == qubits
    assert circuit.input_nodes == list(range(qubits))
    assert isinstance(circuit.gflow, dict)
    assert len(circuit.gflow) == qubits


def test_j_gate():
    qubits = 2
    angle = np.pi / 4
    circuit = ZXMBQCCircuit(qubits)
    circuit.j(0, angle)

    assert len(circuit.gflow) == qubits + 1
    assert isinstance(circuit.gflow[circuit.input_nodes[0]], set)
    assert len(circuit.gflow[circuit.input_nodes[0]]) == 1


def test_cz_gate():
    qubits = 2
    circuit = ZXMBQCCircuit(qubits)
    circuit.cz(0, 1)

    assert len(circuit.gflow) == qubits


def test_phase_gadget():
    qubits = 2
    angle = np.pi / 3
    circuit = ZXMBQCCircuit(qubits)
    circuit.phase_gadget([0, 1], angle)

    assert len(circuit.gflow) == qubits + 1


def test_simulator_initialization():
    qubits = 2
    circuit = ZXMBQCCircuit(qubits)
    simulator = MBQCCircuitSimulator(CircuitSimulatorBackend.StateVector, circuit)

    assert isinstance(simulator.get_state(), StateVector)
    assert simulator.get_state().num_qubits == qubits


def test_simulator_apply_gate():
    qubits = 2
    circuit = ZXMBQCCircuit(qubits)
    circuit.j(0, np.pi / 4)
    simulator = MBQCCircuitSimulator(CircuitSimulatorBackend.StateVector, circuit)

    simulator.simulate()
    state = simulator.get_state()

    # Here we would compare the expected state vector
    # Assuming initial state is |++>
    expected_state = np.array(
        [
            (1 + np.exp(1j * np.pi / 4)),
            (1 + np.exp(1j * np.pi / 4)),
            (1 - np.exp(1j * np.pi / 4)),
            (1 - np.exp(1j * np.pi / 4)),
        ]
    ) / np.sqrt(2)
    assert np.allclose(state.get_state_vector(), expected_state, atol=1e-6)


def test_simulator_apply_cz_gate():
    qubits = 2
    circuit = ZXMBQCCircuit(qubits)
    circuit.cz(0, 1)
    simulator = MBQCCircuitSimulator(CircuitSimulatorBackend.StateVector, circuit)

    simulator.simulate()
    state = simulator.get_state()

    # Assuming initial state is |++>
    expected_state = np.array([1, 1, 1, -1]) / np.sqrt(2)
    assert np.allclose(state.get_state_vector(), expected_state, atol=1e-6)


def test_simulator_invalid_backend():
    qubits = 2
    circuit = ZXMBQCCircuit(qubits)

    with pytest.raises(ValueError):
        MBQCCircuitSimulator("InvalidBackend", circuit)


def test_simulator_phase_gadget_not_implemented():
    qubits = 2
    circuit = ZXMBQCCircuit(qubits)
    circuit.phase_gadget([0, 1], np.pi / 3)
    simulator = MBQCCircuitSimulator(CircuitSimulatorBackend.StateVector, circuit)

    with pytest.raises(NotImplementedError):
        simulator.simulate()


if __name__ == "__main__":
    pytest.main()
