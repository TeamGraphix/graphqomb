import pytest
import numpy as np

from graphix_zx.circuit import MBQCCircuit, J, CZ, PhaseGadget
from graphix_zx.simulator import SimulatorBackend, MBQCCircuitSimulator
from graphix_zx.statevec import StateVector


# Test BasicMBQCCircuit
def test_basic_mbqc_circuit():
    circuit = MBQCCircuit(0)

    assert circuit.num_qubits == 0

    circuit.j(0, np.pi / 4)
    assert len(circuit.get_instructions()) == 1
    assert isinstance(circuit.get_instructions()[0], J)

    circuit.cz(0, 1)
    assert len(circuit.get_instructions()) == 2
    assert isinstance(circuit.get_instructions()[1], CZ)

    circuit.phase_gadget([0, 1], np.pi / 3)
    assert len(circuit.get_instructions()) == 3
    assert isinstance(circuit.get_instructions()[2], PhaseGadget)


# Test MBQCCircuitSimulator
def test_mbqc_circuit_simulator():
    circuit = MBQCCircuit(2)
    circuit.j(0, np.pi / 4)
    circuit.cz(0, 1)

    simulator = MBQCCircuitSimulator(circuit, SimulatorBackend.StateVector)
    simulator.simulate()

    state = simulator.get_state()
    assert isinstance(state, StateVector)

    # Test for expected state vector (simplified example)
    expected_state = (
        np.array(
            [
                (1 + np.exp(-1j * np.pi / 4)),
                (1 + np.exp(-1j * np.pi / 4)),
                (1 - np.exp(-1j * np.pi / 4)),
                -(1 - np.exp(-1j * np.pi / 4)),
            ],
            dtype=complex,
        )
        / 2**1.5
    )  # Adjust based on actual logic
    assert np.allclose(state.get_state_vector(), expected_state)

    # Test for invalid backend
    with pytest.raises(ValueError):
        MBQCCircuitSimulator(circuit, "InvalidBackend")

    # Test for not implemented backend
    with pytest.raises(NotImplementedError):
        MBQCCircuitSimulator(circuit, SimulatorBackend.DensityMatrix)


if __name__ == "__main__":
    pytest.main()
