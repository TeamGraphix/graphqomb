import pytest
import numpy as np
from graphix_zx.statevec import StateVector


@pytest.fixture
def state_vector():
    return StateVector(2)


def test_initialization(state_vector):
    expected_state = np.ones(2**state_vector.num_qubits) / np.sqrt(2**state_vector.num_qubits)
    assert state_vector.num_qubits == 2
    assert np.allclose(state_vector.get_state_vector(), expected_state)


def test_evolve(state_vector):
    operator = np.array([[1, 0], [0, -1]])  # Z gate
    state_vector.evolve(operator, [0])
    expected_state = np.array([1 / 2, 1 / 2, -1 / 2, -1 / 2])  # (|00> + |01> - |10> - |11>)/2
    assert np.allclose(state_vector.get_state_vector(), expected_state)


def test_measure(state_vector):
    state_vector.measure(0, "XZ", 0, 0)  # project onto |0> state
    expected_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])  # |+>

    assert state_vector.num_qubits == 1
    assert np.allclose(state_vector.get_state_vector(), expected_state)


def test_tensor_product(state_vector):
    other_vector = StateVector(1)
    state_vector.tensor_product(other_vector)
    expected_state = np.ones(2 ** (state_vector.num_qubits)) / np.sqrt(2 ** (state_vector.num_qubits))
    assert state_vector.num_qubits == 3
    assert np.allclose(state_vector.get_state_vector(), expected_state)


def test_normalize(state_vector):
    state_vector.normalize()
    expected_norm = 1.0
    assert np.isclose(state_vector.get_norm(), expected_norm)


def test_is_isolated(state_vector):
    isolated = state_vector.is_isolated(0)
    assert isolated


def test_is_not_isolated(state_vector):
    cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    state_vector.evolve(cz, [0, 1])
    isolated = state_vector.is_isolated(0)
    assert not isolated


def test_get_norm(state_vector):
    expected_norm = 1.0
    assert np.isclose(state_vector.get_norm(), expected_norm)


def test_expectation_value(state_vector):
    operator = np.array([[1, 0], [0, -1]])  # Z gate
    exp_value = state_vector.expectation_value(operator, [0])
    expected_value = 0.0  # <psi|Z|psi> = 0 for initial state
    assert np.isclose(exp_value, expected_value)


def test_get_density_matrix_not_implemented(state_vector):
    with pytest.raises(NotImplementedError):
        state_vector.get_density_matrix()


if __name__ == "__main__":
    pytest.main()
