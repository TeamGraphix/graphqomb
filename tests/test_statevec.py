import pytest
import numpy as np
from graphix_zx.common import Plane
from graphix_zx.statevec import StateVector


@pytest.fixture
def plus_state():
    return StateVector(2)


@pytest.fixture
def state_vector():
    num_qubits = 3
    state = np.arange(2**num_qubits, dtype=np.float64)
    return StateVector(num_qubits, state)


def test_initialization(state_vector):
    expected_state = np.arange(2**state_vector.num_qubits)
    assert state_vector.num_qubits == 3
    assert np.allclose(state_vector.get_state_vector(), expected_state)


def test_evolve(state_vector):
    operator = np.array([[1, 0], [0, -1]])  # Z gate
    state_vector.evolve(operator, [0])
    expected_state = np.arange(2**state_vector.num_qubits)
    expected_state[len(expected_state) // 2 :] *= -1  # apply Z gate to qubit 0
    assert np.allclose(state_vector.get_state_vector(), expected_state)


def test_measure(state_vector):
    expected_state = np.arange(2 ** (state_vector.num_qubits - 1))

    state_vector.measure(0, Plane.ZX, 0, 0)  # project onto |0> state

    assert state_vector.num_qubits == 2
    assert np.allclose(state_vector.get_state_vector(), expected_state)


def test_tensor_product(state_vector):
    expected_state = np.array([i // 2 for i in range(2 ** (state_vector.num_qubits + 1))] / np.sqrt(2))
    other_vector = StateVector(1)
    state_vector.tensor_product(other_vector)

    assert state_vector.num_qubits == 4
    assert np.allclose(state_vector.get_state_vector(), expected_state)


def test_normalize(state_vector):
    state_vector.normalize()
    expected_norm = 1.0
    assert np.isclose(state_vector.get_norm(), expected_norm)


def test_reorder(state_vector):
    permutation = [2, 0, 1]
    state_vector.reorder(permutation)

    expected_state = np.arange(2**state_vector.num_qubits)
    expected_state = expected_state.reshape((2, 2, 2)).transpose(permutation).flatten()
    assert np.allclose(state_vector.get_state_vector(), expected_state)


def test_is_isolated(plus_state):
    isolated = plus_state.is_isolated(0)
    assert isolated


def test_is_not_isolated(plus_state):
    cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    plus_state.evolve(cz, [0, 1])
    isolated = plus_state.is_isolated(0)
    assert not isolated


def test_get_norm(state_vector):
    state = np.arange(2**state_vector.num_qubits)
    expected_norm = np.linalg.norm(state)
    assert np.isclose(state_vector.get_norm(), expected_norm)


def test_expectation_value(plus_state):
    operator = np.array([[1, 0], [0, -1]])  # Z gate
    exp_value = plus_state.expectation_value(operator, [0])
    expected_value = 0.0  # <++|Z|++> = 0
    assert np.isclose(exp_value, expected_value)


def test_get_density_matrix_not_implemented(state_vector):
    with pytest.raises(NotImplementedError):
        state_vector.get_density_matrix()


if __name__ == "__main__":
    pytest.main()
