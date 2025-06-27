from collections.abc import Mapping

import numpy as np
import pytest
from numpy.typing import NDArray

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.statevec import StateVector


def kron_n(ops: Mapping[int, NDArray[np.complex128]], num_qubits: int) -> NDArray[np.complex128]:
    """Compute the Kronecker product of a sequence of operators, filling in identity matrices for missing qubits.

    Parameters
    ----------
    ops : Mapping[int, NDArray[np.complex128]]
        The operators to include in the Kronecker product.
    num_qubits : int
        The total number of qubits in the resulting state vector.

    Returns
    -------
    NDArray[np.complex128]
        The resulting Kronecker product as a state vector.
    """
    identity = np.eye(2, dtype=np.complex128)
    mats = [(ops.get(i, identity)) for i in range(num_qubits)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m).astype(np.complex128)
    return out


@pytest.fixture
def plus_state() -> StateVector:
    return StateVector(2)


@pytest.fixture
def state_vector() -> StateVector:
    num_qubits = 3
    state = np.arange(2**num_qubits, dtype=np.complex128)
    return StateVector(num_qubits, state)


def test_initialization(state_vector: StateVector) -> None:
    expected_state = np.arange(2**state_vector.num_qubits, dtype=np.complex128)
    assert state_vector.num_qubits == 3
    assert np.allclose(state_vector.state, expected_state)


def test_evolve(state_vector: StateVector) -> None:
    operator = np.asarray([[1, 0], [0, -1]])  # Z gate
    state_vector.evolve(operator, [0])
    expected_state = np.arange(2**state_vector.num_qubits, dtype=np.complex128)
    expected_state[len(expected_state) // 2 :] *= -1  # apply Z gate to qubit 0
    assert np.allclose(state_vector.state, expected_state)


def test_two_qubit_z_on_qubit1() -> None:
    bell = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    qs = StateVector(2, bell.copy())
    z = np.diag([1, -1]).astype(np.complex128)

    qs.evolve(z, [1])

    expected = np.array([1, 0, 0, -1], dtype=np.complex128) / np.sqrt(2)
    assert np.allclose(qs.state, expected)


def test_noncontiguous_qubit_selection() -> None:
    n = 3
    rng = np.random.default_rng(42)
    psi0 = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi0 /= np.linalg.norm(psi0)

    qs = StateVector(n, psi0.copy())

    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    qs.evolve(h, [2])

    u_full = kron_n({2: h}, n)
    expected = u_full @ psi0

    assert np.allclose(qs.state, expected, atol=1e-12)


@pytest.mark.parametrize(("n", "k"), [(3, 1), (3, 2)])
def test_norm_preserved_random_unitary(n: int, k: int) -> None:
    rng = np.random.default_rng(123)
    psi0 = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi0 /= np.linalg.norm(psi0)

    a = rng.normal(size=(2**k, 2**k)) + 1j * rng.normal(size=(2**k, 2**k))
    q, _ = np.linalg.qr(a)
    u = q.astype(np.complex128)

    qubits = tuple(int(x) for x in sorted(rng.choice(n, k, replace=False).astype(int)))
    qs = StateVector(n, psi0.copy())
    qs.evolve(u, list(qubits))

    assert np.allclose(np.linalg.norm(qs.state), 1.0, atol=1e-12)


def test_measure(state_vector: StateVector) -> None:
    # Initial state: [0, 1, 2, 3, 4, 5, 6, 7] representing |000⟩, |001⟩, ..., |111⟩
    # Measuring qubit 0 in XZ plane with angle 0 (|0⟩ basis) and result 0
    # This selects states where qubit 0 = 0: |000⟩, |001⟩, |010⟩, |011⟩
    # Corresponding to coefficients [0, 1, 2, 3]
    expected_state = np.array([0, 1, 2, 3], dtype=np.complex128)
    expected_state /= np.linalg.norm(expected_state)

    state_vector.measure(0, PlannerMeasBasis(Plane.XZ, 0), 0)  # project onto |0⟩ state

    assert state_vector.num_qubits == 2
    assert np.allclose(state_vector.state, expected_state)


def test_tensor_product(state_vector: StateVector) -> None:
    expected_state = np.asarray([i // 2 for i in range(2 ** (state_vector.num_qubits + 1))]) / np.sqrt(2)
    other_vector = StateVector(1)
    result = StateVector.tensor_product(state_vector, other_vector)

    assert result.num_qubits == 4
    assert np.allclose(result.state, expected_state)


def test_normalize(state_vector: StateVector) -> None:
    state_vector.normalize()
    expected_norm = 1.0
    assert np.isclose(state_vector.norm(), expected_norm)


# def test_reorder(state_vector: StateVector):
#     pass


def test_is_isolated(plus_state: StateVector) -> None:
    isolated = plus_state.is_isolated(0)
    assert isolated


def test_is_not_isolated(plus_state: StateVector) -> None:
    cz = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    plus_state.evolve(cz, [0, 1])
    isolated = plus_state.is_isolated(0)
    assert not isolated


def test_get_norm(state_vector: StateVector) -> None:
    state = np.arange(2**state_vector.num_qubits)
    expected_norm = np.linalg.norm(state)
    assert np.isclose(state_vector.norm(), expected_norm)


def test_expectation_value(plus_state: StateVector) -> None:
    operator = np.asarray([[1, 0], [0, -1]])  # Z gate
    exp_value = plus_state.expectation(operator, [0])
    expected_value = 0.0  # <++|Z|++> = 0
    assert np.isclose(exp_value, expected_value)
