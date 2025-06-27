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


def test_get_norm(state_vector: StateVector) -> None:
    state = np.arange(2**state_vector.num_qubits)
    expected_norm = np.linalg.norm(state)
    assert np.isclose(state_vector.norm(), expected_norm)


def test_expectation(plus_state: StateVector) -> None:
    operator = np.asarray([[1, 0], [0, -1]])  # Z gate
    exp_value = plus_state.expectation(operator, [0])
    expected_value = 0.0  # <++|Z|++> = 0
    assert np.isclose(exp_value, expected_value)


def test_expectation_computational_basis() -> None:
    """Test expectation values with computational basis states."""
    # Test |0⟩ state with Z operator
    zero_state = StateVector(1, [1, 0])
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    exp_val = zero_state.expectation(z_op, [0])
    assert np.isclose(exp_val, 1.0)  # ⟨0|Z|0⟩ = 1

    # Test |1⟩ state with Z operator
    one_state = StateVector(1, [0, 1])
    exp_val = one_state.expectation(z_op, [0])
    assert np.isclose(exp_val, -1.0)  # ⟨1|Z|1⟩ = -1

    # Test |0⟩ state with X operator
    x_op = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    exp_val = zero_state.expectation(x_op, [0])
    assert np.isclose(exp_val, 0.0)  # ⟨0|X|0⟩ = 0

    # Test |1⟩ state with X operator
    exp_val = one_state.expectation(x_op, [0])
    assert np.isclose(exp_val, 0.0)  # ⟨1|X|1⟩ = 0


def test_expectation_superposition_states() -> None:
    """Test expectation values with superposition states."""
    # Test |+⟩ state with X operator
    plus_state = StateVector(1, [1, 1] / np.sqrt(2))
    x_op = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    exp_val = plus_state.expectation(x_op, [0])
    assert np.isclose(exp_val, 1.0)  # ⟨+|X|+⟩ = 1

    # Test |-⟩ state with X operator
    minus_state = StateVector(1, [1, -1] / np.sqrt(2))
    exp_val = minus_state.expectation(x_op, [0])
    assert np.isclose(exp_val, -1.0)  # ⟨-|X|-⟩ = -1

    # Test |+⟩ state with Y operator
    y_op = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    exp_val = plus_state.expectation(y_op, [0])
    assert np.isclose(exp_val, 0.0)  # ⟨+|Y|+⟩ = 0


def test_expectation_two_qubit_states() -> None:
    """Test expectation values with two-qubit states."""
    # Bell state: (|00⟩ + |11⟩)/√2
    bell_state = StateVector(2, [1, 0, 0, 1] / np.sqrt(2))

    # Single qubit operators
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    exp_val_0 = bell_state.expectation(z_op, [0])
    exp_val_1 = bell_state.expectation(z_op, [1])
    assert np.isclose(exp_val_0, 0.0)  # ⟨Bell|Z₀|Bell⟩ = 0
    assert np.isclose(exp_val_1, 0.0)  # ⟨Bell|Z₁|Bell⟩ = 0

    # Two-qubit operator: Z⊗Z
    zz_op = np.kron(z_op, z_op).astype(np.complex128)
    exp_val_zz = bell_state.expectation(zz_op, [0, 1])
    assert np.isclose(exp_val_zz, 1.0)  # ⟨Bell|Z⊗Z|Bell⟩ = 1

    # Two-qubit operator: X⊗X
    x_op = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    xx_op = np.kron(x_op, x_op).astype(np.complex128)
    exp_val_xx = bell_state.expectation(xx_op, [0, 1])
    assert np.isclose(exp_val_xx, 1.0)  # ⟨Bell|X⊗X|Bell⟩ = 1


def test_expectation_non_contiguous_qubits() -> None:
    """Test expectation values with non-contiguous qubit selection."""
    # 3-qubit state: |000⟩ + |111⟩
    state = StateVector(3, [1, 0, 0, 0, 0, 0, 0, 1] / np.sqrt(2))

    # Two-qubit operator on qubits 0 and 2
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    zz_op = np.kron(z_op, z_op).astype(np.complex128)
    exp_val = state.expectation(zz_op, [0, 2])
    assert np.isclose(exp_val, 1.0)  # ⟨state|Z₀⊗Z₂|state⟩ = 1


def test_expectation_unnormalized_state() -> None:
    """Test expectation values with unnormalized states."""
    # Unnormalized |0⟩ state with amplitude 2
    unnormalized_state = StateVector(1, [2, 0])
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    exp_val = unnormalized_state.expectation(z_op, [0])
    assert np.isclose(exp_val, 1.0)  # Should still give ⟨0|Z|0⟩ = 1

    # Unnormalized superposition state
    unnormalized_plus = StateVector(1, [3, 3])  # 3(|0⟩ + |1⟩)
    x_op = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    exp_val = unnormalized_plus.expectation(x_op, [0])
    assert np.isclose(exp_val, 1.0)  # Should still give ⟨+|X|+⟩ = 1


def test_expectation_identity_operator() -> None:
    """Test expectation values with identity operator."""
    # Any state should give expectation value 1 for identity
    state_vector = StateVector(2, [0.5, 0.5, 0.5, 0.5])
    identity = np.eye(2, dtype=np.complex128)
    exp_val = state_vector.expectation(identity, [0])
    assert np.isclose(exp_val, 1.0)

    # Two-qubit identity
    identity_2q = np.eye(4, dtype=np.complex128)
    exp_val = state_vector.expectation(identity_2q, [0, 1])
    assert np.isclose(exp_val, 1.0)


def test_expectation_hermitian_check() -> None:
    """Test that non-Hermitian operators raise ValueError."""
    state = StateVector(1, [1, 0])
    non_hermitian = np.array([[1, 1], [0, 1]], dtype=np.complex128)  # Not Hermitian

    with pytest.raises(ValueError, match="Operator must be Hermitian"):
        state.expectation(non_hermitian, [0])
