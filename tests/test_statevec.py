from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.statevec import StateVector

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


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
    return StateVector.from_num_qubits(2)


@pytest.fixture
def state_vector() -> StateVector:
    num_qubits = 3
    state = np.arange(2**num_qubits, dtype=np.complex128)
    return StateVector(state)


def test_initialization(state_vector: StateVector) -> None:
    expected_state = np.arange(2**state_vector.num_qubits, dtype=np.complex128)
    assert state_vector.num_qubits == 3
    assert np.allclose(state_vector.state().flatten(), expected_state)


def test_evolve(state_vector: StateVector) -> None:
    operator = np.asarray([[1, 0], [0, -1]])  # Z gate
    state_vector.evolve(operator, [0])
    expected_state = np.arange(2**state_vector.num_qubits, dtype=np.complex128)
    expected_state[len(expected_state) // 2 :] *= -1  # apply Z gate to qubit 0
    assert np.allclose(state_vector.state().flatten(), expected_state)


def test_two_qubit_z_on_qubit1() -> None:
    bell = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    qs = StateVector(bell.copy())
    z = np.diag([1, -1]).astype(np.complex128)

    qs.evolve(z, [1])

    expected = np.array([1, 0, 0, -1], dtype=np.complex128) / np.sqrt(2)
    assert np.allclose(qs.state().flatten(), expected)


def test_noncontiguous_qubit_selection() -> None:
    n = 3
    rng = np.random.default_rng(42)
    psi0 = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi0 /= np.linalg.norm(psi0)

    qs = StateVector(psi0.copy())

    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    qs.evolve(h, [2])

    u_full = kron_n({2: h}, n)
    expected = u_full @ psi0

    assert np.allclose(qs.state().flatten(), expected, atol=1e-12)


@pytest.mark.parametrize(("n", "k"), [(3, 1), (3, 2)])
def test_norm_preserved_random_unitary(n: int, k: int) -> None:
    rng = np.random.default_rng(123)
    psi0 = rng.normal(size=(2**n,)) + 1j * rng.normal(size=(2**n,))
    psi0 /= np.linalg.norm(psi0)

    a = rng.normal(size=(2**k, 2**k)) + 1j * rng.normal(size=(2**k, 2**k))
    q, _ = np.linalg.qr(a)
    u = q.astype(np.complex128)

    qubits = tuple(int(x) for x in sorted(rng.choice(n, k, replace=False).astype(int)))
    qs = StateVector(psi0.copy())
    qs.evolve(u, list(qubits))

    assert np.allclose(np.linalg.norm(qs.state()), 1.0, atol=1e-12)


def test_measure(state_vector: StateVector) -> None:
    # Initial state: [0, 1, 2, 3, 4, 5, 6, 7] representing |000⟩, |001⟩, ..., |111⟩
    # Measuring qubit 0 in XZ plane with angle 0 (|0⟩ basis) and result 0
    # This selects states where qubit 0 = 0: |000⟩, |001⟩, |010⟩, |011⟩
    # Corresponding to coefficients [0, 1, 2, 3]
    expected_state = np.array([0, 1, 2, 3], dtype=np.complex128)
    expected_state /= np.linalg.norm(expected_state)

    state_vector.measure(0, PlannerMeasBasis(Plane.XZ, 0), 0)  # project onto |0⟩ state

    assert state_vector.num_qubits == 2
    assert np.allclose(state_vector.state().flatten(), expected_state)


def test_tensor_product(state_vector: StateVector) -> None:
    expected_state = np.asarray([i // 2 for i in range(2 ** (state_vector.num_qubits + 1))]) / np.sqrt(2)
    other_vector = StateVector.from_num_qubits(1)
    result = StateVector.tensor_product(state_vector, other_vector)

    assert result.num_qubits == 4
    assert np.allclose(result.state().flatten(), expected_state)


def test_normalize(state_vector: StateVector) -> None:
    state_vector.normalize()
    expected_norm = 1.0
    assert np.isclose(state_vector.norm(), expected_norm)


def test_get_norm(state_vector: StateVector) -> None:
    state = np.arange(2**state_vector.num_qubits, dtype=np.complex128)
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
    zero_state = StateVector([1, 0])
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    exp_val = zero_state.expectation(z_op, [0])
    assert np.isclose(exp_val, 1.0)  # ⟨0|Z|0⟩ = 1

    # Test |1⟩ state with Z operator
    one_state = StateVector([0, 1])
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
    plus_state = StateVector([1, 1] / np.sqrt(2))
    x_op = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    exp_val = plus_state.expectation(x_op, [0])
    assert np.isclose(exp_val, 1.0)  # ⟨+|X|+⟩ = 1

    # Test |-⟩ state with X operator
    minus_state = StateVector([1, -1] / np.sqrt(2))
    exp_val = minus_state.expectation(x_op, [0])
    assert np.isclose(exp_val, -1.0)  # ⟨-|X|-⟩ = -1

    # Test |+⟩ state with Y operator
    y_op = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    exp_val = plus_state.expectation(y_op, [0])
    assert np.isclose(exp_val, 0.0)  # ⟨+|Y|+⟩ = 0


def test_expectation_two_qubit_states() -> None:
    """Test expectation values with two-qubit states."""
    # Bell state: (|00⟩ + |11⟩)/√2
    bell_state = StateVector([1, 0, 0, 1] / np.sqrt(2))

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
    state = StateVector([1, 0, 0, 0, 0, 0, 0, 1] / np.sqrt(2))

    # Two-qubit operator on qubits 0 and 2
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    zz_op = np.kron(z_op, z_op).astype(np.complex128)
    exp_val = state.expectation(zz_op, [0, 2])
    assert np.isclose(exp_val, 1.0)  # ⟨state|Z₀⊗Z₂|state⟩ = 1


def test_expectation_unnormalized_state() -> None:
    """Test expectation values with unnormalized states."""
    # Unnormalized |0⟩ state with amplitude 2
    unnormalized_state = StateVector([2, 0])
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    exp_val = unnormalized_state.expectation(z_op, [0])
    assert np.isclose(exp_val, 1.0)  # Should still give ⟨0|Z|0⟩ = 1

    # Unnormalized superposition state
    unnormalized_plus = StateVector([3, 3])  # 3(|0⟩ + |1⟩)
    x_op = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    exp_val = unnormalized_plus.expectation(x_op, [0])
    assert np.isclose(exp_val, 1.0)  # Should still give ⟨+|X|+⟩ = 1


def test_expectation_identity_operator() -> None:
    """Test expectation values with identity operator."""
    # Any state should give expectation value 1 for identity
    state_vector = StateVector([0.5, 0.5, 0.5, 0.5])
    identity = np.eye(2, dtype=np.complex128)
    exp_val = state_vector.expectation(identity, [0])
    assert np.isclose(exp_val, 1.0)

    # Two-qubit identity
    identity_2q = np.eye(4, dtype=np.complex128)
    exp_val = state_vector.expectation(identity_2q, [0, 1])
    assert np.isclose(exp_val, 1.0)


def test_expectation_hermitian_check() -> None:
    """Test that non-Hermitian operators raise ValueError."""
    state = StateVector([1, 0])
    non_hermitian = np.array([[1, 1], [0, 1]], dtype=np.complex128)  # Not Hermitian

    with pytest.raises(ValueError, match="Operator must be Hermitian"):
        state.expectation(non_hermitian, [0])


def test_reorder_identity() -> None:
    """Test reorder with identity permutation."""
    state = StateVector([1, 2, 3, 4, 5, 6, 7, 8])
    original_state = state.state().copy()

    # Identity permutation should not change anything
    state.reorder([0, 1, 2])
    assert np.allclose(state.state(), original_state)


def test_reorder_two_qubit() -> None:
    """Test reorder with 2-qubit state."""
    # Initial state: |00⟩=1, |01⟩=2, |10⟩=3, |11⟩=4
    state = StateVector([1, 2, 3, 4])
    original_state = state.state().flatten().copy()

    # Swap qubits: [0,1] -> [1,0]
    state.reorder([1, 0])
    expected_state = np.array([1, 3, 2, 4])  # |00⟩=1, |01⟩=3, |10⟩=2, |11⟩=4
    assert np.allclose(state.state().flatten(), expected_state)

    # Reorder back should restore original
    state.reorder([1, 0])
    assert np.allclose(state.state().flatten(), original_state)


def test_reorder_three_qubit() -> None:
    """Test reorder with 3-qubit state."""
    # Initial state: |abc⟩ = a*4 + b*2 + c + 1 for indices
    state = StateVector([1, 2, 3, 4, 5, 6, 7, 8])
    original_state = state.state().flatten().copy()

    # Cyclic permutation: [0,1,2] -> [2,0,1]
    state.reorder([2, 0, 1])
    # After reorder: qubit 0 becomes old qubit 2, qubit 1 becomes old qubit 0, qubit 2 becomes old qubit 1
    # |abc⟩ -> |cab⟩: |000⟩->|000⟩=1, |001⟩->|010⟩=5, |010⟩->|100⟩=2, |011⟩->|110⟩=6, etc.
    expected_state = np.array([1, 5, 2, 6, 3, 7, 4, 8])
    assert np.allclose(state.state().flatten(), expected_state)

    # Apply twice more to complete the cycle
    state.reorder([2, 0, 1])
    state.reorder([2, 0, 1])
    assert np.allclose(state.state().flatten(), original_state)


def test_reorder_preserves_physical_state() -> None:
    """Test that reorder preserves the physical quantum state."""
    # Create a Bell state
    bell_state = StateVector([1, 0, 0, 1] / np.sqrt(2))

    # Calculate expectation values before reorder
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    exp_z0_before = bell_state.expectation(z_op, [0])
    exp_z1_before = bell_state.expectation(z_op, [1])

    # Reorder qubits
    bell_state.reorder([1, 0])

    # Physical state should be the same, but qubit labels are swapped
    exp_z0_after = bell_state.expectation(z_op, [0])  # This is now the old qubit 1
    exp_z1_after = bell_state.expectation(z_op, [1])  # This is now the old qubit 0

    # Expectation values should be swapped but still valid
    assert np.isclose(exp_z0_after, exp_z1_before)
    assert np.isclose(exp_z1_after, exp_z0_before)


def test_reorder_with_operations() -> None:
    """Test that operations work correctly after reorder."""
    state = StateVector([0, 0, 0, 1])  # |11⟩ state

    # Apply Z gate to qubit 1
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    state.evolve(z_op, [1])
    assert np.allclose(state.state().flatten(), [0, 0, 0, -1])  # Should be changed into -|11⟩

    # Reorder qubits
    state.reorder([1, 0])

    # Now apply Z gate to what is labeled as qubit 1 (but internally was original qubit 0)
    state.evolve(z_op, [1])
    # The state should still be back to |11⟩
    assert np.allclose(state.state().flatten(), [0, 0, 0, 1])


def test_reorder_with_measurement() -> None:
    """Test reorder followed by measurement."""
    # 3-qubit state
    state = StateVector([1, 1, 0, 0, 0, 0, 1, 1])  # |000⟩ + |001⟩ + |110⟩ + |111⟩

    # Reorder: [0,1,2] -> [2,1,0]
    state.reorder([2, 1, 0])

    # Check that we can still measure and get consistent results
    original_num_qubits = state.num_qubits

    # Measure external qubit 0 (which corresponds to original qubit 2)
    state.measure(0, PlannerMeasBasis(Plane.XZ, 0), 0)

    # Should have one less qubit
    assert state.num_qubits == original_num_qubits - 1

    # State should still be valid (normalized)
    assert np.isclose(state.norm(), 1.0)


def test_reorder_copy_independence() -> None:
    """Test that reorder operations on copies are independent."""
    state = StateVector([1, 2, 3, 4])
    state_copy = state.copy()

    # Reorder original
    state.reorder([1, 0])

    # Copy should be unchanged
    assert np.allclose(state_copy.state().flatten(), [1, 2, 3, 4])
    assert not np.allclose(state.state(), state_copy.state())

    # Reorder copy differently
    state_copy.reorder([0, 1])  # Identity - should remain same
    assert np.allclose(state_copy.state().flatten(), [1, 2, 3, 4])


def test_reorder_with_tensor_product() -> None:
    """Test reorder with tensor product operations."""
    state_a = StateVector([1, 0])  # |0⟩
    state_b = StateVector([0, 1])  # |1⟩

    # Tensor product: |0⟩ ⊗ |1⟩ = |01⟩
    combined = StateVector.tensor_product(state_a, state_b)
    assert np.allclose(combined.state().flatten(), [0, 1, 0, 0])

    # Reorder: [0,1] -> [1,0] should give |10⟩
    combined.reorder([1, 0])
    assert np.allclose(combined.state().flatten(), [0, 0, 1, 0])


def test_expectation_invariance_under_permutation() -> None:
    """Test that expectation values are invariant under qubit permutation when operator indices are adjusted."""
    # Create a 3-qubit entangled state: |000⟩ + |011⟩ + |101⟩ + |110⟩
    state = StateVector([1, 0, 0, 1, 0, 1, 1, 0])

    # Single qubit operators
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Test: Single qubit expectation values
    exp_z0_before = state.expectation(z_op, [0])
    exp_z1_before = state.expectation(z_op, [1])
    exp_z2_before = state.expectation(z_op, [2])

    # Apply permutation [0,1,2] -> [2,0,1]
    state_copy = state.copy()
    state_copy.reorder([2, 0, 1])

    # After reorder: external qubit 0 was original qubit 2, external qubit 1 was original qubit 0, etc.
    # So expectation on external qubit i should match original expectation on qubit permutation[i]
    exp_z0_after = state_copy.expectation(z_op, [0])  # Should match original qubit 2
    exp_z1_after = state_copy.expectation(z_op, [1])  # Should match original qubit 0
    exp_z2_after = state_copy.expectation(z_op, [2])  # Should match original qubit 1

    assert np.isclose(exp_z0_after, exp_z2_before)  # external 0 = original 2
    assert np.isclose(exp_z1_after, exp_z0_before)  # external 1 = original 0
    assert np.isclose(exp_z2_after, exp_z1_before)  # external 2 = original 1


def test_expectation_invariance_asymmetric_state() -> None:
    """Test expectation invariance with asymmetric states where order matters."""
    # Create an asymmetric 3-qubit state: |001⟩ + |010⟩
    state = StateVector(np.array([0, 1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2**3))

    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Original expectation values
    exp_z0_original = state.expectation(z_op, [0])
    exp_z1_original = state.expectation(z_op, [1])
    exp_z2_original = state.expectation(z_op, [2])

    # Apply permutation [0,1,2] -> [2,1,0]
    state.reorder([2, 1, 0])

    # After permutation, the state becomes: |100⟩ + |010⟩ in external qubit order
    # So external qubit expectation values should be:
    # - External qubit 0 (was qubit 2): should match original Z_2
    # - External qubit 1 (was qubit 1): should match original Z_1
    # - External qubit 2 (was qubit 0): should match original Z_0
    exp_z0_after = state.expectation(z_op, [0])
    exp_z1_after = state.expectation(z_op, [1])
    exp_z2_after = state.expectation(z_op, [2])

    assert np.isclose(exp_z0_after, exp_z2_original)
    assert np.isclose(exp_z1_after, exp_z1_original)
    assert np.isclose(exp_z2_after, exp_z0_original)


def test_expectation_invariance_mixed_operators() -> None:
    """Test expectation invariance with mixed X and Z operators."""
    # Create |+0⟩ state: (|00⟩ + |10⟩)/√2
    state = StateVector(np.array([1, 0, 1, 0]) / np.sqrt(2))

    x_op = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    z_op = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    xz_op = np.kron(x_op, z_op).astype(np.complex128)
    zx_op = np.kron(z_op, x_op).astype(np.complex128)

    # Original expectation values
    exp_xz_original = state.expectation(xz_op, [0, 1])  # X on qubit 0, Z on qubit 1
    exp_zx_original = state.expectation(zx_op, [0, 1])  # Z on qubit 0, X on qubit 1

    # Apply swap permutation [0,1] -> [1,0]
    state.reorder([1, 0])

    # After swap, X⊗Z on external [0,1] should match original Z⊗X on [0,1] (indices swapped)
    # And Z⊗X on external [0,1] should match original X⊗Z on [0,1] (indices swapped)
    exp_xz_after = state.expectation(xz_op, [0, 1])
    exp_zx_after = state.expectation(zx_op, [0, 1])

    assert np.isclose(exp_xz_after, exp_zx_original)
    assert np.isclose(exp_zx_after, exp_xz_original)


def test_array_method() -> None:
    """Test the __array__ method for numpy array conversion."""
    # Test basic conversion
    state = np.array([1, 2, 3, 4], dtype=np.complex128)
    sv = StateVector(state)

    # Convert to numpy array
    arr = np.array(sv)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.complex128
    assert np.allclose(arr.flatten(), state)

    # Test with different state
    plus_state = StateVector.from_num_qubits(2)  # |++⟩ state
    plus_arr = np.array(plus_state)
    expected = np.full(4, 0.5, dtype=np.complex128)  # All amplitudes are 1/2
    assert np.allclose(plus_arr.flatten(), expected)


def test_array_method_with_dtype() -> None:
    """Test __array__ method with different dtype specifications."""
    state = np.array([1, 2, 3, 4], dtype=np.complex128)
    sv = StateVector(state)

    # Test with explicit dtype
    arr_complex64 = np.array(sv, dtype=np.complex64)
    assert arr_complex64.dtype == np.complex64
    assert np.allclose(arr_complex64.flatten(), state.astype(np.complex64))

    # Test with float64 dtype (should keep real part only)
    arr_float = np.array(sv, dtype=np.float64)
    assert arr_float.dtype == np.float64
    assert np.allclose(arr_float.flatten(), state.real.astype(np.float64))


def test_array_method_preserves_shape() -> None:
    """Test that __array__ method preserves the tensor structure for multi-qubit states."""
    # 3-qubit state
    state = StateVector(np.arange(8, dtype=np.complex128))
    arr = np.array(state)

    # Should be able to reshape to tensor form
    tensor_form = arr.reshape((2, 2, 2))
    expected_tensor = np.arange(8, dtype=np.complex128).reshape((2, 2, 2))
    assert np.allclose(tensor_form, expected_tensor)
