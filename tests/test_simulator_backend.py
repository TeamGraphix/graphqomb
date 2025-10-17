"""Tests for the simulator backend module."""

from __future__ import annotations

import pytest

from graphqomb.simulator_backend import QubitIndexManager


@pytest.fixture
def manager() -> QubitIndexManager:
    """Create a QubitIndexManager with initial indices [0, 1, 2].

    Returns
    -------
    QubitIndexManager
        Initialized manager with indices [0, 1, 2].
    """
    return QubitIndexManager(3)


def test_qubit_index_manager_init() -> None:
    """Test QubitIndexManager initialization."""
    manager = QubitIndexManager(3)
    assert manager.match([0, 1, 2])

    # Test with empty list
    empty_manager = QubitIndexManager(0)
    assert empty_manager.match([])


def test_add_qubits(manager: QubitIndexManager) -> None:
    """Test adding qubits to the manager."""
    # Add 2 qubits to initial [0, 1, 2]
    manager.add_qubits(2)
    assert manager.match([0, 1, 2, 3, 4])

    # Add more qubits
    manager.add_qubits(1)
    assert manager.match([0, 1, 2, 3, 4, 5])


def test_add_qubits_empty() -> None:
    """Test adding qubits to an empty manager."""
    manager = QubitIndexManager(0)
    manager.add_qubits(3)
    assert manager.match([0, 1, 2])


def test_remove_qubit(manager: QubitIndexManager) -> None:
    """Test removing qubits from the manager."""
    # Remove middle qubit (index 1)
    manager.remove_qubit(1)
    assert manager.match([0, 1])  # [0, 2] becomes [0, 1]

    # Remove first qubit
    manager.remove_qubit(0)
    assert manager.match([0])  # [0, 1] becomes [0]


def test_remove_qubit_edge_cases() -> None:
    """Test edge cases for removing qubits."""
    manager = QubitIndexManager(4)

    # Remove last qubit
    manager.remove_qubit(3)
    assert manager.match([0, 1, 2])

    # Remove first qubit
    manager.remove_qubit(0)
    assert manager.match([0, 1])  # [1, 2] becomes [0, 1]


def test_match(manager: QubitIndexManager) -> None:
    """Test the match method."""
    assert manager.match([0, 1, 2])
    assert not manager.match([2, 1, 0])
    assert not manager.match([0, 1])
    assert not manager.match([0, 1, 2, 3])


def test_reorder(manager: QubitIndexManager) -> None:
    """Test reordering indices."""
    # Test permutation [2, 0, 1] -> [q1, q2, q0]
    manager.reorder([2, 0, 1])
    assert manager.match([2, 0, 1])

    # Test identity permutation
    manager.reorder([0, 1, 2])
    assert manager.match([2, 0, 1])  # No change since we apply [0,1,2] to [2,0,1]


def test_reorder_invalid_length() -> None:
    """Test reorder with invalid permutation length."""
    manager = QubitIndexManager(3)

    with pytest.raises(ValueError, match="Permutation length must match the number of indices"):
        manager.reorder([0, 1])  # Too short

    with pytest.raises(ValueError, match="Permutation length must match the number of indices"):
        manager.reorder([0, 1, 2, 3])  # Too long


def test_inverse_permutation(manager: QubitIndexManager) -> None:
    """Test getting recovery permutation."""
    # Initial state [0, 1, 2], recovery should be [0, 1, 2]
    assert manager.inverse_permutation() == [0, 1, 2]

    # After reordering [2, 0, 1] -> indices become [2, 0, 1]
    manager.reorder([2, 0, 1])
    recovery = manager.inverse_permutation()
    assert recovery == [1, 2, 0]  # To get back to [0, 1, 2] from [2, 0, 1]


def test_external_to_internal_single(manager: QubitIndexManager) -> None:
    """Test external to internal conversion for single qubit."""
    assert manager.external_to_internal(0) == 0
    assert manager.external_to_internal(1) == 1
    assert manager.external_to_internal(2) == 2

    # After reordering
    manager.reorder([2, 0, 1])  # [0, 1, 2] -> [2, 0, 1]
    assert manager.external_to_internal(0) == 2
    assert manager.external_to_internal(1) == 0
    assert manager.external_to_internal(2) == 1


def test_external_to_internal_sequence(manager: QubitIndexManager) -> None:
    """Test external to internal conversion for sequence of qubits."""
    assert manager.external_to_internal([0, 1, 2]) == (0, 1, 2)
    assert manager.external_to_internal([2, 0]) == (2, 0)

    # After reordering
    manager.reorder([2, 0, 1])  # [0, 1, 2] -> [2, 0, 1]
    assert manager.external_to_internal([0, 1, 2]) == (2, 0, 1)
    assert manager.external_to_internal([1, 0]) == (0, 2)


def test_internal_to_external_single(manager: QubitIndexManager) -> None:
    """Test internal to external conversion for single qubit."""
    assert manager.internal_to_external(0) == 0
    assert manager.internal_to_external(1) == 1
    assert manager.internal_to_external(2) == 2

    # After reordering
    manager.reorder([2, 0, 1])  # [0, 1, 2] -> [2, 0, 1]
    assert manager.internal_to_external(0) == 1
    assert manager.internal_to_external(1) == 2
    assert manager.internal_to_external(2) == 0


def test_internal_to_external_sequence(manager: QubitIndexManager) -> None:
    """Test internal to external conversion for sequence of qubits."""
    assert manager.internal_to_external([0, 1, 2]) == (0, 1, 2)
    assert manager.internal_to_external([2, 0]) == (2, 0)

    # After reordering
    manager.reorder([2, 0, 1])  # [0, 1, 2] -> [2, 0, 1]
    assert manager.internal_to_external([0, 1, 2]) == (1, 2, 0)
    assert manager.internal_to_external([1, 0]) == (2, 1)


def test_complex_operations() -> None:
    """Test complex sequence of operations."""
    manager = QubitIndexManager(2)

    # Add qubits
    manager.add_qubits(2)  # [0, 1, 2, 3]
    assert manager.match([0, 1, 2, 3])

    # Remove middle qubit
    manager.remove_qubit(1)  # [0, 2, 3] -> [0, 1, 2]
    assert manager.match([0, 1, 2])

    # Reorder
    manager.reorder([1, 2, 0])  # [0, 1, 2] -> [1, 2, 0]
    assert manager.match([1, 2, 0])

    # Test external to internal mapping
    assert manager.external_to_internal([0, 1, 2]) == (1, 2, 0)


def test_edge_case_single_qubit() -> None:
    """Test operations with single qubit."""
    manager = QubitIndexManager(1)

    assert manager.match([0])
    assert manager.inverse_permutation() == [0]
    assert manager.external_to_internal(0) == 0
    assert manager.external_to_internal([0]) == (0,)

    # Reorder with identity
    manager.reorder([0])
    assert manager.match([0])
