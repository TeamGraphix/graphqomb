"""Tests for importing supported Stim circuits into GraphQOMB patterns."""

from __future__ import annotations

import pytest

from graphqomb.stim_importer import stim_text_to_pattern

pytest.importorskip("stim")


def test_stim_text_to_pattern_imports_unitary_clifford_block() -> None:
    result = stim_text_to_pattern(
        """
        H 10
        CX 10 12
        S_DAG 12
        """
    )

    assert result.stim_to_qubit == {10: 0, 12: 1}
    assert result.qubit_to_stim == {0: 10, 1: 12}
    assert result.mpp_extractions == ()
    assert set(result.pattern.input_node_indices.values()) == {0, 1}
    assert set(result.pattern.output_node_indices.values()) == {0, 1}


def test_stim_text_to_pattern_preserves_sparse_qubit_coordinates() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(1, 2) 10
        QUBIT_COORDS(3, 4) 99
        H 99
        """
    )

    assert result.stim_to_qubit == {10: 0, 99: 1}
    assert result.pattern.input_coordinates
    assert set(result.pattern.input_coordinates.values()) == {(1.0, 2.0), (3.0, 4.0)}


def test_stim_text_to_pattern_imports_tick_separated_mpp_block() -> None:
    result = stim_text_to_pattern(
        """
        H 10
        TICK
        MPP X10*Z12
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(3) rec[-1]
        TICK
        CZ 10 12
        """
    )

    assert result.stim_to_qubit == {10: 0, 12: 1}
    assert len(result.mpp_extractions) == 1
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert set(result.pattern.pauli_frame.logical_observables) == {3}
    assert set(result.pattern.output_node_indices.values()) == {0, 1}


def test_stim_text_to_pattern_rejects_mixed_mpp_and_unitary_block() -> None:
    with pytest.raises(ValueError, match="separated from unitary gate instructions by TICK"):
        stim_text_to_pattern("H 0\nMPP X0\n")


def test_stim_text_to_pattern_rejects_measurement_instruction() -> None:
    with pytest.raises(ValueError, match="Unsupported Stim instruction"):
        stim_text_to_pattern("M 0\n")
