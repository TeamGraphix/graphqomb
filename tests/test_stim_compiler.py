"""Tests for stim_compiler module."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from graphqomb.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.qompiler import qompile
from graphqomb.stim_compiler import stim_compile

if TYPE_CHECKING:
    from graphqomb.pattern import Pattern


def create_simple_pattern_x_measurement() -> tuple[Pattern, int, int]:
    """Create a simple pattern with X measurement for testing.

    Returns
    -------
    tuple[Pattern, int, int]
        Pattern and expected node for X measurement
    """
    graph = GraphState()
    in_node = graph.add_physical_node()
    meas_node = graph.add_physical_node()
    out_node = graph.add_physical_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_physical_edge(in_node, meas_node)
    graph.add_physical_edge(meas_node, out_node)

    # X measurement: XY plane with angle 0
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    return pattern, meas_node, in_node


def create_simple_pattern_y_measurement() -> tuple[Pattern, int, int]:
    """Create a simple pattern with Y measurement for testing.

    Returns
    -------
    tuple[Pattern, int, int]
        Pattern and expected node for Y measurement
    """
    graph = GraphState()
    in_node = graph.add_physical_node()
    meas_node = graph.add_physical_node()
    out_node = graph.add_physical_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_physical_edge(in_node, meas_node)
    graph.add_physical_edge(meas_node, out_node)

    # Y measurement: XY plane with angle π/2
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, math.pi / 2))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, math.pi / 2))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    return pattern, meas_node, in_node


def create_simple_pattern_z_measurement() -> tuple[Pattern, int, int]:
    """Create a simple pattern with Z measurement for testing.

    Returns
    -------
    tuple[Pattern, int, int]
        Pattern and expected node for Z measurement
    """
    graph = GraphState()
    in_node = graph.add_physical_node()
    meas_node = graph.add_physical_node()
    out_node = graph.add_physical_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_physical_edge(in_node, meas_node)
    graph.add_physical_edge(meas_node, out_node)

    # Z measurement: XZ plane with angle 0
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XZ, 0.0))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XZ, 0.0))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    return pattern, meas_node, in_node


def test_stim_compile_basic_pattern() -> None:
    """Test basic pattern compilation to stim format."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    stim_str = stim_compile(pattern)

    # Check basic structure
    assert "RX" in stim_str
    assert "CZ" in stim_str
    assert "MX" in stim_str
    assert stim_str.count("\n") > 0


def test_stim_compile_x_measurement() -> None:
    """Test X measurement compilation."""
    pattern, meas_node, in_node = create_simple_pattern_x_measurement()

    stim_str = stim_compile(pattern)

    # X measurement should generate MX command
    assert "MX" in stim_str
    assert f"MX {meas_node}" in stim_str or f"MX {in_node}" in stim_str


def test_stim_compile_y_measurement() -> None:
    """Test Y measurement compilation."""
    pattern, meas_node, in_node = create_simple_pattern_y_measurement()

    stim_str = stim_compile(pattern)

    # Y measurement should generate MY command
    assert "MY" in stim_str
    assert f"MY {meas_node}" in stim_str or f"MY {in_node}" in stim_str


def test_stim_compile_z_measurement() -> None:
    """Test Z measurement compilation."""
    pattern, meas_node, in_node = create_simple_pattern_z_measurement()

    stim_str = stim_compile(pattern)

    # Z measurement should generate MZ command
    assert "MZ" in stim_str
    assert f"MZ {meas_node}" in stim_str or f"MZ {in_node}" in stim_str


def test_stim_compile_with_depolarization() -> None:
    """Test that depolarization error is correctly inserted."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    stim_str = stim_compile(pattern, after_clifford_depolarization=0.01)

    # Check DEPOLARIZE instructions are present
    assert "DEPOLARIZE1(0.01)" in stim_str
    assert "DEPOLARIZE2(0.01)" in stim_str


def test_stim_compile_with_measurement_errors_x() -> None:
    """Test that X measurement errors are correctly inserted."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    stim_str = stim_compile(pattern, before_measure_flip_probability=0.01)

    # For X measurement, Z_ERROR should be inserted before MX
    assert "Z_ERROR(0.01)" in stim_str
    lines = stim_str.split("\n")
    for i, line in enumerate(lines):
        if "Z_ERROR(0.01)" in line and i + 1 < len(lines):
            # Next non-empty line should be MX
            next_line = lines[i + 1]
            assert "MX" in next_line


def test_stim_compile_with_measurement_errors_y() -> None:
    """Test that Y measurement errors are correctly inserted."""
    pattern, _, _ = create_simple_pattern_y_measurement()

    stim_str = stim_compile(pattern, before_measure_flip_probability=0.01)

    # For Y measurement, both X_ERROR and Z_ERROR should be inserted before MY
    assert "X_ERROR(0.01)" in stim_str
    assert "Z_ERROR(0.01)" in stim_str


def test_stim_compile_with_measurement_errors_z() -> None:
    """Test that Z measurement errors are correctly inserted."""
    pattern, _, _ = create_simple_pattern_z_measurement()

    stim_str = stim_compile(pattern, before_measure_flip_probability=0.01)

    # For Z measurement, X_ERROR should be inserted before MZ
    assert "X_ERROR(0.01)" in stim_str
    lines = stim_str.split("\n")
    for i, line in enumerate(lines):
        if "X_ERROR(0.01)" in line and i + 1 < len(lines):
            # Next non-empty line should be MZ
            next_line = lines[i + 1]
            assert "MZ" in next_line


def test_stim_compile_with_detectors() -> None:
    """Test DETECTOR generation with parity check groups."""
    graph = GraphState()
    in_node = graph.add_physical_node()
    meas_node = graph.add_physical_node()
    out_node = graph.add_physical_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_physical_edge(in_node, meas_node)
    graph.add_physical_edge(meas_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    # Add parity check groups
    parity_check_group = [{in_node}]
    pattern = qompile(graph, xflow, parity_check_group=parity_check_group)

    stim_str = stim_compile(pattern)

    # Check DETECTOR instruction is present
    assert "DETECTOR" in stim_str
    # DETECTOR may be empty if the dependent chain resolves to empty set
    # This is valid behavior for certain graph configurations


def test_stim_compile_with_logical_observables() -> None:
    """Test OBSERVABLE_INCLUDE generation."""
    pattern, meas_node, _ = create_simple_pattern_x_measurement()

    # Define logical observables
    logical_observables = {0: [meas_node]}

    stim_str = stim_compile(pattern, logical_observables=logical_observables)

    # Check OBSERVABLE_INCLUDE instruction is present
    assert "OBSERVABLE_INCLUDE(0)" in stim_str
    # OBSERVABLE_INCLUDE may be empty if the dependent chain resolves to empty set
    # This is valid behavior for certain graph configurations


def test_stim_compile_unsupported_basis() -> None:
    """Test that unsupported measurement basis raises ValueError."""
    graph = GraphState()
    in_node = graph.add_physical_node()
    meas_node = graph.add_physical_node()
    out_node = graph.add_physical_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_physical_edge(in_node, meas_node)
    graph.add_physical_edge(meas_node, out_node)

    # Non-Pauli measurement: XY plane with arbitrary angle
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.1))
    graph.assign_meas_basis(meas_node, PlannerMeasBasis(Plane.XY, 0.1))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    # Should raise ValueError for unsupported measurement basis
    with pytest.raises(ValueError, match="Unsupported measurement basis"):
        stim_compile(pattern)


def test_stim_compile_empty_pattern() -> None:
    """Test compilation of minimal pattern."""
    graph = GraphState()
    in_node = graph.add_physical_node()
    out_node = graph.add_physical_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_physical_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in_node: {out_node}}
    pattern = qompile(graph, xflow)

    stim_str = stim_compile(pattern)

    # Should compile without errors
    assert isinstance(stim_str, str)
    assert len(stim_str) > 0


def test_stim_compile_axis_meas_basis() -> None:
    """Test compilation with AxisMeasBasis."""
    graph = GraphState()
    in_node = graph.add_physical_node()
    meas_node = graph.add_physical_node()
    out_node = graph.add_physical_node()

    q_idx = 0
    graph.register_input(in_node, q_idx)
    graph.register_output(out_node, q_idx)

    graph.add_physical_edge(in_node, meas_node)
    graph.add_physical_edge(meas_node, out_node)

    # Use AxisMeasBasis instead of PlannerMeasBasis
    graph.assign_meas_basis(in_node, AxisMeasBasis(Axis.X, Sign.PLUS))
    graph.assign_meas_basis(meas_node, AxisMeasBasis(Axis.Y, Sign.PLUS))

    xflow = {in_node: {meas_node}, meas_node: {out_node}}
    pattern = qompile(graph, xflow)

    stim_str = stim_compile(pattern)

    # Should compile with both MX and MY
    assert "MX" in stim_str
    assert "MY" in stim_str
