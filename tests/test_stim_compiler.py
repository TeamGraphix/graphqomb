"""Tests for stim_compiler module."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from graphqomb.command import TICK, E
from graphqomb.common import Axis, AxisMeasBasis, Plane, PlannerMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.qompiler import qompile
from graphqomb.schedule_solver import ScheduleConfig, Strategy
from graphqomb.scheduler import Scheduler
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

    stim_str = stim_compile(pattern, p_depol_after_clifford=0.01)

    # Check DEPOLARIZE instructions are present
    assert "DEPOLARIZE1(0.01)" in stim_str
    assert "DEPOLARIZE2(0.01)" in stim_str


def test_stim_compile_with_measurement_errors_x() -> None:
    """Test that X measurement errors are correctly inserted."""
    pattern, _, _ = create_simple_pattern_x_measurement()

    stim_str = stim_compile(pattern, p_before_meas_flip=0.01)

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

    stim_str = stim_compile(pattern, p_before_meas_flip=0.01)

    # For Y measurement, both X_ERROR and Z_ERROR should be inserted before MY
    assert "X_ERROR(0.01)" in stim_str
    assert "Z_ERROR(0.01)" in stim_str


def test_stim_compile_with_measurement_errors_z() -> None:
    """Test that Z measurement errors are correctly inserted."""
    pattern, _, _ = create_simple_pattern_z_measurement()

    stim_str = stim_compile(pattern, p_before_meas_flip=0.01)

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


def test_stim_compile_with_tick_commands() -> None:
    """Test that TICK commands are properly compiled to Stim format."""
    # Create a simple graph and compile with TICK commands
    graph = GraphState()
    node0 = graph.add_physical_node()
    node1 = graph.add_physical_node()
    node2 = graph.add_physical_node()
    graph.add_physical_edge(node0, node1)
    graph.add_physical_edge(node1, node2)
    qindex = 0
    graph.register_input(node0, qindex)
    graph.register_output(node2, qindex)

    graph.assign_meas_basis(node0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(node1, PlannerMeasBasis(Plane.XY, 0.0))

    flow = {node0: {node1}, node1: {node2}}
    scheduler = Scheduler(graph, flow)
    config = ScheduleConfig(strategy=Strategy.MINIMIZE_TIME)
    scheduler.solve_schedule(config)

    # Compile with scheduler-driven TICK commands (entanglement auto-scheduled by solve_schedule)
    pattern = qompile(graph, flow, scheduler=scheduler)

    # Verify TICK commands are present in pattern
    tick_count = sum(1 for cmd in pattern if isinstance(cmd, TICK))
    assert tick_count > 0, "Pattern should contain TICK commands"
    assert tick_count == scheduler.num_slices(), "Each time slice should yield one TICK command"

    # Compile to Stim format
    stim_str = stim_compile(pattern)

    # Verify TICK instructions are present in Stim output
    assert "TICK" in stim_str, "Stim output should contain TICK instructions"

    # Count TICK instructions in output
    stim_tick_count = stim_str.count("TICK")
    assert stim_tick_count == tick_count, "Number of TICK instructions should match pattern"


def _entanglement_slices_from_pattern(pattern: Pattern) -> dict[tuple[int, int], int]:
    r"""Extract entanglement time slices from a pattern.

    Parameters
    ----------
    pattern : `Pattern`
        The pattern to inspect.

    Returns
    -------
    `dict`\[`tuple`\[`int`, `int`\], `int`\]
        A mapping from entanglement edge ``(u, v)`` to the time slice index, represented as the
        number of preceding `TICK` commands.
    """
    ticks = 0
    entangle_slice: dict[tuple[int, int], int] = {}
    for cmd in pattern:
        if isinstance(cmd, TICK):
            ticks += 1
        elif isinstance(cmd, E):
            entangle_slice[cmd.nodes] = ticks
    return entangle_slice


def _cz_slices_from_stim(stim_str: str) -> dict[tuple[int, int], int]:
    r"""Extract CZ time slices from a stim circuit string.

    Parameters
    ----------
    stim_str : `str`
        The stim circuit string to inspect.

    Returns
    -------
    `dict`\[`tuple`\[`int`, `int`\], `int`\]
        A mapping from CZ targets ``(q1, q2)`` to the time slice index, represented as the number
        of preceding ``TICK`` instructions.
    """
    ticks = 0
    cz_slice: dict[tuple[int, int], int] = {}
    for line in stim_str.splitlines():
        if line == "TICK":
            ticks += 1
        elif line.startswith("CZ "):
            _, q1, q2 = line.split()
            cz_slice[int(q1), int(q2)] = ticks
    return cz_slice


def test_stim_compile_respects_manual_entangle_time() -> None:
    """Manual entanglement times should determine the CZ slice in both Pattern and Stim output."""
    graph = GraphState()
    in_node = graph.add_physical_node()
    mid_node = graph.add_physical_node()
    out_node = graph.add_physical_node()

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)

    graph.add_physical_edge(in_node, mid_node)
    graph.add_physical_edge(mid_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(mid_node, PlannerMeasBasis(Plane.XY, 0.0))

    scheduler = Scheduler(graph, {in_node: {mid_node}, mid_node: {out_node}})

    scheduler.manual_schedule(
        prepare_time={mid_node: 0, out_node: 0},
        measure_time={in_node: 3, mid_node: 4},
        # Intentionally provide edges in reversed order to ensure they are still accepted.
        entangle_time={
            (mid_node, in_node): 2,
            (out_node, mid_node): 1,
        },
    )
    scheduler.validate_schedule()

    pattern = qompile(graph, {in_node: {mid_node}, mid_node: {out_node}}, scheduler=scheduler)
    entangle_slice = _entanglement_slices_from_pattern(pattern)

    assert entangle_slice[in_node, mid_node] == 2
    assert entangle_slice[mid_node, out_node] == 1

    cz_slice = _cz_slices_from_stim(stim_compile(pattern))

    assert cz_slice[in_node, mid_node] == 2
    assert cz_slice[mid_node, out_node] == 1
