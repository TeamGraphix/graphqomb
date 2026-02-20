"""Tests for ptn_format module."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from graphqomb.command import TICK, E, M, N, X, Z
from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.pattern import Pattern
from graphqomb.ptn_format import (
    dump,
    dumps,
    load,
    loads,
)
from graphqomb.qompiler import qompile


def create_simple_pattern() -> Pattern:
    """Create a simple pattern for testing.

    Returns
    -------
    Pattern
        A compiled MBQC pattern for testing.
    """
    graph = GraphState()
    in_node = graph.add_physical_node(coordinate=(0.0, 0.0))
    mid_node = graph.add_physical_node(coordinate=(1.0, 0.0))
    out_node = graph.add_physical_node(coordinate=(2.0, 0.0))

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)

    graph.add_physical_edge(in_node, mid_node)
    graph.add_physical_edge(mid_node, out_node)

    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(mid_node, PlannerMeasBasis(Plane.XY, math.pi / 2))

    xflow = {in_node: {mid_node}, mid_node: {out_node}}
    return qompile(graph, xflow)


def test_dumps_basic() -> None:
    """Test basic pattern serialization."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    assert ".version 1" in ptn_str
    assert ".input" in ptn_str
    assert ".output" in ptn_str
    assert "#======== QUANTUM ========" in ptn_str
    assert "#======== CLASSICAL ========" in ptn_str


def test_dumps_contains_commands() -> None:
    """Test that dumps includes all command types."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    # Check for command types
    assert "N " in ptn_str  # Node creation
    assert "E " in ptn_str  # Entanglement
    assert "M " in ptn_str  # Measurement


def test_dumps_coordinates() -> None:
    """Test that coordinates are correctly serialized."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    assert ".coord 0 0.0 0.0" in ptn_str


def test_dumps_pauli_measurements() -> None:
    """Test that Pauli measurements are correctly formatted with +/- signs."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    # X measurement (XY plane, angle 0) should be formatted as "X +"
    assert "M 0 X +" in ptn_str
    # Y measurement (XY plane, angle pi/2) should be formatted as "Y +"
    assert "M 1 Y +" in ptn_str


def test_loads_basic() -> None:
    """Test basic pattern deserialization."""
    ptn_str = """# Test pattern
.version 1
.input 0:0
.output 2:0
.coord 0 0.0 0.0

#======== QUANTUM ========
[0]
N 1
N 2
E 0 1
M 0 XY 0
M 1 XY 0

#======== CLASSICAL ========
    .xflow 0 -> 1
    """
    result = loads(ptn_str)

    assert isinstance(result, Pattern)
    assert result.input_node_indices == {0: 0}
    assert result.output_node_indices == {2: 0}
    assert result.input_coordinates == {0: (0.0, 0.0)}


def test_loads_commands() -> None:
    """Test that commands are correctly parsed."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
N 1
N 3 1.0 2.0
N 2
E 0 1
M 0 XY 0
M 1 XY pi/2
M 3 XY 0
X 2
Z 2
"""
    result = loads(ptn_str)

    commands = result.commands
    # Check command types
    assert any(isinstance(c, N) and c.node == 1 for c in commands)
    assert any(isinstance(c, N) and c.node == 3 and c.coordinate == (1.0, 2.0) for c in commands)
    assert any(isinstance(c, E) and c.nodes == (0, 1) for c in commands)
    assert any(isinstance(c, M) and c.node == 0 for c in commands)
    assert any(isinstance(c, X) and c.node == 2 for c in commands)
    assert any(isinstance(c, Z) and c.node == 2 for c in commands)


def test_loads_timeslices() -> None:
    """Test that timeslice markers generate TICK commands."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
N 1
E 0 1
[1]
M 0 XY 0
[2]
"""
    result = loads(ptn_str)

    # Count TICK commands
    tick_count = sum(1 for c in result.commands if isinstance(c, TICK))
    assert tick_count == 2


def test_loads_angle_parsing() -> None:
    """Test various angle format parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 5:0

[0]
N 1
N 2
N 3
N 4
N 5
M 0 XY 0
M 1 XY pi
M 2 XY pi/2
M 3 XY pi/4
M 4 XY 3pi/4
"""
    result = loads(ptn_str)

    measurements = [c for c in result.commands if isinstance(c, M)]
    angles = {m.node: m.meas_basis.angle for m in measurements}

    assert math.isclose(angles[0], 0.0)
    assert math.isclose(angles[1], math.pi)
    assert math.isclose(angles[2], math.pi / 2)
    assert math.isclose(angles[3], math.pi / 4)
    assert math.isclose(angles[4], 3 * math.pi / 4)


def test_loads_pauli_measurements() -> None:
    """Test parsing of Pauli measurement format (X/Y/Z +/-)."""
    ptn_str = """
.version 1
.input 0:0
.output 6:0

[0]
N 1
N 2
N 3
N 4
N 5
N 6
M 0 X +
M 1 X -
M 2 Y +
M 3 Y -
M 4 Z +
M 5 Z -
"""
    result = loads(ptn_str)

    measurements = [c for c in result.commands if isinstance(c, M)]
    assert len(measurements) == 6

    # Check that Pauli measurements are parsed correctly
    m0 = next(m for m in measurements if m.node == 0)
    assert math.isclose(m0.meas_basis.angle, 0.0)  # X +

    m1 = next(m for m in measurements if m.node == 1)
    assert math.isclose(m1.meas_basis.angle, math.pi)  # X -

    m2 = next(m for m in measurements if m.node == 2)
    assert math.isclose(m2.meas_basis.angle, math.pi / 2)  # Y +

    m3 = next(m for m in measurements if m.node == 3)
    assert math.isclose(m3.meas_basis.angle, 3 * math.pi / 2)  # Y -

    m4 = next(m for m in measurements if m.node == 4)
    assert math.isclose(m4.meas_basis.angle, 0.0)  # Z +

    m5 = next(m for m in measurements if m.node == 5)
    assert math.isclose(m5.meas_basis.angle, math.pi)  # Z -


def test_loads_flow_parsing() -> None:
    """Test xflow and zflow parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
N 1
N 2
E 0 1
M 0 XY 0
M 1 XY 0

.xflow 0 -> 1 2
.zflow 0 -> 0 1
"""
    result = loads(ptn_str)

    assert result.pauli_frame.xflow == {0: {1, 2}}
    assert result.pauli_frame.zflow == {0: {0, 1}}


def test_loads_detector_parsing() -> None:
    """Test detector (parity check group) parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
N 1
N 2
M 0 XY 0
M 1 XY 0

.detector 0 1
.detector 1
"""
    result = loads(ptn_str)

    assert len(result.pauli_frame.parity_check_group) == 2
    assert result.pauli_frame.parity_check_group[0] == {0, 1}
    assert result.pauli_frame.parity_check_group[1] == {1}


def test_loads_observable_parsing() -> None:
    """Test logical observable parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
N 1
N 2
M 0 XY 0
M 1 XY 0

.observable 0 0 1
.observable 1
"""
    result = loads(ptn_str)

    assert result.pauli_frame.logical_observables[0] == {0, 1}
    assert result.pauli_frame.logical_observables[1] == set()


def test_loads_missing_version() -> None:
    """Test that missing version raises ValueError."""
    ptn_str = """
.input 0:0
.output 1:0
[0]
M 0 XY 0
"""
    with pytest.raises(ValueError, match=r"Missing \.version directive"):
        loads(ptn_str)


def test_loads_unsupported_version() -> None:
    """Test that unsupported version raises ValueError."""
    ptn_str = """
.version 99
.input 0:0
.output 1:0
"""
    with pytest.raises(ValueError, match=r"Unsupported \.ptn version"):
        loads(ptn_str)


def test_loads_unknown_command() -> None:
    """Test that unknown command raises ValueError."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
UNKNOWN 0
"""
    with pytest.raises(ValueError, match="Unknown command"):
        loads(ptn_str)


def test_roundtrip() -> None:
    """Test that dumps followed by loads preserves data."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)
    result = loads(ptn_str)

    # Check input/output nodes match
    assert result.input_node_indices == pattern.input_node_indices
    assert result.output_node_indices == pattern.output_node_indices

    # Check command count matches (excluding internal differences)
    original_count = len([c for c in pattern.commands if not isinstance(c, (X, Z))])
    parsed_count = len([c for c in result.commands if not isinstance(c, (X, Z))])
    assert original_count == parsed_count


def test_dump_and_load_file() -> None:
    """Test file I/O operations."""
    pattern = create_simple_pattern()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.ptn"

        # Write to file
        dump(pattern, filepath)

        # Verify file exists
        assert filepath.exists()

        # Read from file
        result = load(filepath)

        # Verify content
        assert result.input_node_indices == pattern.input_node_indices
        assert result.output_node_indices == pattern.output_node_indices


def test_dump_contains_observable_directive() -> None:
    """Test that dumps includes observable directives."""
    graph = GraphState()
    in_node = graph.add_physical_node()
    out_node = graph.add_physical_node()

    graph.register_input(in_node, 0)
    graph.register_output(out_node, 0)
    graph.add_physical_edge(in_node, out_node)
    graph.assign_meas_basis(in_node, PlannerMeasBasis(Plane.XY, 0.0))

    pattern = qompile(graph, {in_node: {out_node}}, logical_observables={0: {in_node}})
    ptn_str = dumps(pattern)

    assert ".observable 0 0" in ptn_str


def test_multiple_input_output_qubits() -> None:
    """Test pattern with multiple input/output qubits."""
    graph = GraphState()
    in0 = graph.add_physical_node()
    in1 = graph.add_physical_node()
    out0 = graph.add_physical_node()
    out1 = graph.add_physical_node()

    graph.register_input(in0, 0)
    graph.register_input(in1, 1)
    graph.register_output(out0, 0)
    graph.register_output(out1, 1)

    graph.add_physical_edge(in0, out0)
    graph.add_physical_edge(in1, out1)

    graph.assign_meas_basis(in0, PlannerMeasBasis(Plane.XY, 0.0))
    graph.assign_meas_basis(in1, PlannerMeasBasis(Plane.XY, 0.0))

    xflow = {in0: {out0}, in1: {out1}}
    pattern = qompile(graph, xflow)

    ptn_str = dumps(pattern)
    result = loads(ptn_str)

    assert len(result.input_node_indices) == 2
    assert len(result.output_node_indices) == 2


def test_3d_coordinates() -> None:
    """Test 3D coordinate serialization and parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0
.coord 0 1.0 2.0 3.0

[0]
N 1 4.0 5.0 6.0
M 0 XY 0
"""
    result = loads(ptn_str)

    assert result.input_coordinates[0] == (1.0, 2.0, 3.0)

    # Check N command coordinate
    n_cmd = next(c for c in result.commands if isinstance(c, N))
    assert n_cmd.coordinate == (4.0, 5.0, 6.0)


def test_different_measurement_planes() -> None:
    """Test all measurement planes are correctly handled."""
    ptn_str = """
.version 1
.input 0:0
.output 3:0

[0]
N 1
N 2
N 3
M 0 XY pi/4
M 1 XZ pi/4
M 2 YZ pi/4
"""
    result = loads(ptn_str)

    measurements = [c for c in result.commands if isinstance(c, M)]
    planes = {m.node: m.meas_basis.plane for m in measurements}

    assert planes[0] == Plane.XY
    assert planes[1] == Plane.XZ
    assert planes[2] == Plane.YZ


def test_empty_flow() -> None:
    """Test pattern with empty flow mappings."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
N 1
M 0 XY 0

#======== CLASSICAL ========
"""
    result = loads(ptn_str)

    assert result.pauli_frame.xflow == {}
    assert result.pauli_frame.zflow == {}


def test_comments_ignored() -> None:
    """Test that comments are properly ignored."""
    ptn_str = """
# This is a comment
.version 1
# Another comment
.input 0:0  # inline comment should be parsed as part of content
.output 1:0

# Comment in quantum section
[0]
N 1
M 0 XY 0
"""
    result = loads(ptn_str)
    # Should parse without error
    assert result.input_node_indices is not None


def test_loads_allows_dag_violating_measurement_order() -> None:
    """Test that importer skips DAG/schedule causality checks."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
N 1
N 2
M 1 XY 0
M 0 XY 0

.xflow 0 -> 1
"""
    result = loads(ptn_str)
    assert result.pauli_frame.xflow == {0: {1}}


def test_loads_reject_duplicate_measurement() -> None:
    """Test that duplicate measurement on the same node is rejected."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
N 1
M 0 XY 0
M 0 XY pi
"""
    with pytest.raises(ValueError, match="measured more than once"):
        loads(ptn_str)


def test_loads_reject_unknown_flow_target() -> None:
    """Test that flows referencing unknown nodes are rejected."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
N 1
M 0 XY 0

.xflow 0 -> 99
"""
    with pytest.raises(ValueError, match="unknown target nodes"):
        loads(ptn_str)


def test_loads_reject_observable_non_measured_node() -> None:
    """Test that observables referencing non-measured nodes are rejected."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
N 1
M 0 XY 0

.observable 0 1
"""
    with pytest.raises(ValueError, match="contains non-measured nodes"):
        loads(ptn_str)
