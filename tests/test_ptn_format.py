"""Tests for ptn_format module."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from graphqomb.command import TICK, E, M, N, X, Z
from graphqomb.common import Plane, PlannerMeasBasis
from graphqomb.graphstate import GraphState
from graphqomb.ptn_format import (
    PatternData,
    dump,
    dumps,
    load,
    loads,
)
from graphqomb.qompiler import qompile


def create_simple_pattern():
    """Create a simple pattern for testing."""
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
    pattern = qompile(graph, xflow)

    return pattern


def test_dumps_basic():
    """Test basic pattern serialization."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    assert ".version 1" in ptn_str
    assert ".input" in ptn_str
    assert ".output" in ptn_str
    assert "#======== QUANTUM ========" in ptn_str
    assert "#======== CLASSICAL ========" in ptn_str


def test_dumps_contains_commands():
    """Test that dumps includes all command types."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    # Check for command types
    assert "N " in ptn_str  # Node creation
    assert "E " in ptn_str  # Entanglement
    assert "M " in ptn_str  # Measurement


def test_dumps_coordinates():
    """Test that coordinates are correctly serialized."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    assert ".coord 0 0.0 0.0" in ptn_str


def test_dumps_measurement_angles():
    """Test that measurement angles are correctly formatted."""
    pattern = create_simple_pattern()
    ptn_str = dumps(pattern)

    # pi/2 should be formatted as "pi/2"
    assert "pi/2" in ptn_str
    # 0.0 should be formatted as "0"
    assert "XY 0" in ptn_str


def test_loads_basic():
    """Test basic pattern deserialization."""
    ptn_str = """# Test pattern
.version 1
.input 0:0
.output 2:0
.coord 0 0.0 0.0

#======== QUANTUM ========
[0]
N 1
E 0 1
M 0 XY 0

#======== CLASSICAL ========
.xflow 0 -> 1
"""
    result = loads(ptn_str)

    assert isinstance(result, PatternData)
    assert result.input_node_indices == {0: 0}
    assert result.output_node_indices == {2: 0}
    assert result.input_coordinates == {0: (0.0, 0.0)}


def test_loads_commands():
    """Test that commands are correctly parsed."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
N 1
N 3 1.0 2.0
E 0 1
M 0 XY 0
M 1 XY pi/2
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


def test_loads_timeslices():
    """Test that timeslice markers generate TICK commands."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
E 0 1
[1]
M 0 XY 0
[2]
M 1 XY 0
"""
    result = loads(ptn_str)

    # Count TICK commands
    tick_count = sum(1 for c in result.commands if isinstance(c, TICK))
    assert tick_count == 2


def test_loads_angle_parsing():
    """Test various angle format parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 5:0

[0]
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


def test_loads_flow_parsing():
    """Test xflow and zflow parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
N 1
E 0 1
M 0 XY 0

.xflow 0 -> 1 2
.zflow 0 -> 3 4
"""
    result = loads(ptn_str)

    assert result.xflow == {0: {1, 2}}
    assert result.zflow == {0: {3, 4}}


def test_loads_detector_parsing():
    """Test detector (parity check group) parsing."""
    ptn_str = """
.version 1
.input 0:0
.output 2:0

[0]
M 0 XY 0

.detector 0 1 2
.detector 3 4
"""
    result = loads(ptn_str)

    assert len(result.parity_check_groups) == 2
    assert result.parity_check_groups[0] == {0, 1, 2}
    assert result.parity_check_groups[1] == {3, 4}


def test_loads_missing_version():
    """Test that missing version raises ValueError."""
    ptn_str = """
.input 0:0
.output 1:0
[0]
M 0 XY 0
"""
    with pytest.raises(ValueError, match="Missing .version directive"):
        loads(ptn_str)


def test_loads_unsupported_version():
    """Test that unsupported version raises ValueError."""
    ptn_str = """
.version 99
.input 0:0
.output 1:0
"""
    with pytest.raises(ValueError, match="Unsupported .ptn version"):
        loads(ptn_str)


def test_loads_unknown_command():
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


def test_roundtrip():
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


def test_dump_and_load_file():
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


def test_multiple_input_output_qubits():
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


def test_3d_coordinates():
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


def test_different_measurement_planes():
    """Test all measurement planes are correctly handled."""
    ptn_str = """
.version 1
.input 0:0
.output 3:0

[0]
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


def test_empty_flow():
    """Test pattern with empty flow mappings."""
    ptn_str = """
.version 1
.input 0:0
.output 1:0

[0]
M 0 XY 0

#======== CLASSICAL ========
"""
    result = loads(ptn_str)

    assert result.xflow == {}
    assert result.zflow == {}


def test_comments_ignored():
    """Test that comments are properly ignored."""
    ptn_str = """
# This is a comment
.version 1
# Another comment
.input 0:0  # inline comment should be parsed as part of content
.output 1:0

# Comment in quantum section
[0]
M 0 XY 0
"""
    result = loads(ptn_str)
    # Should parse without error
    assert result.input_node_indices is not None
