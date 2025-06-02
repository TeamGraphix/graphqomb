from __future__ import annotations

import pytest

from graphix_zx.command import Command, D, E, M, N, X
from graphix_zx.common import default_meas_basis
from graphix_zx.decoder_backend import XORDecoder
from graphix_zx.pattern import Pattern, is_runnable


def test_len_iter_getitem() -> None:
    cmds: list[Command] = [N(1), M(1, default_meas_basis())]
    input_node_indices = {1: 0}
    output_node_indices: dict[int, int] = {}
    pattern = Pattern(input_node_indices, output_node_indices, tuple(cmds))

    # __len__
    assert len(pattern) == 2

    # __iter__
    assert list(iter(pattern)) == cmds

    # __getitem__ single index and slice
    assert pattern[0] is cmds[0]
    assert pattern[:] == tuple(cmds)


def test_space_and_max_space() -> None:
    # Start with one input qubit (0).  Prepare a new qubit (1),
    # and then measure it(1).  The space usage over time should be: 1, 2, 1, 1.
    cmds: list[Command] = [N(1), E((0, 1)), M(0, default_meas_basis())]
    input_node_indices = {0: 0}
    output_node_indices = {1: 0}
    pattern = Pattern(input_node_indices, output_node_indices, tuple(cmds))

    assert pattern.space == [1, 2, 1]
    assert pattern.max_space == 2


def test_is_runnable_happy_path() -> None:
    """Test that `is_runnable` returns True for a runnable pattern."""
    input_node_indices = {0: 0}
    output_node_indices: dict[int, int] = {}
    cmds: list[Command] = [N(1), E((0, 1)), M(0, default_meas_basis()), M(1, default_meas_basis())]
    pattern = Pattern(input_node_indices, output_node_indices, tuple(cmds))

    is_runnable(pattern)


def test_dependency_on_unmeasured_qubit_raises() -> None:
    """Using a non-measured c-bit inside a D command must raise ValueError."""
    input_node_indices = {0: 0}
    output_node_indices = {0: 0}
    cmds = [D([0], [1], XORDecoder())]
    pattern = Pattern(input_node_indices, output_node_indices, tuple(cmds))

    with pytest.raises(ValueError, match="D command depends on an output that hasn't been measured yet:"):
        is_runnable(pattern)


def test_operation_on_measured_qubit_raises() -> None:
    input_node_indices = {0: 0}
    output_node_indices: dict[int, int] = {}
    cmds: list[Command] = [M(0, default_meas_basis()), X(0)]
    pattern = Pattern(input_node_indices, output_node_indices, tuple(cmds))

    with pytest.raises(ValueError, match="Operation on a measured qubit:"):
        is_runnable(pattern)


def test_operation_on_unprepared_qubit_raises() -> None:
    # Qubit 2 is neither an input nor has it been prepared by an N command.
    input_node_indices = {0: 0}
    output_node_indices = {0: 0}
    cmds = [X(2)]
    pattern = Pattern(input_node_indices, output_node_indices, tuple(cmds))

    with pytest.raises(ValueError, match="Operation on a qubit that hasn't been prepared yet:"):
        is_runnable(pattern)


def test_measure_output_qubit_raises() -> None:
    input_node_indices = {0: 0}
    output_node_indices = {0: 0}
    cmds = [M(0, default_meas_basis())]
    pattern = Pattern(input_node_indices, output_node_indices, tuple(cmds))

    with pytest.raises(ValueError, match="The command measures an output qubit:"):
        is_runnable(pattern)


def test_missing_measurement_raises() -> None:
    # Non-output qubit 1 is prepared but never measured.
    input_node_indices = {0: 0}
    output_node_indices = {0: 0}
    cmds = [N(1)]
    pattern = Pattern(input_node_indices, output_node_indices, tuple(cmds))

    with pytest.raises(ValueError, match="Missing measurements on qubit"):
        is_runnable(pattern)
