import pytest

from graphix_zx.command import Command, M, N

# Assuming Command and CommandKind are defined somewhere in the codebase
from graphix_zx.common import default_meas_basis
from graphix_zx.pattern import MutablePattern


def test_pattern_initialization() -> None:
    input_node_indices = {0: 0, 1: 1, 2: 2}
    output_node_indices = {1: 1, 2: 2}
    pattern = MutablePattern(input_node_indices, output_node_indices)
    assert pattern.input_node_indices == input_node_indices
    assert pattern.output_node_indices == output_node_indices
    assert len(pattern) == 0


def test_pattern_add_command() -> None:
    pattern = MutablePattern({0: 0, 1: 1}, {1: 1, 2: 2})
    cmd = N(node=2)
    pattern.add(cmd)
    assert len(pattern) == 1


def test_pattern_add_command_already_prepared() -> None:
    pattern = MutablePattern({0: 0, 1: 1}, {1: 1, 2: 2})
    cmd = N(node=1)
    with pytest.raises(ValueError, match="already been used"):
        pattern.add(cmd)


def test_pattern_get_space_list() -> None:
    pattern = MutablePattern({0: 0}, {2: 0})
    cmds: list[Command] = [
        N(node=1),
        N(node=2),
        M(node=1, meas_basis=default_meas_basis(), s_cbit=0, t_cbit=1),
    ]
    pattern.extend(cmds)
    space_list = pattern.space_list
    assert space_list == [1, 2, 3, 2]  # Space changes with each N and M command


def test_pattern_calc_max_space() -> None:
    pattern = MutablePattern({0: 0}, {2: 0})
    cmds: list[Command] = [
        N(node=1),
        N(node=2),
        M(node=1, meas_basis=default_meas_basis(), s_cbit=0, t_cbit=1),
    ]
    pattern.extend(cmds)
    max_space = pattern.max_space
    assert max_space == 3  # Maximum space needed during execution


if __name__ == "__main__":
    pytest.main()
