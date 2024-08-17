import pytest

# Assuming Command and CommandKind are defined somewhere
from graphix_zx.command import (
    E,
    M,
    N,
    X,
)
from graphix_zx.pattern import MutablePattern, NodeAlreadyPreparedError, is_standardized


def test_pattern_initialization():
    pattern = MutablePattern({0, 1, 2})
    assert pattern.get_input_nodes() == {0, 1, 2}
    assert pattern.get_output_nodes() == {0, 1, 2}
    assert len(pattern) == 0


def test_pattern_add_command():
    pattern = MutablePattern({0, 1})
    cmd = N(node=2)
    pattern.add(cmd)
    assert len(pattern) == 1
    assert pattern.get_output_nodes() == {0, 1, 2}


def test_pattern_add_command_already_prepared():
    pattern = MutablePattern({0, 1})
    cmd = N(node=1)
    with pytest.raises(NodeAlreadyPreparedError):
        pattern.add(cmd)


def test_pattern_add_measurement_command():
    pattern = MutablePattern({0, 1})
    cmd = M(node=1)
    pattern.add(cmd)
    assert len(pattern) == 1
    assert pattern.get_output_nodes() == {0}  # 1 is measured and removed from output_nodes


def test_pattern_clear():
    pattern = MutablePattern({0, 1})
    cmd = N(node=2)
    pattern.add(cmd)
    pattern.clear()
    assert len(pattern) == 0
    assert pattern.get_output_nodes() == {0, 1}


def test_pattern_replace():
    pattern = MutablePattern({0, 1})
    cmds = [N(node=2), M(node=3)]
    pattern.replace(cmds, input_nodes={3, 4})
    assert pattern.get_input_nodes() == {3, 4}
    assert pattern.get_output_nodes() == {4, 2}
    assert len(pattern) == 2  # 2 replaced commands


def test_pattern_get_space_list():
    pattern = MutablePattern({0})
    cmds = [
        N(node=1),
        N(node=2),
        M(node=1),
    ]
    pattern.extend(cmds)
    space_list = pattern.get_space_list()
    assert space_list == [1, 2, 3, 2]  # Space changes with each N and M command


def test_pattern_calc_max_space():
    pattern = MutablePattern({0})
    cmds = [
        N(node=1),
        N(node=2),
        M(node=1),
    ]
    pattern.extend(cmds)
    max_space = pattern.calc_max_space()
    assert max_space == 3  # Maximum space needed during execution


def test_is_standardized():
    pattern = MutablePattern({0})
    cmds = [
        N(node=1),
        E(nodes=(0, 1)),
        M(node=1),
        X(node=0, domain=[1]),
    ]
    pattern.extend(cmds)
    assert is_standardized(pattern) == True

    # Adding an out-of-order command should break standardization
    cmd_out_of_order = N(node=2)
    pattern.add(cmd_out_of_order)
    assert is_standardized(pattern) == False


if __name__ == "__main__":
    pytest.main()
