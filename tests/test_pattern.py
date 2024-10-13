import pytest

# Assuming Command and CommandKind are defined somewhere
from graphix_zx.command import Command, E, M, N, X
from graphix_zx.pattern import MutablePattern, NodeAlreadyPreparedError, is_standardized


def test_pattern_initialization() -> None:
    pattern = MutablePattern({0, 1, 2})
    assert pattern.input_nodes == {0, 1, 2}
    assert pattern.output_nodes == {0, 1, 2}
    assert len(pattern) == 0


def test_pattern_add_command() -> None:
    pattern = MutablePattern({0, 1})
    cmd = N(node=2)
    pattern.add(cmd)
    assert len(pattern) == 1
    assert pattern.output_nodes == {0, 1, 2}


def test_pattern_add_command_already_prepared() -> None:
    pattern = MutablePattern({0, 1})
    cmd = N(node=1)
    with pytest.raises(NodeAlreadyPreparedError):
        pattern.add(cmd)


def test_pattern_add_measurement_command() -> None:
    pattern = MutablePattern({0, 1})
    cmd = M(node=1)
    pattern.add(cmd)
    assert len(pattern) == 1
    assert pattern.output_nodes == {0}  # 1 is measured and removed from output_nodes


def test_pattern_clear() -> None:
    pattern = MutablePattern({0, 1})
    cmd = N(node=2)
    pattern.add(cmd)
    pattern.clear()
    assert len(pattern) == 0
    assert pattern.output_nodes == {0, 1}


def test_pattern_replace() -> None:
    pattern = MutablePattern({0, 1})
    cmds: list[Command] = [N(node=2), M(node=3)]
    pattern.replace(cmds, input_nodes={3, 4})
    assert pattern.input_nodes == {3, 4}
    assert pattern.output_nodes == {4, 2}
    assert len(pattern) == 2  # 2 replaced commands


def test_pattern_get_space_list() -> None:
    pattern = MutablePattern({0})
    cmds: list[Command] = [
        N(node=1),
        N(node=2),
        M(node=1),
    ]
    pattern.extend(cmds)
    space_list = pattern.space_list
    assert space_list == [1, 2, 3, 2]  # Space changes with each N and M command


def test_pattern_calc_max_space() -> None:
    pattern = MutablePattern({0})
    cmds: list[Command] = [
        N(node=1),
        N(node=2),
        M(node=1),
    ]
    pattern.extend(cmds)
    max_space = pattern.max_space
    assert max_space == 3  # Maximum space needed during execution


def test_is_standardized() -> None:
    pattern = MutablePattern({0})
    cmds: list[Command] = [
        N(node=1),
        E(nodes=(0, 1)),
        M(node=1),
        X(node=0, domain={1}),
    ]
    pattern.extend(cmds)
    assert is_standardized(pattern)

    # Adding an out-of-order command should break standardization
    cmd_out_of_order = N(node=2)
    pattern.add(cmd_out_of_order)
    assert not is_standardized(pattern)


if __name__ == "__main__":
    pytest.main()
