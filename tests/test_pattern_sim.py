from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from graphix_zx.command import E, M, N, X, Z
from graphix_zx.common import Plane
from graphix_zx.pattern import MutablePattern
from graphix_zx.simulator import PatternSimulator, SimulatorBackend

if TYPE_CHECKING:
    from graphix_zx.pattern import ImmutablePattern


@pytest.fixture
def setup_pattern() -> ImmutablePattern:
    pattern = MutablePattern({0}, {0: 0})
    cmds = [
        N(node=1),
        E(nodes=(0, 1)),
        M(node=1, plane=Plane.XY, angle=0.5, s_domain=set(), t_domain=set()),
        X(node=0, domain={1}),
        Z(node=0, domain={1}),
    ]
    pattern.extend(cmds)
    pattern.mark_runnable()
    pattern.mark_deterministic()
    return pattern.freeze()


def test_apply_command_add_node(setup_pattern: ImmutablePattern) -> None:
    pattern = setup_pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    cmd = N(node=2)
    simulator.apply_cmd(cmd)
    assert len(simulator.node_indices) == len(pattern.get_input_nodes()) + 1


def test_apply_command_measure(setup_pattern: ImmutablePattern) -> None:
    pattern = setup_pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    cmd = M(node=0, plane=Plane.XY, angle=0.5, s_domain=set(), t_domain=set())
    simulator.apply_cmd(cmd)
    assert 1 not in simulator.node_indices
