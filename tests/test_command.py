"""Test command.py"""

import pytest
import dataclasses

import numpy as np
from graphix_zx.common import Plane
from graphix_zx.command import N, E, M, X, Z, Pattern, is_standardized


@dataclasses.dataclass(frozen=True)
class PatternTestCase:
    pattern: Pattern
    standardized: bool


pattern0 = Pattern(input_nodes=[0])
pattern0.add(N(node=1))
pattern0.add(N(node=2))
pattern0.add(E(nodes=(0, 1)))
pattern0.add(E(nodes=(1, 2)))
pattern0.add(M(node=0, plane=Plane.XY, angle=0.0, s_domain=[], t_domain=[]))
pattern0.add(M(node=1, plane=Plane.XY, angle=0.0, s_domain=[0], t_domain=[]))
pattern0.add(X(node=2, domain=[1]))
pattern0.add(Z(node=2, domain=[1]))

CASE0 = PatternTestCase(pattern=pattern0, standardized=True)

pattern1 = Pattern(input_nodes=[0])
pattern1.add(N(node=1))
pattern1.add(E(nodes=(0, 1)))
pattern1.add(M(node=0, plane=Plane.XY, angle=0.0, s_domain=[], t_domain=[]))
pattern1.add(X(node=1, domain=[0]))
pattern1.add(N(node=2))
pattern1.add(E(nodes=(1, 2)))
pattern1.add(M(node=1, plane=Plane.XY, angle=0.0, s_domain=[], t_domain=[]))
pattern1.add(X(node=2, domain=[1]))

CASE1 = PatternTestCase(pattern=pattern1, standardized=False)


@pytest.mark.parametrize("case", [CASE0, CASE1])
def test_is_standardized(case: PatternTestCase):
    assert is_standardized(case.pattern) == case.standardized
