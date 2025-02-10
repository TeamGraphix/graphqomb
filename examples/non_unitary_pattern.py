"""Demonstration of non-Unitary pattern compilation."""

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import GraphState
from graphix_zx.pattern import print_pattern
from graphix_zx.qompiler import qompile_from_xz_flow

# construct a graph state
# 1(X) - 2(epsilon, XZ) - 3

epsilon = 0.01

graph = GraphState()
graph.add_physical_node(1, 0, is_input=True)
graph.add_physical_node(2, 0)
graph.add_physical_node(3, is_output=True)

graph.add_physical_edge(1, 2)
graph.add_physical_edge(2, 3)

graph.set_meas_basis(1, PlannerMeasBasis(Plane.XY, 0))
graph.set_meas_basis(2, PlannerMeasBasis(Plane.XZ, epsilon))

# feedforward design
# 2 (X)-> 3
xflow: dict[int, set] = {2: {3}}
zflow: dict[int, set] = {}

# compile pattern from graphstate and feedforward design

pattern = qompile_from_xz_flow(graph, xflow, zflow)
print_pattern(pattern)
