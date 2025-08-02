"""Scheduling solver."""

from __future__ import annotations

import enum
from enum import Enum
from typing import TYPE_CHECKING

from ortools.sat.python import cp_model

if TYPE_CHECKING:
    from graphix_zx.graphstate import BaseGraphState


class Strategy(Enum):
    """Enumeration for scheduling strategies."""

    MINIMIZE_SPACE = enum.auto()
    MINIMIZE_TIME = enum.auto()


def construct_model(  # noqa: C901, PLR0912
    graph: BaseGraphState,
    dag: dict[int, set[int]],
    strategy: Strategy = Strategy.MINIMIZE_SPACE,
) -> cp_model.CpModel:
    r"""Construct a scheduling model for the given graph and strategy.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state to optimize.
    dag : `dict`\[`int`, `set`\[`int`\]\]
        The directed acyclic graph (DAG) representing the dependencies of the nodes.
    strategy : `Strategy`, optional
        The optimization strategy to use, by default Strategy.MINIMIZE_SPACE

    Returns
    -------
    `ortools.python.cp_model.CpModel`
        The constructed CP model.
    """
    m = cp_model.CpModel()

    # variables
    max_time = 2 * len(graph.physical_nodes)
    node2prep: dict[int, cp_model.IntVar] = {}
    node2meas: dict[int, cp_model.IntVar] = {}
    for node in graph.physical_nodes:
        if node not in graph.input_node_indices:
            node2prep[node] = m.NewIntVar(0, max_time, f"prep_{node}")

        if node not in graph.output_node_indices:
            node2meas[node] = m.NewIntVar(0, max_time, f"meas_{node}")

    # constraints

    # measurement order
    for node, children in dag.items():
        for child in children:
            m.Add(node2meas[node] < node2meas[child])

    # edge constraints
    for node in graph.physical_nodes - set(graph.output_node_indices):
        for neighbor in graph.neighbors(node):
            if neighbor in graph.input_node_indices:
                continue
            m.Add(node2prep[neighbor] < node2meas[node])

    # objectives
    if strategy == Strategy.MINIMIZE_SPACE:
        max_space = m.NewIntVar(0, len(graph.physical_nodes), "max_space")

        for t in range(max_time):
            alive_at_t: list[cp_model.IntVar] = []
            for node in graph.physical_nodes:
                a_pre = m.NewBoolVar(f"alive_pre_{node}_{t}")
                if node in graph.input_node_indices:
                    m.Add(a_pre == 1)
                else:
                    p = node2prep[node]
                    m.Add(p <= t).OnlyEnforceIf(a_pre)
                    m.Add(p > t).OnlyEnforceIf(a_pre.Not())

                a_meas = m.NewBoolVar(f"alive_meas_{node}_{t}")
                if node in graph.output_node_indices:
                    m.Add(a_meas == 0)
                else:
                    q = node2meas[node]
                    m.Add(q <= t).OnlyEnforceIf(a_meas)
                    m.Add(q > t).OnlyEnforceIf(a_meas.Not())

                alive = m.NewBoolVar(f"alive_{node}_{t}")
                m.AddImplication(alive, a_pre)
                m.AddImplication(alive, a_meas.Not())
                m.Add(a_pre - a_meas <= alive)
                alive_at_t.append(alive)

            m.Add(max_space >= sum(alive_at_t))
        m.Minimize(max_space)

    elif strategy == Strategy.MINIMIZE_TIME:
        # minimize the maximum time of all measurements
        meas_vars = list(node2meas.values())
        makespan = m.NewIntVar(0, max_time, "makespan")
        m.AddMaxEquality(makespan, meas_vars)
        m.Minimize(makespan)

    return m
