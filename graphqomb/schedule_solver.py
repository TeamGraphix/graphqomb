"""Scheduling solver for optimizing MBQC pattern execution.

This module provides:

- `Strategy`: Enumeration of scheduling optimization strategies
- `ScheduleConfig`: Configuration for scheduling parameters and constraints
- `solve_schedule`: Solve scheduling optimization using constraint programming
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ortools.sat.python import cp_model

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


class Strategy(Enum):
    """Enumeration for scheduling strategies."""

    MINIMIZE_SPACE = enum.auto()
    MINIMIZE_TIME = enum.auto()


@dataclass
class ScheduleConfig:
    """Configuration for scheduling strategy, constraints, and parameters."""

    strategy: Strategy
    max_qubit_count: int | None = None
    max_time: int | None = None


@dataclass
class _ModelContext:
    """Internal context for model and graph data."""

    model: cp_model.CpModel
    graph: BaseGraphState


def _add_constraints(
    model: cp_model.CpModel,
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
) -> None:
    """Add constraints to the scheduling model."""
    # Measurement order constraints
    for node, children in dag.items():
        for child in children:
            if node in node2meas and child in node2meas:
                model.Add(node2meas[node] < node2meas[child])

    # Edge constraints
    for node in graph.physical_nodes - set(graph.output_node_indices):
        for neighbor in graph.neighbors(node):
            if neighbor in graph.input_node_indices:
                continue
            model.Add(node2prep[neighbor] < node2meas[node])


def _set_objective(
    ctx: _ModelContext,
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
    config: ScheduleConfig,
    max_time: int,
) -> None:
    """Set the objective function for the scheduling model.

    Raises
    ------
    ValueError
        If the scheduling strategy is unknown.
    """
    if config.strategy == Strategy.MINIMIZE_SPACE:
        _set_minimize_space_objective(ctx, node2prep, node2meas, max_time)
    elif config.strategy == Strategy.MINIMIZE_TIME:
        _set_minimize_time_objective(ctx, node2prep, node2meas, max_time, config.max_qubit_count)
    else:
        msg = f"Unknown scheduling strategy: {config.strategy}"
        raise ValueError(msg)


def _compute_alive_nodes_at_time(
    ctx: _ModelContext,
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
    t: int,
) -> list[cp_model.IntVar]:
    """Compute the list of alive nodes at time t.

    Returns
    -------
    list[cp_model.IntVar]
        Boolean variables indicating whether each node is alive at time t.
    """
    alive_at_t: list[cp_model.IntVar] = []
    for node in ctx.graph.physical_nodes:
        a_pre = ctx.model.NewBoolVar(f"alive_pre_{node}_{t}")
        if node in ctx.graph.input_node_indices:
            ctx.model.Add(a_pre == 1)
        else:
            p = node2prep[node]
            ctx.model.Add(p <= t).OnlyEnforceIf(a_pre)
            ctx.model.Add(p > t).OnlyEnforceIf(a_pre.Not())

        a_meas = ctx.model.NewBoolVar(f"alive_meas_{node}_{t}")
        if node in ctx.graph.output_node_indices:
            ctx.model.Add(a_meas == 0)
        else:
            q = node2meas[node]
            ctx.model.Add(q <= t).OnlyEnforceIf(a_meas)
            ctx.model.Add(q > t).OnlyEnforceIf(a_meas.Not())

        alive = ctx.model.NewBoolVar(f"alive_{node}_{t}")
        ctx.model.AddImplication(alive, a_pre)
        ctx.model.AddImplication(alive, a_meas.Not())
        ctx.model.Add(a_pre - a_meas <= alive)
        alive_at_t.append(alive)

    return alive_at_t


def _set_minimize_space_objective(
    ctx: _ModelContext,
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
    max_time: int,
) -> None:
    """Set objective to minimize the maximum number of qubits used at any time."""
    max_space = ctx.model.NewIntVar(0, len(ctx.graph.physical_nodes), "max_space")
    for t in range(max_time):
        alive_at_t = _compute_alive_nodes_at_time(ctx, node2prep, node2meas, t)
        ctx.model.Add(max_space >= sum(alive_at_t))
    ctx.model.Minimize(max_space)


def _set_minimize_time_objective(
    ctx: _ModelContext,
    node2prep: Mapping[int, cp_model.IntVar],
    node2meas: Mapping[int, cp_model.IntVar],
    max_time: int,
    max_qubit_count: int | None = None,
) -> None:
    """Set objective to minimize the total execution time."""
    # Add space constraint if max_qubit_count is specified
    if max_qubit_count is not None:
        for t in range(max_time):
            alive_at_t = _compute_alive_nodes_at_time(ctx, node2prep, node2meas, t)
            ctx.model.Add(sum(alive_at_t) <= max_qubit_count)

    # Time objective: minimize makespan
    meas_vars = list(node2meas.values())
    makespan = ctx.model.NewIntVar(0, max_time, "makespan")
    ctx.model.AddMaxEquality(makespan, meas_vars)
    ctx.model.Minimize(makespan)


def solve_schedule(
    graph: BaseGraphState,
    dag: Mapping[int, AbstractSet[int]],
    config: ScheduleConfig,
    timeout: int = 60,
) -> tuple[dict[int, int], dict[int, int]] | None:
    r"""Solve the scheduling problem for the given graph.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state to optimize.
    dag : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The directed acyclic graph representing dependencies.
    config : `ScheduleConfig`
        The scheduling configuration including strategy and constraints.
    timeout : `int`, optional
        Maximum solve time in seconds, by default 60

    Returns
    -------
    `tuple`\[`dict`\[`int`, `int`\], `dict`\[`int`, `int`\]\] | `None`
        A tuple of (prepare_time, measure_time) dictionaries if solved,
        None if no solution found.
    """
    # Construct model
    model = cp_model.CpModel()

    # Determine max_time from config or calculate default
    max_time = config.max_time if config.max_time is not None else 2 * len(graph.physical_nodes)

    # Create variables
    node2prep: dict[int, cp_model.IntVar] = {}
    node2meas: dict[int, cp_model.IntVar] = {}
    for node in graph.physical_nodes:
        if node not in graph.input_node_indices:
            node2prep[node] = model.NewIntVar(0, max_time, f"prep_{node}")
        if node not in graph.output_node_indices:
            node2meas[node] = model.NewIntVar(0, max_time, f"meas_{node}")

    # Add constraints
    _add_constraints(model, graph, dag, node2prep, node2meas)

    # Set objective
    ctx = _ModelContext(model, graph)
    _set_objective(ctx, node2prep, node2meas, config, max_time)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    status = solver.Solve(model)

    # Note: type: ignore is needed due to a bug in or-tools type annotations
    # The actual runtime type of status is cp_model_pb2.ValueType, but it's incorrectly
    # annotated as CpSolverStatus, causing mypy to report a false positive comparison-overlap error
    if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:  # type: ignore[comparison-overlap]
        prepare_time = {node: solver.Value(var) for node, var in node2prep.items()}
        measure_time = {node: solver.Value(var) for node, var in node2meas.items()}
        return prepare_time, measure_time

    return None
