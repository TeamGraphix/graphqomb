"""Quantum Compiler(qompiler) module for Measurement-Based Quantum Computing (MBQC).

note: `compile` is used in Python built-in functions, so we use `qompile` instead.

This module provides:

- `qompile`: Compile graph state into a pattern with x/z correction flows.
"""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

from graphix_zx.command import Command, E, M, N, X, Z
from graphix_zx.feedforward import check_flow, dag_from_flow
from graphix_zx.graphstate import odd_neighbors
from graphix_zx.pattern import Pattern
from graphix_zx.pauli_frame import PauliFrame

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphix_zx.graphstate import BaseGraphState
    from graphix_zx.scheduler import Scheduler


def qompile(
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    *,
    scheduler: Scheduler | None = None,
    correct_output: bool = True,
) -> Pattern:
    r"""Compile graph state into pattern with x/z correction flows.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state
    xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        x correction flow
    zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        z correction flow
        if `None`, it is generated from xflow by odd neighbors
    scheduler : `Scheduler` | `None`, optional
        scheduler to schedule the graph state preparation and measurements,
        if `None`, the commands are scheduled in a single slice,
        by default `None`
    correct_output : `bool`, optional
        whether to correct outputs or not, by default True

    Returns
    -------
    `Pattern`
        compiled pattern

    Raises
    ------
    ValueError
        1. If the graph state is not in canonical form
        2. If the x flow or z flow is invalid with respect to the graph state
    """
    if not graph.is_canonical_form():
        msg = "Graph state must be in canonical form."
        raise ValueError(msg)
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) for node in xflow}
    check_flow(graph, xflow, zflow)

    pauli_frame = PauliFrame(graph.physical_nodes, xflow, zflow)

    return _qompile(graph, pauli_frame, scheduler=scheduler, correct_output=correct_output)


def _qompile(
    graph: BaseGraphState,
    pauli_frame: PauliFrame,
    *,
    scheduler: Scheduler | None = None,
    correct_output: bool = True,
) -> Pattern:
    """Compile graph state into pattern with a given Pauli frame.

    note: This is an internal function of `qompile`.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state
    pauli_frame : `PauliFrame`
        Pauli frame to track the Pauli state of each node
    scheduler : `Scheduler` | `None`, optional
        scheduler to schedule the graph state preparation and measurements,
        if `None`, the commands are scheduled in a single slice,
        by default `None`
    correct_output : `bool`, optional
        whether to correct outputs or not, by default True

    Returns
    -------
    `Pattern`
        compiled pattern
    """
    meas_bases = graph.meas_bases
    non_input_nodes = graph.physical_nodes - set(graph.input_node_indices)

    dag = dag_from_flow(graph, xflow=pauli_frame.xflow, zflow=pauli_frame.zflow)
    topo_order = list(TopologicalSorter(dag).static_order())
    topo_order.reverse()  # children first

    commands: list[Command] = []
    if not scheduler:
        commands.extend(N(node=node) for node in non_input_nodes)
        commands.extend(E(nodes=edge) for edge in graph.physical_edges)
        commands.extend(M(node, meas_bases[node]) for node in topo_order if node not in graph.output_node_indices)
    else:
        timeline = scheduler.timeline
        prepared_edges: set[frozenset[int]] = set()

        for time in range(scheduler.num_slices()):
            prepare_nodes, measure_nodes = timeline[time]
            for node in measure_nodes:
                for neighbor in graph.neighbors(node):
                    edge = frozenset({node, neighbor})
                    if edge not in prepared_edges:
                        commands.append(E(nodes=(node, neighbor)))
                        prepared_edges.add(edge)
            commands.extend(M(node, meas_bases[node]) for node in measure_nodes)
            commands.extend(N(node) for node in prepare_nodes)
    if correct_output:
        commands.extend(X(node=node) for node in graph.output_node_indices)
        commands.extend(Z(node=node) for node in graph.output_node_indices)

    return Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=tuple(commands),
        pauli_frame=pauli_frame,
    )
