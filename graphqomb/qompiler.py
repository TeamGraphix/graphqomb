"""Quantum Compiler(qompiler) module for Measurement-Based Quantum Computing (MBQC).

note: `compile` is used in Python built-in functions, so we use `qompile` instead.

This module provides:

- `qompile`: Compile graph state into a pattern with x/z correction flows.
"""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

from graphqomb.command import Command, E, M, N, TICK, X, Z
from graphqomb.feedforward import check_flow, dag_from_flow
from graphqomb.graphstate import odd_neighbors
from graphqomb.pattern import Pattern
from graphqomb.pauli_frame import PauliFrame

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState
    from graphqomb.scheduler import Scheduler


def qompile(
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    zflow: Mapping[int, AbstractSet[int]] | None = None,
    *,
    scheduler: Scheduler | None = None,
    insert_tick: bool = True,
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
    insert_tick : `bool`, optional
        whether to insert TICK commands between time slices when using a scheduler,
        by default `True`

    Returns
    -------
    `Pattern`
        compiled pattern
    """
    graph.check_canonical_form()
    if zflow is None:
        zflow = {node: odd_neighbors(xflow[node], graph) for node in xflow}
    check_flow(graph, xflow, zflow)

    pauli_frame = PauliFrame(graph.physical_nodes, xflow, zflow)

    return _qompile(graph, pauli_frame, scheduler=scheduler, insert_tick=insert_tick)


def _qompile(
    graph: BaseGraphState,
    pauli_frame: PauliFrame,
    *,
    scheduler: Scheduler | None = None,
    insert_tick: bool = True,
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
    insert_tick : `bool`, optional
        whether to insert TICK commands between time slices when using a scheduler,
        by default `True`

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
        # Auto-schedule entanglement if not already scheduled
        if all(time is None for time in scheduler.entangle_time.values()):
            scheduler.auto_schedule_entanglement()

        detailed_timeline = scheduler.detailed_timeline

        for time_idx in range(scheduler.num_slices()):
            prepare_nodes, entangle_edges, measure_nodes = detailed_timeline[time_idx]

            # Order within time slice: N -> E -> M
            commands.extend(N(node) for node in prepare_nodes)
            commands.extend(E(nodes=tuple(edge)) for edge in entangle_edges)
            commands.extend(M(node, meas_bases[node]) for node in measure_nodes)

            # Insert TICK between time slices
            if insert_tick and time_idx < scheduler.num_slices() - 1:
                commands.append(TICK())

    for node in graph.output_node_indices:
        if meas_basis := graph.meas_bases.get(node):
            commands.append(M(node, meas_basis))
        else:
            commands.extend((X(node=node), Z(node=node)))

    return Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=tuple(commands),
        pauli_frame=pauli_frame,
    )
