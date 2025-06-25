"""Quantum Compiler(qompiler) module for Measurement-Based Quantum Computing (MBQC).

note: `compile` is used in Python built-in functions, so we use `qompile` instead.

This module provides:

- `qompile`: Compile graph state into a pattern with x/z correction flows.
"""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

from graphix_zx.command import Command, E, M, N, X, Z
from graphix_zx.feedforward import check_causality
from graphix_zx.graphstate import odd_neighbors
from graphix_zx.pattern import Pattern
from graphix_zx.pauli_frame import PauliFrame

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphix_zx.graphstate import BaseGraphState


def qompile(
    graph: BaseGraphState,
    x_flow: Mapping[int, AbstractSet[int]],
    z_flow: Mapping[int, AbstractSet[int]] | None = None,
    *,
    correct_output: bool = True,
) -> Pattern:
    r"""Compile graph state into pattern with x/z correction flows.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state
    x_flow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        x correction flow
    z_flow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`
        z correction flow
        if `None`, it is generated from `x_flow` by odd neighbors
    correct_output : `bool`, optional
        whether to correct outputs or not, by default True

    Returns
    -------
    `Pattern`
        compiled pattern
    """
    if not check_causality(graph, x_flow):
        msg = "Invalid x flow"
        raise ValueError(msg)
    if z_flow is None:
        z_flow = {node: odd_neighbors(x_flow[node], graph) for node in x_flow}
    if not check_causality(graph, z_flow):
        msg = "Invalid z flow"
        raise ValueError(msg)

    pauli_frame = PauliFrame.from_xzflow(graph, x_flow, z_flow)

    return _qompile(graph, pauli_frame, correct_output=correct_output)


def _qompile(
    graph: BaseGraphState,
    pauli_frame: PauliFrame,
    *,
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
    correct_output : `bool`, optional
        whether to correct outputs or not, by default True

    Returns
    -------
    `Pattern`
        compiled pattern
    """
    meas_bases = graph.meas_bases
    non_input_nodes = graph.physical_nodes - set(graph.input_node_indices)
    non_output_nodes = graph.physical_nodes - set(graph.output_node_indices)

    dag = {
        node: (
            pauli_frame.x2x_dag.get(node, set())
            | pauli_frame.x2z_dag.get(node, set())
            | pauli_frame.z2x_dag.get(node, set())
            | pauli_frame.z2z_dag.get(node, set())
        )
        - {node}
        for node in non_output_nodes
    }
    topo_order = reversed(list(TopologicalSorter(dag).static_order()))  # children first

    commands: list[Command] = []
    commands.extend(N(node=node) for node in non_input_nodes)
    commands.extend(E(nodes=edge) for edge in graph.physical_edges)
    commands.extend(M(node, meas_bases[node]) for node in topo_order if node not in graph.output_node_indices)
    if correct_output:
        commands.extend(X(node=node) for node in graph.output_node_indices)
        commands.extend(Z(node=node) for node in graph.output_node_indices)

    # NOTE: currently, we remove local Clifford commands

    return Pattern(
        input_node_indices=graph.input_node_indices,
        output_node_indices=graph.output_node_indices,
        commands=tuple(commands),
        pauli_frame=pauli_frame,
    )
