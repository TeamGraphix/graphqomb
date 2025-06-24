"""Quantum Compiler(qompiler) module for Measurement-Based Quantum Computing (MBQC).

note: `compile` is used in Python built-in functions, so we use `qompile` instead.

This module provides:

- `initialize_pauli_frame`: Initialize a Pauli frame from the graph and correction flows.
- `qompile_from_flow`: Compile graph state into pattern with gflow.
- `qompile_from_xz_flow`: Compile graph state into pattern with x/z correction flows.
- `qompile`: Compile graph state into pattern with correctionmaps and directed acyclic graph.
"""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

from graphix_zx.command import Command, E, M, N, X, Z
from graphix_zx.common import Plane
from graphix_zx.feedforward import check_causality
from graphix_zx.graphstate import odd_neighbors
from graphix_zx.pattern import Pattern
from graphix_zx.pauli_frame import PauliFrame

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphix_zx.graphstate import BaseGraphState


def initialize_pauli_frame(
    graph: BaseGraphState,
    x_flow: Mapping[int, AbstractSet[int]],
    z_flow: Mapping[int, AbstractSet[int]],
) -> PauliFrame:
    r"""Initialize a Pauli frame from the graph and correction flows.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state.
    x_flow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The X correction flow.
    z_flow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        The Z correction flow.

    Returns
    -------
    `PauliFrame`
        The initialized Pauli frame.
    """
    nodes = graph.physical_nodes
    non_output_nodes = nodes - set(graph.output_node_indices)

    x2x_dag = {}
    x2z_dag = {}
    z2x_dag = {}
    z2z_dag = {}

    for node in non_output_nodes:
        if graph.meas_bases[node].plane == Plane.XY:
            # Z byproduct
            z2x_dag[node] = set(x_flow.get(node, set()))
            z2z_dag[node] = set(z_flow.get(node, set()))
        elif graph.meas_bases[node].plane == Plane.XZ:
            # Y byproduct
            x2x_dag[node] = set(x_flow.get(node, set()))
            x2z_dag[node] = set(z_flow.get(node, set()))
            z2x_dag[node] = set(x_flow.get(node, set()))
            z2z_dag[node] = set(z_flow.get(node, set()))
        else:
            # X byproduct
            x2x_dag[node] = set(x_flow.get(node, set()))
            x2z_dag[node] = set(z_flow.get(node, set()))

    return PauliFrame(
        nodes=nodes,
        x2x_dag=x2x_dag,
        x2z_dag=x2z_dag,
        z2x_dag=z2x_dag,
        z2z_dag=z2z_dag,
    )


def qompile_from_flow(
    graph: BaseGraphState, gflow: Mapping[int, AbstractSet[int]], *, correct_output: bool = True
) -> Pattern:
    r"""Compile graph state into pattern with flowlike object.

    Parameters
    ----------
    graph : `BaseGraphState`
        graph state
    gflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        flowlike object, which is a mapping from node to its target nodes
    correct_output : `bool`, optional
        whether to correct outputs or not. Defaults to True.

    Returns
    -------
    `Pattern`
        compiled pattern

    Raises
    ------
    ValueError
        if the flow is invalid
    """
    if not check_causality(graph, gflow):
        msg = "Invalid flow"
        raise ValueError(msg)

    # generate corrections
    x_flow = gflow
    z_flow = {node: odd_neighbors(gflow[node], graph) for node in gflow}
    return qompile_from_xz_flow(graph, x_flow, z_flow, correct_output=correct_output)


def qompile_from_xz_flow(
    graph: BaseGraphState,
    x_flow: Mapping[int, AbstractSet[int]],
    z_flow: Mapping[int, AbstractSet[int]],
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
    z_flow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
        z correction flow
    correct_output : `bool`, optional
        whether to correct outputs or not, by default True

    Returns
    -------
    `Pattern`
        compiled pattern
    """
    pauli_frame = initialize_pauli_frame(graph, x_flow, z_flow)

    return qompile(graph, pauli_frame, correct_output=correct_output)


def qompile(
    graph: BaseGraphState,
    pauli_frame: PauliFrame,
    *,
    correct_output: bool = True,
) -> Pattern:
    """Compile graph state into pattern with a given Pauli frame.

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
