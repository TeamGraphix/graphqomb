"""Random object generator.

This module provides:
- get_random_flow_graph: Generate a random flow graph.
- random_circ: Generate a random MBQC circuit with gflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.circuit import MBQCCircuit
from graphix_zx.common import default_meas_basis
from graphix_zx.graphstate import GraphState

if TYPE_CHECKING:
    from numpy.random import Generator
    from collections.abc import Set as AbstractSet


def get_random_flow_graph(
    width: int,
    depth: int,
    edge_p: float = 0.5,
    rng: Generator | None = None,
) -> tuple[GraphState, dict[int, set[int]]]:
    """Generate a random flow graph.

    Parameters
    ----------
    width : int
        The width of the graph.
    depth : int
        The depth of the graph.
    edge_p : float, optional
        The probability of adding an edge between two adjacent nodes.
        Default is 0.5.
    rng : Generator, optional
        The random number generator.
        Default is None.

    Returns
    -------
    GraphState
        The generated graph.
    dict[int, set[int]]
        The flow of the graph.
    """
    graph = GraphState()
    flow: dict[int, set[int]] = {}
    num_nodes = 0

    if rng is None:
        rng = np.random.default_rng()

    # input nodes
    for q_index in range(width):
        graph.add_physical_node(num_nodes, is_input=True)
        graph.set_meas_basis(num_nodes, default_meas_basis())
        graph.set_q_index(num_nodes, q_index)
        num_nodes += 1

    # internal nodes
    for _ in range(depth - 2):
        for q_index in range(width):
            graph.add_physical_node(num_nodes)
            graph.set_meas_basis(num_nodes, default_meas_basis())
            graph.set_q_index(num_nodes, q_index)
            graph.add_physical_edge(num_nodes - width, num_nodes)
            flow[num_nodes - width] = {num_nodes}
            num_nodes += 1

        for w in range(width - 1):
            if rng.random() < edge_p:
                graph.add_physical_edge(num_nodes - width + w, num_nodes - width + w + 1)

    # output nodes
    for q_index in range(width):
        graph.add_physical_node(num_nodes, is_output=True)
        graph.set_q_index(num_nodes, q_index)
        graph.add_physical_edge(num_nodes - width, num_nodes)
        flow[num_nodes - width] = {num_nodes}
        num_nodes += 1

    return graph, flow


def random_circ(
    width: int,
    depth: int,
    rng: np.random.Generator | None = None,
    edge_p: float = 0.5,
    angle_list: AbstractSet[float] = [0, np.pi / 3, 2 * np.pi / 3, np.pi],
) -> MBQCCircuit:
    """Generate a random MBQC circuit.

    Parameters
    ----------
    width : int
        circuit width
    depth : int
        circuit depth
    rng : np.random.Generator, optional
        random number generator, by default np.random.default_rng()
    edge_p : float, optional
        probability of adding CZ gate, by default 0.5
    angle_list : AbstractSet[float], optional
        list of angles, by default [0, np.pi / 3, 2 * np.pi / 3, np.pi]

    Returns
    -------
    MBQCCircuit
        generated MBQC circuit
    """
    if rng is None:
        rng = np.random.default_rng()
    circ = MBQCCircuit(width)
    for d in range(depth):
        for j in range(width):
            circ.j(j, rng.choice(angle_list))
        if d < depth - 1:
            for j in range(width):
                if rng.random() < edge_p:
                    circ.cz(j, (j + 1) % width)
            num = rng.integers(0, width)
            if num > 0:
                target = set(rng.choice(list(range(width)), num))
                circ.phase_gadget(target, rng.choice(angle_list))

    return circ
