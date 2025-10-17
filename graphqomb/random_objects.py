"""Random object generator.

This module provides:

- `generate_random_flow_graph`: Generate a random flow graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from graphqomb.circuit import MBQCCircuit
from graphqomb.common import default_meas_basis
from graphqomb.graphstate import GraphState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator


def generate_random_flow_graph(
    width: int,
    depth: int,
    edge_p: float = 0.5,
    rng: Generator | None = None,
) -> tuple[GraphState, dict[int, set[int]]]:
    r"""Generate a random flow graph.

    Parameters
    ----------
    width : `int`
        The width of the graph.
    depth : `int`
        The depth of the graph.
    edge_p : `float`, optional
        The probability of adding an edge between two adjacent nodes.
        Default is 0.5.
    rng : `numpy.random.Generator`, optional
        The random number generator.
        Default is `None`.

    Returns
    -------
    `GraphState`
        The generated graph.
    `dict`\[`int`, `set`\[`int`\]\]
        The flow of the graph.
    """
    graph = GraphState()
    flow: dict[int, set[int]] = {}

    if rng is None:
        rng = np.random.default_rng()

    # input nodes
    for i in range(width):
        node_index = graph.add_physical_node()
        graph.register_input(node_index, i)
        graph.assign_meas_basis(node_index, default_meas_basis())

    # internal nodes
    for _ in range(depth - 2):
        node_indices_layer: list[int] = []
        for _ in range(width):
            node_index = graph.add_physical_node()
            graph.assign_meas_basis(node_index, default_meas_basis())
            graph.add_physical_edge(node_index - width, node_index)
            flow[node_index - width] = {node_index}
            node_indices_layer.append(node_index)

        for w in range(width - 1):
            if rng.random() < edge_p:
                graph.add_physical_edge(node_indices_layer[w], node_indices_layer[w + 1])

    # output nodes
    for i in range(width):
        node_index = graph.add_physical_node()
        graph.register_output(node_index, i)
        graph.add_physical_edge(node_index - width, node_index)
        flow[node_index - width] = {node_index}

    return graph, flow


def random_circ(
    width: int,
    depth: int,
    rng: np.random.Generator | None = None,
    edge_p: float = 0.5,
    angle_candidates: Sequence[float] = (0.0, np.pi / 3, 2 * np.pi / 3, np.pi),
) -> MBQCCircuit:
    r"""Generate a random MBQC circuit.

    Parameters
    ----------
    width : `int`
        circuit width
    depth : `int`
        circuit depth
    rng : `numpy.random.Generator`, optional
        random number generator, by default numpy.random.default_rng()
    edge_p : `float`, optional
        probability of adding CZ gate, by default 0.5
    angle_candidates : `collections.abc.Sequence[float]`, optional
        sequence of angles, by default (0, np.pi / 3, 2 * np.pi / 3, np.pi)

    Returns
    -------
    `MBQCCircuit`
        generated MBQC circuit
    """
    if rng is None:
        rng = np.random.default_rng()
    circ = MBQCCircuit(width)
    for d in range(depth):
        for j in range(width):
            circ.j(j, rng.choice(angle_candidates))
        if d < depth - 1:
            for j in range(width):
                if rng.random() < edge_p:
                    circ.cz(j, (j + 1) % width)
            num = rng.integers(0, width)
            if num > 0:
                target = sorted(set(rng.choice(range(width), num)))
                circ.phase_gadget(target, rng.choice(angle_candidates))

    return circ
