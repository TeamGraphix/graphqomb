"""ZXGraph State classes for Measurement-based Quantum Computing.

This module provides:
- ZXGraphState: Graph State for the ZX-calculus.
- bipartite_edges: Return a set of edges for the complete bipartite graph between two sets of nodes.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import Plane, PlannerMeasBasis, is_clifford_angle, is_close_angle
from graphix_zx.euler import LocalClifford
from graphix_zx.graphstate import BaseGraphState, GraphState, bipartite_edges

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class ZXGraphState(GraphState):
    r"""Graph State for the ZX-calculus.

    Attributes
    ----------
    input_nodes : `set`\[`int`\]
        set of input nodes
    output_nodes : `set`\[`int`\]
        set of output nodes
    physical_nodes : `set`\[`int`\]
        set of physical nodes
    physical_edges : `set`\[`tuple`\[`int`, `int`\]\]
        physical edges
    meas_bases : `dict`\[`int`, `MeasBasis`\]
        measurement bases
    q_indices : `dict`\[`int`, `int`\]
        qubit indices
    local_cliffords : `dict`\[`int`, `LocalClifford`\]
        local clifford operators
    """

    def __init__(self) -> None:
        super().__init__()

    def _update_connections(self, rmv_edges: Iterable[tuple[int, int]], new_edges: Iterable[tuple[int, int]]) -> None:
        r"""Update the physical edges of the graph state.

        Parameters
        ----------
        rmv_edges : `Iterable`\[`tuple`\[`int`, `int`\]\]
            edges to remove
        new_edges : `Iterable`\[`tuple`\[`int`, `int`\]\]
            edges to add
        """
        for edge in rmv_edges:
            self.remove_physical_edge(edge[0], edge[1])
        for edge in new_edges:
            self.add_physical_edge(edge[0], edge[1])

    def local_complement(self, node: int) -> None:
        r"""Local complement operation on the graph state: G*u.

        Non-input node u gets Rx(pi/2) and its neighbors get Rz(-pi/2).
        The edges between the neighbors of u are complemented.

        Parameters
        ----------
        node : `int`
            node index. The node must not be an input node.

        Raises
        ------
        ValueError
            If the node does not exist, is an input node, or the graph is not a ZX-diagram.
        """
        self._ensure_node_exists(node)
        if node in self.input_node_indices:
            msg = "Cannot apply local complement to input node."
            raise ValueError(msg)
        self._check_meas_basis()

        nbrs: set[int] = self.neighbors(node)
        nbr_pairs = complete_graph_edges(nbrs)
        new_edges = nbr_pairs - self.physical_edges
        rmv_edges = self.physical_edges & nbr_pairs

        self._update_connections(rmv_edges, new_edges)

        # apply local clifford to node and its neighbors
        lc = LocalClifford(0, np.pi / 2, 0)
        self.apply_local_clifford(node, lc)
        lc = LocalClifford(-np.pi / 2, 0, 0)
        for v in nbrs:
            self.apply_local_clifford(v, lc)


def complete_graph_edges(nodes: Iterable[int]) -> set[tuple[int, int]]:
    r"""Return a set of edges for the complete graph on the given nodes.

    Parameters
    ----------
    nodes : `Iterable`\[`int`\]
        nodes

    Returns
    -------
    `set`\[`tuple`\[`int`, `int`\]\]
        edges of the complete graph
    """
    return {tuple(sorted((u, v))) for u, v in combinations(nodes, 2)}
