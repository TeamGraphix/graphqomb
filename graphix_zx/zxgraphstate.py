"""ZXGraph State classes for Measurement-based Quantum Computing.

This module provides:
- ZXGraphState: Graph State for the ZX-calculus.
- bipartite_edges: Return a set of edges for the complete bipartite graph between two sets of nodes.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.euler import LocalClifford
from graphix_zx.graphstate import GraphState, bipartite_edges

if TYPE_CHECKING:
    from collections.abc import Iterable


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

    def _swap(self, node1: int, node2: int) -> None:
        """Swap nodes u and v in the graph state.

        Parameters
        ----------
        node1  : `int`
            node index
        node2 : `int`
            node index
        """
        node1_nbrs = self.neighbors(node1) - {node2}
        node2_nbrs = self.neighbors(node2) - {node1}
        nbr_b = node1_nbrs - node2_nbrs
        nbr_c = node2_nbrs - node1_nbrs
        for b in nbr_b:
            self.remove_physical_edge(node1, b)
            self.add_physical_edge(node2, b)
        for c in nbr_c:
            self.remove_physical_edge(node2, c)
            self.add_physical_edge(node1, c)

    def pivot(self, node1: int, node2: int) -> None:
        """Pivot operation on the graph state: G∧(uv) (= G*u*v*u = G*v*u*v) for neighboring nodes u and v.

        In order to maintain the ZX-diagram simple, pi-spiders are shifted properly.

        Parameters
        ----------
        node1 : `int`
            node index. The node must not be an input node.
        node2 : `int`
            node index. The node must not be an input node.

        Raises
        ------
        ValueError
            If the nodes are input nodes, or the graph is not a ZX-diagram.
        """
        self._ensure_node_exists(node1)
        self._ensure_node_exists(node2)
        if node1 in self.input_node_indices or node2 in self.input_node_indices:
            msg = "Cannot apply pivot to input node"
            raise ValueError(msg)
        self._check_meas_basis()

        node1_nbrs = self.neighbors(node1) - {node2}
        node2_nbrs = self.neighbors(node2) - {node1}
        nbr_a = node1_nbrs & node2_nbrs
        nbr_b = node1_nbrs - node2_nbrs
        nbr_c = node2_nbrs - node1_nbrs
        nbr_pairs = [
            bipartite_edges(nbr_a, nbr_b),
            bipartite_edges(nbr_a, nbr_c),
            bipartite_edges(nbr_b, nbr_c),
        ]
        rmv_edges = set().union(*(p & self.physical_edges for p in nbr_pairs))
        add_edges = set().union(*(p - self.physical_edges for p in nbr_pairs))

        self._update_connections(rmv_edges, add_edges)
        self._swap(node1, node2)

        # update node1 and node2 measurement
        lc = LocalClifford(np.pi / 2, np.pi / 2, np.pi / 2)
        for a in {node1, node2} - set(self.output_node_indices):
            self.apply_local_clifford(a, lc)

        # update nodes measurement of nbr_a
        lc = LocalClifford(np.pi, 0, 0)
        for w in nbr_a - set(self.output_node_indices):
            self.apply_local_clifford(w, lc)


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
