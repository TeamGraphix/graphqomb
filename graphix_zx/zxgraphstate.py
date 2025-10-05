"""ZXGraph State classes for Measurement-based Quantum Computing.

This module provides:
- ZXGraphState: Graph State for the ZX-calculus.
- bipartite_edges: Return a set of edges for the complete bipartite graph between two sets of nodes.
"""

from __future__ import annotations

import operator
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import Plane, is_clifford_angle, is_close_angle
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

    @property
    def _clifford_rules(self) -> tuple[tuple[Callable[[int, float], bool], Callable[[int], None]], ...]:
        """List of rules (check_func, action_func) for removing local clifford nodes.

        The rules are applied in the order they are defined.
        """
        return (
            (self._needs_lc, self.local_complement),
            (self._needs_nop, lambda _: None),
            (
                self._needs_pivot,
                lambda node: self.pivot(node, min(self.neighbors(node) - set(self.input_node_indices))),
            ),
        )

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

    def _needs_nop(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node does not need any operation in order to perform _remove_clifford.

        For this operation, the measurement measurement angle must be 0 or pi (mod 2pi)
        and the measurement plane must be YZ or XZ.

        Parameters
        ----------
        node : `int`
            node index
        atol : `float`, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        `bool`
            True if the node is a removable Clifford node.
        """
        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        return is_close_angle(2 * alpha, 0, atol) and (self.meas_bases[node].plane in {Plane.YZ, Plane.XZ})

    def _needs_lc(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node needs a local complementation in order to perform _remove_clifford.

        For this operation, the measurement angle must be 0.5 pi or 1.5 pi (mod 2pi)
        and the measurement plane must be YZ or XY.

        Parameters
        ----------
        node : `int`
            node index
        atol : `float`, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        `bool`
            True if the node needs a local complementation.
        """
        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        return is_close_angle(2 * (alpha - np.pi / 2), 0, atol) and (
            self.meas_bases[node].plane in {Plane.YZ, Plane.XY}
        )

    def _needs_pivot(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the nodes need a pivot operation in order to perform _remove_clifford.

        The pivot operation is performed on the non-input neighbor of the node.
        For this operation,
        (a) the measurement angle must be 0 or pi (mod 2pi) and the measurement plane must be XY,
        or
        (b) the measurement angle must be 0.5 pi or 1.5 pi (mod 2pi) and the measurement plane must be XZ.

        Parameters
        ----------
        node : `int`
            node index
        atol : `float`, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        `bool`
            True if the nodes need a pivot operation.
        """
        if not (self.neighbors(node) - set(self.input_node_indices)):
            nbrs = self.neighbors(node)
            if not (nbrs.issubset(set(self.output_node_indices)) and nbrs):
                return False

        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        # (a) the measurement angle is 0 or pi (mod 2pi) and the measurement plane is XY
        case_a = is_close_angle(2 * alpha, 0, atol) and self.meas_bases[node].plane == Plane.XY
        # (b) the measurement angle is 0.5 pi or 1.5 pi (mod 2pi) and the measurement plane is XZ
        case_b = is_close_angle(2 * (alpha - np.pi / 2), 0, atol) and self.meas_bases[node].plane == Plane.XZ
        return case_a or case_b

    def _remove_clifford(self, node: int, atol: float = 1e-9) -> None:
        """Perform the Clifford node removal.

        Parameters
        ----------
        node : `int`
            node index
        atol : `float`, optional
            absolute tolerance, by default 1e-9
        """
        a_pi = self.meas_bases[node].angle % (2.0 * np.pi)
        coeff = 0.0 if is_close_angle(a_pi, 0, atol) else 1.0
        lc = LocalClifford(coeff * np.pi, 0, 0)
        for v in self.neighbors(node) - set(self.output_node_indices):
            self.apply_local_clifford(v, lc)

        self.remove_physical_node(node)

    def remove_clifford(self, node: int, atol: float = 1e-9) -> None:
        """Remove the local Clifford node.

        Parameters
        ----------
        node : `int`
            node index
        atol : `float`, optional
            absolute tolerance, by default 1e-9

        Raises
        ------
        ValueError
            1. If the node is an input node.
            2. If the node is not a Clifford node.
            3. If all neighbors are input nodes
                in some special cases ((meas_plane, meas_angle) = (XY, a pi), (XZ, a pi/2) for a = 0, 1).
            4. If the node has no neighbors that are not connected only to output nodes.
        """
        self._ensure_node_exists(node)
        if node in self.input_node_indices or node in self.output_node_indices:
            msg = "Clifford node removal not allowed for input node"
            raise ValueError(msg)

        if not (
            is_clifford_angle(self.meas_bases[node].angle, atol)
            and self.meas_bases[node].plane in {Plane.XY, Plane.XZ, Plane.YZ}
        ):
            msg = "This node is not a Clifford node."
            raise ValueError(msg)

        for check, action in self._clifford_rules:
            if not check(node, atol):
                continue
            action(node)
            self._remove_clifford(node, atol)
            return

        msg = "This Clifford node is unremovable."
        raise ValueError(msg)

    def is_removable_clifford(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node is a removable Clifford node.

        Parameters
        ----------
        node : `int`
            node index
        atol : `float`, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        `bool`
            True if the node is a removable Clifford node.
        """
        return any(
            [
                self._needs_nop(node, atol),
                self._needs_lc(node, atol),
                self._needs_pivot(node, atol),
            ]
        )


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
