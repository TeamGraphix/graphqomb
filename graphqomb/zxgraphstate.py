"""ZXGraph State classes for Measurement-based Quantum Computing.

This module provides:

- `ZXGraphState`: Graph State for the ZX-calculus.
- `to_zx_graphstate`: Convert input GraphState to ZXGraphState.
- `complete_graph_edges`: Return a set of edges for the complete graph on the given nodes.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from graphqomb.common import Plane, PlannerMeasBasis, is_clifford_angle, is_close_angle
from graphqomb.euler import LocalClifford
from graphqomb.graphstate import BaseGraphState, GraphState, bipartite_edges

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from collections.abc import Set as AbstractSet

    CliffordRule: TypeAlias = tuple[Callable[[int, float], bool], Callable[[int], None]]


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
    def _clifford_rules(self) -> tuple[CliffordRule, ...]:
        r"""Tuple of rules (check_func, action_func) for removing local clifford nodes.

        The rules are applied in the order they are defined.

        Returns
        -------
        `tuple`\[`CliffordRule`, ...\]
            Tuple of rules (check_func, action_func) before removing local clifford nodes.
            If check_func(node) returns True, action_func(node) is executed.
            Then, the removal of the local clifford node is performed if possible.
        """
        return (
            (self._needs_lc, self.local_complement),
            (self._needs_nop, lambda _: None),
            (
                self._needs_pivot,
                lambda node: self.pivot(node, min(self.neighbors(node) - set(self.input_node_indices))),
            ),
        )

    def _update_connections(
        self, rmv_edges: AbstractSet[tuple[int, int]], new_edges: AbstractSet[tuple[int, int]]
    ) -> None:
        r"""Update the physical edges of the graph state.

        Parameters
        ----------
        rmv_edges : `collections.abc.Set`\[`tuple`\[`int`, `int`\]\]
            edges to remove
        new_edges : `collections.abc.Set`\[`tuple`\[`int`, `int`\]\]
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
        nbr_pairs: set[tuple[int, int]] = complete_graph_edges(nbrs)
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
        rmv_edges: set[tuple[int, int]] = set()
        rmv_edges.update(*(p & self.physical_edges for p in nbr_pairs))
        add_edges: set[tuple[int, int]] = set()
        add_edges.update(*(p - self.physical_edges for p in nbr_pairs))

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
        """Check if the node needs a pivot operation in order to perform _remove_clifford.

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
            True if the node needs a pivot operation.
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

    def remove_cliffords(self, atol: float = 1e-9) -> None:
        """Remove all local clifford nodes which are removable.

        Parameters
        ----------
        atol : `float`, optional
            absolute tolerance, by default 1e-9
        """
        self._check_meas_basis()
        while any(
            self.is_removable_clifford(n, atol)
            for n in (self.physical_nodes - set(self.input_node_indices) - set(self.output_node_indices))
        ):
            for check, action in self._clifford_rules:
                while True:
                    candidates = self.physical_nodes - set(self.input_node_indices) - set(self.output_node_indices)
                    clifford_node = next((node for node in candidates if check(node, atol)), None)
                    if clifford_node is None:
                        break
                    action(clifford_node)
                    self._remove_clifford(clifford_node, atol)

    def _extract_yz_adjacent_pair(self) -> tuple[int, int] | None:
        r"""Call inside convert_to_phase_gadget.

        Find a pair of adjacent nodes that are both measured in the YZ-plane.

        Returns
        -------
        `tuple`\[`int`, `int`\] | `None`
            A pair of adjacent nodes that are both measured in the YZ-plane, or None if no such pair exists.
        """
        yz_nodes = {node for node, basis in self.meas_bases.items() if basis.plane == Plane.YZ}
        for u, v in self.physical_edges:
            if u in yz_nodes and v in yz_nodes:
                return (min(u, v), max(u, v))
        return None

    def _extract_xz_node(self) -> int | None:
        """Call inside convert_to_phase_gadget.

        Find a node that is measured in the XZ-plane.

        Returns
        -------
        `int` | `None`
            A node that is measured in the XZ-plane, or None if no such node exists.
        """
        for node, basis in self.meas_bases.items():
            if basis.plane == Plane.XZ:
                return node
        return None

    def convert_to_phase_gadget(self) -> None:
        """Convert a ZX-diagram with gflow in MBQC+LC form into its phase-gadget form while preserving gflow."""
        while True:
            if pair := self._extract_yz_adjacent_pair():
                self.pivot(*pair)
                continue
            if u := self._extract_xz_node():
                self.local_complement(u)
                continue
            break

    def merge_yz_to_xy(self) -> None:
        """Merge YZ-measured nodes that have only one neighbor with an XY-measured node.

        If a node u is measured in the YZ-plane and u has only one neighbor v with a XY-measurement,
        then the node u can be merged into the node v.
        """
        target_candidates = {
            u for u, basis in self.meas_bases.items() if (basis.plane == Plane.YZ and len(self.neighbors(u)) == 1)
        }
        target_nodes = {
            u
            for u in target_candidates
            if (
                (v := next(iter(self.neighbors(u))))
                and (mb := self.meas_bases.get(v, None)) is not None
                and mb.plane == Plane.XY
            )
        }
        for u in target_nodes:
            (v,) = self.neighbors(u)
            new_angle = (self.meas_bases[u].angle + self.meas_bases[v].angle) % (2.0 * np.pi)
            self.assign_meas_basis(v, PlannerMeasBasis(Plane.XY, new_angle))
            self.remove_physical_node(u)

    def merge_yz_nodes(self) -> None:
        """Merge isolated YZ-measured nodes into a single node.

        If u, v nodes are measured in the YZ-plane and u, v have the same neighbors,
        then u, v can be merged into a single node.
        """
        min_nodes = 2
        yz_nodes = {u for u, basis in self.meas_bases.items() if basis.plane == Plane.YZ}
        if len(yz_nodes) < min_nodes:
            return
        neighbor_groups: dict[frozenset[int], list[int]] = defaultdict(list)
        for u in yz_nodes:
            neighbors = frozenset(self.neighbors(u))
            neighbor_groups[neighbors].append(u)

        for neighbors, nodes in neighbor_groups.items():
            if len(nodes) < min_nodes or len(neighbors) < min_nodes:
                continue
            new_angle = sum(self.meas_bases[v].angle for v in nodes) % (2.0 * np.pi)
            self.assign_meas_basis(nodes[0], PlannerMeasBasis(Plane.YZ, new_angle))
            for v in nodes[1:]:
                self.remove_physical_node(v)

    def full_reduce(self, atol: float = 1e-9) -> None:
        """Reduce all Clifford nodes and some non-Clifford nodes.

        Repeat the following steps until there are no non-Clifford nodes:
            1. remove_cliffords
            2. convert_to_phase_gadget
            3. merge_yz_to_xy
            4. merge_yz_nodes
            5. if there are some removable Clifford nodes, back to step 1.

        Parameters
        ----------
        atol : `float`, optional
            absolute tolerance, by default 1e-9
        """
        while True:
            self.remove_cliffords(atol)
            self.convert_to_phase_gadget()
            self.merge_yz_to_xy()
            self.merge_yz_nodes()
            if not any(
                self.is_removable_clifford(node, atol)
                for node in self.physical_nodes - set(self.input_node_indices) - set(self.output_node_indices)
            ):
                break


def to_zx_graphstate(graph: BaseGraphState) -> tuple[ZXGraphState, dict[int, int]]:
    r"""Convert input graph to ZXGraphState.

    Parameters
    ----------
    graph : `BaseGraphState`
        The graph state to convert.

    Returns
    -------
    `tuple`\[`ZXGraphState`, `dict`\[`int`, `int`\]\]
        Converted ZXGraphState and node map for old node index to new node index.

    Raises
    ------
    TypeError
        If the input graph is not an instance of GraphState.
    """
    graph.check_canonical_form()
    if not isinstance(graph, GraphState):
        msg = "The input graph must be an instance of GraphState."
        raise TypeError(msg)

    node_map: dict[int, int] = {}
    zx_graph = ZXGraphState()

    # Copy all physical nodes and measurement bases
    for node in graph.physical_nodes:
        node_index = zx_graph.add_physical_node()
        node_map[node] = node_index
        meas_basis = graph.meas_bases.get(node, None)
        if meas_basis is not None:
            zx_graph.assign_meas_basis(node_index, meas_basis)

    # Register input nodes
    for node, q_index in graph.input_node_indices.items():
        zx_graph.register_input(node_map[node], q_index)

    # Register output nodes
    for node, q_index in graph.output_node_indices.items():
        zx_graph.register_output(node_map[node], q_index)

    # Copy all physical edges
    for u, v in graph.physical_edges:
        zx_graph.add_physical_edge(node_map[u], node_map[v])

    # Copy local Clifford operators
    for node, lc in graph.local_cliffords.items():
        zx_graph.apply_local_clifford(node_map[node], lc)

    return zx_graph, node_map


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
    return {(min(u, v), max(u, v)) for u, v in combinations(nodes, 2)}
