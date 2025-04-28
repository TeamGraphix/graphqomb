"""ZXGraph State classes for Measurement-based Quantum Computing.

This module provides:
- ZXGraphState: Graph State for the ZX-calculus.
- bipartite_edges: Return a set of edges for the complete bipartite graph between two sets of nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.euler import LocalClifford, is_clifford_angle
from graphix_zx.graphstate import GraphState, bipartite_edges

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


class ZXGraphState(GraphState):
    """Graph State for the ZX-calculus.

    Attributes
    ----------
    input_nodes : set[int]
        set of input nodes
    output_nodes : set[int]
        set of output nodes
    physical_nodes : set[int]
        set of physical nodes
    physical_edges : set[tuple[int]]
        physical edges
    meas_bases : dict[int, MeasBasis]
    q_indices : dict[int, int]
        qubit indices
    local_cliffords : dict[int, LocalClifford]
        local clifford operators
    """

    def __init__(self) -> None:
        super().__init__()

    def _update_connections(self, rmv_edges: Iterable[tuple[int, int]], new_edges: Iterable[tuple[int, int]]) -> None:
        """Update the physical edges of the graph state.

        Parameters
        ----------
        rmv_edges : Iterable[tuple[int, int]]
            edges to remove
        new_edges : Iterable[tuple[int, int]]
            edges to add
        """
        for edge in rmv_edges:
            self.remove_physical_edge(edge[0], edge[1])
        for edge in new_edges:
            self.add_physical_edge(edge[0], edge[1])

    def _update_node_measurement(
        self, measurement_action: Mapping[Plane, tuple[Plane, Callable[[float], float]]], v: int
    ) -> None:
        """Update the measurement action of the node.

        Parameters
        ----------
        measurement_action : Mapping[Plane, tuple[Plane, Callable[[float], float]]]
            mapping of the measurement plane to the new measurement plane and function to update the angle
        v : int
            node index
        """
        new_plane, new_angle_func = measurement_action[self.meas_bases[v].plane]
        if new_plane:
            new_angle = new_angle_func(v) % (2.0 * np.pi)
            self.set_meas_basis(v, PlannerMeasBasis(new_plane, new_angle))

    def local_complement(self, node: int) -> None:
        """Local complement operation on the graph state: G*u.

        Parameters
        ----------
        node : int
            node index. The node must not be an input node.

        Raises
        ------
        ValueError
            If the node does not exist, is an input node, or the graph is not a ZX-diagram.
        """
        self.ensure_node_exists(node)
        if node in self.input_nodes:
            msg = "Cannot apply local complement to input node."
            raise ValueError(msg)
        self.check_meas_basis()

        nbrs: set[int] = self.get_neighbors(node)
        nbr_pairs = bipartite_edges(nbrs, nbrs)
        new_edges = nbr_pairs - self.physical_edges
        rmv_edges = self.physical_edges & nbr_pairs

        self._update_connections(rmv_edges, new_edges)

        # update node measurement if not output node
        lc = LocalClifford(0, np.pi / 2, 0)
        if node not in self.output_nodes:
            self.apply_local_clifford(node, lc)

        # update neighbors measurement if not output node
        lc = LocalClifford(-np.pi / 2, 0, 0)
        for v in nbrs - self.output_nodes:
            self.apply_local_clifford(v, lc)

    def _swap(self, node1: int, node2: int) -> None:
        """Swap nodes u and v in the graph state.

        Parameters
        ----------
        node1  : int
            node index
        node2 : int
            node index
        """
        node1_nbrs = self.get_neighbors(node1) - {node2}
        node2_nbrs = self.get_neighbors(node2) - {node1}
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
        node1 : int
            node index. The node must not be an input node.
        node2 : int
            node index. The node must not be an input node.

        Raises
        ------
        ValueError
            If the nodes are input nodes, or the graph is not a ZX-diagram.
        """
        self.ensure_node_exists(node1)
        self.ensure_node_exists(node2)
        if node1 in self.input_nodes or node2 in self.input_nodes:
            msg = "Cannot apply pivot to input node"
            raise ValueError(msg)
        self.check_meas_basis()

        node1_nbrs = self.get_neighbors(node1) - {node2}
        node2_nbrs = self.get_neighbors(node2) - {node1}
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
        for a in {node1, node2} - self.output_nodes:
            self.apply_local_clifford(a, lc)

        # update nodes measurement of nbr_a
        lc = LocalClifford(np.pi, 0, 0)
        for w in nbr_a - self.output_nodes:
            self.apply_local_clifford(w, lc)

    def _needs_nop(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node does not need any operation in order to perform _remove_clifford.

        For this operation, the measurement measurement angle must be 0 or pi (mod 2pi)
        and the measurement plane must be YZ or XZ.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        bool
            True if the node is a removable Clifford vertex.
        """
        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        return abs(alpha % np.pi) < atol and (self.meas_bases[node].plane in {Plane.YZ, Plane.XZ})

    def _needs_lc(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node needs a local complementation in order to perform _remove_clifford.

        For this operation, the measurement angle must be 0.5 pi or 1.5 pi (mod 2pi)
        and the measurement plane must be YZ or XY.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        bool
            True if the node needs a local complementation.
        """
        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        return abs((alpha + 0.5 * np.pi) % np.pi) < atol and self.meas_bases[node].plane in {Plane.YZ, Plane.XY}

    def _needs_pivot_1(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the nodes need a pivot operation in order to perform _remove_clifford.

        The pivot operation is performed on the non-input neighbor of the node.
        For this operation,
        (i) the measurement angle must be 0 or pi (mod 2pi) and the measurement plane must be XY,
        or
        (ii) the measurement angle must be 0.5 pi or 1.5 pi (mod 2pi) and the measurement plane must be XZ.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        bool
            True if the nodes need a pivot operation.
        """
        if not self.get_neighbors(node) - self.input_nodes:
            return False

        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        case_a = abs(alpha % np.pi) < atol and self.meas_bases[node].plane == Plane.XY
        case_b = abs((alpha + 0.5 * np.pi) % np.pi) < atol and self.meas_bases[node].plane == Plane.XZ
        return case_a or case_b

    def _needs_pivot_2(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node needs a pivot operation on output nodes in order to perform _remove_clifford.

        The pivot operation is performed on the non-input but output neighbor of the node.
        For this operation,
        (i) the measurement angle must be 0 or pi (mod 2pi) and the measurement plane must be XY,
        or
        (ii) the measurement angle must be 0.5 pi or 1.5 pi (mod 2pi) and the measurement plane must be XZ.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        bool
            True if the node needs a pivot operation on output nodes.
        """
        nbrs = self.get_neighbors(node)
        if not (nbrs.issubset(self.output_nodes) and nbrs):
            return False

        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        case_a = abs(alpha % np.pi) < atol and self.meas_bases[node].plane == Plane.XY
        case_b = abs((alpha + 0.5 * np.pi) % np.pi) < atol and self.meas_bases[node].plane == Plane.XZ
        return case_a or case_b

    def _remove_clifford(self, node: int, atol: float = 1e-9) -> None:
        """Perform the Clifford vertex removal.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9
        """
        a_pi = self.meas_bases[node].angle % (2.0 * np.pi)
        coeff = 1.0
        if abs(a_pi % (2 * np.pi)) < atol:
            coeff = 0.0
        lc = LocalClifford(coeff * np.pi, 0, 0)
        for v in self.get_neighbors(node) - self.output_nodes:
            self.apply_local_clifford(v, lc)

        self.remove_physical_node(node)

    def remove_clifford(self, node: int, atol: float = 1e-9) -> None:
        """Remove the local clifford node.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9

        Raises
        ------
        ValueError
            1. If the node is an input node.
            2. If the node is not a Clifford vertex.
            3. If all neighbors are input nodes
                in some special cases ((meas_plane, meas_angle) = (XY, a pi), (XZ, a pi/2) for a = 0, 1).
            4. If the node has no neighbors that are not connected only to output nodes.
        """
        self.ensure_node_exists(node)
        if node in self.input_nodes or node in self.output_nodes:
            msg = "Clifford vertex removal not allowed for input node"
            raise ValueError(msg)

        if not (
            is_clifford_angle(self.meas_bases[node].angle, atol)
            and self.meas_bases[node].plane in {Plane.XY, Plane.XZ, Plane.YZ}
        ):
            msg = "This node is not a Clifford vertex."
            raise ValueError(msg)

        if self._needs_nop(node, atol):
            pass
        elif self._needs_lc(node, atol):
            self.local_complement(node)
        elif self._needs_pivot_1(node, atol) or self._needs_pivot_2(node, atol):
            nbrs = self.get_neighbors(node) - self.input_nodes
            v = min(nbrs)
            nbrs.remove(v)
            self.pivot(node, v)
        else:
            msg = "This Clifford vertex is unremovable."
            raise ValueError(msg)

        self._remove_clifford(node, atol)

    def is_removable_clifford(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node is a removable Clifford vertex.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        bool
            True if the node is a removable Clifford vertex.
        """
        return any(
            [
                self._needs_nop(node, atol),
                self._needs_lc(node, atol),
                self._needs_pivot_1(node, atol),
                self._needs_pivot_2(node, atol),
            ]
        )

    def _remove_cliffords(
        self, action_func: Callable[[int, float], None], check_func: Callable[[int, float], bool], atol: float = 1e-9
    ) -> None:
        """Remove all local clifford nodes which are specified by the check_func and action_func.

        Parameters
        ----------
        action_func : Callable[[int, float], None]
            action to perform on the node
        check_func : Callable[[int, float], bool]
            check if the node is a removable Clifford vertex
        """
        self.check_meas_basis()
        while True:
            nodes = self.physical_nodes - self.input_nodes - self.output_nodes
            clifford_nodes = [node for node in nodes if check_func(node, atol)]
            clifford_node = min(clifford_nodes, default=None)
            if clifford_node is None:
                break
            action_func(clifford_node, atol)

    def _step1_action(self, node: int, atol: float = 1e-9) -> None:
        """If _needs_lc is True, apply local complement to the node, and remove it.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9
        """
        self.local_complement(node)
        self._remove_clifford(node, atol)

    def _step2_action(self, node: int, atol: float = 1e-9) -> None:
        """If _needs_nop is True, remove the node.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9
        """
        self._remove_clifford(node, atol)

    def _step3_action(self, node: int, atol: float = 1e-9) -> None:
        """If _needs_pivot_1 is True, apply pivot operation to the node, and remove it.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9
        """
        nbrs = self.get_neighbors(node) - self.input_nodes
        nbr = min(nbrs)
        self.pivot(node, nbr)
        self._remove_clifford(node, atol)

    def _step4_action(self, node: int, atol: float = 1e-9) -> None:
        """If _needs_pivot_2 is True, apply pivot operation to the node, and remove it.

        Parameters
        ----------
        node : int
            node index
        atol : float, optional
            absolute tolerance, by default 1e-9
        """
        nbrs = self.get_neighbors(node) - self.input_nodes
        nbr = min(nbrs)
        self.pivot(node, nbr)
        self._remove_clifford(node, atol)

    def remove_cliffords(self, atol: float = 1e-9) -> None:
        """Remove all local clifford nodes which are removable.

        Parameters
        ----------
        atol : float, optional
            absolute tolerance, by default 1e-9
        """
        self.check_meas_basis()
        while any(
            self.is_removable_clifford(n, atol) for n in (self.physical_nodes - self.input_nodes - self.output_nodes)
        ):
            steps = [
                (self._step1_action, self._needs_lc),
                (self._step2_action, self._needs_nop),
                (self._step3_action, self._needs_pivot_1),
                (self._step4_action, self._needs_pivot_2),
            ]
            for action_func, check_func in steps:
                self._remove_cliffords(action_func, check_func, atol)

    def _extract_yz_adjacent_pair(self) -> tuple[int, int] | None:
        """Call inside convert_to_phase_gadget.

        Find a pair of adjacent nodes that are both measured in the YZ-plane.

        Returns
        -------
        tuple[int, int] | None
            A pair of adjacent nodes that are both measured in the YZ-plane, or None if no such pair exists.
        """
        yz_nodes = {node for node, basis in self.meas_bases.items() if basis.plane == Plane.YZ}
        for u in yz_nodes:
            for v in self.get_neighbors(u):
                if v in yz_nodes:
                    return (min(u, v), max(u, v))
        return None

    def _extract_xz_node(self) -> int | None:
        """Call inside convert_to_phase_gadget.

        Find a node that is measured in the XZ-plane.

        Returns
        -------
        int | None
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
            u for u, basis in self.meas_bases.items() if (basis.plane == Plane.YZ and len(self.get_neighbors(u)) == 1)
        }
        target_nodes = {
            u
            for u in target_candidates
            if (v := next(iter(self.get_neighbors(u)))) and self.meas_bases[v].plane == Plane.XY
        }
        for u in target_nodes:
            v = self.get_neighbors(u).pop()
            new_angle = (self.meas_bases[u].angle + self.meas_bases[v].angle) % (2.0 * np.pi)
            self.set_meas_basis(v, PlannerMeasBasis(Plane.XY, new_angle))
            self.remove_physical_node(u)

    def merge_yz_nodes(self) -> None:
        """Merge isolated YZ-measured nodes into a single node.

        If u, v nodes are measured in the YZ-plane and u, v have the same neighbors,
        then u, v can be merged into a single node.
        """
        min_yz_nodes = 2
        while (yz_nodes := {u for u, basis in self.meas_bases.items() if basis.plane == Plane.YZ}) and len(
            yz_nodes
        ) >= min_yz_nodes:
            merged = False
            for u in sorted(yz_nodes):
                for v in sorted(yz_nodes - {u}):
                    if self.get_neighbors(u) != self.get_neighbors(v):
                        continue

                    new_angle = (self.meas_bases[u].angle + self.meas_bases[v].angle) % (2.0 * np.pi)
                    self.set_meas_basis(u, PlannerMeasBasis(Plane.YZ, new_angle))
                    self.remove_physical_node(v)

                    merged = True
                    break
                if merged:
                    break
            if not merged:
                break

    def full_reduce(self, atol: float = 1e-9) -> None:
        """Reduce removable non-Clifford vertices from the graph state.

        Repeat the following steps until there are no non-Clifford vertices:
            1. remove_cliffords
            2. convert_to_phase_gadget
            3. merge_yz_to_xy
            4. merge_yz_nodes
            5. if there are some removable Clifford vertices, back to step 1.

        Parameters
        ----------
        atol : float, optional
            absolute tolerance, by default 1e-9
        """
        while True:
            self.remove_cliffords(atol)
            self.convert_to_phase_gadget()
            self.merge_yz_to_xy()
            self.merge_yz_nodes()
            if not any(
                self.is_removable_clifford(node, atol)
                for node in self.physical_nodes - self.input_nodes - self.output_nodes
            ):
                break
