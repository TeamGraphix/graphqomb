"""ZXGraph State classes for Measurement-based Quantum Computing.

This module provides:

- `ZXGraphState`: Graph State for the ZX-calculus.
- `complete_graph_edges`: Return a set of edges for the complete graph on the given nodes.
- `LocalCliffordExpansion`: Local Clifford expansion.
- `ExpansionMaps`: Expansion maps for local clifford operators.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import typing_extensions

from graphqomb.common import (
    Plane,
    PlannerMeasBasis,
    basis2tuple,
    is_clifford_angle,
    is_close_angle,
    round_clifford_angle,
)
from graphqomb.euler import LocalClifford, update_lc_basis, update_lc_lc
from graphqomb.gflow_utils import _EQUIV_MEAS_BASIS_MAP
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
    """

    __local_cliffords: dict[int, LocalClifford]

    def __init__(self) -> None:
        super().__init__()
        self.__local_cliffords = {}

    @property
    def local_cliffords(self) -> dict[int, LocalClifford]:
        r"""Return local clifford nodes.

        Returns
        -------
        `dict`\[`int`, `LocalClifford`\]
            local clifford nodes.
        """
        return self.__local_cliffords.copy()

    def apply_local_clifford(self, node: int, lc: LocalClifford) -> None:
        """Apply a local clifford to the node.

        Parameters
        ----------
        node : `int`
            node index
        lc : `LocalClifford`
            local clifford operator
        """
        self._ensure_node_exists(node)
        if node in self.input_node_indices or node in self.output_node_indices:
            original_lc = self._pop_local_clifford(node)
            if original_lc is not None:
                new_lc = update_lc_lc(lc, original_lc)
                self.__local_cliffords[node] = new_lc
            else:
                self.__local_cliffords[node] = lc
        else:
            self._check_meas_basis()
            new_meas_basis = update_lc_basis(lc, self.meas_bases[node])
            self.assign_meas_basis(node, new_meas_basis)

    def remove_physical_node(self, node: int) -> None:
        """Remove a physical node from the zxgraph state.

        Parameters
        ----------
        node : `int`
            node index
        """
        super().remove_physical_node(node)
        self.__local_cliffords.pop(node, None)

    @typing_extensions.override
    def check_canonical_form(self) -> None:
        r"""Check if the graph state is in canonical form.

        The definition of canonical form is:
        1. No Clifford operators applied.
        2. All non-output nodes have measurement basis

        Raises
        ------
        ValueError
            If the graph state is not in canonical form.
        """
        if self.__local_cliffords:
            msg = "Clifford operators are applied."
            raise ValueError(msg)
        super().check_canonical_form()

    def _pop_local_clifford(self, node: int) -> LocalClifford | None:
        """Pop local clifford of the node.

        Parameters
        ----------
        node : `int`
            node index to remove local clifford.

        Returns
        -------
        `LocalClifford` | `None`
            removed local clifford
        """
        return self.__local_cliffords.pop(node, None)

    @classmethod
    def from_base_graph_state(
        cls,
        base: BaseGraphState,
        copy_local_cliffords: bool = True,
    ) -> tuple[ZXGraphState, dict[int, int]]:
        r"""Create a new ZXGraphState from an existing BaseGraphState instance.

        This method creates a complete copy of the graph structure, including nodes,
        edges, input/output registrations, and measurement bases. Useful for creating
        mutable copies or converting between ZXGraphState implementations.

        Parameters
        ----------
        base : `BaseGraphState`
            The source graph state to copy from.
        copy_local_cliffords : `bool`, optional
            Whether to copy local Clifford operators if the source is a ZXGraphState.
            If True and the source has local Cliffords, they are copied.
            If False, local Cliffords are not copied (canonical form only).
            Default is True.

        Returns
        -------
        `tuple`\[`ZXGraphState`, `dict`\[`int`, `int`\]\]
            - Created ZXGraphState instance
            - Mapping from source node indices to new node indices
        """
        zxgraph_state, node_map = super().from_base_graph_state(base)

        # Copy local Clifford operators if requested and source is ZXGraphState
        if not copy_local_cliffords or not isinstance(base, ZXGraphState) or not hasattr(base, "local_cliffords"):
            return zxgraph_state, node_map
        for node, lc in base.local_cliffords.items():
            zxgraph_state.apply_local_clifford(node_map[node], lc)

        return zxgraph_state, node_map

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
            (self._is_trivial_meas, lambda _: None),
            (
                self._needs_pivot,
                lambda node: self.pivot(node, min(self.neighbors(node) - set(self.input_node_indices))),
            ),
            (self._needs_pivot_on_boundary, self.pivot_on_boundary),
        )

    def _assure_gflow(self, node: int, plane_map: dict[Plane, Plane], old_basis: tuple[Plane, float]) -> None:
        r"""Transform the measurement basis after applying operation to assure gflow existence.

        This method is used to assure gflow existence
        after the Clifford angle measurement basis is transformed by LocalClifford.

        Parameters
        ----------
        node : `int`
            node index
        plane_map : `dict`\[`Plane`, `Plane`\]
            mapping of planes
        old_basis : `tuple[Plane, float]`
            basis before applying the operation (such as local complement, pivot etc.)
        """
        # Round first
        cur = self.meas_bases[node]
        rounded = round_clifford_angle(cur.angle)
        self.assign_meas_basis(node, PlannerMeasBasis(cur.plane, rounded))

        # Re-read after rounding
        cur = self.meas_bases[node]
        cur_key = basis2tuple(cur)

        # Convert to an equivalent basis if plane mismatch
        if plane_map[old_basis[0]] != cur.plane:
            self.assign_meas_basis(node, _EQUIV_MEAS_BASIS_MAP[cur_key])

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

        Notes
        -----
        Here we adopt the definition (lemma) of local complementation from [1].
        In some literature, local complementation is defined with Rx(-pi/2) on the target node
        and Rz(pi/2) on the neighbors, which is strictly equivalent.

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021); arXiv:2003.01664v3 [quant-ph]. Lemma 2.31 and lemma 4.3
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

        # apply local clifford to node and assure gflow existence
        lc = LocalClifford(0, np.pi / 2, 0)
        old_meas_basis = self.meas_bases.get(node, None)
        if old_meas_basis is None:
            self.apply_local_clifford(node, lc)
        else:
            old_basis = basis2tuple(old_meas_basis)
            self.apply_local_clifford(node, lc)
            plane_map: dict[Plane, Plane] = {Plane.XY: Plane.XZ, Plane.XZ: Plane.XY, Plane.YZ: Plane.YZ}
            self._assure_gflow(node, plane_map, old_basis)

        # apply local clifford to neighbors and assure gflow existence
        lc = LocalClifford(-np.pi / 2, 0, 0)
        plane_map = {Plane.XY: Plane.XY, Plane.XZ: Plane.YZ, Plane.YZ: Plane.XZ}
        for v in nbrs:
            old_meas_basis = self.meas_bases.get(v, None)
            if old_meas_basis is None:
                self.apply_local_clifford(v, lc)
                continue

            self.apply_local_clifford(v, lc)
            old_basis = basis2tuple(old_meas_basis)
            self._assure_gflow(v, plane_map, old_basis)

    def _pivot(self, node1: int, node2: int) -> None:
        """Pivot edges around nodes u and v in the graph state.

        Parameters
        ----------
        node1 : `int`
            node index
        node2 : `int`
            node index
        """
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

        # complement edges between nbr_a, nbr_b, nbr_c
        rmv_edges: set[tuple[int, int]] = set()
        rmv_edges.update(*(p & self.physical_edges for p in nbr_pairs))
        add_edges: set[tuple[int, int]] = set()
        add_edges.update(*(p - self.physical_edges for p in nbr_pairs))
        self._update_connections(rmv_edges, add_edges)

        # swap node u and node v
        for b in nbr_b:
            self.remove_physical_edge(node1, b)
            self.add_physical_edge(node2, b)
        for c in nbr_c:
            self.remove_physical_edge(node2, c)
            self.add_physical_edge(node1, c)

    def pivot(self, node1: int, node2: int) -> None:
        """Pivot operation on the graph state: G∧(uv) (= G*u*v*u = G*v*u*v) for neighboring nodes u and v.

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

        Notes
        -----
        Here we adopt the definition (lemma) of pivot from [1].
        In some literature, pivot is defined as below::

            Rz(pi/2) Rx(-pi/2) Rz(pi/2) on the target nodes,
            Rz(pi) on all the neighbors of both target nodes (not including the target nodes).

        This definition is strictly equivalent to the one adopted here.

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021); arXiv:2003.01664v3 [quant-ph]. Lemma 2.32 and lemma 4.5
        """
        self._ensure_node_exists(node1)
        self._ensure_node_exists(node2)
        if node1 in self.input_node_indices or node2 in self.input_node_indices:
            msg = "Cannot apply pivot to input node"
            raise ValueError(msg)
        self._check_meas_basis()
        self._pivot(node1, node2)

        # update node1 and node2 measurement
        plane_map: dict[Plane, Plane] = {Plane.XY: Plane.YZ, Plane.XZ: Plane.XZ, Plane.YZ: Plane.XY}
        lc = LocalClifford(np.pi / 2, np.pi / 2, np.pi / 2)
        for a in {node1, node2}:
            old_meas_basis = self.meas_bases.get(a, None)
            if old_meas_basis is None:
                self.apply_local_clifford(a, lc)
                continue

            self.apply_local_clifford(a, lc)
            old_basis = basis2tuple(old_meas_basis)
            self._assure_gflow(a, plane_map, old_basis)

        # update nodes measurement of neighbors
        plane_map = {Plane.XY: Plane.XY, Plane.XZ: Plane.XZ, Plane.YZ: Plane.YZ}
        lc = LocalClifford(np.pi, 0, 0)
        for w in self.neighbors(node1) & self.neighbors(node2):
            old_meas_basis = self.meas_bases.get(w, None)
            if old_meas_basis is None:
                self.apply_local_clifford(w, lc)
                continue

            self.apply_local_clifford(w, lc)
            old_basis = basis2tuple(old_meas_basis)
            self._assure_gflow(w, plane_map, old_basis)

    def _is_trivial_meas(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node does not need any operation in order to perform _remove_clifford.

        For this operation, the followings must hold:
            measurement plane = YZ or XZ
            measurement angle = 0 or pi (mod 2pi)

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

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021); arXiv:2003.01664v3 [quant-ph]. Lemma 4.7
        """
        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        return (self.meas_bases[node].plane in {Plane.YZ, Plane.XZ}) and is_close_angle(2 * alpha, 0, atol)

    def _needs_lc(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node needs a local complementation in order to perform _remove_clifford.

        For this operation, the followings must hold:
            measurement plane = XY or YZ
            measurement angle = 0.5 pi or 1.5 pi (mod 2pi)

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

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021); arXiv:2003.01664v3 [quant-ph]. Lemma 4.8
        """
        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        return self.meas_bases[node].plane in {Plane.XY, Plane.YZ} and is_close_angle(2 * (alpha - np.pi / 2), 0, atol)

    def _needs_pivot(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node needs a pivot operation in order to perform _remove_clifford.

        The pivot operation is performed on the node and its non-input neighbor.
        For this operation, either of the following must hold:
            (a) measurement plane = XY and measurement angle = 0 or pi (mod 2pi)
            (b) measurement plane = XZ and measurement angle = 0.5 pi or 1.5 pi (mod 2pi)

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

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021); arXiv:2003.01664v3 [quant-ph]. Lemma 4.9
        """
        non_input_nbrs = self.neighbors(node) - set(self.input_node_indices)
        if not non_input_nbrs:
            return False

        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        # (a) measurement plane = XY and measurement angle = 0 or pi (mod 2pi)
        case_a = self.meas_bases[node].plane == Plane.XY and is_close_angle(2 * alpha, 0, atol)
        # (b) measurement plane = XZ and measurement angle = 0.5 pi or 1.5 pi (mod 2pi)
        case_b = self.meas_bases[node].plane == Plane.XZ and is_close_angle(2 * (alpha - np.pi / 2), 0, atol)
        return case_a or case_b

    def _needs_pivot_on_boundary(self, node: int, atol: float = 1e-9) -> bool:
        """Check if the node is non-input and all neighbors are input or output nodes.

        If True, pivot operation is performed on the node and its non-input neighbor, and then the node will be removed.

        For this operation, one of the following must hold:
        (a) measurement plane = XY and measurement angle = 0 or pi (mod 2pi)
        (b) measurement plane = XZ and measurement angle = 0.5 pi or 1.5 pi (mod 2pi)

        Parameters
        ----------
        node : `int`
            node index
        atol : `float`, optional
            absolute tolerance, by default 1e-9

        Returns
        -------
        `bool`
            True if the node is non-input and all neighbors are input or output nodes.

        Notes
        -----
        In order to follow the algorithm in Theorem 4.12 of Quantum 5, 421 (2021),
        this function is not commonalized into _needs_pivot.

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021); arXiv:2003.01664v3 [quant-ph]. Lemma 4.11
        """
        non_input_nbrs = self.neighbors(node) - set(self.input_node_indices)
        # check non_input_nbrs is composed of only output nodes and is not empty
        if not (non_input_nbrs.issubset(set(self.output_node_indices)) and non_input_nbrs):
            return False

        alpha = self.meas_bases[node].angle % (2.0 * np.pi)
        # (a) measurement plane = XY and measurement angle = 0 or pi (mod 2pi)
        case_a = self.meas_bases[node].plane == Plane.XY and is_close_angle(2 * alpha, 0, atol)
        # (b) measurement plane = XZ and measurement angle = 0.5 pi or 1.5 pi (mod 2pi)
        case_b = self.meas_bases[node].plane == Plane.XZ and is_close_angle(2 * (alpha - np.pi / 2), 0, atol)
        return case_a or case_b

    def pivot_on_boundary(self, node: int) -> None:
        """Perform the Clifford node removal on a corner case.

        Parameters
        ----------
        node : `int`
            node index

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021); arXiv:2003.01664v3 [quant-ph]. Lemma 4.11
        """
        output_nbr = min(self.neighbors(node) - set(self.input_node_indices))
        self.pivot(node, output_nbr)

    def _remove_clifford(self, node: int, atol: float = 1e-9) -> None:
        """Perform the Clifford node removal.

        Parameters
        ----------
        node : `int`
            node index
        atol : `float`, optional
            absolute tolerance, by default 1e-9

        Raises
        ------
        ValueError
            If the node is not a Clifford node.

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021); arXiv:2003.01664v3 [quant-ph]. Lemma 4.7
        """
        a_pi = self.meas_bases[node].angle % (2.0 * np.pi)
        if not is_close_angle(2 * a_pi, 0, atol):
            msg = "This node cannot be removed by _remove_clifford."
            raise ValueError(msg)

        lc = LocalClifford(a_pi, 0, 0)
        plane_map = {Plane.XY: Plane.XY, Plane.XZ: Plane.XZ, Plane.YZ: Plane.YZ}
        for v in self.neighbors(node):
            old_meas_basis = self.meas_bases.get(v, None)
            if old_meas_basis is None:
                self.apply_local_clifford(v, lc)
                continue

            self.apply_local_clifford(v, lc)
            old_basis = basis2tuple(old_meas_basis)
            self._assure_gflow(v, plane_map, old_basis)

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

        References
        ----------
        [1] Backens et al., Quantum 5, 421 (2021); arXiv:2003.01664v3 [quant-ph]. Theorem 4.12
        """
        self._ensure_node_exists(node)
        if node in self.input_node_indices or node in self.output_node_indices:
            msg = "Clifford node removal not allowed for input or output nodes."
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
                self._is_trivial_meas(node, atol),
                self._needs_lc(node, atol),
                self._needs_pivot(node, atol),
                self._needs_pivot_on_boundary(node, atol),
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

    def to_xy(self) -> None:
        r"""Update some special measurement basis to logically equivalent XY-basis.

        - (Plane.XZ, \pm pi/2) -> (Plane.XY, 0 or pi)
        - (Plane.YZ, \pm pi/2) -> (Plane.XY, \pm pi/2)

        This method is mainly used in convert_to_phase_gadget.
        """
        for node, basis in self.meas_bases.items():
            if basis.plane == Plane.XZ and is_close_angle(basis.angle, np.pi / 2):
                self.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, 0.0))
            elif basis.plane == Plane.XZ and is_close_angle(basis.angle, -np.pi / 2):
                self.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, np.pi))
            elif basis.plane == Plane.YZ and is_close_angle(basis.angle, np.pi / 2):
                self.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, np.pi / 2))
            elif basis.plane == Plane.YZ and is_close_angle(basis.angle, -np.pi / 2):
                self.assign_meas_basis(node, PlannerMeasBasis(Plane.XY, -np.pi / 2))

    def to_yz(self) -> None:
        r"""Update some special measurement basis to logically equivalent YZ-basis.

        - (Plane.XZ, 0) -> (Plane.YZ, 0)
        - (Plane.XZ, pi) -> (Plane.YZ, pi)

        This method is mainly used in convert_to_phase_gadget.
        """
        for node, basis in self.meas_bases.items():
            if basis.plane == Plane.XZ and is_close_angle(basis.angle, 0.0):
                self.assign_meas_basis(node, PlannerMeasBasis(Plane.YZ, 0.0))
            elif basis.plane == Plane.XZ and is_close_angle(basis.angle, np.pi):
                self.assign_meas_basis(node, PlannerMeasBasis(Plane.YZ, np.pi))

    def to_xz(self) -> None:
        r"""Update some special measurement basis to logically equivalent XZ-basis.

        This method is mainly used when we want to find a gflow.
        """
        inputs = set(self.input_node_indices)
        for node, basis in self.meas_bases.items():
            if node in inputs:
                continue
            if basis.plane == Plane.YZ and is_close_angle(basis.angle, 0.0):
                self.assign_meas_basis(node, PlannerMeasBasis(Plane.XZ, 0.0))
            elif basis.plane == Plane.YZ and is_close_angle(basis.angle, np.pi):
                self.assign_meas_basis(node, PlannerMeasBasis(Plane.XZ, np.pi))

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
            self.to_xy()
            self.to_yz()
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

    def expand_local_cliffords(self) -> ExpansionMaps:
        r"""Expand local Clifford operators applied on the input and output nodes.

        Returns
        -------
        `ExpansionMaps`
            A tuple of dictionaries mapping input and output node indices to the new node indices created.
        """
        input_node_map = self._expand_input_local_cliffords()
        output_node_map = self._expand_output_local_cliffords()
        return ExpansionMaps(input_node_map, output_node_map)

    def _expand_input_local_cliffords(self) -> dict[int, LocalCliffordExpansion]:
        r"""Expand local Clifford operators applied on the input nodes.

        Returns
        -------
        `dict`\[`int`, `LocalCliffordExpansion`\]
            A dictionary mapping input node indices to the new node indices created.
        """
        node_index_addition_map: dict[int, LocalCliffordExpansion] = {}
        new_input_indices: dict[int, int] = {}
        for old_input_node, q_index in self.input_node_indices.items():
            lc = self._pop_local_clifford(old_input_node)
            if lc is None:
                lc = LocalClifford(0.0, 0.0, 0.0)

            new_input = self.add_physical_node()
            new_node = self.add_physical_node()
            new_input_indices[new_input] = q_index

            self.add_physical_edge(new_input, new_node)
            self.add_physical_edge(new_node, old_input_node)

            self.assign_meas_basis(new_input, PlannerMeasBasis(Plane.XY, 0.0))
            self.assign_meas_basis(new_node, PlannerMeasBasis(Plane.XY, 0.0))
            meas_basis = self.meas_bases[old_input_node]
            new_meas_basis = update_lc_basis(lc, meas_basis)
            self.assign_meas_basis(old_input_node, new_meas_basis)
            self._assure_gflow_input_expansion(old_input_node)
            node_index_addition_map[old_input_node] = LocalCliffordExpansion(new_input, new_node)

        self._input_node_indices = {}
        for new_input_index, q_index in new_input_indices.items():
            self.register_input(new_input_index, q_index)

        return node_index_addition_map

    def _assure_gflow_input_expansion(self, node: int) -> None:
        r"""Assure gflow existence after input local Clifford expansion.

        Parameters
        ----------
        node : `int`
            node index
        """
        cur = self.meas_bases[node]
        rounded = round_clifford_angle(cur.angle)
        self.assign_meas_basis(node, PlannerMeasBasis(cur.plane, rounded))

        cur = self.meas_bases[node]
        cur_key = basis2tuple(cur)

        # if the updated basis is self-inclusion type, push it to an XY-equivalent one.
        if (cur.plane in {Plane.XZ, Plane.YZ} and is_close_angle(cur.angle, 0.0)) or (
            is_close_angle(cur.angle, np.pi) and cur_key in _EQUIV_MEAS_BASIS_MAP
        ):
            self.assign_meas_basis(node, _EQUIV_MEAS_BASIS_MAP[cur_key])

        # ensure XY if possible.
        cur = self.meas_bases[node]
        if cur.plane != Plane.XY and cur_key in _EQUIV_MEAS_BASIS_MAP:
            self.assign_meas_basis(node, _EQUIV_MEAS_BASIS_MAP[cur_key])

    def _expand_output_local_cliffords(self) -> dict[int, LocalCliffordExpansion]:
        r"""Expand local Clifford operators applied on the output nodes.

        Returns
        -------
        `dict`\[`int`, `LocalCliffordExpansion`\]
            A dictionary mapping output node indices to the new node indices created.
        """
        node_index_addition_map: dict[int, LocalCliffordExpansion] = {}
        new_output_node_index_map: dict[int, int] = {}
        for old_output_node, q_index in self.output_node_indices.items():
            lc = self._pop_local_clifford(old_output_node)
            if lc is None:
                new_output_node_index_map[old_output_node] = q_index
                continue

            new_node = self.add_physical_node()
            new_output_node = self.add_physical_node()
            new_output_node_index_map[new_output_node] = q_index

            self._output_node_indices.pop(old_output_node)
            self.register_output(new_output_node, q_index)

            self.add_physical_edge(old_output_node, new_node)
            self.add_physical_edge(new_node, new_output_node)

            self.assign_meas_basis(new_node, PlannerMeasBasis(Plane.XY, 0.0))
            meas_basis = update_lc_basis(lc, PlannerMeasBasis(Plane.XY, 0.0))
            self.assign_meas_basis(old_output_node, meas_basis)

            node_index_addition_map[old_output_node] = LocalCliffordExpansion(new_node, new_output_node)

        return node_index_addition_map


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


class LocalCliffordExpansion(NamedTuple):
    """Local Clifford expansion map."""

    node1: int
    node2: int


class ExpansionMaps(NamedTuple):
    """Expansion maps for inputs and outputs with Local Clifford."""

    input_node_map: dict[int, LocalCliffordExpansion]
    output_node_map: dict[int, LocalCliffordExpansion]
