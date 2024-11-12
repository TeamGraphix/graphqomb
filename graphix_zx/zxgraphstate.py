"""ZXGraph State classes for Measurement-based Quantum Computing.

This module provides:
- ZXGraphState: Graph State for the ZX-calculus.
- bipartite_edges: Return a set of edges for the complete bipartite graph between two sets of nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from graphix_zx.common import Plane
from graphix_zx.euler import is_clifford_angle
from graphix_zx.graphstate import GraphState, bipartite_edges

if TYPE_CHECKING:
    from typing import Callable


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
    physical_edges : dict[int, set[int]]
        physical edges
    meas_planes : dict[int, Plane]
        measurement planes
    meas_angles : dict[int, float]
        measurement angles
    q_indices : dict[int, int]
        qubit indices
    local_cliffords : dict[int, LocalClifford]
        local clifford operators
    """

    def __init__(self) -> None:
        super().__init__()

    def _update_connections(self, rmv_edges: set[tuple[int, int]], new_edges: set[tuple[int, int]]) -> None:
        for edge in rmv_edges:
            self.remove_physical_edge(edge[0], edge[1])
        for edge in new_edges:
            self.add_physical_edge(edge[0], edge[1])

    def _update_node_measurement(
        self, measurement_action: dict[Plane, tuple[Plane, Callable[[float], float]]], v: int
    ) -> None:
        new_plane, new_angle_func = measurement_action[self.meas_planes[v]]
        if new_plane:
            self.set_meas_plane(v, new_plane)
            self.set_meas_angle(v, new_angle_func(v) % (2.0 * np.pi))

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
        measurement_action = {
            Plane.XY: (Plane.XZ, lambda v: (0.5 * np.pi - self.meas_angles[v]) % (2.0 * np.pi)),
            Plane.XZ: (Plane.XY, lambda v: (self.meas_angles[v] - 0.5 * np.pi) % (2.0 * np.pi)),
            Plane.YZ: (Plane.YZ, lambda v: (self.meas_angles[v] + 0.5 * np.pi) % (2.0 * np.pi)),
        }
        if node not in self.output_nodes:
            self._update_node_measurement(measurement_action, node)

        # update neighbors measurement if not output node
        measurement_action = {
            Plane.XY: (Plane.XY, lambda v: (self.meas_angles[v] - 0.5 * np.pi) % (2.0 * np.pi)),
            Plane.XZ: (Plane.YZ, lambda v: (self.meas_angles[v]) % (2.0 * np.pi)),
            Plane.YZ: (Plane.XZ, lambda v: (-self.meas_angles[v]) % (2.0 * np.pi)),
        }

        for v in nbrs - self.output_nodes:
            self._update_node_measurement(measurement_action, v)

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
        measurement_action = {
            Plane.XY: (Plane.YZ, lambda v: self.meas_angles[v]),
            Plane.XZ: (Plane.XZ, lambda v: (0.5 * np.pi - self.meas_angles[v]) % (2.0 * np.pi)),
            Plane.YZ: (Plane.XY, lambda v: self.meas_angles[v]),
        }

        for a in {node1, node2} - self.output_nodes:
            self._update_node_measurement(measurement_action, a)

        # update nodes measurement of nbr_a
        measurement_action = {
            Plane.XY: (Plane.XY, lambda v: (self.meas_angles[v] + np.pi) % (2.0 * np.pi)),
            Plane.XZ: (Plane.YZ, lambda v: -self.meas_angles[v] % (2.0 * np.pi)),
            Plane.YZ: (Plane.XZ, lambda v: -self.meas_angles[v] % (2.0 * np.pi)),
        }

        for w in nbr_a - self.output_nodes:
            self._update_node_measurement(measurement_action, w)

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
        alpha = self.meas_angles[node] % (2 * np.pi)
        return abs(alpha % np.pi) < atol and (self.meas_planes[node] in {Plane.YZ, Plane.XZ})

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
        alpha = self.meas_angles[node] % (2 * np.pi)
        return abs((alpha + 0.5 * np.pi) % np.pi) < atol and self.meas_planes[node] in {Plane.YZ, Plane.XY}

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

        alpha = self.meas_angles[node] % (2 * np.pi)
        case_a = abs(alpha % np.pi) < atol and self.meas_planes[node] == Plane.XY
        case_b = abs((alpha + 0.5 * np.pi) % np.pi) < atol and self.meas_planes[node] == Plane.XZ
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

        alpha = self.meas_angles[node] % (2 * np.pi)
        case_a = abs(alpha % np.pi) < atol and self.meas_planes[node] == Plane.XY
        case_b = abs((alpha + 0.5 * np.pi) % np.pi) < atol and self.meas_planes[node] == Plane.XZ
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
        alpha = self.meas_angles[node] % (2 * np.pi)
        measurement_action = {
            Plane.XY: (Plane.XY, lambda v: (alpha + self.meas_angles[v]) % (2.0 * np.pi)),
            Plane.XZ: (
                Plane.YZ,
                lambda v: -self.meas_angles[v] % (2 * np.pi) if abs(alpha % np.pi) < atol else self.meas_angles[v],
            ),
            Plane.YZ: (
                Plane.XZ,
                lambda v: self.meas_angles[v] if abs(alpha % np.pi) < atol else -self.meas_angles[v] % (2 * np.pi),
            ),
        }
        for v in self.get_neighbors(node) - self.output_nodes:
            self._update_node_measurement(measurement_action, v)

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
            is_clifford_angle(self.meas_angles[node], atol) and self.meas_planes[node] in {Plane.XY, Plane.XZ, Plane.YZ}
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

    def _is_removable_clifford(self, node: int, atol: float = 1e-9) -> bool:
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
            clifford_node = next((node for node in nodes if check_func(node, atol)), None)
            if not clifford_node:
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
        while True:
            nodes = self.physical_nodes - self.input_nodes - self.output_nodes
            clifford_node = next(
                (
                    node
                    for node in nodes
                    if is_clifford_angle(self.meas_angles[node], atol) and self._is_removable_clifford(node, atol)
                ),
                None,
            )
            if clifford_node is None:
                break
            steps = [
                (self._step1_action, self._needs_lc),
                (self._step2_action, self._needs_nop),
                (self._step3_action, self._needs_pivot_1),
                (self._step4_action, self._needs_pivot_2),
            ]
            for action_func, check_func in steps:
                self._remove_cliffords(action_func, check_func, atol)
