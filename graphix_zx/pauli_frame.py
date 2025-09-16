"""Pauli frame for Measurement-based Quantum Computing.

This module provides:

- `PauliFrame`: A class to track the Pauli frame of a quantum computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.common import Axis, Plane

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphix_zx.graphstate import BaseGraphState


class PauliFrame:
    r"""Pauli frame tracker.

    Attributes
    ----------
    graphstate : `BaseGraphState`
        Set of nodes in the resource graph
    xflow : `dict`\[`int`, `set`\[`int`\]
        X correction flow for each measurement flip
    zflow : `dict`\[`int`, `set`\[`int`\]
        Z correction flow for each  measurement flip
    x_pauli : `dict`\[`int`, `bool`\]
        Current X Pauli state for each node
    z_pauli : `dict`\[`int`, `bool`\]
        Current Z Pauli state for each node
    """

    graphstate: BaseGraphState
    xflow: dict[int, set[int]]
    zflow: dict[int, set[int]]
    x_pauli: dict[int, bool]
    z_pauli: dict[int, bool]
    x_parity_check_group: list[set[int]]
    z_parity_check_group: list[set[int]]

    def __init__(
        self,
        graphstate: BaseGraphState,
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]],
        x_parity_check_group: Sequence[AbstractSet[int]] | None = None,
        z_parity_check_group: Sequence[AbstractSet[int]] | None = None,
    ) -> None:
        if x_parity_check_group is None:
            x_parity_check_group = []
        if z_parity_check_group is None:
            z_parity_check_group = []
        self.graphstate = graphstate
        self.xflow = {node: set(targets) for node, targets in xflow.items()}
        self.zflow = {node: set(targets) for node, targets in zflow.items()}
        self.x_pauli = dict.fromkeys(graphstate.physical_nodes, False)
        self.z_pauli = dict.fromkeys(graphstate.physical_nodes, False)

        if not x_parity_check_group:
            x_parity_check_group = []
        if not z_parity_check_group:
            z_parity_check_group = []
        self.x_parity_check_group = [set(item) for item in x_parity_check_group]
        self.z_parity_check_group = [set(item) for item in z_parity_check_group]

    def x_flip(self, node: int) -> None:
        """Flip the X Pauli mask for the given node.

        Parameters
        ----------
        node : `int`
            The node to flip.
        """
        self.x_pauli[node] = not self.x_pauli[node]

    def z_flip(self, node: int) -> None:
        """Flip the Z Pauli mask for the given node.

        Parameters
        ----------
        node : `int`
            The node to flip.
        """
        self.z_pauli[node] = not self.z_pauli[node]

    def meas_flip(self, node: int) -> None:
        """Update the Pauli frame for a measurement flip based on the given correction flows.

        Parameters
        ----------
        node : `int`
            The node to flip.
        """
        for target in self.xflow.get(node, set()):
            self.x_pauli[target] = not self.x_pauli[target]
        for target in self.zflow.get(node, set()):
            self.z_pauli[target] = not self.z_pauli[target]

    def children(self, node: int) -> set[int]:
        r"""Get the children of a node in the Pauli frame.

        Parameters
        ----------
        node : `int`
            The node to get children for.

        Returns
        -------
        `set`\[`int`\]
            The set of child nodes.
        """
        return (self.xflow.get(node, set()) | self.zflow.get(node, set())) - {node}

    def detector_groups(self) -> tuple[list[set[int]], list[set[int]]]:
        r"""Get the X and Z parity check groups.

        Returns
        -------
        `tuple`\[`list`\[`set`\[`int`\]\], `list`\[`set`\[`int`\]\]\]
            The X and Z parity check groups.
        """
        inv_x_flow, inv_z_flow = self._build_inverse_flows()

        x_groups: list[set[int]] = []
        z_groups: list[set[int]] = []

        for syndrome_group in self.x_parity_check_group:
            mbqc_group: set[int] = set()
            for node in syndrome_group:
                mbqc_group ^= self._collect_dependent_chain(inv_x_flow, inv_z_flow, node, output_basis=Axis.X)
            x_groups.append(mbqc_group)
        for syndrome_group in self.z_parity_check_group:
            mbqc_group = set()
            for node in syndrome_group:
                mbqc_group ^= self._collect_dependent_chain(inv_x_flow, inv_z_flow, node, output_basis=Axis.Z)
            z_groups.append(mbqc_group)

        return x_groups, z_groups

    def logical_observables_group(self, target_nodes_with_axes: Mapping[int, Axis]) -> set[int]:
        r"""Get the logical observables group for the given target nodes.

        Parameters
        ----------
        target_nodes : `collections.abc.Mapping`\[`int`, `Axis`\]
            The target nodes to get the logical observables group for, with their corresponding measurement axes.

        Returns
        -------
        `set`\[`int`\]
            The logical observables group for the given target nodes.
        """
        inv_x_flow, inv_z_flow = self._build_inverse_flows()

        group: set[int] = set()
        for node, axis in target_nodes_with_axes.items():

            group ^= self._collect_dependent_chain(
                inv_x_flow=inv_x_flow,
                inv_z_flow=inv_z_flow,
                node=node,
                output_basis=axis,
            )

        return group

    def _build_inverse_flows(self) -> tuple[Mapping[int, set[int]], Mapping[int, set[int]]]:
        r"""Build inverse x/z flows (parent sets) for all physical nodes.

        Returns
        -------
        `tuple`\[`collections.abc.Mapping`\[`int`, `set`\[`int`\]\], `collections.abc.Mapping`\[`int`, `set`\[`int`\]\]\]
            The inverse x and z flows.
        """
        inv_x_flow: Mapping[int, set[int]] = {n: set() for n in self.graphstate.physical_nodes}
        inv_z_flow: Mapping[int, set[int]] = {n: set() for n in self.graphstate.physical_nodes}

        for node, targets in self.xflow.items():
            for t in targets:
                inv_x_flow[t].add(node)
            inv_x_flow[node] -= {node}

        for node, targets in self.zflow.items():
            for t in targets:
                inv_z_flow[t].add(node)
            inv_z_flow[node] -= {node}

        return inv_x_flow, inv_z_flow

    def _collect_dependent_chain(  # noqa: C901
        self,
        inv_x_flow: Mapping[int, set[int]],
        inv_z_flow: Mapping[int, set[int]],
        node: int,
        output_basis: Axis | None,
    ) -> set[int]:
        r"""Generalized dependent-chain collector that respects measurement planes.

        Parameters
        ----------
        inv_x_flow : `collections.abc.Mapping`\[`int`, `set`\[`int`\]\]
            Inverse X flow mapping.
        inv_z_flow : `collections.abc.Mapping`\[`int`, `set`\[`int`\]\]
            Inverse Z flow mapping.
        node : `int`
            The starting node.
        output_basis : `str` or `None`
            The basis of the output node ("X", "Y", "Z", or None). If None, defaults to "X".

        Returns
        -------
        `set`\[`int`\]
            The set of dependent nodes in the chain.
        """
        chain: set[int] = set()
        untracked = {node}
        tracked: set[int] = set()

        outputs = set(self.graphstate.output_node_indices)

        while untracked:
            current = untracked.pop()
            chain ^= {current}

            if current in outputs:
                if output_basis == Axis.X or output_basis is None:
                    parents = inv_z_flow.get(current, set())
                elif output_basis == Axis.Z:
                    parents = inv_x_flow.get(current, set())
                elif output_basis == Axis.Y:
                    parents = inv_x_flow.get(current, set()) ^ inv_z_flow.get(current, set())

            else:
                plane = self.graphstate.meas_bases[current].plane
                if plane == Plane.XY:
                    parents = inv_z_flow.get(current, set())
                elif plane == Plane.YZ:
                    parents = inv_x_flow.get(current, set())
                elif plane == Plane.XZ:
                    parents = inv_x_flow.get(current, set()) ^ inv_z_flow.get(current, set())

            for p in parents:
                if p not in tracked:
                    untracked.add(p)
            tracked.add(current)

        return chain
