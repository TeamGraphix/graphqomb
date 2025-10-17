"""Pauli frame for Measurement-based Quantum Computing.

This module provides:

- `PauliFrame`: A class to track the Pauli frame of a quantum computation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from graphqomb.common import Axis, determine_pauli_axis

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


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
    parity_check_group : `list`\[`set`\[`int`\]\]
        Parity check group for FTQC
    inv_xflow : `dict`\[`int`, `int`\]
        Inverse X correction flow for each measurement flip
    inv_zflow : `dict`\[`int`, `int`\]
        Inverse Z correction flow for each measurement flip
    """

    graphstate: BaseGraphState
    xflow: dict[int, set[int]]
    zflow: dict[int, set[int]]
    x_pauli: dict[int, bool]
    z_pauli: dict[int, bool]
    parity_check_group: list[set[int]]
    inv_xflow: dict[int, set[int]]
    inv_zflow: dict[int, set[int]]

    def __init__(
        self,
        graphstate: BaseGraphState,
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]],
        parity_check_group: Sequence[AbstractSet[int]] | None = None,
    ) -> None:
        if parity_check_group is None:
            parity_check_group = []
        self.graphstate = graphstate
        self.xflow = {node: set(targets) for node, targets in xflow.items()}
        self.zflow = {node: set(targets) for node, targets in zflow.items()}
        self.x_pauli = dict.fromkeys(graphstate.physical_nodes, False)
        self.z_pauli = dict.fromkeys(graphstate.physical_nodes, False)
        self.parity_check_group = [set(item) for item in parity_check_group]

        self.inv_xflow = defaultdict(set)
        self.inv_zflow = defaultdict(set)
        for node, targets in self.xflow.items():
            for target in targets:
                self.inv_xflow[target].add(node)
            self.inv_xflow[node] -= {node}
        for node, targets in self.zflow.items():
            for target in targets:
                self.inv_zflow[target].add(node)
            self.inv_zflow[node] -= {node}

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

    def parents(self, node: int) -> set[int]:
        r"""Get the parents of a node in the Pauli frame.

        Parameters
        ----------
        node : `int`
            The node to get parents for.

        Returns
        -------
        `set`\[`int`\]
            The set of parent nodes.
        """
        return self.inv_xflow.get(node, set()) | self.inv_zflow.get(node, set())

    def detector_groups(self) -> list[set[int]]:
        r"""Get the parity check groups.

        Returns
        -------
        `list`\[`set`\[`int`\]\]
            The parity check groups.
        """
        groups: list[set[int]] = []

        for syndrome_group in self.parity_check_group:
            mbqc_group: set[int] = set()
            for node in syndrome_group:
                mbqc_group ^= self._collect_dependent_chain(node)
            groups.append(mbqc_group)

        return groups

    def logical_observables_group(self, target_nodes: Collection[int]) -> set[int]:
        r"""Get the logical observables group for the given target nodes.

        Parameters
        ----------
        target_nodes : `collections.abc.Collection`\[`int`\]
            The target nodes to get the logical observables group for.

        Returns
        -------
        `set`\[`int`\]
            The logical observables group for the given target nodes.
        """
        group: set[int] = set()
        for node in target_nodes:
            group ^= self._collect_dependent_chain(node=node)

        return group

    def _collect_dependent_chain(self, node: int) -> set[int]:
        r"""Generalized dependent-chain collector that respects measurement planes.

        Parameters
        ----------
        node : `int`
            The starting node.

        Returns
        -------
        `set`\[`int`\]
            The set of dependent nodes in the chain.

        Raises
        ------
        ValueError
            If an unexpected output basis or measurement plane is encountered.
        """
        chain: set[int] = set()
        untracked = {node}
        tracked: set[int] = set()

        while untracked:
            current = untracked.pop()
            chain ^= {current}

            parents: set[int] = set()

            # NOTE: might have to support plane instead of axis
            axis = determine_pauli_axis(self.graphstate.meas_bases[current])
            if axis == Axis.X:
                parents = self.inv_zflow.get(current, set())
            elif axis == Axis.Y:
                parents = self.inv_xflow.get(current, set()) ^ self.inv_zflow.get(current, set())
            elif axis == Axis.Z:
                parents = self.inv_xflow.get(current, set())
            else:
                msg = f"Unexpected measurement axis: {axis}"
                raise ValueError(msg)

            for p in parents:
                if p not in tracked:
                    untracked.add(p)
            tracked.add(current)

        return chain
