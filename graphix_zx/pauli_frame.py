"""Pauli frame for Measurement-based Quantum Computing.

This module provides:

- `PauliFrame`: A class to track the Pauli frame of a quantum computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        inv_z_flow: dict[int, set[int]] = {node: set() for node in self.graphstate.physical_nodes}
        for node, targets in self.zflow.items():
            for target in targets:
                inv_z_flow[target].add(node)

        x_groups = []
        z_groups = []
        for syndrome_group in self.x_parity_check_group:
            mbqc_group: set[int] = set()
            for node in syndrome_group:
                mbqc_group ^= _collect_dependent_chain(inv_z_flow, node) ^ {node}
            x_groups.append(mbqc_group)
        for syndrome_group in self.z_parity_check_group:
            mbqc_group = set()
            for node in syndrome_group:
                mbqc_group ^= _collect_dependent_chain(inv_z_flow, node) ^ {node}
            z_groups.append(mbqc_group)

        return x_groups, z_groups

    def logical_observables_group(self, target_nodes: AbstractSet[int]) -> set[int]:
        r"""Get the logical observables group for the given target nodes.

        Parameters
        ----------
        target_nodes : `collections.abc.Set`\[`int`\]
            The target nodes to get the logical observables group for.

        Returns
        -------
        `set`\[`int`\]
            The logical observables group for the given target nodes.
        """
        # NOTE: This logic assumes that all the measurements are X-based.
        group: set[int] = set()
        inv_z_flow: dict[int, set[int]] = {node: set() for node in self.graphstate.physical_nodes}
        for node, targets in self.zflow.items():
            for target in targets:
                inv_z_flow[target].add(node)
        for node in target_nodes:
            group ^= _collect_dependent_chain(inv_z_flow, node)
        return group


def _collect_dependent_chain(inv_flow: dict[int, set[int]], node: int) -> set[int]:
    chain: set[int] = set()
    untracked = {node}
    tracked = set()
    while untracked:
        current = untracked.pop()
        chain ^= {current}
        for parent in inv_flow.get(current, set()):
            if parent not in tracked:
                untracked.add(parent)
        tracked.add(current)
    return chain
