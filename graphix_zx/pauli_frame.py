"""Pauli frame for Measurement-based Quantum Computing.

This module provides:

- `PauliFrame`: A class to track the Pauli frame of a quantum computation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet


class PauliFrame:
    r"""Pauli frame tracker.

    Attributes
    ----------
    nodes : `set`\[`int`\]
        Set of nodes in the resource graph
    xflow : `dict`\[`int`, `set`\[`int`\]
        X correction flow for each measurement flip
    zflow : `dict`\[`int`, `set`\[`int`\]
        Z correction flow for each  measurement flip
    x_pauli : `dict`\[`int`, `bool`\]
        Current X Pauli state for each node
    z_pauli : `dict`\[`int`, `bool`\]
        Current Z Pauli state for each node
    inv_xflow : `dict`\[`int`, `int`\]
        Inverse X correction flow for each measurement flip
    inv_zflow : `dict`\[`int`, `int`\]
        Inverse Z correction flow for each measurement flip
    """

    nodes: set[int]
    xflow: dict[int, set[int]]
    zflow: dict[int, set[int]]
    x_pauli: dict[int, bool]
    z_pauli: dict[int, bool]
    inv_xflow: dict[int, set[int]]
    inv_zflow: dict[int, set[int]]

    def __init__(
        self,
        nodes: AbstractSet[int],
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]],
    ) -> None:
        self.nodes = set(nodes)
        self.xflow = {node: set(targets) for node, targets in xflow.items()}
        self.zflow = {node: set(targets) for node, targets in zflow.items()}
        self.x_pauli = dict.fromkeys(nodes, False)
        self.z_pauli = dict.fromkeys(nodes, False)

        self.inv_xflow = defaultdict(set)
        self.inv_zflow = defaultdict(set)
        for node, targets in self.xflow.items():
            for target in targets:
                self.inv_xflow[target].add(node)
        for node, targets in self.zflow.items():
            for target in targets:
                self.inv_zflow[target].add(node)

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
