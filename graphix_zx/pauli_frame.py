"""Pauli frame for Measurement-based Quantum Computing.

This module provides:

- `PauliFrame`: A class to track the Pauli frame of a quantum computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.common import Plane

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphix_zx.graphstate import BaseGraphState


class PauliFrame:
    r"""Pauli frame tracker.

    Attributes
    ----------
    nodes : `set`\[`int`\]
        Set of nodes in the resource graph
    x2x_dag : `dict`\[`int`, `set`\[`int`\]\]
        Dependency graph for X to X Pauli flips
    x2z_dag : `dict`\[`int`, `set`\[`int`\]\]
        Dependency graph for X to Z Pauli flips
    z2x_dag : `dict`\[`int`, `set`\[`int`\]\]
        Dependency graph for Z to X Pauli flips
    z2z_dag : `dict`\[`int`, `set`\[`int`\]\]
        Dependency graph for Z to Z Pauli flips
    x_pauli : `dict`\[`int`, `bool`\]
        Current X Pauli state for each node
    z_pauli : `dict`\[`int`, `bool`\]
        Current Z Pauli state for each node
    """

    nodes: set[int]
    x2x_dag: dict[int, set[int]]
    x2z_dag: dict[int, set[int]]
    z2x_dag: dict[int, set[int]]
    z2z_dag: dict[int, set[int]]
    x_pauli: dict[int, bool]
    z_pauli: dict[int, bool]

    def __init__(
        self,
        nodes: AbstractSet[int],
        x2x_dag: Mapping[int, AbstractSet[int]],
        x2z_dag: Mapping[int, AbstractSet[int]],
        z2x_dag: Mapping[int, AbstractSet[int]],
        z2z_dag: Mapping[int, AbstractSet[int]],
    ) -> None:
        self.nodes = set(nodes)
        self.x2x_dag = {node: set(targets) for node, targets in x2x_dag.items()}
        self.x2z_dag = {node: set(targets) for node, targets in x2z_dag.items()}
        self.z2x_dag = {node: set(targets) for node, targets in z2x_dag.items()}
        self.z2z_dag = {node: set(targets) for node, targets in z2z_dag.items()}
        self.x_pauli = dict.fromkeys(nodes, False)
        self.z_pauli = dict.fromkeys(nodes, False)

    def x_flip(self, node: int) -> None:
        """Flip the X Pauli frame for a given node.

        Parameters
        ----------
        node : `int`
            The node to flip.
        """
        for target in self.x2x_dag[node]:
            self.x_pauli[target] = not self.x_pauli[target]
        for target in self.x2z_dag[node]:
            self.z_pauli[target] = not self.z_pauli[target]

    def z_flip(self, node: int) -> None:
        """Flip the Z Pauli frame for a given node.

        Parameters
        ----------
        node : `int`
            The node to flip.
        """
        for target in self.z2x_dag[node]:
            self.x_pauli[target] = not self.x_pauli[target]
        for target in self.z2z_dag[node]:
            self.z_pauli[target] = not self.z_pauli[target]

    @staticmethod
    def from_xzflow(
        graph: BaseGraphState,
        x_flow: Mapping[int, AbstractSet[int]],
        z_flow: Mapping[int, AbstractSet[int]],
    ) -> PauliFrame:
        r"""Initialize a Pauli frame from the graph and correction flows.

        Parameters
        ----------
        graph : `BaseGraphState`
            The graph state.
        x_flow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
            The X correction flow.
        z_flow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
            The Z correction flow.

        Returns
        -------
        `PauliFrame`
            The initialized Pauli frame.
        """
        nodes = graph.physical_nodes
        non_output_nodes = nodes - set(graph.output_node_indices)

        x2x_dag: dict[int, set[int]] = {}
        x2z_dag: dict[int, set[int]] = {}
        z2x_dag: dict[int, set[int]] = {}
        z2z_dag: dict[int, set[int]] = {}

        for node in non_output_nodes:
            if graph.meas_bases[node].plane == Plane.XY:
                # Z byproduct
                z2x_dag[node] = set(x_flow.get(node, set()))
                z2z_dag[node] = set(z_flow.get(node, set()))
            elif graph.meas_bases[node].plane == Plane.XZ:
                # Y byproduct
                x2x_dag[node] = set(x_flow.get(node, set()))
                x2z_dag[node] = set(z_flow.get(node, set()))
                z2x_dag[node] = set(x_flow.get(node, set()))
                z2z_dag[node] = set(z_flow.get(node, set()))
            else:
                # X byproduct
                x2x_dag[node] = set(x_flow.get(node, set()))
                x2z_dag[node] = set(z_flow.get(node, set()))

        return PauliFrame(
            nodes=nodes,
            x2x_dag=x2x_dag,
            x2z_dag=x2z_dag,
            z2x_dag=z2x_dag,
            z2z_dag=z2z_dag,
        )
