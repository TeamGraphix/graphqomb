"""Pauli frame for Measurement-based Quantum Computing.

This module provides:

- `PauliFrame`: A class to track the Pauli frame of a quantum computation.
"""

from __future__ import annotations

import typing
from collections import defaultdict
from collections.abc import Collection as AbstractCollection
from typing import TYPE_CHECKING

from graphqomb.common import Axis, determine_pauli_axis

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.common import MeasBasis
    from graphqomb.graphstate import BaseGraphState


class PauliFrame:
    r"""Pauli frame tracker.

    Attributes
    ----------
    nodes : `set`\[`int`\]
        Set of nodes managed by the frame.
    meas_bases : `dict`\[`int`, `MeasBasis`\]
        Measurement bases by node, used for dependent-chain expansion.
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
    logical_observables : `dict`\[`int`, `set`\[`int`\]\]
        Logical observables represented by logical index and seed nodes.
    inv_xflow : `dict`\[`int`, `set`\[`int`\]\]
        Inverse X correction flow for each measurement flip
    inv_zflow : `dict`\[`int`, `set`\[`int`\]\]
        Inverse Z correction flow for each measurement flip
    """

    nodes: set[int]
    meas_bases: dict[int, MeasBasis]
    xflow: dict[int, set[int]]
    zflow: dict[int, set[int]]
    x_pauli: dict[int, bool]
    z_pauli: dict[int, bool]
    parity_check_group: list[set[int]]
    logical_observables: dict[int, set[int]]
    inv_xflow: dict[int, set[int]]
    inv_zflow: dict[int, set[int]]
    _pauli_axis_cache: dict[int, Axis | None]
    _chain_cache: dict[int, frozenset[int]]

    @typing.overload
    def __init__(
        self,
        graphstate_or_nodes: BaseGraphState,
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]],
        parity_check_group: Sequence[AbstractSet[int]] | None = None,
        *,
        meas_bases: None = None,
        logical_observables: Mapping[int, AbstractSet[int]] | None = None,
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        graphstate_or_nodes: Collection[int],
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]],
        parity_check_group: Sequence[AbstractSet[int]] | None = None,
        *,
        meas_bases: Mapping[int, MeasBasis],
        logical_observables: Mapping[int, AbstractSet[int]] | None = None,
    ) -> None: ...

    def __init__(  # noqa: C901, PLR0912, PLR0913
        self,
        graphstate_or_nodes: BaseGraphState | Collection[int],
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]],
        parity_check_group: Sequence[AbstractSet[int]] | None = None,
        *,
        meas_bases: Mapping[int, MeasBasis] | None = None,
        logical_observables: Mapping[int, AbstractSet[int]] | None = None,
    ) -> None:
        if parity_check_group is None:
            parity_check_group = []
        if logical_observables is None:
            logical_observables = {}
        nodes: set[int]
        parsed_meas_bases: dict[int, MeasBasis]
        if _is_graphstate_like(graphstate_or_nodes):
            if meas_bases is not None:
                msg = "meas_bases must not be provided when graphstate is supplied."
                raise ValueError(msg)
            nodes = set(graphstate_or_nodes.physical_nodes)
            parsed_meas_bases = dict(graphstate_or_nodes.meas_bases)
        else:
            if meas_bases is None:
                msg = "meas_bases must be provided when constructing PauliFrame from nodes."
                raise ValueError(msg)
            if not isinstance(graphstate_or_nodes, AbstractCollection):
                msg = "graphstate_or_nodes must be a graph-like object or a collection of node indices."
                raise TypeError(msg)
            nodes = set(graphstate_or_nodes)
            if not all(isinstance(node, int) for node in nodes):
                msg = "All node indices must be integers."
                raise TypeError(msg)
            parsed_meas_bases = dict(meas_bases)

        unknown_basis_nodes = set(parsed_meas_bases) - nodes
        if unknown_basis_nodes:
            unknown = sorted(unknown_basis_nodes)
            msg = f"Measurement bases are defined for unknown nodes: {unknown}"
            raise ValueError(msg)
        unknown_logical_nodes = set().union(*logical_observables.values()) - nodes if logical_observables else set()
        if unknown_logical_nodes:
            unknown = sorted(unknown_logical_nodes)
            msg = f"Logical observables reference unknown nodes: {unknown}"
            raise ValueError(msg)

        self.nodes = nodes
        self.meas_bases = parsed_meas_bases
        self.xflow = {node: set(targets) for node, targets in xflow.items()}
        self.zflow = {node: set(targets) for node, targets in zflow.items()}
        self.x_pauli = dict.fromkeys(self.nodes, False)
        self.z_pauli = dict.fromkeys(self.nodes, False)
        self.parity_check_group = [set(item) for item in parity_check_group]
        self.logical_observables = {logical_idx: set(nodes) for logical_idx, nodes in logical_observables.items()}

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

        # Pre-compute Pauli axes for performance optimization
        # Nodes without a measurement basis are cached as None.
        # NOTE: if non-Pauli measurements are involved, the stim_compile func will error out earlier
        self._pauli_axis_cache = {
            node: determine_pauli_axis(self.meas_bases[node]) if node in self.meas_bases else None
            for node in self.nodes
        }
        # Cache for memoization of dependent chains
        self._chain_cache = {}

    @classmethod
    def from_nodes(  # noqa: PLR0913, PLR0917
        cls,
        nodes: Collection[int],
        meas_bases: Mapping[int, MeasBasis],
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]],
        parity_check_group: Sequence[AbstractSet[int]] | None = None,
        logical_observables: Mapping[int, AbstractSet[int]] | None = None,
    ) -> PauliFrame:
        r"""Build a Pauli frame from node and measurement-basis mappings.

        Parameters
        ----------
        nodes : `collections.abc.Collection`\[`int`\]
            Physical nodes of the frame.
        meas_bases : `collections.abc.Mapping`\[`int`, `MeasBasis`\]
            Measurement basis mapping for nodes that are measured.
        xflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
            X correction flow for each measurement flip.
        zflow : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\]
            Z correction flow for each measurement flip.
        parity_check_group : `collections.abc.Sequence`\[`collections.abc.Set`\[`int`\]\] | `None`, optional
            Parity check group for FTQC.
        logical_observables : `collections.abc.Mapping`\[`int`, `collections.abc.Set`\[`int`\]\] | `None`, optional
            Logical observables represented by logical index and seed nodes.

        Returns
        -------
        `PauliFrame`
            Constructed Pauli frame.
        """
        return cls(
            nodes,
            xflow=xflow,
            zflow=zflow,
            parity_check_group=parity_check_group,
            meas_bases=meas_bases,
            logical_observables=logical_observables,
        )

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

    def observable_groups(self) -> dict[int, set[int]]:
        r"""Get logical observable groups for all stored logical observables.

        Returns
        -------
        `dict`\[`int`, `set`\[`int`\]\]
            Expanded logical observable groups keyed by logical index.
        """
        return {
            logical_idx: self.logical_observables_group(seed_nodes)
            for logical_idx, seed_nodes in sorted(self.logical_observables.items())
        }

    def _collect_dependent_chain(self, node: int) -> set[int]:
        r"""Generalized dependent-chain collector that respects measurement planes.

        Uses recursive memoization to correctly XOR nodes reached via multiple paths.

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
        # Check memoization cache
        if node in self._chain_cache:
            return set(self._chain_cache[node])

        chain: set[int] = {node}

        # Use pre-computed Pauli axis from cache
        if node not in self._pauli_axis_cache:
            msg = f"Unknown node in PauliFrame: {node}"
            raise ValueError(msg)
        axis = self._pauli_axis_cache[node]

        # NOTE: might have to support plane instead of axis
        if axis == Axis.X:
            parents = self.inv_zflow[node]
        elif axis == Axis.Y:
            parents = self.inv_xflow[node].symmetric_difference(self.inv_zflow[node])
        elif axis == Axis.Z:
            parents = self.inv_xflow[node]
        else:
            msg = f"Node {node} is not measured in a Pauli basis."
            raise ValueError(msg)

        # Recursively collect and XOR parent chains
        for parent in parents:
            parent_chain = self._collect_dependent_chain(parent)
            chain ^= parent_chain

        # Store result in cache for future calls
        self._chain_cache[node] = frozenset(chain)

        return chain


def _is_graphstate_like(value: object) -> typing.TypeGuard[BaseGraphState]:
    r"""Check whether an object exposes graph-state-like attributes.

    Parameters
    ----------
    value : `object`
        Candidate object.

    Returns
    -------
    `typing.TypeGuard`\[`BaseGraphState`\]
        True if ``value`` behaves like a graph state.
    """
    return hasattr(value, "physical_nodes") and hasattr(value, "meas_bases")
