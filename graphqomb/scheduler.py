"""Graph scheduler for measurement and preparation timing in MBQC patterns.

This module provides:

- `compress_schedule`: Compress preparation and measurement times by removing gaps.
- `Scheduler`: Schedule graph node preparation and measurement operations
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from graphqomb.feedforward import dag_from_flow
from graphqomb.schedule_solver import ScheduleConfig, Strategy, solve_schedule

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState


def compress_schedule(  # noqa: C901, PLR0912
    prepare_time: Mapping[int, int | None],
    measure_time: Mapping[int, int | None],
    entangle_time: Mapping[frozenset[int], int | None] | None = None,
) -> tuple[dict[int, int | None], dict[int, int | None], dict[frozenset[int], int | None]]:
    r"""Compress a schedule by removing gaps in time indices.

    This function shifts all time indices forward to remove unused time slots,
    reducing the total number of slices without changing the relative ordering.

    Parameters
    ----------
    prepare_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
        A mapping from node indices to their preparation time.
    measure_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
        A mapping from node indices to their measurement time.
    entangle_time : `collections.abc.Mapping`\[`frozenset`\[`int`\], `int` | `None`\] | `None`, optional
        A mapping from edges (as frozensets) to their entanglement time.

    Returns
    -------
    tuple
        A tuple containing:

        - `dict`\[`int`, `int` | `None`\]: compressed prepare_time
        - `dict`\[`int`, `int` | `None`\]: compressed measure_time
        - `dict`\[`frozenset`\[`int`\], `int` | `None`\]: compressed entangle_time
    """
    # Collect all used time indices
    all_times: set[int] = set()

    for time in prepare_time.values():
        if time is not None:
            all_times.add(time)

    for time in measure_time.values():
        if time is not None:
            all_times.add(time)

    if entangle_time is not None:
        for time in entangle_time.values():
            if time is not None:
                all_times.add(time)

    if not all_times:
        compressed_entangle_time = dict(entangle_time) if entangle_time is not None else {}
        return dict(prepare_time), dict(measure_time), compressed_entangle_time

    # Create mapping from old time to new compressed time
    sorted_times = sorted(all_times)
    time_mapping = {old_time: new_time for new_time, old_time in enumerate(sorted_times)}

    # Apply compression to preparation times
    compressed_prepare_time: dict[int, int | None] = {}
    for node, old_time in prepare_time.items():
        if old_time is not None:
            compressed_prepare_time[node] = time_mapping[old_time]
        else:
            compressed_prepare_time[node] = None

    # Apply compression to measurement times
    compressed_measure_time: dict[int, int | None] = {}
    for node, old_time in measure_time.items():
        if old_time is not None:
            compressed_measure_time[node] = time_mapping[old_time]
        else:
            compressed_measure_time[node] = None

    # Apply compression to entanglement times
    compressed_entangle_time = {}
    if entangle_time is not None:
        for edge, old_time in entangle_time.items():
            if old_time is not None:
                compressed_entangle_time[edge] = time_mapping[old_time]
            else:
                compressed_entangle_time[edge] = None

    return compressed_prepare_time, compressed_measure_time, compressed_entangle_time


class Scheduler:
    r"""Schedule graph preparation and measurements.

    Attributes
    ----------
    graph : `BaseGraphState`
        The graph state to be scheduled.
    dag : `dict`\[`int`, `set`\[`int`\]\]
        The directed acyclic graph representing dependencies.
    prepare_time : `dict`\[`int`, `int` | `None`\]
        A mapping from node indices to their preparation time.
    measure_time : `dict`\[`int`, `int` | `None`\]
        A mapping from node indices to their measurement time.
    entangle_time : `dict`\[`frozenset`\[`int`\], `int` | `None`\]
        A mapping from edge (as frozenset of two node indices) to their entanglement time.
    """

    graph: BaseGraphState
    dag: dict[int, set[int]]
    prepare_time: dict[int, int | None]
    measure_time: dict[int, int | None]
    entangle_time: dict[frozenset[int], int | None]

    def __init__(
        self,
        graph: BaseGraphState,
        xflow: Mapping[int, AbstractSet[int]],
        zflow: Mapping[int, AbstractSet[int]] | None = None,
    ) -> None:
        self.graph = graph
        self.dag = dag_from_flow(graph, xflow, zflow)
        self.prepare_time = dict.fromkeys(graph.physical_nodes - graph.input_node_indices.keys())
        self.measure_time = dict.fromkeys(graph.physical_nodes - graph.output_node_indices.keys())
        # Initialize entangle_time for all physical edges
        self.entangle_time = {frozenset(edge): None for edge in graph.physical_edges}

    def num_slices(self) -> int:
        r"""Return the number of slices in the schedule.

        Returns
        -------
        `int`
            The number of slices, which is the maximum time across all nodes and edges plus one.
        """
        return (
            max(
                max((t for t in self.prepare_time.values() if t is not None), default=0),
                max((t for t in self.measure_time.values() if t is not None), default=0),
                max((t for t in self.entangle_time.values() if t is not None), default=0),
            )
            + 1
        )

    @property
    def timeline(self) -> list[tuple[set[int], set[int]]]:
        r"""Get the timeline as a list of sets of nodes.

        Returns
        -------
        `list`\[`tuple`\[`set`\[`int`\], `set`\[`int`\]\]
            A list where each element is a tuple containing a set of node indices
            scheduled for preparation and a set of node indices scheduled for measurement.
        """
        prep_time: defaultdict[int, set[int]] = defaultdict(set)
        for node, time in self.prepare_time.items():
            if time is not None:
                prep_time[time].add(node)
        meas_time: defaultdict[int, set[int]] = defaultdict(set)
        for node, time in self.measure_time.items():
            if time is not None:
                meas_time[time].add(node)
        return [(prep_time[time], meas_time[time]) for time in range(self.num_slices())]

    @property
    def detailed_timeline(self) -> list[tuple[set[int], set[frozenset[int]], set[int]]]:
        r"""Get the detailed timeline including entanglement operations.

        Returns
        -------
        `list`\[`tuple`\[`set`\[`int`\], `set`\[`frozenset`\[`int`\]\], `set`\[`int`\]\]\]
            A list where each element is a tuple containing:
            - set of node indices scheduled for preparation
            - set of edges (as frozensets) scheduled for entanglement
            - set of node indices scheduled for measurement
        """
        prep_time: defaultdict[int, set[int]] = defaultdict(set)
        for node, time in self.prepare_time.items():
            if time is not None:
                prep_time[time].add(node)

        ent_time: defaultdict[int, set[frozenset[int]]] = defaultdict(set)
        for edge, time in self.entangle_time.items():
            if time is not None:
                ent_time[time].add(edge)

        meas_time: defaultdict[int, set[int]] = defaultdict(set)
        for node, time in self.measure_time.items():
            if time is not None:
                meas_time[time].add(node)

        return [(prep_time[time], ent_time[time], meas_time[time]) for time in range(self.num_slices())]

    def manual_schedule(
        self,
        prepare_time: Mapping[int, int | None],
        measure_time: Mapping[int, int | None],
        entangle_time: Mapping[frozenset[int], int | None] | None = None,
    ) -> None:
        r"""Set the schedule manually.

        Parameters
        ----------
        prepare_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
            A mapping from node indices to their preparation time.
        measure_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
            A mapping from node indices to their measurement time.
        entangle_time : `collections.abc.Mapping`\[`frozenset`\[`int`\], `int` | `None`\] | `None`, optional
            A mapping from edges (as frozensets) to their entanglement time.
            If None, entanglement times remain unchanged.
        """
        self.prepare_time = {
            node: prepare_time.get(node, None)
            for node in self.graph.physical_nodes - self.graph.input_node_indices.keys()
        }
        self.measure_time = {
            node: measure_time.get(node, None)
            for node in self.graph.physical_nodes - self.graph.output_node_indices.keys()
        }
        if entangle_time is not None:
            self.entangle_time = {edge: entangle_time.get(edge, None) for edge in self.entangle_time}

    def _validate_node_sets(self) -> bool:
        """Validate that node sets are correctly configured.

        Returns
        -------
        `bool`
            True if input/output nodes are correctly excluded from prepare/measure times.
        """
        input_nodes = self.graph.input_node_indices.keys()
        output_nodes = self.graph.output_node_indices.keys()
        physical_nodes = self.graph.physical_nodes

        # Input nodes should not be in prepare_time
        if not input_nodes.isdisjoint(self.prepare_time.keys()):
            return False

        # Output nodes should not be in measure_time
        if not output_nodes.isdisjoint(self.measure_time.keys()):
            return False

        # Check expected node sets
        expected_prep_nodes = physical_nodes - input_nodes
        expected_meas_nodes = physical_nodes - output_nodes

        return self.prepare_time.keys() == expected_prep_nodes and self.measure_time.keys() == expected_meas_nodes

    def _validate_all_nodes_scheduled(self) -> bool:
        """Validate that all required nodes are scheduled.

        Returns
        -------
        `bool`
            True if all nodes in prepare_time and measure_time have non-None values.
        """
        # All nodes in prepare_time must have non-None values
        if any(time is None for time in self.prepare_time.values()):
            return False

        # All nodes in measure_time must have non-None values
        return all(time is not None for time in self.measure_time.values())

    def _validate_dag_constraints(self) -> bool:
        """Validate that measurement order respects DAG dependencies.

        Returns
        -------
        `bool`
            True if measurement times respect the DAG ordering constraints.
        """
        for u, successors in self.dag.items():
            u_time = self.measure_time.get(u)
            if u_time is None:
                continue
            for v in successors:
                v_time = self.measure_time.get(v)
                if v_time is not None and u_time >= v_time:
                    return False
        return True

    def auto_schedule_entanglement(self) -> None:
        r"""Automatically schedule entanglement operations based on preparation times.

        Each edge is scheduled at the time when both of its endpoints are prepared.
        For edges involving input nodes, they are scheduled when the non-input node is prepared.
        Input nodes are considered to be prepared at time -1 (before the first time slice).
        """
        for edge in self.graph.physical_edges:
            edge_frozen = frozenset(edge)
            node1, node2 = edge

            # Get preparation times (input nodes are considered prepared at time -1)
            time1 = self.prepare_time.get(node1)
            if time1 is None and node1 in self.graph.input_node_indices:
                time1 = -1

            time2 = self.prepare_time.get(node2)
            if time2 is None and node2 in self.graph.input_node_indices:
                time2 = -1

            # Edge can be created when both nodes are prepared
            if time1 is not None and time2 is not None:
                self.entangle_time[edge_frozen] = max(time1, time2)

    def _validate_entangle_time_constraints(self) -> bool:
        """Validate that entanglement times respect preparation constraints.

        Returns
        -------
        `bool`
            True if entanglement times are valid, False otherwise.
        """
        for edge, ent_time in self.entangle_time.items():
            if ent_time is None:
                # Entanglement not scheduled yet is okay (might be auto-scheduled later)
                continue

            node1, node2 = edge

            # Get preparation times (input nodes are considered prepared at time -1)
            time1 = self.prepare_time.get(node1)
            if time1 is None and node1 in self.graph.input_node_indices:
                time1 = -1

            time2 = self.prepare_time.get(node2)
            if time2 is None and node2 in self.graph.input_node_indices:
                time2 = -1

            # Both nodes must be prepared before or at entanglement time
            if time1 is None or time2 is None:
                # Cannot validate if preparation times are not set
                return False

            if ent_time < time1 or ent_time < time2:
                # Entanglement happens before one of the nodes is prepared
                return False

        return True

    def _validate_time_ordering(self) -> bool:
        """Validate ordering within same time slice.

        Returns
        -------
        `bool`
            True if no node is both prepared and measured at the same time.
        """
        # Within each time slice, all measurements should happen before all preparations
        # Group nodes by time
        time_to_prep_nodes: defaultdict[int, set[int]] = defaultdict(set)
        time_to_meas_nodes: defaultdict[int, set[int]] = defaultdict(set)

        for node, time in self.prepare_time.items():
            if time is not None:
                time_to_prep_nodes[time].add(node)

        for node, time in self.measure_time.items():
            if time is not None:
                time_to_meas_nodes[time].add(node)

        # Check that no node is both prepared and measured at the same time
        all_times = time_to_prep_nodes.keys() | time_to_meas_nodes.keys()
        for time in all_times:
            prep_nodes = time_to_prep_nodes[time]
            meas_nodes = time_to_meas_nodes[time]
            if not prep_nodes.isdisjoint(meas_nodes):
                return False

        return True

    def validate_schedule(self) -> bool:
        r"""Validate that the schedule is consistent with the graph state and DAG.

        Checks:
        - Input nodes are not prepared (assumed to be prepared before time 0)
        - Output nodes are not measured
        - All non-input nodes have a preparation time
        - All non-output nodes have a measurement time
        - Measurement order respects DAG dependencies
        - Within same time slice, measurements happen before preparations
        - Entanglement times respect preparation constraints (if entanglement is scheduled)

        Returns
        -------
        `bool`
            True if the schedule is valid, False otherwise.
        """
        basic_valid = (
            self._validate_node_sets()
            and self._validate_all_nodes_scheduled()
            and self._validate_dag_constraints()
            and self._validate_time_ordering()
        )

        if not basic_valid:
            return False

        # Validate entanglement times only if at least one edge has a scheduled time
        if any(time is not None for time in self.entangle_time.values()):
            return self._validate_entangle_time_constraints()

        return True

    def solve_schedule(
        self,
        config: ScheduleConfig | None = None,
        timeout: int = 60,
    ) -> bool:
        r"""Compute the schedule using the constraint programming solver.

        Parameters
        ----------
        config : `ScheduleConfig` | `None`, optional
            The scheduling configuration. If None, defaults to MINIMIZE_SPACE strategy.
        timeout : `int`, optional
            Maximum solve time in seconds, by default 60

        Returns
        -------
        `bool`
            True if a solution was found and applied, False otherwise.
        """
        if config is None:
            config = ScheduleConfig(Strategy.MINIMIZE_SPACE)

        result = solve_schedule(self.graph, self.dag, config, timeout)
        if result is None:
            return False

        prepare_time, measure_time = result
        prep_time = {
            node: prepare_time.get(node, None)
            for node in self.graph.physical_nodes - self.graph.input_node_indices.keys()
        }
        meas_time = {
            node: measure_time.get(node, None)
            for node in self.graph.physical_nodes - self.graph.output_node_indices.keys()
        }

        # Compress the schedule to minimize time indices
        self.prepare_time, self.measure_time, compressed_entangle = compress_schedule(
            prep_time, meas_time, self.entangle_time
        )
        self.entangle_time = compressed_entangle
        return True
