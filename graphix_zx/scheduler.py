"""Graph scheduler for measurement and preparation timing in MBQC patterns.

This module provides:

- `Scheduler`: Schedule graph node preparation and measurement operations
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from graphix_zx.feedforward import dag_from_flow
from graphix_zx.schedule_solver import ScheduleConfig, Strategy, solve_schedule

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphix_zx.graphstate import BaseGraphState


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
    """

    graph: BaseGraphState
    dag: dict[int, set[int]]
    prepare_time: dict[int, int | None]
    measure_time: dict[int, int | None]

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

    def num_slices(self) -> int:
        r"""Return the number of slices in the schedule.

        Returns
        -------
        `int`
            The number of slices, which is the maximum time across all nodes plus one.
        """
        return (
            max(
                max((t for t in self.prepare_time.values() if t is not None), default=0),
                max((t for t in self.measure_time.values() if t is not None), default=0),
            )
            + 1
        )

    def get_schedule(self) -> list[tuple[set[int], set[int]]]:
        r"""Get the schedule as a list of sets of nodes.

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

    def from_manual_design(
        self,
        prepare_time: Mapping[int, int | None],
        measure_time: Mapping[int, int | None],
    ) -> None:
        r"""Set the schedule manually.

        Parameters
        ----------
        prepare_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
            A mapping from node indices to their preparation time.
        measure_time : `collections.abc.Mapping`\[`int`, `int` | `None`\]
            A mapping from node indices to their measurement time.
        """
        self.prepare_time = {
            node: prepare_time.get(node, None)
            for node in self.graph.physical_nodes - self.graph.input_node_indices.keys()
        }
        self.measure_time = {
            node: measure_time.get(node, None)
            for node in self.graph.physical_nodes - self.graph.output_node_indices.keys()
        }

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
        if input_nodes & self.prepare_time.keys():
            return False

        # Output nodes should not be in measure_time
        if output_nodes & self.measure_time.keys():
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
            if prep_nodes & meas_nodes:
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

        Returns
        -------
        `bool`
            True if the schedule is valid, False otherwise.
        """
        return (
            self._validate_node_sets()
            and self._validate_all_nodes_scheduled()
            and self._validate_dag_constraints()
            and self._validate_time_ordering()
        )

    def from_solver(
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
        self.prepare_time = {
            node: prepare_time.get(node, None)
            for node in self.graph.physical_nodes - self.graph.input_node_indices.keys()
        }
        self.measure_time = {
            node: measure_time.get(node, None)
            for node in self.graph.physical_nodes - self.graph.output_node_indices.keys()
        }

        # Compress the schedule to minimize time indices
        self.compress_schedule()
        return True

    def compress_schedule(self) -> None:
        r"""Compress the schedule by removing gaps in time indices.

        This method shifts all time indices forward to remove unused time slots,
        reducing the total number of slices without changing the relative ordering.
        Can be called manually after `from_manual_design` or is automatically
        called after `from_solver`.
        """
        # Collect all used time indices
        all_times: set[int] = set()

        for time in self.prepare_time.values():
            if time is not None:
                all_times.add(time)

        for time in self.measure_time.values():
            if time is not None:
                all_times.add(time)

        if not all_times:
            return

        # Create mapping from old time to new compressed time
        sorted_times = sorted(all_times)
        time_mapping = {old_time: new_time for new_time, old_time in enumerate(sorted_times)}

        # Apply compression to preparation times
        for node in self.prepare_time:
            old_time = self.prepare_time[node]
            if old_time is not None:
                self.prepare_time[node] = time_mapping[old_time]

        # Apply compression to measurement times
        for node in self.measure_time:
            old_time = self.measure_time[node]
            if old_time is not None:
                self.measure_time[node] = time_mapping[old_time]
