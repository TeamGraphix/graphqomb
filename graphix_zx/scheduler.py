"""Graph Scheduler for compiling graph states into patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.schedule_solver import Strategy, solve_schedule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from graphix_zx.graphstate import BaseGraphState


class Scheduler:
    r"""Schedule graph preparation and measurements.

    Attributes
    ----------
    graph : `BaseGraphState`
        The graph state to be scheduled.
    prepare_time : `dict`\[`int`, `int | None`\]
        A mapping from node indices to their preparation time.
    measure_time : `dict`\[`int`, `int | None`\]
        A mapping from node indices to their measurement time.
    """

    graph: BaseGraphState
    prepare_time: dict[int, int | None]
    measure_time: dict[int, int | None]

    def __init__(self, graph: BaseGraphState) -> None:
        self.graph = graph
        self.prepare_time = dict.fromkeys(graph.physical_nodes - set(graph.input_node_indices))
        self.measure_time = dict.fromkeys(graph.physical_nodes - set(graph.output_node_indices))

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
        prep_time: dict[int, set[int]] = {}
        for node, time in self.prepare_time.items():
            if time is not None:
                prep_time.setdefault(time, set()).add(node)
        meas_time: dict[int, set[int]] = {}
        for node, time in self.measure_time.items():
            if time is not None:
                meas_time.setdefault(time, set()).add(node)
        return [(prep_time.get(time, set()), meas_time.get(time, set())) for time in range(self.num_slices())]

    def from_manual_design(
        self,
        prepare_time: Mapping[int, int],
        measure_time: Mapping[int, int],
    ) -> None:
        r"""Set the schedule manually.

        Parameters
        ----------
        prepare_time : `dict`\[`int`, `int | None`\]
            A mapping from node indices to their preparation time.
        measure_time : `dict`\[`int`, `int | None`\]
            A mapping from node indices to their measurement time.
        """
        self.prepare_time = {
            node: prepare_time.get(node, None)
            for node in self.graph.physical_nodes - set(self.graph.input_node_indices)
        }
        self.measure_time = {
            node: measure_time.get(node, None)
            for node in self.graph.physical_nodes - set(self.graph.output_node_indices)
        }

    def from_solver(
        self,
        dag: dict[int, set[int]],
        strategy: Strategy | None = None,
        timeout: int = 60,
    ) -> bool:
        r"""Compute the schedule using the constraint programming solver.

        Parameters
        ----------
        dag : `dict`\[`int`, `set`\[`int`\]
            The directed acyclic graph representing dependencies between nodes.
        strategy : `Strategy`, optional
            The optimization strategy to use. If None, defaults to MINIMIZE_SPACE.
        timeout : `int`, optional
            Maximum solve time in seconds, by default 60

        Returns
        -------
        `bool`
            True if a solution was found and applied, False otherwise.
        """
        if strategy is None:
            strategy = Strategy.MINIMIZE_SPACE

        result = solve_schedule(self.graph, dag, strategy, timeout)
        if result is None:
            return False

        prepare_time, measure_time = result
        self.prepare_time = {
            node: prepare_time.get(node, None)
            for node in self.graph.physical_nodes - set(self.graph.input_node_indices)
        }
        self.measure_time = {
            node: measure_time.get(node, None)
            for node in self.graph.physical_nodes - set(self.graph.output_node_indices)
        }
        return True
