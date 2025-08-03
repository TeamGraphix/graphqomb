"""Graph Scheduler for compiling graph states into patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        schedule: list[tuple[set[int], set[int]]] = []
        for time in range(self.num_slices()):
            prep_nodes = {node for node, t in self.prepare_time.items() if t == time}
            meas_nodes = {node for node, t in self.measure_time.items() if t == time}
            schedule.append((prep_nodes, meas_nodes))
        return schedule

    def on_the_fly_from_grouping(self, grouping: Sequence[set[int]]) -> None:
        r"""Schedule graph preparation and measurements based on a grouping.

        Parameters
        ----------
        grouping : `collections.abc.Sequence`\[`set`\[`int`\]\]
            A sequence of sets, where each set contains node indices that can be prepared and measured together.
        """
        for i, group in enumerate(grouping):
            for node in group:
                if node not in self.graph.input_node_indices:
                    self.prepare_time[node] = i
                if node not in self.graph.output_node_indices:
                    self.measure_time[node] = i + 2
