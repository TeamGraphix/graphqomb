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
    time_schedule : `dict`\[`int`, `tuple`\[`int`, `int`\]\]
        A mapping from node indices to tuples representing the start and end times of each node's operations.
        In this context, the node is prepared at the start time and measured at the end time.
    """

    graph: BaseGraphState
    __time_schedule: dict[int, tuple[int, int]]

    def __init__(self, graph: BaseGraphState) -> None:
        self.graph = graph
        self.__time_schedule = dict.fromkeys(graph.physical_nodes, (0, 1))

    def num_slices(self) -> int:
        """Get the number of slices in the schedule.

        Returns
        -------
        `int`
            The maximum end time across all nodes in the time schedule.
        """
        return max(end for _, end in self.__time_schedule.values())

    @property
    def time_schedule(self) -> dict[int, tuple[int, int]]:
        """Get the time schedule."""
        return self.__time_schedule

    def on_the_fly_from_grouping(self, grouping: Sequence[set[int]]) -> None:
        r"""Schedule graph preparation and measurements based on a grouping.

        Parameters
        ----------
        grouping : `collections.abc.Sequence`\[`set`\[`int`\]\]
            A sequence of sets, where each set contains node indices that can be prepared and measured together.
        """
        for i, group in enumerate(grouping):
            for node in group:
                if i != len(grouping) - 1:
                    self.__time_schedule[node] = (i, i + 2)
                else:
                    self.__time_schedule[node] = (i, i + 1)
