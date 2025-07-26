"""Graph Scheduler for compiling graph states into patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    time_schedule: dict[int, tuple[int, int]]

    def __init__(self, graph: BaseGraphState) -> None:
        self.graph = graph
        self.time_schedule = dict.fromkeys(graph.physical_nodes, (0, 1))

    @property
    def num_slices(self) -> int:
        """Number of slices in the schedule."""
        return len(self.time_schedule)
