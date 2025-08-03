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
