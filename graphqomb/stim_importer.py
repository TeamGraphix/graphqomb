"""Import supported Stim circuits into GraphQOMB patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

from graphqomb.circuit import Circuit, CircuitScheduleStrategy, circuit2graph
from graphqomb.gates import CNOT, CZ, SWAP, Gate, H, Rz, S, X, Y, Z
from graphqomb.graphstate import GraphState, compose, odd_neighbors
from graphqomb.qec.qeccode import StabilizerGraphStateBuildResult, build_graph_state
from graphqomb.qec.stim_mpp import StimMppExtraction, stabilizer_code_from_stim_text
from graphqomb.qec.stim_mpp import _load_stim as _load_stim_module
from graphqomb.qompiler import qompile

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from graphqomb.pattern import Pattern


_UNITARY_GATES = frozenset({"H", "S", "SQRT_Z", "S_DAG", "SQRT_Z_DAG", "X", "Y", "Z", "CX", "CNOT", "CZ", "SWAP"})
_ANNOTATION_GATES = frozenset({"DETECTOR", "OBSERVABLE_INCLUDE"})
_SINGLE_QUBIT_GATE_FACTORIES: dict[str, Callable[[int], Gate]] = {
    "H": H,
    "S": S,
    "SQRT_Z": S,
    "S_DAG": lambda qubit: Rz(qubit, -math.pi / 2),
    "SQRT_Z_DAG": lambda qubit: Rz(qubit, -math.pi / 2),
    "X": X,
    "Y": Y,
    "Z": Z,
}
_TWO_QUBIT_GATE_FACTORIES: dict[str, Callable[[tuple[int, int]], Gate]] = {
    "CX": CNOT,
    "CNOT": CNOT,
    "CZ": CZ,
    "SWAP": SWAP,
}


class _StimGateTarget(Protocol):
    value: int
    qubit_value: int | None
    is_qubit_target: bool


class _StimCircuitInstruction(Protocol):
    name: str

    def targets_copy(self) -> list[_StimGateTarget]:
        """Return instruction targets."""
        ...

    def target_groups(self) -> list[list[_StimGateTarget]]:
        """Return grouped instruction targets."""
        ...

    def gate_args_copy(self) -> list[float]:
        """Return instruction arguments."""
        ...


class _StimRepeatBlock(Protocol):
    """Marker protocol for Stim repeat blocks."""


class _StimCircuit(Protocol):
    def __iter__(self) -> Iterator[_StimCircuitInstruction | _StimRepeatBlock]:
        """Iterate over circuit instructions and repeat blocks."""
        ...

    def flattened(self) -> _StimCircuit:
        """Return a flattened circuit."""
        ...


class _StimModule(Protocol):
    Circuit: type[Any]
    CircuitInstruction: type[Any]
    CircuitRepeatBlock: type[Any]


@dataclass(frozen=True)
class StimImportResult:
    """Result of importing a supported Stim circuit."""

    pattern: Pattern
    stim_to_qubit: dict[int, int]
    qubit_to_stim: dict[int, int]
    mpp_extractions: tuple[StimMppExtraction, ...]


@dataclass(frozen=True)
class _Fragment:
    graph: GraphState
    xflow: dict[int, set[int]]
    zflow: dict[int, set[int]]
    parity_check_groups: list[set[int]]
    logical_observables: dict[int, set[int]]
    mpp_extractions: tuple[StimMppExtraction, ...] = ()


@dataclass(frozen=True)
class _ImportContext:
    stim_to_qubit: Mapping[int, int]
    coordinate_by_stim_id: Mapping[int, tuple[float, ...]]
    coord_dims: int
    schedule_strategy: CircuitScheduleStrategy


def stim_file_to_pattern(
    path: str | Path,
    *,
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
) -> StimImportResult:
    """Import a supported Stim file into a GraphQOMB pattern.

    Returns
    -------
    `StimImportResult`
        Imported pattern, qubit mapping, and MPP extraction metadata.
    """
    return stim_text_to_pattern(
        Path(path).read_text(encoding="utf-8"),
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
    )


def stim_text_to_pattern(
    text: str,
    *,
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
) -> StimImportResult:
    """Import supported Stim text into a GraphQOMB pattern.

    Returns
    -------
    `StimImportResult`
        Imported pattern, qubit mapping, and MPP extraction metadata.
    """
    stim_module = _load_stim()
    return stim_circuit_to_pattern(
        stim_module.Circuit(text),
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
    )


def stim_circuit_to_pattern(
    circuit: _StimCircuit,
    *,
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
) -> StimImportResult:
    """Import a supported Stim circuit into a GraphQOMB pattern.

    The v1 importer supports noiseless Clifford unitary blocks and MPP blocks.
    Blocks with MPP measurements must be separated from unitary blocks by TICK.

    Returns
    -------
    `StimImportResult`
        Imported pattern, qubit mapping, and MPP extraction metadata.

    Raises
    ------
    ValueError
        If the circuit uses unsupported instructions or invalid coordinates.
    """
    if coord_dims not in {2, 3}:
        msg = "coord_dims must be 2 or 3."
        raise ValueError(msg)

    stim_module = _load_stim()
    flat_circuit = circuit.flattened()
    coordinate_by_stim_id = _extract_qubit_coordinates(flat_circuit, coord_dims=coord_dims, stim_module=stim_module)
    stim_to_qubit = _stim_to_qubit_map(flat_circuit, stim_module=stim_module)
    qubit_to_stim = {qubit: stim_id for stim_id, qubit in stim_to_qubit.items()}
    context = _ImportContext(
        stim_to_qubit=stim_to_qubit,
        coordinate_by_stim_id=coordinate_by_stim_id,
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
    )

    fragments = [
        _fragment_from_block(
            block,
            block_index=block_index,
            context=context,
        )
        for block_index, block in enumerate(_tick_blocks(flat_circuit, stim_module=stim_module))
        if block
    ]
    if not fragments:
        fragments = [
            _unitary_fragment(
                (),
                context=context,
            )
        ]

    fragment = _compose_fragments(fragments)
    pattern = qompile(
        fragment.graph,
        fragment.xflow,
        fragment.zflow,
        parity_check_group=fragment.parity_check_groups,
        logical_observables=fragment.logical_observables,
    )
    return StimImportResult(
        pattern=pattern,
        stim_to_qubit=stim_to_qubit,
        qubit_to_stim=qubit_to_stim,
        mpp_extractions=fragment.mpp_extractions,
    )


def _load_stim() -> _StimModule:
    return _load_stim_module()


def _is_circuit_instruction(
    instruction: _StimCircuitInstruction | _StimRepeatBlock,
    stim_module: _StimModule,
) -> TypeGuard[_StimCircuitInstruction]:
    return isinstance(instruction, stim_module.CircuitInstruction)


def _tick_blocks(circuit: _StimCircuit, *, stim_module: _StimModule) -> list[tuple[_StimCircuitInstruction, ...]]:
    blocks: list[tuple[_StimCircuitInstruction, ...]] = []
    current: list[_StimCircuitInstruction] = []
    for instruction in circuit:
        if not _is_circuit_instruction(instruction, stim_module):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)
        if instruction.name == "TICK":
            blocks.append(tuple(current))
            current = []
            continue
        if instruction.name == "QUBIT_COORDS":
            continue
        if instruction.name == "SHIFT_COORDS":
            msg = "SHIFT_COORDS is not supported by stim_circuit_to_pattern."
            raise ValueError(msg)
        current.append(instruction)
    blocks.append(tuple(current))
    return blocks


def _fragment_from_block(
    block: Sequence[_StimCircuitInstruction],
    *,
    block_index: int,
    context: _ImportContext,
) -> _Fragment:
    has_mpp = any(instruction.name == "MPP" for instruction in block)
    has_unitary = any(instruction.name in _UNITARY_GATES for instruction in block)
    unsupported = [
        instruction.name
        for instruction in block
        if (
            instruction.name not in _UNITARY_GATES
            and instruction.name not in _ANNOTATION_GATES
            and instruction.name != "MPP"
        )
    ]
    if unsupported:
        msg = f"Unsupported Stim instruction(s): {', '.join(sorted(set(unsupported)))}."
        raise ValueError(msg)
    if has_mpp and has_unitary:
        msg = "MPP instructions must be separated from unitary gate instructions by TICK."
        raise ValueError(msg)
    if has_mpp:
        return _mpp_fragment(
            block,
            block_index=block_index,
            context=context,
        )
    if any(instruction.name in _ANNOTATION_GATES for instruction in block):
        msg = "DETECTOR and OBSERVABLE_INCLUDE are only supported in MPP blocks."
        raise ValueError(msg)
    return _unitary_fragment(
        block,
        context=context,
    )


def _unitary_fragment(
    block: Sequence[_StimCircuitInstruction],
    *,
    context: _ImportContext,
) -> _Fragment:
    circuit = Circuit(len(context.stim_to_qubit))
    for instruction in block:
        _append_unitary_instruction(circuit, instruction, context.stim_to_qubit)

    graph, xflow, _scheduler = circuit2graph(circuit, schedule_strategy=context.schedule_strategy)
    _apply_stim_coordinates(
        graph,
        stim_to_qubit=context.stim_to_qubit,
        coordinate_by_stim_id=context.coordinate_by_stim_id,
    )
    xflow_sets = {node: set(targets) for node, targets in xflow.items()}
    zflow = {node: odd_neighbors(targets, graph) for node, targets in xflow_sets.items()}
    return _Fragment(graph=graph, xflow=xflow_sets, zflow=zflow, parity_check_groups=[], logical_observables={})


def _append_unitary_instruction(
    circuit: Circuit,
    instruction: _StimCircuitInstruction,
    stim_to_qubit: Mapping[int, int],
) -> None:
    for group in instruction.target_groups():
        qubits = [_plain_qubit_target(target, instruction.name) for target in group]
        mapped = [stim_to_qubit[qubit] for qubit in qubits]
        single_factory = _SINGLE_QUBIT_GATE_FACTORIES.get(instruction.name)
        two_factory = _TWO_QUBIT_GATE_FACTORIES.get(instruction.name)
        if single_factory is not None:
            circuit.apply_macro_gate(single_factory(mapped[0]))
        elif two_factory is not None:
            circuit.apply_macro_gate(two_factory((mapped[0], mapped[1])))
        else:
            msg = f"Unsupported unitary Stim instruction: {instruction.name}."
            raise ValueError(msg)


def _mpp_fragment(
    block: Sequence[_StimCircuitInstruction],
    *,
    block_index: int,
    context: _ImportContext,
) -> _Fragment:
    text = _stim_text_with_coords(block, coordinate_by_stim_id=context.coordinate_by_stim_id)
    extraction = stabilizer_code_from_stim_text(text, coord_dims=context.coord_dims)
    qubit_indices = {
        column: context.stim_to_qubit[stim_id]
        for column, stim_id in extraction.column_to_stim.items()
    }
    z_base = 2 * block_index
    result = build_graph_state(extraction.code, z_base=z_base, data_as_io=True, qubit_indices=qubit_indices)
    xflow, zflow = _mpp_flow(result, z_base=z_base)
    return _Fragment(
        graph=result.graph,
        xflow=xflow,
        zflow=zflow,
        parity_check_groups=extraction.detector_groups(result.ancilla_nodes),
        logical_observables=extraction.logical_observables(result.ancilla_nodes),
        mpp_extractions=(extraction,),
    )


def _mpp_flow(
    result: StabilizerGraphStateBuildResult,
    *,
    z_base: int,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    xflow: dict[int, set[int]] = {}
    zflow: dict[int, set[int]] = {}
    for qubit in {key[0] for key in result.data_nodes}:
        lower_node = result.data_nodes[qubit, z_base]
        upper_node = result.data_nodes[qubit, z_base + 1]
        xflow[lower_node] = {upper_node}
        zflow[lower_node] = set()
    for ancilla_node in result.ancilla_nodes.values():
        xflow[ancilla_node] = {ancilla_node}
        zflow[ancilla_node] = set()
    return xflow, zflow


def _compose_fragments(fragments: Sequence[_Fragment]) -> _Fragment:
    current = fragments[0]
    for fragment in fragments[1:]:
        graph, node_map1, node_map2 = compose(current.graph, fragment.graph)
        current = _Fragment(
            graph=graph,
            xflow=_remap_flow(current.xflow, node_map1) | _remap_flow(fragment.xflow, node_map2),
            zflow=_remap_flow(current.zflow, node_map1) | _remap_flow(fragment.zflow, node_map2),
            parity_check_groups=[
                *[_remap_node_set(group, node_map1) for group in current.parity_check_groups],
                *[_remap_node_set(group, node_map2) for group in fragment.parity_check_groups],
            ],
            logical_observables=_merge_logical_observables(
                _remap_logical_observables(current.logical_observables, node_map1),
                _remap_logical_observables(fragment.logical_observables, node_map2),
            ),
            mpp_extractions=(*current.mpp_extractions, *fragment.mpp_extractions),
        )
    return current


def _remap_flow(flow: Mapping[int, set[int]], node_map: Mapping[int, int]) -> dict[int, set[int]]:
    return {node_map[node]: _remap_node_set(targets, node_map) for node, targets in flow.items()}


def _remap_node_set(nodes: set[int], node_map: Mapping[int, int]) -> set[int]:
    return {node_map[node] for node in nodes}


def _remap_logical_observables(
    logical_observables: Mapping[int, set[int]],
    node_map: Mapping[int, int],
) -> dict[int, set[int]]:
    return {logical_idx: _remap_node_set(nodes, node_map) for logical_idx, nodes in logical_observables.items()}


def _merge_logical_observables(*items: Mapping[int, set[int]]) -> dict[int, set[int]]:
    merged: dict[int, set[int]] = {}
    for item in items:
        for logical_idx, nodes in item.items():
            merged.setdefault(logical_idx, set()).symmetric_difference_update(nodes)
    return merged


def _apply_stim_coordinates(
    graph: GraphState,
    *,
    stim_to_qubit: Mapping[int, int],
    coordinate_by_stim_id: Mapping[int, tuple[float, ...]],
) -> None:
    qubit_to_stim = {qubit: stim_id for stim_id, qubit in stim_to_qubit.items()}
    for node, q_index in graph.input_node_indices.items() | graph.output_node_indices.items():
        stim_id = qubit_to_stim[q_index]
        coord = coordinate_by_stim_id.get(stim_id)
        if coord is not None:
            graph.set_coordinate(node, coord)


def _extract_qubit_coordinates(
    circuit: _StimCircuit,
    *,
    coord_dims: int,
    stim_module: _StimModule,
) -> dict[int, tuple[float, ...]]:
    coordinates: dict[int, tuple[float, ...]] = {}
    for instruction in circuit:
        if not _is_circuit_instruction(instruction, stim_module):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)
        if instruction.name != "QUBIT_COORDS":
            continue
        args = instruction.gate_args_copy()
        if len(args) < coord_dims:
            msg = f"QUBIT_COORDS has {len(args)} coordinate(s), fewer than requested coord_dims={coord_dims}."
            raise ValueError(msg)
        coord = tuple(float(value) for value in args[:coord_dims])
        for target in instruction.targets_copy():
            coordinates[_plain_qubit_target(target, instruction.name)] = coord
    return coordinates


def _stim_to_qubit_map(circuit: _StimCircuit, *, stim_module: _StimModule) -> dict[int, int]:
    stim_ids: set[int] = set()
    for instruction in circuit:
        if not _is_circuit_instruction(instruction, stim_module):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)
        if instruction.name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}:
            continue
        for target in instruction.targets_copy():
            qubit_value = target.qubit_value
            if qubit_value is not None:
                stim_ids.add(int(qubit_value))
    return {stim_id: qubit for qubit, stim_id in enumerate(sorted(stim_ids))}


def _plain_qubit_target(target: _StimGateTarget, instruction_name: str) -> int:
    qubit_value = target.qubit_value
    if qubit_value is None or not target.is_qubit_target:
        msg = f"{instruction_name} contains unsupported target {target!r}; only plain qubit targets are supported."
        raise ValueError(msg)
    return int(qubit_value)


def _stim_text_with_coords(
    instructions: Sequence[_StimCircuitInstruction],
    *,
    coordinate_by_stim_id: Mapping[int, tuple[float, ...]],
) -> str:
    stim_ids = sorted(
        int(target.qubit_value)
        for instruction in instructions
        for target in instruction.targets_copy()
        if target.qubit_value is not None
    )
    coord_lines = [
        f"QUBIT_COORDS({', '.join(str(value) for value in coordinate_by_stim_id[stim_id])}) {stim_id}"
        for stim_id in dict.fromkeys(stim_ids)
        if stim_id in coordinate_by_stim_id
    ]
    return "\n".join([*coord_lines, *(str(instruction) for instruction in instructions)])
