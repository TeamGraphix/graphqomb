"""Import supported Stim circuits into GraphQOMB patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import stim

from graphqomb.circuit import Circuit, CircuitScheduleStrategy, circuit2graph
from graphqomb.gates import CNOT, CZ, SWAP, Gate, H, Rz, S, X, Y, Z
from graphqomb.graphstate import GraphState, compose, odd_neighbors
from graphqomb.qec.qeccode import StabilizerGraphStateBuildResult, build_graph_state
from graphqomb.qec.stim_mpp import StimMppExtraction, stabilizer_code_from_stim_text
from graphqomb.qompiler import qompile

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

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
    auto_zflow_nodes: set[int]
    record_nodes: dict[int, int]
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
    return stim_circuit_to_pattern(
        stim.Circuit(text),
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
    )


def stim_circuit_to_pattern(
    circuit: stim.Circuit,
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

    flat_circuit = circuit.flattened()
    coordinate_by_stim_id = _extract_qubit_coordinates(flat_circuit, coord_dims=coord_dims)
    stim_to_qubit = _stim_to_qubit_map(flat_circuit)
    qubit_to_stim = {qubit: stim_id for stim_id, qubit in stim_to_qubit.items()}
    context = _ImportContext(
        stim_to_qubit=stim_to_qubit,
        coordinate_by_stim_id=coordinate_by_stim_id,
        coord_dims=coord_dims,
        schedule_strategy=schedule_strategy,
    )

    fragments = _fragments_from_blocks(_tick_blocks(flat_circuit), context=context)
    fragment = _compose_fragments(fragments)
    zflow = _resolve_zflow(fragment)
    parity_check_groups, logical_observables = _mpp_annotations(
        flat_circuit,
        record_nodes=fragment.record_nodes,
        coord_dims=coord_dims,
    )
    pattern = qompile(
        fragment.graph,
        fragment.xflow,
        zflow,
        parity_check_group=parity_check_groups,
        logical_observables=logical_observables,
    )
    return StimImportResult(
        pattern=pattern,
        stim_to_qubit=stim_to_qubit,
        qubit_to_stim=qubit_to_stim,
        mpp_extractions=fragment.mpp_extractions,
    )


def _fragments_from_blocks(
    blocks: Sequence[Sequence[stim.CircuitInstruction]],
    *,
    context: _ImportContext,
) -> list[_Fragment]:
    has_mpp = any(instruction.name == "MPP" for block in blocks for instruction in block)
    has_annotations = any(instruction.name in _ANNOTATION_GATES for block in blocks for instruction in block)
    if has_annotations and not has_mpp:
        msg = "DETECTOR and OBSERVABLE_INCLUDE require at least one MPP instruction."
        raise ValueError(msg)

    fragments: list[_Fragment] = []
    measurement_offset = 0
    for block_index, block in enumerate(blocks):
        if not block:
            continue
        fragment = _fragment_from_block(
            block,
            block_index=block_index,
            measurement_offset=measurement_offset,
            context=context,
        )
        if fragment is not None:
            fragments.append(fragment)
        measurement_offset += sum(instruction.num_measurements for instruction in block)

    return fragments or [_unitary_fragment((), context=context)]


def _tick_blocks(circuit: stim.Circuit) -> list[tuple[stim.CircuitInstruction, ...]]:
    blocks: list[tuple[stim.CircuitInstruction, ...]] = []
    current: list[stim.CircuitInstruction] = []
    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
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
    block: Sequence[stim.CircuitInstruction],
    *,
    block_index: int,
    measurement_offset: int,
    context: _ImportContext,
) -> _Fragment | None:
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
            measurement_offset=measurement_offset,
            context=context,
        )
    unitary_instructions = tuple(instruction for instruction in block if instruction.name in _UNITARY_GATES)
    if not unitary_instructions:
        return None
    return _unitary_fragment(
        unitary_instructions,
        context=context,
    )


def _unitary_fragment(
    block: Sequence[stim.CircuitInstruction],
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
    return _Fragment(
        graph=graph,
        xflow=xflow_sets,
        zflow={},
        auto_zflow_nodes=set(xflow_sets),
        record_nodes={},
    )


def _append_unitary_instruction(
    circuit: Circuit,
    instruction: stim.CircuitInstruction,
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
    block: Sequence[stim.CircuitInstruction],
    *,
    block_index: int,
    measurement_offset: int,
    context: _ImportContext,
) -> _Fragment:
    text = _stim_text_with_coords(block, coordinate_by_stim_id=context.coordinate_by_stim_id)
    extraction = stabilizer_code_from_stim_text(text, mpp_layer=None, coord_dims=context.coord_dims)
    qubit_indices = {column: context.stim_to_qubit[stim_id] for column, stim_id in extraction.column_to_stim.items()}
    z_base = 2 * block_index
    result = build_graph_state(extraction.code, z_base=z_base, data_as_io=True, qubit_indices=qubit_indices)
    xflow, zflow = _mpp_flow(result, z_base=z_base)
    return _Fragment(
        graph=result.graph,
        xflow=xflow,
        zflow=zflow,
        auto_zflow_nodes=set(),
        record_nodes={measurement_offset + row: node for row, node in result.ancilla_nodes.items()},
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
            auto_zflow_nodes=_remap_node_set(current.auto_zflow_nodes, node_map1)
            | _remap_node_set(fragment.auto_zflow_nodes, node_map2),
            record_nodes=_remap_record_nodes(current.record_nodes, node_map1)
            | _remap_record_nodes(fragment.record_nodes, node_map2),
            mpp_extractions=(*current.mpp_extractions, *fragment.mpp_extractions),
        )
    return current


def _resolve_zflow(fragment: _Fragment) -> dict[int, set[int]]:
    zflow = {node: set(targets) for node, targets in fragment.zflow.items()}
    for node in fragment.auto_zflow_nodes:
        zflow[node] = odd_neighbors(fragment.xflow[node], fragment.graph)
    return zflow


def _mpp_annotations(
    circuit: stim.Circuit,
    *,
    record_nodes: Mapping[int, int],
    coord_dims: int,
) -> tuple[list[set[int]], dict[int, set[int]]]:
    if not record_nodes:
        return [], {}

    extraction = stabilizer_code_from_stim_text(str(circuit), mpp_layer=None, coord_dims=coord_dims)
    parity_check_groups = [
        _record_indices_to_nodes(record_indices, record_nodes) for record_indices in extraction.detector_record_indices
    ]
    logical_observables = {
        logical_idx: _record_indices_to_nodes(record_indices, record_nodes)
        for logical_idx, record_indices in extraction.logical_observable_record_indices.items()
    }
    return parity_check_groups, logical_observables


def _record_indices_to_nodes(record_indices: frozenset[int], record_nodes: Mapping[int, int]) -> set[int]:
    missing_records = sorted(record_index for record_index in record_indices if record_index not in record_nodes)
    if missing_records:
        msg = f"Cannot map Stim measurement record(s) to imported MPP nodes: {missing_records}."
        raise ValueError(msg)
    return {record_nodes[record_index] for record_index in record_indices}


def _remap_flow(flow: Mapping[int, set[int]], node_map: Mapping[int, int]) -> dict[int, set[int]]:
    return {node_map[node]: _remap_node_set(targets, node_map) for node, targets in flow.items()}


def _remap_node_set(nodes: set[int], node_map: Mapping[int, int]) -> set[int]:
    return {node_map[node] for node in nodes}


def _remap_record_nodes(record_nodes: Mapping[int, int], node_map: Mapping[int, int]) -> dict[int, int]:
    return {record_index: node_map[node] for record_index, node in record_nodes.items()}


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
    circuit: stim.Circuit,
    *,
    coord_dims: int,
) -> dict[int, tuple[float, ...]]:
    coordinates: dict[int, tuple[float, ...]] = {}
    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
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


def _stim_to_qubit_map(circuit: stim.Circuit) -> dict[int, int]:
    stim_ids: set[int] = set()
    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)
        if instruction.name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}:
            continue
        for target in instruction.targets_copy():
            qubit_value = target.qubit_value
            if qubit_value is not None:
                stim_ids.add(int(qubit_value))
    return {stim_id: qubit for qubit, stim_id in enumerate(sorted(stim_ids))}


def _plain_qubit_target(target: stim.GateTarget, instruction_name: str) -> int:
    qubit_value = target.qubit_value
    if qubit_value is None or not target.is_qubit_target:
        msg = f"{instruction_name} contains unsupported target {target!r}; only plain qubit targets are supported."
        raise ValueError(msg)
    return int(qubit_value)


def _stim_text_with_coords(
    instructions: Sequence[stim.CircuitInstruction],
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
