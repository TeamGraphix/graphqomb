"""Import supported Stim circuits into GraphQOMB patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import stim

from graphqomb.circuit import Circuit, CircuitScheduleStrategy, circuit2graph
from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.feedforward import dag_from_flow
from graphqomb.gates import CNOT, CZ, SWAP, Gate, H, Rz, S, X, Y, Z
from graphqomb.graphstate import GraphState, compose
from graphqomb.qec.qeccode import StabilizerGraphStateBuildResult, YFoliation, build_graph_state
from graphqomb.qec.stim_mpp import (
    StimMppExtraction,
    _mpp_targets_to_products,
    _stim_mpp_extraction_from_records,
)
from graphqomb.qompiler import qompile

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from graphqomb.pattern import Pattern


_UNITARY_GATES = frozenset({"H", "S", "SQRT_Z", "S_DAG", "SQRT_Z_DAG", "X", "Y", "Z", "CX", "CNOT", "CZ", "SWAP"})
_SINGLE_PAULI_MEASUREMENT_AXES = {"M": Axis.Z, "MX": Axis.X, "MY": Axis.Y}
_PAIR_PAULI_MEASUREMENT_AXES = {"MXX": "X", "MYY": "Y", "MZZ": "Z"}
_PAULI_PRODUCT_MEASUREMENT_GATES = frozenset({"MPP", *_PAIR_PAULI_MEASUREMENT_AXES})
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
    record_nodes: dict[int, int]
    mpp_extractions: tuple[StimMppExtraction, ...] = ()


@dataclass(frozen=True)
class _ImportContext:
    stim_to_qubit: Mapping[int, int]
    coordinate_by_stim_id: Mapping[int, tuple[float, ...]]
    detector_record_indices: Sequence[frozenset[int]]
    logical_observable_record_indices: Mapping[int, frozenset[int]]
    schedule_strategy: CircuitScheduleStrategy
    y_foliation: YFoliation


@dataclass(frozen=True)
class _IdealizedCircuit:
    circuit: stim.Circuit
    zero_record_indices: frozenset[int]


@dataclass(frozen=True)
class _AnalyzedInstruction:
    instruction: stim.CircuitInstruction
    record_indices: tuple[int, ...]


@dataclass(frozen=True)
class _CircuitAnalysis:
    blocks: tuple[tuple[_AnalyzedInstruction, ...], ...]
    detector_record_indices: tuple[frozenset[int], ...]
    logical_observable_record_indices: dict[int, frozenset[int]]
    measurement_count: int


def stim_file_to_pattern(
    path: str | Path,
    *,
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
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
        y_foliation=y_foliation,
    )


def stim_text_to_pattern(
    text: str,
    *,
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
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
        y_foliation=y_foliation,
    )


def stim_circuit_to_pattern(
    circuit: stim.Circuit,
    *,
    coord_dims: int = 2,
    schedule_strategy: CircuitScheduleStrategy = CircuitScheduleStrategy.PARALLEL,
    y_foliation: YFoliation = YFoliation.TYPE_I,
) -> StimImportResult:
    """Import a supported Stim circuit into a GraphQOMB pattern.

    The importer supports Clifford unitary blocks and Pauli measurement blocks.
    Stim noise instructions and measurement-error probabilities are omitted
    because circuit-level noise is outside the GraphQOMB import model. Pauli
    measurement blocks must be separated from unitary blocks by TICK.

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
    idealized = _idealize_circuit(flat_circuit)
    analysis = _analyze_circuit(idealized.circuit)
    if analysis.measurement_count == 0 and (
        analysis.detector_record_indices or analysis.logical_observable_record_indices
    ):
        msg = "DETECTOR and OBSERVABLE_INCLUDE require at least one imported measurement instruction."
        raise ValueError(msg)
    coordinate_by_stim_id = _extract_qubit_coordinates(idealized.circuit, coord_dims=coord_dims)
    stim_to_qubit = _stim_to_qubit_map(idealized.circuit)
    qubit_to_stim = {qubit: stim_id for stim_id, qubit in stim_to_qubit.items()}
    context = _ImportContext(
        stim_to_qubit=stim_to_qubit,
        coordinate_by_stim_id=coordinate_by_stim_id,
        detector_record_indices=analysis.detector_record_indices,
        logical_observable_record_indices=analysis.logical_observable_record_indices,
        schedule_strategy=schedule_strategy,
        y_foliation=y_foliation,
    )

    fragments = _fragments_from_blocks(analysis.blocks, context=context)
    fragment = _compose_fragments(fragments)
    parity_check_groups, logical_observables = _measurement_annotations_from_analysis(
        analysis,
        record_nodes=fragment.record_nodes,
        zero_record_indices=idealized.zero_record_indices,
    )
    pattern = qompile(
        fragment.graph,
        fragment.xflow,
        None,
        parity_check_group=parity_check_groups,
        logical_observables=logical_observables,
    )
    return StimImportResult(
        pattern=pattern,
        stim_to_qubit=stim_to_qubit,
        qubit_to_stim=qubit_to_stim,
        mpp_extractions=fragment.mpp_extractions,
    )


def _idealize_circuit(circuit: stim.Circuit) -> _IdealizedCircuit:
    """Remove circuit-level noise while preserving ideal measurement records.

    Returns
    -------
    `_IdealizedCircuit`
        Ideal circuit and absolute indices of zero-valued records.

    Raises
    ------
    TypeError
        If a flattened circuit unexpectedly contains a repeat block.
    ValueError
        If a constant true record cannot be represented.
    """
    result = stim.Circuit()
    zero_record_indices: set[int] = set()
    measurement_offset = 0

    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)

        gate_data = stim.gate_data(instruction.name)
        if instruction.name in _SINGLE_PAULI_MEASUREMENT_AXES:
            result.append(instruction.name, instruction.targets_copy())
        elif instruction.name in _PAULI_PRODUCT_MEASUREMENT_GATES:
            _append_ideal_pauli_measurements(result, instruction)
        elif instruction.name == "MPAD":
            targets = instruction.targets_copy()
            if any(int(target.value) != 0 for target in targets):
                msg = "MPAD 1 records are not supported because detector parity offsets are not represented."
                raise ValueError(msg)
            result.append("MPAD", targets)
            zero_record_indices.update(range(measurement_offset, measurement_offset + instruction.num_measurements))
        elif gate_data.is_noisy_gate and not gate_data.is_reset:
            if gate_data.produces_measurements:
                result.append("MPAD", [0] * instruction.num_measurements)
                zero_record_indices.update(range(measurement_offset, measurement_offset + instruction.num_measurements))
        else:
            result.append(instruction)

        measurement_offset += instruction.num_measurements

    return _IdealizedCircuit(result, frozenset(zero_record_indices))


def _analyze_circuit(circuit: stim.Circuit) -> _CircuitAnalysis:
    """Index measurement records and split a flattened circuit at TICKs.

    Returns
    -------
    `_CircuitAnalysis`
        TICK-separated instructions and whole-circuit record annotations.

    Raises
    ------
    TypeError
        If a flattened circuit unexpectedly contains a repeat block.
    ValueError
        If an annotation has an invalid record target or SHIFT_COORDS is used.
    """
    blocks: list[tuple[_AnalyzedInstruction, ...]] = []
    current_block: list[_AnalyzedInstruction] = []
    detector_record_indices: list[frozenset[int]] = []
    logical_record_indices: dict[int, set[int]] = {}
    measurement_count = 0

    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)
        if instruction.name == "TICK":
            blocks.append(tuple(current_block))
            current_block = []
            continue
        if instruction.name == "QUBIT_COORDS":
            continue
        if instruction.name == "SHIFT_COORDS":
            msg = "SHIFT_COORDS is not supported by stim_circuit_to_pattern."
            raise ValueError(msg)
        if instruction.name == "DETECTOR":
            detector_record_indices.append(_annotation_record_indices(instruction, measurement_count=measurement_count))
        elif instruction.name == "OBSERVABLE_INCLUDE":
            logical_idx = _observable_index(instruction)
            logical_record_indices.setdefault(logical_idx, set()).symmetric_difference_update(
                _annotation_record_indices(instruction, measurement_count=measurement_count)
            )
        elif instruction.name != "MPAD":
            record_indices = tuple(range(measurement_count, measurement_count + instruction.num_measurements))
            current_block.append(_AnalyzedInstruction(instruction, record_indices))

        measurement_count += instruction.num_measurements

    blocks.append(tuple(current_block))
    return _CircuitAnalysis(
        blocks=tuple(blocks),
        detector_record_indices=tuple(detector_record_indices),
        logical_observable_record_indices={
            logical_idx: frozenset(records) for logical_idx, records in sorted(logical_record_indices.items())
        },
        measurement_count=measurement_count,
    )


def _append_ideal_pauli_measurements(
    circuit: stim.Circuit,
    instruction: stim.CircuitInstruction,
) -> None:
    """Append ideal MPP equivalents of Stim Pauli measurements.

    Raises
    ------
    ValueError
        If an instruction has an invalid target group.
    """
    if instruction.name == "MPP":
        circuit.append("MPP", instruction.targets_copy())
        return

    axis = _PAIR_PAULI_MEASUREMENT_AXES[instruction.name]
    expected_group_size = 2

    target_factory = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}[axis]
    for group in instruction.target_groups():
        if len(group) != expected_group_size:
            msg = f"{instruction.name} target group must contain {expected_group_size} qubit(s)."
            raise ValueError(msg)
        product_targets: list[stim.GateTarget] = []
        for index, target in enumerate(group):
            if index:
                product_targets.append(stim.target_combiner())
            product_targets.append(
                target_factory(
                    _plain_qubit_target(target, instruction.name),
                    invert=target.is_inverted_result_target,
                )
            )
        circuit.append("MPP", product_targets)


def _fragments_from_blocks(
    blocks: Sequence[Sequence[_AnalyzedInstruction]],
    *,
    context: _ImportContext,
) -> list[_Fragment]:
    _validate_blocks(blocks)
    _validate_single_measurement_lifetimes(blocks)

    fragments: list[_Fragment] = []
    mpp_layer_index = 0
    for block in blocks:
        unitary_instructions = tuple(
            analyzed.instruction for analyzed in block if analyzed.instruction.name in _UNITARY_GATES
        )
        if unitary_instructions:
            fragments.append(_unitary_fragment(unitary_instructions, context=context))
        else:
            measurement_fragments = _measurement_fragments_from_block(
                block,
                mpp_layer_index=mpp_layer_index,
                context=context,
            )
            fragments.extend(measurement_fragments)
            if any(analyzed.instruction.name == "MPP" for analyzed in block):
                mpp_layer_index += sum(
                    len(analyzed.record_indices) for analyzed in block if analyzed.instruction.name == "MPP"
                )

    return fragments or [_unitary_fragment((), context=context)]


def _validate_blocks(blocks: Sequence[Sequence[_AnalyzedInstruction]]) -> None:
    """Validate supported instructions and required TICK separation.

    Raises
    ------
    ValueError
        If an instruction is unsupported or a block mixes unitary gates with
        Pauli measurements.
    """
    for block in blocks:
        unsupported = [
            analyzed.instruction.name
            for analyzed in block
            if (
                analyzed.instruction.name not in _UNITARY_GATES
                and analyzed.instruction.name not in _SINGLE_PAULI_MEASUREMENT_AXES
                and analyzed.instruction.name != "MPP"
            )
        ]
        if unsupported:
            msg = f"Unsupported Stim instruction(s): {', '.join(sorted(set(unsupported)))}."
            raise ValueError(msg)

        has_unitary = any(analyzed.instruction.name in _UNITARY_GATES for analyzed in block)
        has_pauli_measurement = any(
            analyzed.instruction.name == "MPP" or analyzed.instruction.name in _SINGLE_PAULI_MEASUREMENT_AXES
            for analyzed in block
        )
        if has_unitary and has_pauli_measurement:
            msg = "Pauli measurement instructions must be separated from unitary gate instructions by TICK."
            raise ValueError(msg)


def _validate_single_measurement_lifetimes(
    blocks: Sequence[Sequence[_AnalyzedInstruction]],
) -> None:
    """Reject quantum operations after a directly measured qubit terminates.

    Raises
    ------
    ValueError
        If a measured qubit is reused, or a unitary block follows a direct
        measurement before reset support can establish new qubit lifetimes.
    """
    measured_qubits: set[int] = set()
    for block in blocks:
        for analyzed in block:
            instruction = analyzed.instruction
            if instruction.name in _UNITARY_GATES and measured_qubits:
                msg = (
                    "Unitary instructions after a single-qubit measurement are not supported; "
                    "reset import is required to establish the following qubit lifetimes."
                )
                raise ValueError(msg)
            if instruction.name != "MPP" and instruction.name not in _SINGLE_PAULI_MEASUREMENT_AXES:
                continue

            instruction_qubits = {
                int(target.qubit_value) for target in instruction.targets_copy() if target.qubit_value is not None
            }
            reused_qubits = measured_qubits & instruction_qubits
            if reused_qubits:
                msg = (
                    f"Stim qubit(s) {sorted(reused_qubits)} are used after a single-qubit measurement; "
                    "reset import is required before reuse."
                )
                raise ValueError(msg)
            if instruction.name in _SINGLE_PAULI_MEASUREMENT_AXES:
                measured_qubits.update(instruction_qubits)


def _measurement_fragments_from_block(
    block: Sequence[_AnalyzedInstruction],
    *,
    mpp_layer_index: int,
    context: _ImportContext,
) -> list[_Fragment]:
    """Build direct and product measurement fragments in record order.

    Returns
    -------
    `list`[`_Fragment`]
        Measurement fragments in source order.
    """
    fragments: list[_Fragment] = []
    mpp_items = tuple(analyzed for analyzed in block if analyzed.instruction.name == "MPP")
    mpp_added = False
    for analyzed in block:
        instruction = analyzed.instruction
        if instruction.name == "MPP" and not mpp_added:
            fragments.append(
                _mpp_fragment(
                    mpp_items,
                    mpp_layer_index=mpp_layer_index,
                    context=context,
                )
            )
            mpp_added = True
        elif instruction.name in _SINGLE_PAULI_MEASUREMENT_AXES:
            fragments.append(
                _single_measurement_fragment(
                    instruction,
                    record_indices=analyzed.record_indices,
                    context=context,
                )
            )
    return fragments


def _single_measurement_fragment(
    instruction: stim.CircuitInstruction,
    *,
    record_indices: Sequence[int],
    context: _ImportContext,
) -> _Fragment:
    """Build a fragment by assigning a basis directly to each measured node.

    Returns
    -------
    `_Fragment`
        Direct-measurement graph fragment and record-to-node mapping.

    Raises
    ------
    ValueError
        If target counts differ or a qubit is repeated in the instruction.
    """
    targets = instruction.targets_copy()
    if len(targets) != len(record_indices):
        msg = f"{instruction.name} target count does not match its measurement-record count."
        raise ValueError(msg)

    graph = GraphState()
    record_nodes: dict[int, int] = {}
    seen_qubits: set[int] = set()
    axis = _SINGLE_PAULI_MEASUREMENT_AXES[instruction.name]
    for target, record_index in zip(targets, record_indices, strict=True):
        stim_id = _plain_qubit_target(target, instruction.name)
        if stim_id in seen_qubits:
            msg = f"{instruction.name} measures qubit {stim_id} more than once in one instruction."
            raise ValueError(msg)
        seen_qubits.add(stim_id)

        node = graph.add_node(coordinate=context.coordinate_by_stim_id.get(stim_id))
        qubit_index = context.stim_to_qubit[stim_id]
        graph.register_input(node, qubit_index)
        graph.register_output(node, qubit_index)
        sign = Sign.MINUS if target.is_inverted_result_target else Sign.PLUS
        graph.assign_meas_basis(node, AxisMeasBasis(axis, sign))
        record_nodes[record_index] = node

    return _Fragment(
        graph=graph,
        xflow={},
        record_nodes=record_nodes,
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
    block: Sequence[_AnalyzedInstruction],
    *,
    mpp_layer_index: int,
    context: _ImportContext,
) -> _Fragment:
    supports = tuple(
        support for analyzed in block for support in _mpp_targets_to_products(analyzed.instruction.targets_copy())
    )
    record_indices = tuple(record_index for analyzed in block for record_index in analyzed.record_indices)
    extraction = _stim_mpp_extraction_from_records(
        supports,
        record_indices,
        coordinate_by_stim_id=context.coordinate_by_stim_id,
        detector_record_indices=context.detector_record_indices,
        logical_observable_record_indices=context.logical_observable_record_indices,
    )
    z_base = 2 * mpp_layer_index
    fragment = _mpp_graph_fragment(
        extraction,
        record_indices=record_indices,
        z_base=z_base,
        context=context,
    )
    if _has_causal_flow(fragment):
        return _with_mpp_extraction(fragment, extraction)

    serialized_fragments = [
        _mpp_graph_fragment(
            _stim_mpp_extraction_from_records(
                (support,),
                (record_index,),
                coordinate_by_stim_id=context.coordinate_by_stim_id,
                detector_record_indices=context.detector_record_indices,
                logical_observable_record_indices=context.logical_observable_record_indices,
            ),
            record_indices=(record_index,),
            z_base=z_base + 2 * row,
            context=context,
        )
        for row, (support, record_index) in enumerate(zip(supports, record_indices, strict=True))
    ]
    return _with_mpp_extraction(_compose_fragments(serialized_fragments), extraction)


def _mpp_graph_fragment(
    extraction: StimMppExtraction,
    *,
    record_indices: Sequence[int],
    z_base: int,
    context: _ImportContext,
) -> _Fragment:
    qubit_indices = {column: context.stim_to_qubit[stim_id] for column, stim_id in extraction.column_to_stim.items()}
    result = build_graph_state(
        extraction.code,
        z_base=z_base,
        y_foliation=context.y_foliation,
        data_as_io=True,
        qubit_indices=qubit_indices,
    )
    xflow = _mpp_flow(result)
    if len(record_indices) != len(result.ancilla_nodes):
        msg = "Imported MPP record count does not match the generated ancilla-node count."
        raise ValueError(msg)
    return _Fragment(
        graph=result.graph,
        xflow=xflow,
        record_nodes={record_indices[row]: node for row, node in result.ancilla_nodes.items()},
    )


def _has_causal_flow(fragment: _Fragment) -> bool:
    try:
        tuple(TopologicalSorter(dag_from_flow(fragment.graph, fragment.xflow)).static_order())
    except CycleError:
        return False
    return True


def _with_mpp_extraction(fragment: _Fragment, extraction: StimMppExtraction) -> _Fragment:
    return _Fragment(
        graph=fragment.graph,
        xflow=fragment.xflow,
        record_nodes=fragment.record_nodes,
        mpp_extractions=(extraction,),
    )


def _mpp_flow(
    result: StabilizerGraphStateBuildResult,
) -> dict[int, set[int]]:
    xflow: dict[int, set[int]] = {}
    measured_nodes_by_qubit: dict[int, list[int]] = {}
    for qubit in sorted({key[0] for key in result.data_nodes}):
        layer_nodes = [node for (data_qubit, _layer), node in sorted(result.data_nodes.items()) if data_qubit == qubit]
        measured_nodes_by_qubit[qubit] = [node for node in layer_nodes if node in result.graph.meas_bases]
        for current_node, next_node in pairwise(layer_nodes):
            if current_node in result.graph.meas_bases:
                xflow[current_node] = {next_node}
    for ancilla_node in result.ancilla_nodes.values():
        correction_nodes = {ancilla_node}
        for measured_nodes in measured_nodes_by_qubit.values():
            for earlier_node, later_node in pairwise(measured_nodes):
                # Type I Y support touches both data-measurement layers. Including
                # the later data stabilizer cancels the backward dependency in
                # the automatically derived odd-neighborhood zflow.
                if result.graph.has_edge(ancilla_node, earlier_node) and result.graph.has_edge(
                    ancilla_node, later_node
                ):
                    correction_nodes.add(later_node)
        xflow[ancilla_node] = correction_nodes
    return xflow


def _compose_fragments(fragments: Sequence[_Fragment]) -> _Fragment:
    current = fragments[0]
    for fragment in fragments[1:]:
        graph, node_map1, node_map2 = compose(current.graph, fragment.graph)
        current = _Fragment(
            graph=graph,
            xflow=_remap_flow(current.xflow, node_map1) | _remap_flow(fragment.xflow, node_map2),
            record_nodes=_remap_record_nodes(current.record_nodes, node_map1)
            | _remap_record_nodes(fragment.record_nodes, node_map2),
            mpp_extractions=(*current.mpp_extractions, *fragment.mpp_extractions),
        )
    return current


def _measurement_annotations_from_analysis(
    analysis: _CircuitAnalysis,
    *,
    record_nodes: Mapping[int, int],
    zero_record_indices: frozenset[int],
) -> tuple[list[set[int]], dict[int, set[int]]]:
    if not record_nodes and not zero_record_indices:
        return [], {}

    parity_check_groups = [
        _record_indices_to_nodes(record_indices, record_nodes, zero_record_indices=zero_record_indices)
        for record_indices in analysis.detector_record_indices
    ]
    logical_observables = {
        logical_idx: _record_indices_to_nodes(
            record_indices,
            record_nodes,
            zero_record_indices=zero_record_indices,
        )
        for logical_idx, record_indices in analysis.logical_observable_record_indices.items()
    }
    return parity_check_groups, logical_observables


def _annotation_record_indices(
    instruction: stim.CircuitInstruction,
    *,
    measurement_count: int,
) -> frozenset[int]:
    """Resolve an annotation's relative record targets to absolute indices.

    Returns
    -------
    `frozenset`[`int`]
        Absolute measurement-record indices after parity cancellation.

    Raises
    ------
    ValueError
        If the annotation contains a non-record target.
    """
    record_indices: set[int] = set()
    for target in instruction.targets_copy():
        if not target.is_measurement_record_target:
            msg = f"{instruction.name} contains unsupported target {target!r}; only rec targets are supported."
            raise ValueError(msg)
        record_index = measurement_count + int(target.value)
        if not 0 <= record_index < measurement_count:
            msg = f"{instruction.name} refers to measurement record {record_index} before the beginning of time."
            raise ValueError(msg)
        if record_index in record_indices:
            record_indices.remove(record_index)
        else:
            record_indices.add(record_index)
    return frozenset(record_indices)


def _observable_index(instruction: stim.CircuitInstruction) -> int:
    """Return the logical-observable index from a Stim annotation.

    Returns
    -------
    `int`
        Logical-observable index.

    Raises
    ------
    ValueError
        If the instruction does not contain one integer argument.
    """
    args = instruction.gate_args_copy()
    if len(args) != 1 or not args[0].is_integer():
        msg = "OBSERVABLE_INCLUDE must have one integer observable index."
        raise ValueError(msg)
    return int(args[0])


def _record_indices_to_nodes(
    record_indices: frozenset[int],
    record_nodes: Mapping[int, int],
    *,
    zero_record_indices: frozenset[int],
) -> set[int]:
    missing_records = sorted(
        record_index
        for record_index in record_indices
        if record_index not in record_nodes and record_index not in zero_record_indices
    )
    if missing_records:
        msg = f"Cannot map Stim measurement record(s) to imported Pauli-measurement nodes: {missing_records}."
        raise ValueError(msg)
    return {record_nodes[record_index] for record_index in record_indices if record_index in record_nodes}


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
        if instruction.name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD", "SHIFT_COORDS"}:
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
