"""Import supported Stim circuits into GraphQOMB patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import stim

from graphqomb.circuit import Circuit, CircuitScheduleStrategy, circuit2graph
from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.gates import CNOT, CZ, SWAP, Gate, H, Rz, S, X, Y, Z
from graphqomb.graphstate import GraphState, compose, odd_neighbors
from graphqomb.qec._stim import (
    PauliSupport,
    StimMppExtraction,
    extract_qubit_coordinates,
    mpp_targets_to_products,
    observable_index,
    pauli_products_commute,
    plain_qubit_target,
    record_targets_to_absolute_indices,
    stim_mpp_extraction_from_records,
)
from graphqomb.qec.qeccode import StabilizerGraphStateBuildResult, YFoliation, build_graph_state
from graphqomb.qompiler import qompile

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from graphqomb.pattern import Pattern


_UNITARY_GATES = frozenset({"H", "S", "SQRT_Z", "S_DAG", "SQRT_Z_DAG", "X", "Y", "Z", "CX", "CNOT", "CZ", "SWAP"})
_RESET_AXES = {"R": Axis.Z, "RX": Axis.X, "RY": Axis.Y}
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
    input_initialization_axes: Mapping[int, Axis]
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

    The importer supports initial Pauli resets, Clifford unitary blocks, and
    Pauli measurement blocks. Stim ``R``, ``RX``, and ``RY`` instructions are
    imported as positive Z-, X-, and Y-eigenstate input initialization,
    respectively, when they occur before any other quantum operation on the
    target qubit. Stim noise instructions and measurement-error probabilities
    are omitted because circuit-level noise is outside the GraphQOMB import
    model. Pauli measurement blocks must be separated from unitary blocks by
    TICK. A direct single-qubit measurement terminates that qubit's lifetime;
    other qubits may continue, but the measured qubit cannot be used by a later
    operation.

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
    coordinate_by_stim_id = extract_qubit_coordinates(idealized.circuit, coord_dims=coord_dims)
    stim_to_qubit = _stim_to_qubit_map(idealized.circuit)
    qubit_to_stim = {qubit: stim_id for stim_id, qubit in stim_to_qubit.items()}
    input_initialization_axes = _input_initialization_axes(analysis.blocks)
    context = _ImportContext(
        stim_to_qubit=stim_to_qubit,
        coordinate_by_stim_id=coordinate_by_stim_id,
        input_initialization_axes=input_initialization_axes,
        detector_record_indices=analysis.detector_record_indices,
        logical_observable_record_indices=analysis.logical_observable_record_indices,
        schedule_strategy=schedule_strategy,
        y_foliation=y_foliation,
    )

    fragments = _fragments_from_blocks(analysis.blocks, context=context)
    fragment = _compose_fragments(fragments)
    fragment = _apply_single_measurements(fragment, analysis.blocks, context=context)
    parity_check_groups, logical_observables = _measurement_annotations_from_analysis(
        analysis,
        record_nodes=fragment.record_nodes,
        zero_record_indices=idealized.zero_record_indices,
    )
    pattern = qompile(
        fragment.graph,
        fragment.xflow,
        {node: odd_neighbors(targets, fragment.graph) for node, targets in fragment.xflow.items()},
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
        if instruction.name == "DETECTOR":
            detector_record_indices.append(
                record_targets_to_absolute_indices(
                    instruction.targets_copy(),
                    measurement_count=measurement_count,
                    instruction_name=instruction.name,
                )
            )
        elif instruction.name == "OBSERVABLE_INCLUDE":
            logical_idx = observable_index(instruction)
            logical_record_indices.setdefault(logical_idx, set()).symmetric_difference_update(
                record_targets_to_absolute_indices(
                    instruction.targets_copy(),
                    measurement_count=measurement_count,
                    instruction_name=f"OBSERVABLE_INCLUDE({logical_idx})",
                )
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
                    plain_qubit_target(target, instruction.name),
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

    fragments = [_identity_fragment(context)]
    z_base = 0
    live_stim_ids = set(context.stim_to_qubit)
    for block in blocks:
        directly_measured_stim_ids = {
            plain_qubit_target(target, analyzed.instruction.name)
            for analyzed in block
            if analyzed.instruction.name in _SINGLE_PAULI_MEASUREMENT_AXES
            for target in analyzed.instruction.targets_copy()
        }
        unitary_instructions = tuple(
            analyzed.instruction for analyzed in block if analyzed.instruction.name in _UNITARY_GATES
        )
        if unitary_instructions:
            fragment, z_base = _unitary_fragment(
                unitary_instructions,
                live_stim_ids=live_stim_ids,
                z_base=z_base,
                context=context,
            )
            fragments.append(fragment)
        else:
            mpp_stim_ids = {
                stim_id
                for analyzed in block
                if analyzed.instruction.name == "MPP"
                for support in mpp_targets_to_products(analyzed.instruction.targets_copy())
                for stim_id, _axis in support
            }
            mpp_fragment = _mpp_fragment_from_block(
                block,
                z_base=z_base,
                io_stim_ids=(live_stim_ids - directly_measured_stim_ids) | mpp_stim_ids,
                context=context,
            )
            if mpp_fragment is not None:
                fragments.append(mpp_fragment)
                z_base += 2
            elif directly_measured_stim_ids:
                continuing_stim_ids = live_stim_ids - directly_measured_stim_ids
                if continuing_stim_ids:
                    z_base += 1
                    fragments.append(_relocation_fragment(continuing_stim_ids, z=z_base, context=context))

        live_stim_ids.difference_update(directly_measured_stim_ids)

    return fragments


def _input_initialization_axes(
    blocks: Sequence[Sequence[_AnalyzedInstruction]],
) -> dict[int, Axis]:
    """Collect leading Stim resets as input initialization axes.

    Repeated resets before a qubit's first non-reset quantum operation are
    allowed; the final reset determines the imported initialization axis.

    Returns
    -------
    `dict`[`int`, `Axis`]
        Map from Stim qubit ids to positive Pauli initialization axes.

    Raises
    ------
    ValueError
        If a reset occurs after another quantum operation on the same qubit.
    """
    initialization_axes: dict[int, Axis] = {}
    used_qubits: set[int] = set()
    quantum_operation_names = {
        *_UNITARY_GATES,
        *_SINGLE_PAULI_MEASUREMENT_AXES,
        "MPP",
    }

    for block in blocks:
        for analyzed in block:
            instruction = analyzed.instruction
            instruction_qubits = {
                int(target.qubit_value) for target in instruction.targets_copy() if target.qubit_value is not None
            }
            reset_axis = _RESET_AXES.get(instruction.name)
            if reset_axis is not None:
                reset_after_use = used_qubits & instruction_qubits
                if reset_after_use:
                    msg = (
                        f"Stim reset instruction {instruction.name} targets qubit(s) {sorted(reset_after_use)} "
                        "after another quantum operation; only initial resets are supported."
                    )
                    raise ValueError(msg)
                for stim_id in instruction_qubits:
                    initialization_axes[stim_id] = reset_axis
            elif instruction.name in quantum_operation_names:
                used_qubits.update(instruction_qubits)

    return initialization_axes


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
                and analyzed.instruction.name not in _RESET_AXES
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
        If a quantum operation reuses a directly measured qubit.
    """
    measured_qubits: set[int] = set()
    for block in blocks:
        for analyzed in block:
            instruction = analyzed.instruction
            if (
                instruction.name not in _UNITARY_GATES
                and instruction.name != "MPP"
                and instruction.name not in _SINGLE_PAULI_MEASUREMENT_AXES
            ):
                continue

            instruction_qubits = {
                int(target.qubit_value) for target in instruction.targets_copy() if target.qubit_value is not None
            }
            reused_qubits = measured_qubits & instruction_qubits
            if reused_qubits:
                msg = (
                    f"Stim qubit(s) {sorted(reused_qubits)} are used after a single-qubit measurement; "
                    "single-qubit measurements terminate those qubit lifetimes."
                )
                raise ValueError(msg)
            if instruction.name in _SINGLE_PAULI_MEASUREMENT_AXES:
                measured_qubits.update(instruction_qubits)


def _mpp_fragment_from_block(
    block: Sequence[_AnalyzedInstruction],
    *,
    z_base: int,
    io_stim_ids: set[int],
    context: _ImportContext,
) -> _Fragment | None:
    """Build the combined MPP fragment in a measurement block.

    Returns
    -------
    `_Fragment` | `None`
        Combined MPP fragment, or `None` when the block has no MPP instruction.
    """
    mpp_items = tuple(analyzed for analyzed in block if analyzed.instruction.name == "MPP")
    if not mpp_items:
        return None
    return _mpp_fragment(
        mpp_items,
        z_base=z_base,
        io_stim_ids=io_stim_ids,
        context=context,
    )


def _identity_fragment(context: _ImportContext) -> _Fragment:
    graph = GraphState()
    for stim_id, qubit_index in sorted(context.stim_to_qubit.items()):
        coord = context.coordinate_by_stim_id.get(stim_id)
        node = graph.add_node(coordinate=_coordinate_at_z(coord, 0) if coord is not None else None)
        graph.register_input(
            node,
            qubit_index,
            init_axis=context.input_initialization_axes.get(stim_id, Axis.X),
        )
        graph.register_output(node, qubit_index)
    return _Fragment(
        graph=graph,
        xflow={},
        record_nodes={},
    )


def _unitary_fragment(
    block: Sequence[stim.CircuitInstruction],
    *,
    live_stim_ids: set[int],
    z_base: int,
    context: _ImportContext,
) -> tuple[_Fragment, int]:
    ordered_stim_ids = sorted(live_stim_ids)
    stim_to_local = {stim_id: local_index for local_index, stim_id in enumerate(ordered_stim_ids)}
    local_to_global = {local_index: context.stim_to_qubit[stim_id] for stim_id, local_index in stim_to_local.items()}
    circuit = Circuit(len(ordered_stim_ids))
    for instruction in block:
        _append_unitary_instruction(circuit, instruction, stim_to_local)

    local_graph, local_xflow, _ = circuit2graph(circuit, schedule_strategy=context.schedule_strategy)
    graph, node_map = _copy_graph_with_qindices(local_graph, local_to_global)
    xflow = _remap_flow(local_xflow, node_map)
    z_span = _apply_unitary_coordinates(
        graph,
        xflow,
        z_base=z_base,
        context=context,
    )
    return (
        _Fragment(
            graph=graph,
            xflow=xflow,
            record_nodes={},
        ),
        z_base + z_span,
    )


def _apply_unitary_coordinates(
    graph: GraphState,
    xflow: Mapping[int, set[int]],
    *,
    z_base: int,
    context: _ImportContext,
) -> int:
    """Place a transpiled gate block between common input and output Z layers.

    Qubit chains with gates are stretched across the block's maximum transpiled
    depth. An idle qubit remains a single input/output node and is relocated to
    the common output layer without adding graph nodes or edges.

    Returns
    -------
    `int`
        Z span occupied by the gate block.

    Raises
    ------
    ValueError
        If the transpiled unitary graph does not consist of data-wire chains.
    """
    output_node_by_qubit = {qubit: node for node, qubit in graph.output_node_indices.items()}
    chains: dict[int, list[int]] = {}
    chain_nodes: set[int] = set()
    for input_node, qubit in graph.input_node_indices.items():
        output_node = output_node_by_qubit[qubit]
        chain = [input_node]
        visited = {input_node}
        while chain[-1] != output_node:
            targets = xflow.get(chain[-1])
            if targets is None or len(targets) != 1:
                msg = f"Transpiled unitary wire for qubit {qubit} is not a single X-flow chain."
                raise ValueError(msg)
            next_node = next(iter(targets))
            if next_node in visited:
                msg = f"Transpiled unitary wire for qubit {qubit} contains an X-flow cycle."
                raise ValueError(msg)
            chain.append(next_node)
            visited.add(next_node)
        chains[qubit] = chain
        chain_nodes.update(chain)

    if chain_nodes != graph.nodes:
        msg = "Transpiled unitary graph contains a node outside its data-wire X-flow chains."
        raise ValueError(msg)

    max_chain_depth = max((len(chain) - 1 for chain in chains.values()), default=0)
    z_span = max(1, max_chain_depth)
    qubit_to_stim = {qubit: stim_id for stim_id, qubit in context.stim_to_qubit.items()}
    for qubit, chain in chains.items():
        coord = context.coordinate_by_stim_id.get(qubit_to_stim[qubit])
        if coord is None:
            continue
        depth = len(chain) - 1
        if depth == 0:
            graph.set_coordinate(chain[0], _coordinate_at_z(coord, z_base + z_span))
            continue
        for layer, node in enumerate(chain):
            z = z_base + z_span * layer / depth
            graph.set_coordinate(node, _coordinate_at_z(coord, z))
    return z_span


def _copy_graph_with_qindices(
    graph: GraphState,
    local_to_global: Mapping[int, int],
) -> tuple[GraphState, dict[int, int]]:
    """Copy a graph while replacing its input and output qindices.

    Returns
    -------
    tuple[GraphState, dict[int, int]]
        Copied graph and source-to-copy node map.
    """
    copied = GraphState()
    node_map = {node: copied.add_node() for node in sorted(graph.nodes)}
    for node1, node2 in graph.edges:
        copied.add_edge(node_map[node1], node_map[node2])
    for node, q_index in graph.input_node_indices.items():
        copied.register_input(node_map[node], local_to_global[q_index])
    for node, q_index in graph.output_node_indices.items():
        copied.register_output(node_map[node], local_to_global[q_index])
    for node, meas_basis in graph.meas_bases.items():
        copied.assign_meas_basis(node_map[node], meas_basis)
    for node, local_clifford in graph.local_cliffords.items():
        copied.apply_local_clifford(node_map[node], local_clifford)
    for node, coordinate in graph.coordinates.items():
        copied.set_coordinate(node_map[node], coordinate)
    return copied, node_map


def _append_unitary_instruction(
    circuit: Circuit,
    instruction: stim.CircuitInstruction,
    stim_to_qubit: Mapping[int, int],
) -> None:
    for group in instruction.target_groups():
        qubits = [plain_qubit_target(target, instruction.name) for target in group]
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
    z_base: int,
    io_stim_ids: set[int],
    context: _ImportContext,
) -> _Fragment:
    supports = tuple(
        support for analyzed in block for support in mpp_targets_to_products(analyzed.instruction.targets_copy())
    )
    _validate_commuting_mpp_supports(supports)
    record_indices = tuple(record_index for analyzed in block for record_index in analyzed.record_indices)
    extraction = stim_mpp_extraction_from_records(
        supports,
        record_indices,
        coordinate_by_stim_id=context.coordinate_by_stim_id,
        detector_record_indices=context.detector_record_indices,
        logical_observable_record_indices=context.logical_observable_record_indices,
    )
    fragment = _mpp_graph_fragment(
        extraction,
        record_indices=record_indices,
        z_base=z_base,
        io_stim_ids=io_stim_ids,
        context=context,
    )
    return _with_mpp_extraction(fragment, extraction)


def _validate_commuting_mpp_supports(supports: Sequence[PauliSupport]) -> None:
    for (left_index, left), (right_index, right) in combinations(enumerate(supports), 2):
        if not pauli_products_commute(left, right):
            msg = (
                f"MPP products within one TICK block must commute; products {left_index} and {right_index} anticommute."
            )
            raise ValueError(msg)


def _mpp_graph_fragment(
    extraction: StimMppExtraction,
    *,
    record_indices: Sequence[int],
    z_base: int,
    io_stim_ids: set[int],
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
    active_stim_ids = set(extraction.stim_to_column)
    _add_relocated_io_nodes(
        result.graph,
        io_stim_ids - active_stim_ids,
        z=z_base + 2,
        context=context,
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


def _relocation_fragment(stim_ids: set[int], *, z: int, context: _ImportContext) -> _Fragment:
    graph = GraphState()
    _add_relocated_io_nodes(graph, stim_ids, z=z, context=context)
    return _Fragment(graph=graph, xflow={}, record_nodes={})


def _add_relocated_io_nodes(
    graph: GraphState,
    stim_ids: set[int],
    *,
    z: int,
    context: _ImportContext,
) -> None:
    for stim_id in sorted(stim_ids):
        coord = context.coordinate_by_stim_id.get(stim_id)
        node = graph.add_node(coordinate=_coordinate_at_z(coord, z) if coord is not None else None)
        qubit = context.stim_to_qubit[stim_id]
        graph.register_input(node, qubit)
        graph.register_output(node, qubit)


def _coordinate_at_z(coord: tuple[float, ...], z: float) -> tuple[float, float, float]:
    return (float(coord[0]), float(coord[1]), float(z))


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
    for qubit in sorted({key[0] for key in result.data_nodes}):
        layer_nodes = [node for (data_qubit, _layer), node in sorted(result.data_nodes.items()) if data_qubit == qubit]
        for current_node, next_node in pairwise(layer_nodes):
            if current_node in result.graph.meas_bases:
                xflow[current_node] = {next_node}
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


def _apply_single_measurements(
    fragment: _Fragment,
    blocks: Sequence[Sequence[_AnalyzedInstruction]],
    *,
    context: _ImportContext,
) -> _Fragment:
    """Assign direct measurements to existing data-lane output nodes.

    Returns
    -------
    `_Fragment`
        Fragment with terminal measurement bases and record mappings applied.

    Raises
    ------
    ValueError
        If target counts differ or a qubit is repeated in one instruction.
    """
    output_node_by_qubit = {qubit: node for node, qubit in fragment.graph.output_node_indices.items()}
    record_nodes = dict(fragment.record_nodes)

    for block in blocks:
        for analyzed in block:
            instruction = analyzed.instruction
            if instruction.name not in _SINGLE_PAULI_MEASUREMENT_AXES:
                continue
            targets = instruction.targets_copy()
            if len(targets) != len(analyzed.record_indices):
                msg = f"{instruction.name} target count does not match its measurement-record count."
                raise ValueError(msg)

            seen_qubits: set[int] = set()
            axis = _SINGLE_PAULI_MEASUREMENT_AXES[instruction.name]
            for target, record_index in zip(targets, analyzed.record_indices, strict=True):
                stim_id = plain_qubit_target(target, instruction.name)
                if stim_id in seen_qubits:
                    msg = f"{instruction.name} measures qubit {stim_id} more than once in one instruction."
                    raise ValueError(msg)
                seen_qubits.add(stim_id)

                node = output_node_by_qubit[context.stim_to_qubit[stim_id]]
                sign = Sign.MINUS if target.is_inverted_result_target else Sign.PLUS
                fragment.graph.assign_meas_basis(node, AxisMeasBasis(axis, sign))
                record_nodes[record_index] = node

    return _Fragment(
        graph=fragment.graph,
        xflow=fragment.xflow,
        record_nodes=record_nodes,
        mpp_extractions=fragment.mpp_extractions,
    )


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


def _stim_to_qubit_map(circuit: stim.Circuit) -> dict[int, int]:
    stim_ids: set[int] = set()
    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)
        if instruction.name in {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD"}:
            continue
        for target in instruction.targets_copy():
            qubit_value = target.qubit_value
            if qubit_value is not None:
                stim_ids.add(int(qubit_value))
    return {stim_id: qubit for qubit, stim_id in enumerate(sorted(stim_ids))}
