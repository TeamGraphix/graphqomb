"""Build stabilizer-code inputs from Stim MPP layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import stim
from scipy.sparse import csr_array, lil_array

from graphqomb.qec.qeccode import StabilizerCode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


PauliSupport = tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class _MppProductRecord:
    record_index: int
    support: PauliSupport


def _empty_logical_observable_record_indices() -> dict[int, frozenset[int]]:
    return {}


@dataclass(frozen=True)
class StimMppExtraction:
    """Stabilizer-code data extracted from Stim MPP products.

    Attributes
    ----------
    code : `StabilizerCode`
        Dense-column stabilizer code using the ``[Hx | Hz]`` convention.
    stim_to_column : `dict`[`int`, `int`]
        Mapping from original Stim qubit ids to dense matrix columns.
    column_to_stim : `dict`[`int`, `int`]
        Inverse dense-column mapping.
    supports : `tuple`[`PauliSupport`, ...]
        Original Stim Pauli supports, one support per stabilizer row.
    detector_rows : `tuple`[`frozenset`[`int`], ...]
        Detector groups as selected-MPP stabilizer row indices. If a Stim
        detector also references measurements outside the selected MPP products,
        only rows represented in this extraction are included here.
    logical_observable_rows : `dict`[`int`, `frozenset`[`int`]]
        Logical observables as selected-MPP stabilizer row indices, keyed by
        Stim logical observable index. External measurement records are ignored
        in this row view.
    detector_record_indices : `tuple`[`frozenset`[`int`], ...]
        Absolute Stim measurement-record indices for detectors that touch at
        least one MPP product represented in this extraction.
    logical_observable_record_indices : `dict`[`int`, `frozenset`[`int`]]
        Absolute Stim measurement-record indices for logical observables that
        touch at least one MPP product represented in this extraction.
    """

    code: StabilizerCode
    stim_to_column: dict[int, int]
    column_to_stim: dict[int, int]
    supports: tuple[PauliSupport, ...]
    detector_rows: tuple[frozenset[int], ...]
    logical_observable_rows: dict[int, frozenset[int]]
    detector_record_indices: tuple[frozenset[int], ...] = ()
    logical_observable_record_indices: dict[int, frozenset[int]] = field(
        default_factory=_empty_logical_observable_record_indices
    )

    def detector_groups(self, ancilla_nodes: Mapping[int, int]) -> list[set[int]]:
        """Return detector groups mapped to graph node ids for ``qompile``.

        Parameters
        ----------
        ancilla_nodes : `collections.abc.Mapping`[`int`, `int`]
            Mapping from selected-MPP stabilizer rows to graph node ids.

        Returns
        -------
        `list`[`set`[`int`]]
            Detector groups suitable for ``qompile(..., parity_check_group=...)``.
        """
        return [_map_rows_to_nodes(rows, ancilla_nodes, "detector") for rows in self.detector_rows]

    def logical_observables(self, ancilla_nodes: Mapping[int, int]) -> dict[int, set[int]]:
        """Return logical observables mapped to graph node ids for ``qompile``.

        Parameters
        ----------
        ancilla_nodes : `collections.abc.Mapping`[`int`, `int`]
            Mapping from selected-MPP stabilizer rows to graph node ids.

        Returns
        -------
        `dict`[`int`, `set`[`int`]]
            Logical observables suitable for ``qompile(..., logical_observables=...)``.
        """
        return {
            logical_idx: _map_rows_to_nodes(rows, ancilla_nodes, f"logical observable {logical_idx}")
            for logical_idx, rows in self.logical_observable_rows.items()
        }


@dataclass(frozen=True)
class _StimMppAnnotations:
    detector_rows: tuple[frozenset[int], ...]
    logical_observable_rows: dict[int, frozenset[int]]
    detector_record_indices: tuple[frozenset[int], ...]
    logical_observable_record_indices: dict[int, frozenset[int]]


def stabilizer_code_from_stim_file(
    path: str | Path,
    *,
    mpp_layer: int | None = 0,
    coord_dims: int = 2,
) -> StimMppExtraction:
    """Build a stabilizer code from MPP products in a Stim file.

    Returns
    -------
    `StimMppExtraction`
        Extracted stabilizer code, qubit mapping, and Pauli supports.
    """
    return stabilizer_code_from_stim_text(
        Path(path).read_text(encoding="utf-8"),
        mpp_layer=mpp_layer,
        coord_dims=coord_dims,
    )


def stabilizer_code_from_stim_text(
    text: str,
    *,
    mpp_layer: int | None = 0,
    coord_dims: int = 2,
) -> StimMppExtraction:
    """Build a stabilizer code from MPP products in Stim text.

    The selected MPP products are interpreted as a parity-check matrix using
    the ``[Hx | Hz]`` convention. ``X`` and ``Y`` targets set entries in
    ``Hx``; ``Z`` and ``Y`` targets set entries in ``Hz``. By default,
    ``mpp_layer=0`` selects the first contiguous MPP layer. Pass
    ``mpp_layer=None`` to select all MPP products in the flattened Stim file.

    Returns
    -------
    `StimMppExtraction`
        Extracted stabilizer code, qubit mapping, and Pauli supports.

    Raises
    ------
    ValueError
        If the requested MPP layer or coordinate format is invalid.
    """
    if mpp_layer is not None and mpp_layer < 0:
        msg = "mpp_layer must be non-negative."
        raise ValueError(msg)
    if coord_dims not in {2, 3}:
        msg = "coord_dims must be 2 or 3."
        raise ValueError(msg)

    circuit = stim.Circuit(text).flattened()
    coordinate_by_stim_id = _extract_qubit_coordinates(circuit, coord_dims=coord_dims)
    layers = _extract_mpp_layers(circuit)

    selected_layer = _select_mpp_products(layers, mpp_layer=mpp_layer)
    supports = tuple(product.support for product in selected_layer)
    if not supports:
        layer_label = "file" if mpp_layer is None else f"layer {mpp_layer}"
        msg = f"MPP {layer_label} is empty."
        raise ValueError(msg)
    record_to_row = {product.record_index: row for row, product in enumerate(selected_layer)}
    annotations = _extract_selected_mpp_annotations(
        circuit,
        record_to_row=record_to_row,
    )

    matrix, stim_to_column, column_to_stim, qubit_coords = _build_stabilizer_data(
        supports,
        coordinate_by_stim_id,
    )

    return StimMppExtraction(
        code=StabilizerCode(matrix, qubit_coords=qubit_coords),
        stim_to_column=stim_to_column,
        column_to_stim=column_to_stim,
        supports=supports,
        detector_rows=annotations.detector_rows,
        logical_observable_rows=annotations.logical_observable_rows,
        detector_record_indices=annotations.detector_record_indices,
        logical_observable_record_indices=annotations.logical_observable_record_indices,
    )


def _build_stabilizer_data(
    supports: Sequence[PauliSupport],
    coordinate_by_stim_id: Mapping[int, tuple[float, ...]],
) -> tuple[csr_array, dict[int, int], dict[int, int], dict[int, tuple[float, ...]]]:
    stim_ids = sorted({qid for support in supports for qid, _pauli in support})
    stim_to_column = {qid: column for column, qid in enumerate(stim_ids)}
    column_to_stim = {column: qid for qid, column in stim_to_column.items()}

    num_qubits = len(stim_ids)
    matrix = lil_array((len(supports), 2 * num_qubits), dtype=bool)
    for row, support in enumerate(supports):
        for stim_id, pauli in support:
            column = stim_to_column[stim_id]
            if pauli in {"X", "Y"}:
                matrix[row, column] = True
            if pauli in {"Z", "Y"}:
                matrix[row, num_qubits + column] = True

    qubit_coords = {stim_to_column[qid]: coord for qid, coord in coordinate_by_stim_id.items() if qid in stim_to_column}
    return matrix.tocsr(), stim_to_column, column_to_stim, qubit_coords


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
            coordinates[int(target.value)] = coord
    return coordinates


def _extract_mpp_layers(circuit: stim.Circuit) -> list[list[_MppProductRecord]]:
    layers: list[list[_MppProductRecord]] = []
    current_layer: list[_MppProductRecord] | None = None
    measurement_count = 0

    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)
        if instruction.name == "MPP":
            if current_layer is None:
                current_layer = []
            products = _mpp_targets_to_products(instruction.targets_copy())
            if len(products) != instruction.num_measurements:
                msg = "Stim MPP instruction measurement count does not match its parsed product count."
                raise ValueError(msg)
            current_layer.extend(
                _MppProductRecord(record_index=measurement_count + offset, support=support)
                for offset, support in enumerate(products)
            )
        elif current_layer is not None:
            layers.append(current_layer)
            current_layer = None
        measurement_count += instruction.num_measurements

    if current_layer is not None:
        layers.append(current_layer)
    return layers


def _select_mpp_products(
    layers: Sequence[Sequence[_MppProductRecord]],
    *,
    mpp_layer: int | None,
) -> list[_MppProductRecord]:
    if mpp_layer is None:
        return [product for layer in layers for product in layer]
    if mpp_layer >= len(layers):
        msg = f"Stim circuit has {len(layers)} MPP layer(s); cannot select layer {mpp_layer}."
        raise ValueError(msg)
    return list(layers[mpp_layer])


def _extract_selected_mpp_annotations(
    circuit: stim.Circuit,
    *,
    record_to_row: Mapping[int, int],
) -> _StimMppAnnotations:
    detector_rows: list[frozenset[int]] = []
    detector_record_indices: list[frozenset[int]] = []
    logical_observable_rows: dict[int, set[int]] = {}
    logical_observable_record_indices: dict[int, set[int]] = {}
    measurement_count = 0

    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)

        if instruction.name == "DETECTOR":
            rows, record_indices = _record_targets_to_selected_mpp_rows(
                instruction.targets_copy(),
                measurement_count=measurement_count,
                record_to_row=record_to_row,
                instruction_name=instruction.name,
            )
            if rows is not None:
                detector_rows.append(frozenset(rows))
                detector_record_indices.append(record_indices)
        elif instruction.name == "OBSERVABLE_INCLUDE":
            logical_idx = _observable_index(instruction)
            rows, record_indices = _record_targets_to_selected_mpp_rows(
                instruction.targets_copy(),
                measurement_count=measurement_count,
                record_to_row=record_to_row,
                instruction_name=f"OBSERVABLE_INCLUDE({logical_idx})",
            )
            if rows is not None:
                logical_observable_rows.setdefault(logical_idx, set()).symmetric_difference_update(rows)
                logical_observable_record_indices.setdefault(logical_idx, set()).symmetric_difference_update(
                    record_indices
                )

        measurement_count += instruction.num_measurements

    return _StimMppAnnotations(
        detector_rows=tuple(detector_rows),
        logical_observable_rows={
            logical_idx: frozenset(rows) for logical_idx, rows in sorted(logical_observable_rows.items())
        },
        detector_record_indices=tuple(detector_record_indices),
        logical_observable_record_indices={
            logical_idx: frozenset(records)
            for logical_idx, records in sorted(logical_observable_record_indices.items())
        },
    )


def _record_targets_to_selected_mpp_rows(
    targets: Sequence[stim.GateTarget],
    *,
    measurement_count: int,
    record_to_row: Mapping[int, int],
    instruction_name: str,
) -> tuple[set[int] | None, frozenset[int]]:
    rows: set[int] = set()
    record_indices: set[int] = set()
    saw_selected_record = False

    for target in targets:
        if not target.is_measurement_record_target:
            msg = f"{instruction_name} contains unsupported target {target!r}; only rec targets are supported."
            raise ValueError(msg)
        record_index = measurement_count + int(target.value)
        if record_index in record_indices:
            record_indices.remove(record_index)
        else:
            record_indices.add(record_index)
        row = record_to_row.get(record_index)
        if row is None:
            continue
        saw_selected_record = True
        if row in rows:
            rows.remove(row)
        else:
            rows.add(row)

    if not saw_selected_record:
        return None, frozenset(record_indices)
    return rows, frozenset(record_indices)


def _observable_index(instruction: stim.CircuitInstruction) -> int:
    args = instruction.gate_args_copy()
    if len(args) != 1 or not args[0].is_integer():
        msg = "OBSERVABLE_INCLUDE must have one integer observable index."
        raise ValueError(msg)
    return int(args[0])


def _mpp_targets_to_products(targets: Sequence[stim.GateTarget]) -> list[PauliSupport]:
    products: list[PauliSupport] = []
    current: list[tuple[int, str]] = []
    seen_in_current: set[int] = set()
    expect_pauli = True

    for target in targets:
        if target.is_combiner:
            if expect_pauli:
                msg = "Invalid MPP target list: unexpected combiner."
                raise ValueError(msg)
            expect_pauli = True
            continue

        pauli = _target_pauli(target)
        if current and not expect_pauli:
            products.append(tuple(current))
            current = []
            seen_in_current = set()

        qid = int(target.value)
        if qid in seen_in_current:
            msg = f"Invalid MPP product: qubit {qid} appears more than once."
            raise ValueError(msg)
        current.append((qid, pauli))
        seen_in_current.add(qid)
        expect_pauli = False

    if expect_pauli:
        msg = "Invalid MPP target list: trailing combiner or empty product."
        raise ValueError(msg)
    products.append(tuple(current))
    return products


def _target_pauli(target: stim.GateTarget) -> str:
    if target.is_x_target:
        return "X"
    if target.is_y_target:
        return "Y"
    if target.is_z_target:
        return "Z"
    msg = f"Unsupported MPP target: {target!r}."
    raise ValueError(msg)


def _map_rows_to_nodes(rows: frozenset[int], ancilla_nodes: Mapping[int, int], label: str) -> set[int]:
    missing_rows = sorted(row for row in rows if row not in ancilla_nodes)
    if missing_rows:
        msg = f"Cannot map {label}; ancilla node map is missing stabilizer row(s): {missing_rows}."
        raise ValueError(msg)
    return {ancilla_nodes[row] for row in rows}
