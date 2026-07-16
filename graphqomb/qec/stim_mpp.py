"""Build stabilizer-code inputs from Stim MPP layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import stim

from graphqomb.qec._stim import (
    PauliSupport,
    StimMppExtraction,
    extract_qubit_coordinates,
    mpp_targets_to_products,
    observable_index,
    record_targets_to_absolute_indices,
    stim_mpp_extraction_from_records,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class _MppProductRecord:
    record_index: int
    support: PauliSupport


@dataclass(frozen=True)
class _StimRecordAnnotations:
    detector_record_indices: tuple[frozenset[int], ...]
    logical_observable_record_indices: dict[int, frozenset[int]]


def stabilizer_code_from_stim_file(
    path: str | Path,
    *,
    mpp_layer: int | None = 0,
    coord_dims: int = 2,
) -> StimMppExtraction:
    """Build a stabilizer code from MPP products in a Stim file.

    Signed MPP products are not supported. A Stim target inversion such as
    ``!X0`` raises ``ValueError`` because stabilizer signs are not retained.

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
    Signed MPP products are not supported because ``StabilizerCode`` does not
    retain stabilizer signs. A Stim target inversion such as ``!X0`` is
    therefore rejected instead of being silently discarded.

    Returns
    -------
    `StimMppExtraction`
        Extracted stabilizer code, qubit mapping, and Pauli supports.

    Raises
    ------
    ValueError
        If the requested MPP layer or coordinate format is invalid, or if an
        MPP product is signed using an inverted Pauli target.
    """
    if mpp_layer is not None and mpp_layer < 0:
        msg = "mpp_layer must be non-negative."
        raise ValueError(msg)
    circuit = stim.Circuit(text).flattened()
    coordinate_by_stim_id = extract_qubit_coordinates(circuit, coord_dims=coord_dims)
    layers = _extract_mpp_layers(circuit)

    selected_layer = _select_mpp_products(layers, mpp_layer=mpp_layer)
    supports = tuple(product.support for product in selected_layer)
    if not supports:
        layer_label = "file" if mpp_layer is None else f"layer {mpp_layer}"
        msg = f"MPP {layer_label} is empty."
        raise ValueError(msg)
    annotations = _extract_stim_record_annotations(circuit)
    return stim_mpp_extraction_from_records(
        supports,
        tuple(product.record_index for product in selected_layer),
        coordinate_by_stim_id=coordinate_by_stim_id,
        detector_record_indices=annotations.detector_record_indices,
        logical_observable_record_indices=annotations.logical_observable_record_indices,
    )


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
            products = mpp_targets_to_products(instruction.targets_copy())
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


def _extract_stim_record_annotations(circuit: stim.Circuit) -> _StimRecordAnnotations:
    detector_record_indices: list[frozenset[int]] = []
    logical_observable_record_indices: dict[int, set[int]] = {}
    measurement_count = 0

    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)

        if instruction.name == "DETECTOR":
            record_indices = record_targets_to_absolute_indices(
                instruction.targets_copy(),
                measurement_count=measurement_count,
                instruction_name=instruction.name,
            )
            detector_record_indices.append(record_indices)
        elif instruction.name == "OBSERVABLE_INCLUDE":
            logical_idx = observable_index(instruction)
            record_indices = record_targets_to_absolute_indices(
                instruction.targets_copy(),
                measurement_count=measurement_count,
                instruction_name=f"OBSERVABLE_INCLUDE({logical_idx})",
            )
            logical_observable_record_indices.setdefault(logical_idx, set()).symmetric_difference_update(record_indices)

        measurement_count += instruction.num_measurements

    return _StimRecordAnnotations(
        detector_record_indices=tuple(detector_record_indices),
        logical_observable_record_indices={
            logical_idx: frozenset(records)
            for logical_idx, records in sorted(logical_observable_record_indices.items())
        },
    )
