"""Build stabilizer-code inputs from Stim MPP layers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeGuard, cast

from scipy.sparse import lil_array

from graphqomb.qec.qeccode import StabilizerCode

if TYPE_CHECKING:
    from collections.abc import Sequence

    import stim


PauliSupport = tuple[tuple[int, str], ...]


class _StimModule(Protocol):
    Circuit: type[stim.Circuit]
    CircuitInstruction: type[stim.CircuitInstruction]
    CircuitRepeatBlock: type[stim.CircuitRepeatBlock]


@dataclass(frozen=True)
class StimMppExtraction:
    """Stabilizer-code data extracted from one Stim MPP layer.

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
    """

    code: StabilizerCode
    stim_to_column: dict[int, int]
    column_to_stim: dict[int, int]
    supports: tuple[PauliSupport, ...]


def stabilizer_code_from_stim_file(
    path: str | Path,
    *,
    mpp_layer: int = 0,
    coord_dims: int = 2,
) -> StimMppExtraction:
    """Build a stabilizer code from an MPP layer in a Stim file.

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
    mpp_layer: int = 0,
    coord_dims: int = 2,
) -> StimMppExtraction:
    """Build a stabilizer code from an MPP layer in Stim text.

    The selected MPP layer is interpreted as a parity-check matrix using the
    ``[Hx | Hz]`` convention. ``X`` and ``Y`` targets set entries in ``Hx``;
    ``Z`` and ``Y`` targets set entries in ``Hz``.

    Returns
    -------
    `StimMppExtraction`
        Extracted stabilizer code, qubit mapping, and Pauli supports.

    Raises
    ------
    ValueError
        If the requested MPP layer or coordinate format is invalid.
    """
    if mpp_layer < 0:
        msg = "mpp_layer must be non-negative."
        raise ValueError(msg)
    if coord_dims not in {2, 3}:
        msg = "coord_dims must be 2 or 3."
        raise ValueError(msg)

    stim_module = _load_stim()
    circuit = stim_module.Circuit(text).flattened()
    coordinate_by_stim_id = _extract_qubit_coordinates(circuit, coord_dims=coord_dims, stim_module=stim_module)
    layers = _extract_mpp_layers(circuit, stim_module=stim_module)
    if mpp_layer >= len(layers):
        msg = f"Stim circuit has {len(layers)} MPP layer(s); cannot select layer {mpp_layer}."
        raise ValueError(msg)

    supports = tuple(layers[mpp_layer])
    if not supports:
        msg = f"MPP layer {mpp_layer} is empty."
        raise ValueError(msg)

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

    return StimMppExtraction(
        code=StabilizerCode(matrix.tocsr(), qubit_coords=qubit_coords),
        stim_to_column=stim_to_column,
        column_to_stim=column_to_stim,
        supports=supports,
    )


def _load_stim() -> _StimModule:
    try:
        module = importlib.import_module("stim")
    except ImportError as exc:
        msg = "Stim MPP parsing requires the optional 'stim' dependency. Install stim or graphqomb[stim]."
        raise ImportError(msg) from exc
    return cast("_StimModule", module)


def _is_circuit_instruction(
    instruction: stim.CircuitInstruction | stim.CircuitRepeatBlock,
    stim_module: _StimModule,
) -> TypeGuard[stim.CircuitInstruction]:
    return isinstance(instruction, stim_module.CircuitInstruction)


def _extract_qubit_coordinates(
    circuit: stim.Circuit,
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
            coordinates[int(target.value)] = coord
    return coordinates


def _extract_mpp_layers(circuit: stim.Circuit, *, stim_module: _StimModule) -> list[list[PauliSupport]]:
    layers: list[list[PauliSupport]] = []
    current_layer: list[PauliSupport] | None = None

    for instruction in circuit:
        if not _is_circuit_instruction(instruction, stim_module):
            msg = "Flattened Stim circuit unexpectedly contains a repeat block."
            raise TypeError(msg)
        if instruction.name == "MPP":
            if current_layer is None:
                current_layer = []
            current_layer.extend(_mpp_targets_to_products(instruction.targets_copy()))
        elif current_layer is not None:
            layers.append(current_layer)
            current_layer = None

    if current_layer is not None:
        layers.append(current_layer)
    return layers


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
