"""Core Clifford transpilation and basis-gate optimization logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cache
from typing import Literal

import stim

_BASIS_GATES = frozenset({"H", "S", "CZ"})
_PRESERVED_ANNOTATIONS = frozenset({"QUBIT_COORDS", "SHIFT_COORDS", "TICK"})
_BOUNDARY_OPERATIONS = frozenset({"R", "RX", "RY", "M", "MX", "MY"})
_SIGNED_PAULI_COUNT = 6
HS_STIM_GATE = "C_XNYZ"
_Basis = Literal["H_S_CZ", "H_HS_CZ"]


class UnsupportedInstructionError(ValueError):
    """Raised when a circuit instruction is not a unitary Clifford operation."""


def transpile(
    circuit: stim.Circuit | str,
    *,
    optimize: bool = False,
) -> stim.Circuit:
    """Transpile a Stim Clifford circuit into the H/HS/CZ gate basis.

    Parameters
    ----------
    circuit : ``stim.Circuit`` | `str`
        A Stim circuit or Stim circuit text. REPEAT blocks are handled
        recursively. TICK, QUBIT_COORDS, and SHIFT_COORDS annotations are
        preserved because they do not affect the quantum operation.
    optimize : `bool`, optional
        Remove redundant basis gates and simplify gates adjacent to R/M
        boundaries after transpilation.

    Returns
    -------
    ``stim.Circuit``
        A new circuit containing only H, logical HS, CZ, supported boundaries,
        and preserved annotations. Stim represents ``HS = J(pi/2)`` as
        ``C_XNYZ``.

    Raises
    ------
    TypeError
        If ``circuit`` is neither Stim text nor a Stim circuit.
    """
    if isinstance(circuit, str):
        circuit = stim.Circuit(circuit)
    elif not isinstance(circuit, stim.Circuit):
        msg = "circuit must be a stim.Circuit or Stim circuit text"
        raise TypeError(msg)

    h_s_cz = _transpile_block(circuit, context="circuit")
    if optimize:
        # Remove diagonal H/S/CZ redundancies before replacing each S by HS·H.
        # This catches identities such as S^4 without obscuring them behind the
        # composite-gate encoding.
        h_s_cz = _optimize_h_s_cz(h_s_cz)
    h_hs_cz = _convert_s_to_hs(h_s_cz)
    return optimize_h_hs_cz(h_hs_cz) if optimize else h_hs_cz


def _optimize_h_s_cz(circuit: stim.Circuit | str) -> stim.Circuit:
    """Remove redundant H, S, and CZ gates from an H/S/CZ circuit.

    The identities H^2 = I, S^4 = I, and CZ^2 = I are removed. Cancellation
    is also performed across intervening gates when all of them commute with
    the gate being cancelled. Single-qubit gates at R and M boundaries are
    folded into R/RX/RY preparations or M/MX/MY measurements when safe.
    Annotations are treated as optimization barriers.

    Returns
    -------
    ``stim.Circuit``
        The optimized circuit.
    """
    return _optimize_basis_circuit(
        _coerce_circuit(circuit),
        basis="H_S_CZ",
        context="circuit",
        terminal_measurements=True,
    )


def optimize_h_hs_cz(circuit: stim.Circuit | str) -> stim.Circuit:
    """Remove redundant H, logical HS, and CZ gates.

    Logical ``HS = J(pi/2)`` is represented by Stim's C_XNYZ instruction. The
    identities H^2 = I, (HS)^3 = I, and CZ^2 = I are removed, including across
    commuting operations on disjoint qubits. Single-qubit gates at R and M
    boundaries are folded into R/RX/RY preparations or M/MX/MY measurements
    when safe. Annotations are treated as optimization barriers.

    Returns
    -------
    ``stim.Circuit``
        The optimized circuit.
    """
    return _optimize_basis_circuit(
        _coerce_circuit(circuit),
        basis="H_HS_CZ",
        context="circuit",
        terminal_measurements=True,
    )


def _coerce_circuit(circuit: stim.Circuit | str) -> stim.Circuit:
    if isinstance(circuit, str):
        return stim.Circuit(circuit)
    if not isinstance(circuit, stim.Circuit):
        msg = "circuit must be a stim.Circuit or Stim circuit text"
        raise TypeError(msg)
    return circuit


def _convert_s_to_hs(circuit: stim.Circuit) -> stim.Circuit:
    result = stim.Circuit()
    for instruction_index in range(len(circuit)):
        instruction = circuit[instruction_index]
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(
                stim.CircuitRepeatBlock(
                    instruction.repeat_count,
                    _convert_s_to_hs(instruction.body_copy()),
                    tag=instruction.tag,
                )
            )
        elif instruction.name == "S":
            for target in instruction.targets_copy():
                qubit = _qubit_value(target)
                # Chronological HS (= H S in matrix order) followed by H
                # implements S, because H H S = S.
                result.append(HS_STIM_GATE, [qubit], tag=instruction.tag)
                result.append("H", [qubit], tag=instruction.tag)
        else:
            result.append(instruction)
    return result


@dataclass(frozen=True)
class _AtomicGate:
    name: str
    targets: tuple[int, ...]
    tag: str
    gate_args: tuple[float, ...] = ()
    inverted: bool = False

    @property
    def qubits(self) -> frozenset[int]:
        return frozenset(self.targets)

    @property
    def cancellation_key(self) -> tuple[str, tuple[int, ...]]:
        targets = tuple(sorted(self.targets)) if self.name == "CZ" else self.targets
        return self.name, targets

    def append_to(self, circuit: stim.Circuit) -> None:
        targets: tuple[int | stim.GateTarget, ...]
        if self.name in {"M", "MX", "MY"} and self.inverted:
            targets = (stim.target_inv(self.targets[0]),)
        else:
            targets = self.targets
        circuit.append(
            self.name,
            targets,
            self.gate_args or None,
            tag=self.tag,
        )


def _optimize_basis_circuit(
    circuit: stim.Circuit,
    *,
    basis: _Basis,
    context: str,
    terminal_measurements: bool,
) -> stim.Circuit:
    result = stim.Circuit()
    pending: list[_AtomicGate] = []
    basis_gates = (
        {"H", "S", "CZ"}
        if basis == "H_S_CZ"
        else {
            "H",
            HS_STIM_GATE,
            "CZ",
        }
    )
    allowed = basis_gates | _BOUNDARY_OPERATIONS

    def flush(*, allow_terminal_measurement_fold: bool) -> None:
        if not pending:
            return
        optimized = _simplify_boundaries(
            pending,
            basis=basis,
            allow_terminal_measurement_fold=allow_terminal_measurement_fold,
        )
        optimized = _cancel_redundant_gates(optimized, basis=basis)
        optimized = _simplify_boundaries(
            optimized,
            basis=basis,
            allow_terminal_measurement_fold=allow_terminal_measurement_fold,
        )
        for gate in optimized:
            gate.append_to(result)
        pending.clear()

    for instruction_index in range(len(circuit)):
        instruction = circuit[instruction_index]
        location = f"{context}, instruction {instruction_index}"
        if isinstance(instruction, stim.CircuitRepeatBlock):
            flush(allow_terminal_measurement_fold=False)
            body = _optimize_basis_circuit(
                instruction.body_copy(),
                basis=basis,
                context=f"{location}, REPEAT body",
                terminal_measurements=False,
            )
            result.append(
                stim.CircuitRepeatBlock(
                    instruction.repeat_count,
                    body,
                    tag=instruction.tag,
                )
            )
            continue

        if instruction.name in _PRESERVED_ANNOTATIONS:
            flush(allow_terminal_measurement_fold=False)
            result.append(instruction)
            continue
        if instruction.name not in allowed:
            rendered = ", ".join(sorted(allowed))
            msg = f"Instruction {instruction.name!r} at {location} is not in the {basis} basis ({rendered})."
            raise UnsupportedInstructionError(msg)

        _validate_plain_qubit_targets(instruction, location=location)
        for group in instruction.target_groups():
            targets = tuple(_qubit_value(target) for target in group)
            if len(targets) != (2 if instruction.name == "CZ" else 1):
                msg = f"Instruction {instruction.name!r} at {location} has unsupported target grouping."
                raise UnsupportedInstructionError(msg)
            target = group[0]
            pending.append(
                _AtomicGate(
                    instruction.name,
                    targets,
                    instruction.tag,
                    tuple(instruction.gate_args_copy()),
                    instruction.name in {"M", "MX", "MY"} and target.is_inverted_result_target,
                )
            )

    flush(allow_terminal_measurement_fold=terminal_measurements)
    return result


def _cancel_redundant_gates(
    gates: list[_AtomicGate],
    *,
    basis: _Basis,
) -> list[_AtomicGate]:
    result = list(gates)
    orders = {"H": 2, "CZ": 2}
    orders["S" if basis == "H_S_CZ" else HS_STIM_GATE] = 4 if basis == "H_S_CZ" else 3

    while True:
        removed = False
        for start, gate in enumerate(result):
            if gate.name not in orders:
                continue
            matching_positions = [start]
            for candidate_index in range(start + 1, len(result)):
                candidate = result[candidate_index]
                if candidate.cancellation_key == gate.cancellation_key:
                    matching_positions.append(candidate_index)
                    if len(matching_positions) == orders[gate.name]:
                        removed_positions = set(matching_positions)
                        result = [item for index, item in enumerate(result) if index not in removed_positions]
                        removed = True
                        break
                elif not _commutes_for_optimization(gate, candidate, basis=basis):
                    break
            if removed:
                break
        if not removed:
            return result


def _simplify_boundaries(  # ruff:ignore[complex-structure, too-many-branches, too-many-statements]
    gates: list[_AtomicGate],
    *,
    basis: _Basis,
    allow_terminal_measurement_fold: bool,
) -> list[_AtomicGate]:
    result = list(gates)
    local_gates = {"H", "S" if basis == "H_S_CZ" else HS_STIM_GATE}

    while True:
        changed = False

        # A reset discards prior single-qubit operations on the reset qubit.
        for reset_index, reset in enumerate(result):
            if reset.name not in {"R", "RX", "RY"}:
                continue
            qubit = reset.targets[0]
            removable: list[int] = []
            for index in range(reset_index - 1, -1, -1):
                candidate = result[index]
                if qubit not in candidate.qubits:
                    continue
                if candidate.name in local_gates and candidate.targets == (qubit,):
                    removable.append(index)
                    continue
                break
            if removable:
                result = _replace_gate_positions(result, removable, ())
                changed = True
                break
        if changed:
            continue

        # Absorb a positive-axis state preparation into Stim's reset basis.
        for reset_index, reset in enumerate(result):
            if reset.name != "R":
                continue
            positions = _local_word_after(
                result,
                reset_index,
                qubit=reset.targets[0],
                local_gates=local_gates,
            )
            if not positions:
                continue
            word = tuple(result[index].name for index in positions)
            state_key = _boundary_key(word, boundary="state")
            reset_name = {"+Z": "R", "+X": "RX", "+Y": "RY"}.get(state_key)
            if reset_name is None:
                continue
            replacement = _AtomicGate(
                reset_name,
                reset.targets,
                reset.tag,
                reset.gate_args,
            )
            result = _replace_reset_and_word(
                result,
                reset_index=reset_index,
                word_positions=positions,
                replacement=replacement,
            )
            changed = True
            break
        if changed:
            continue

        # Absorb a signed measurement basis into M/MX/MY and target inversion.
        for measurement_index, measurement in enumerate(result):
            if measurement.name != "M":
                continue
            if not _measurement_state_is_discarded(
                result,
                measurement_index=measurement_index,
                allow_end=allow_terminal_measurement_fold,
            ):
                continue
            positions = _local_word_before(
                result,
                measurement_index,
                qubit=measurement.targets[0],
                local_gates=local_gates,
            )
            if not positions:
                continue
            word = tuple(result[index].name for index in positions)
            measurement_key = _boundary_key(word, boundary="measurement")
            measurement_name = {
                "+Z": "M",
                "-Z": "M",
                "+X": "MX",
                "-X": "MX",
                "+Y": "MY",
                "-Y": "MY",
            }[measurement_key]
            replacement = _AtomicGate(
                measurement_name,
                measurement.targets,
                measurement.tag,
                measurement.gate_args,
                measurement.inverted ^ measurement_key.startswith("-"),
            )
            result = _replace_measurement_and_word(
                result,
                measurement_index=measurement_index,
                word_positions=positions,
                replacement=replacement,
            )
            changed = True
            break
        if changed:
            continue

        # After R, only the prepared stabilizer state matters.
        for reset_index, reset in enumerate(result):
            if reset.name != "R":
                continue
            positions = _local_word_after(
                result,
                reset_index,
                qubit=reset.targets[0],
                local_gates=local_gates,
            )
            if positions:
                normalized = _normalize_boundary_positions(
                    result,
                    positions,
                    qubit=reset.targets[0],
                    basis=basis,
                    boundary="state",
                )
                if normalized is not None:
                    result = normalized
                    changed = True
                    break
        if changed:
            continue

        # Before M, only the signed measured Pauli observable matters.
        for measurement_index, measurement in enumerate(result):
            if measurement.name != "M":
                continue
            positions = _local_word_before(
                result,
                measurement_index,
                qubit=measurement.targets[0],
                local_gates=local_gates,
            )
            if positions:
                normalized = _normalize_boundary_positions(
                    result,
                    positions,
                    qubit=measurement.targets[0],
                    basis=basis,
                    boundary="measurement",
                )
                if normalized is not None:
                    result = normalized
                    changed = True
                    break

        if not changed:
            return result


def _local_word_after(
    gates: list[_AtomicGate],
    boundary_index: int,
    *,
    qubit: int,
    local_gates: set[str],
) -> list[int]:
    positions: list[int] = []
    for index in range(boundary_index + 1, len(gates)):
        candidate = gates[index]
        if qubit not in candidate.qubits:
            continue
        if candidate.name in local_gates and candidate.targets == (qubit,):
            positions.append(index)
            continue
        break
    return positions


def _local_word_before(
    gates: list[_AtomicGate],
    boundary_index: int,
    *,
    qubit: int,
    local_gates: set[str],
) -> list[int]:
    positions: list[int] = []
    for index in range(boundary_index - 1, -1, -1):
        candidate = gates[index]
        if qubit not in candidate.qubits:
            continue
        if candidate.name in local_gates and candidate.targets == (qubit,):
            positions.append(index)
            continue
        break
    positions.reverse()
    return positions


def _normalize_boundary_positions(
    gates: list[_AtomicGate],
    positions: list[int],
    *,
    qubit: int,
    basis: _Basis,
    boundary: Literal["state", "measurement"],
) -> list[_AtomicGate] | None:
    word = tuple(gates[index].name for index in positions)
    state_forms, measurement_forms = _boundary_normal_forms(basis)
    key = _boundary_key(word, boundary=boundary)
    replacement = state_forms[key] if boundary == "state" else measurement_forms[key]
    if replacement == word:
        return None
    tag = gates[positions[0]].tag
    replacement_gates = tuple(_AtomicGate(name, (qubit,), tag) for name in replacement)
    return _replace_gate_positions(
        result=gates,
        positions=positions,
        replacement=replacement_gates,
    )


def _replace_gate_positions(
    result: list[_AtomicGate],
    positions: list[int],
    replacement: tuple[_AtomicGate, ...],
) -> list[_AtomicGate]:
    position_set = set(positions)
    first = min(positions)
    output: list[_AtomicGate] = []
    for index, gate in enumerate(result):
        if index == first:
            output.extend(replacement)
        if index not in position_set:
            output.append(gate)
    return output


def _replace_reset_and_word(
    gates: list[_AtomicGate],
    *,
    reset_index: int,
    word_positions: list[int],
    replacement: _AtomicGate,
) -> list[_AtomicGate]:
    removed = set(word_positions)
    return [replacement if index == reset_index else gate for index, gate in enumerate(gates) if index not in removed]


def _replace_measurement_and_word(
    gates: list[_AtomicGate],
    *,
    measurement_index: int,
    word_positions: list[int],
    replacement: _AtomicGate,
) -> list[_AtomicGate]:
    removed = set(word_positions)
    return [
        replacement if index == measurement_index else gate for index, gate in enumerate(gates) if index not in removed
    ]


def _measurement_state_is_discarded(
    gates: list[_AtomicGate],
    *,
    measurement_index: int,
    allow_end: bool,
) -> bool:
    qubit = gates[measurement_index].targets[0]
    for candidate in gates[measurement_index + 1 :]:
        if qubit not in candidate.qubits:
            continue
        return candidate.name in {"R", "RX", "RY"}
    return allow_end


@cache
def _boundary_normal_forms(
    basis: _Basis,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    generators = ("H", "S") if basis == "H_S_CZ" else ("H", HS_STIM_GATE)
    state_forms: dict[str, tuple[str, ...]] = {}
    measurement_forms: dict[str, tuple[str, ...]] = {}
    queue: deque[tuple[str, ...]] = deque([()])
    seen_tableaus: set[tuple[str, str]] = set()

    while queue and (len(state_forms) < _SIGNED_PAULI_COUNT or len(measurement_forms) < _SIGNED_PAULI_COUNT):
        word = queue.popleft()
        tableau = _single_qubit_tableau(word)
        tableau_key = (str(tableau.x_output(0)), str(tableau.z_output(0)))
        if tableau_key in seen_tableaus:
            continue
        seen_tableaus.add(tableau_key)

        state_forms.setdefault(str(tableau.z_output(0)), word)
        measurement_forms.setdefault(str(tableau.inverse().z_output(0)), word)
        for generator in generators:
            queue.append((*word, generator))

    if len(state_forms) != _SIGNED_PAULI_COUNT or len(measurement_forms) != _SIGNED_PAULI_COUNT:
        msg = f"Failed to enumerate boundary forms for {basis}"
        raise AssertionError(msg)
    return state_forms, measurement_forms


def _boundary_key(
    word: tuple[str, ...],
    *,
    boundary: Literal["state", "measurement"],
) -> str:
    tableau = _single_qubit_tableau(word)
    if boundary == "state":
        return str(tableau.z_output(0))
    return str(tableau.inverse().z_output(0))


def _single_qubit_tableau(word: tuple[str, ...]) -> stim.Tableau:
    circuit = stim.Circuit()
    for name in word:
        circuit.append(name, [0])
    return stim.Tableau.from_circuit(circuit) if word else stim.Tableau(1)


def _commutes_for_optimization(
    left: _AtomicGate,
    right: _AtomicGate,
    *,
    basis: _Basis,
) -> bool:
    if left.qubits.isdisjoint(right.qubits):
        return True
    if basis == "H_S_CZ":
        return left.name in {"S", "CZ"} and right.name in {"S", "CZ"}
    return left.name == "CZ" and right.name == "CZ"


def _transpile_block(circuit: stim.Circuit, *, context: str) -> stim.Circuit:  # ruff:ignore[complex-structure]
    result = stim.Circuit()

    for instruction_index in range(len(circuit)):
        instruction = circuit[instruction_index]
        location = f"{context}, instruction {instruction_index}"

        if isinstance(instruction, stim.CircuitRepeatBlock):
            body = _transpile_block(
                instruction.body_copy(),
                context=f"{location}, REPEAT body",
            )
            result.append(
                stim.CircuitRepeatBlock(
                    instruction.repeat_count,
                    body,
                    tag=instruction.tag,
                )
            )
            continue

        name = instruction.name
        if name in _PRESERVED_ANNOTATIONS:
            result.append(instruction)
            continue
        if name in _BOUNDARY_OPERATIONS:
            _validate_plain_qubit_targets(instruction, location=location)
            result.append(instruction)
            continue

        try:
            gate_data = stim.gate_data(name)
        except IndexError as ex:
            msg = f"Unsupported instruction {name!r} at {location}."
            raise UnsupportedInstructionError(msg) from ex

        if not gate_data.is_unitary:
            msg = (
                f"Instruction {name!r} at {location} is not a unitary "
                "Clifford gate and cannot be decomposed into H, HS, and CZ."
            )
            raise UnsupportedInstructionError(msg)

        if name in _BASIS_GATES:
            _validate_plain_qubit_targets(instruction, location=location)
            result.append(instruction)
        elif name == "S_DAG":
            _validate_plain_qubit_targets(instruction, location=location)
            targets = instruction.targets_copy()
            for _ in range(3):
                result.append("S", targets, tag=instruction.tag)
        elif name in {"SPP", "SPP_DAG"}:
            _append_spp_decomposition(
                result,
                instruction,
                location=location,
            )
        elif gate_data.is_single_qubit_gate or gate_data.is_two_qubit_gate:
            _append_fixed_gate_decomposition(
                result,
                instruction,
                gate_data=gate_data,
                location=location,
            )
        else:
            msg = f"Unitary instruction {name!r} at {location} has unsupported target semantics."
            raise UnsupportedInstructionError(msg)

    return result


def _validate_plain_qubit_targets(
    instruction: stim.CircuitInstruction,
    *,
    location: str,
) -> None:
    for target in instruction.targets_copy():
        if not target.is_qubit_target:
            msg = (
                f"Instruction {instruction.name!r} at {location} uses non-qubit target {target!s}; "
                "classical controls cannot be represented using the supported unitary gate basis."
            )
            raise UnsupportedInstructionError(msg)


def _qubit_value(target: stim.GateTarget) -> int:
    """Return a validated qubit target value.

    Returns
    -------
    `int`
        The qubit index.

    Raises
    ------
    AssertionError
        If an internally validated target has no qubit value.
    """
    value = target.qubit_value
    if value is None:
        msg = f"Expected a qubit target, got {target!s}."
        raise AssertionError(msg)
    return int(value)


def _append_fixed_gate_decomposition(
    output: stim.Circuit,
    instruction: stim.CircuitInstruction,
    *,
    gate_data: stim.GateData,
    location: str,
) -> None:
    _validate_plain_qubit_targets(instruction, location=location)
    for group in instruction.target_groups():
        expected_size = 1 if gate_data.is_single_qubit_gate else 2
        if len(group) != expected_size:
            rendered_targets = " ".join(str(target) for target in group)
            msg = f"Instruction {instruction.name!r} at {location} uses unsupported targets {rendered_targets!r}."
            raise UnsupportedInstructionError(msg)

        qubits = tuple(_qubit_value(target) for target in group)
        _append_template(
            output,
            _fixed_gate_template(instruction.name),
            qubits=qubits,
            tag=instruction.tag,
        )


@cache
def _fixed_gate_template(name: str) -> stim.Circuit:
    tableau = stim.gate_data(name).tableau
    if tableau is None:
        msg = f"Stim did not provide tableau data for {name!r}"
        raise AssertionError(msg)
    return tableau.to_circuit(method="elimination")


def _append_spp_decomposition(
    output: stim.Circuit,
    instruction: stim.CircuitInstruction,
    *,
    location: str,
) -> None:
    for group in instruction.target_groups():
        if not group:
            msg = f"Instruction {instruction.name!r} at {location} has an empty Pauli product."
            raise UnsupportedInstructionError(msg)

        original_qubits: list[int] = []
        local_targets: list[stim.GateTarget] = []
        seen_qubits: set[int] = set()

        for local_index, target in enumerate(group):
            pauli = target.pauli_type
            if pauli not in {"X", "Y", "Z"}:
                msg = f"Instruction {instruction.name!r} at {location} has unsupported Pauli target {target!s}."
                raise UnsupportedInstructionError(msg)
            if target.value in seen_qubits:
                msg = (
                    f"Instruction {instruction.name!r} at {location} targets qubit {target.value} "
                    "more than once in one Pauli product."
                )
                raise UnsupportedInstructionError(msg)
            seen_qubits.add(target.value)
            original_qubits.append(target.value)

            factory = {
                "X": stim.target_x,
                "Y": stim.target_y,
                "Z": stim.target_z,
            }[pauli]
            local_targets.append(
                factory(
                    local_index,
                    invert=target.is_inverted_result_target,
                )
            )
            if local_index + 1 != len(group):
                local_targets.append(stim.target_combiner())

        local_circuit = stim.Circuit()
        local_circuit.append(instruction.name, local_targets)
        template = stim.Tableau.from_circuit(local_circuit).to_circuit(method="elimination")
        _append_template(
            output,
            template,
            qubits=tuple(original_qubits),
            tag=instruction.tag,
        )


def _append_template(
    output: stim.Circuit,
    template: stim.Circuit,
    *,
    qubits: tuple[int, ...],
    tag: str,
) -> None:
    for instruction_index in range(len(template)):
        instruction = template[instruction_index]
        if isinstance(instruction, stim.CircuitRepeatBlock):
            msg = "Stim unexpectedly synthesized a REPEAT block"
            raise TypeError(msg)

        targets = instruction.targets_copy()
        if instruction.name in {"H", "S"}:
            for target in targets:
                output.append(
                    instruction.name,
                    [qubits[_qubit_value(target)]],
                    tag=tag,
                )
        elif instruction.name == "CX":
            if len(targets) % 2:
                msg = "Stim synthesized CX with an odd target count"
                raise AssertionError(msg)
            for pair_start in range(0, len(targets), 2):
                control = qubits[_qubit_value(targets[pair_start])]
                target = qubits[_qubit_value(targets[pair_start + 1])]
                output.append("H", [target], tag=tag)
                output.append("CZ", [control, target], tag=tag)
                output.append("H", [target], tag=tag)
        else:
            msg = f"Stim's elimination synthesis unexpectedly emitted {instruction.name!r} instead of H, S, or CX"
            raise AssertionError(msg)
