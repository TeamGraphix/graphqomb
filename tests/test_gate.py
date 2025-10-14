from collections.abc import Sequence

import numpy as np
import pytest

from graphqomb.gates import (
    CCZ,
    CNOT,
    CU3,
    CZ,
    SWAP,
    U3,
    CRx,
    CRz,
    Gate,
    H,
    Identity,
    J,
    MultiGate,
    PhaseGadget,
    Rx,
    Ry,
    Rz,
    S,
    SingleGate,
    T,
    Tdg,
    Toffoli,
    TwoQubitGate,
    X,
    Y,
    Z,
)
from graphqomb.statevec import StateVector

SINGLE_GATES: list[type[SingleGate]] = [J, Identity, X, Y, Z, H, S, T, Tdg, Rx, Ry, Rz, U3]
TWO_GATES: list[type[TwoQubitGate]] = [CZ, CNOT, SWAP, CRz, CRx, CU3]
MULTI_GATES: list[type[MultiGate]] = [PhaseGadget, CCZ, Toffoli]

NUM_ANGLES: dict[type, int] = {
    J: 1,
    Identity: 0,
    X: 0,
    Y: 0,
    Z: 0,
    H: 0,
    S: 0,
    T: 0,
    Tdg: 0,
    Rx: 1,
    Ry: 1,
    Rz: 1,
    U3: 3,
    CZ: 0,
    CNOT: 0,
    SWAP: 0,
    CRz: 1,
    CRx: 1,
    CU3: 3,
    PhaseGadget: 1,
    CCZ: 0,
    Toffoli: 0,
}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()


def apply_gates(state: StateVector, gates: Sequence[Gate], qubits: Sequence[int]) -> StateVector:
    indices: Sequence[int]
    for gate in gates:
        if isinstance(gate, SingleGate):
            indices = [gate.qubit]
        elif isinstance(gate, (TwoQubitGate, MultiGate)):
            indices = gate.qubits
        else:
            msg = f"Unknown gate type: {type(gate)}"
            raise TypeError(msg)
        qubit_indices = [qubits.index(i) for i in indices]
        state.evolve(gate.matrix(), qubit_indices)
    return state


@pytest.mark.parametrize("gate_class", SINGLE_GATES)
def test_single_qubit_gate(gate_class: type, rng: np.random.Generator) -> None:
    num_qubits = 1
    state = StateVector.from_num_qubits(num_qubits)
    qubits = [0]

    gate = gate_class(qubits[0], *[rng.uniform(0, 2 * np.pi)] * NUM_ANGLES[gate_class])

    result1 = apply_gates(state.copy(), [gate], qubits)
    result2 = apply_gates(state.copy(), gate.unit_gates(), qubits)

    inner_product = np.vdot(result1.state(), result2.state())
    assert np.isclose(np.abs(inner_product), 1)


@pytest.mark.parametrize("gate_class", TWO_GATES)
def test_two_qubit_gate(gate_class: type, rng: np.random.Generator) -> None:
    num_qubits = 2
    state = StateVector.from_num_qubits(num_qubits)
    qubits = [0, 1]

    gate = gate_class(qubits, *[rng.uniform(0, 2 * np.pi)] * NUM_ANGLES[gate_class])

    result1 = apply_gates(state.copy(), [gate], qubits)
    result2 = apply_gates(state.copy(), gate.unit_gates(), qubits)

    inner_product = np.vdot(result1.state(), result2.state())
    assert np.isclose(np.abs(inner_product), 1)


@pytest.mark.parametrize("gate_class", MULTI_GATES)
def test_multi_qubit_gate(gate_class: type, rng: np.random.Generator) -> None:
    num_qubits = 3
    state = StateVector.from_num_qubits(num_qubits)
    qubits = [0, 1, 2]

    gate = gate_class(qubits, *[rng.uniform(0, 2 * np.pi)] * NUM_ANGLES[gate_class])

    result1 = apply_gates(state.copy(), [gate], qubits)
    result2 = apply_gates(state.copy(), gate.unit_gates(), qubits)

    inner_product = np.vdot(result1.state(), result2.state())
    assert np.isclose(np.abs(inner_product), 1)
