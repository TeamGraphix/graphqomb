"""Tests for importing supported Stim circuits into GraphQOMB patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphqomb.command import M
from graphqomb.common import Axis, AxisMeasBasis, Sign
from graphqomb.graphstate import odd_neighbors
from graphqomb.qec.qeccode import YFoliation
from graphqomb.simulator import PatternSimulator, SimulatorBackend
from graphqomb.statevec import StateVector
from graphqomb.stim_compiler import stim_compile
from graphqomb.stim_importer import stim_circuit_to_pattern, stim_file_to_pattern, stim_text_to_pattern

stim = pytest.importorskip("stim")

if TYPE_CHECKING:
    from pathlib import Path


def test_stim_text_to_pattern_imports_unitary_clifford_block() -> None:
    result = stim_text_to_pattern(
        """
        H 10
        CX 10 12
        S_DAG 12
        """
    )

    assert result.stim_to_qubit == {10: 0, 12: 1}
    assert result.qubit_to_stim == {0: 10, 1: 12}
    assert result.mpp_extractions == ()
    assert set(result.pattern.input_node_indices.values()) == {0, 1}
    assert set(result.pattern.output_node_indices.values()) == {0, 1}


@pytest.mark.parametrize(
    "gate_name",
    [name for name in sorted(stim.gate_data()) if stim.gate_data(name).is_unitary and name not in {"SPP", "SPP_DAG"}],
)
def test_stim_circuit_to_pattern_imports_every_fixed_clifford_gate(gate_name: str) -> None:
    """Test every fixed Stim Clifford gate through parser and importer integration."""
    gate_data = stim.gate_data(gate_name)
    targets = [0] if gate_data.is_single_qubit_gate else [0, 1]
    source = stim.Circuit()
    source.append(gate_name, targets)

    result = stim_circuit_to_pattern(source)

    assert set(result.pattern.input_node_indices.values()) == set(targets)
    assert set(result.pattern.output_node_indices.values()) == set(targets)


@pytest.mark.parametrize("gate_name", ["SPP", "SPP_DAG"])
def test_stim_text_to_pattern_imports_pauli_product_rotations(gate_name: str) -> None:
    result = stim_text_to_pattern(f"{gate_name} X0*Y1*!Z2")

    assert set(result.pattern.input_node_indices.values()) == {0, 1, 2}
    assert set(result.pattern.output_node_indices.values()) == {0, 1, 2}


def test_stim_text_to_pattern_cancels_repeated_cz_in_one_tick_block() -> None:
    result = stim_text_to_pattern("CZ 0 1\nCZ 0 1")
    graph = result.pattern.pauli_frame.graphstate

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 0


def test_stim_text_to_pattern_rejects_classically_controlled_clifford() -> None:
    with pytest.raises(ValueError, match="non-qubit target"):
        stim_text_to_pattern("M 0\nTICK\nCX rec[-1] 1")


def test_stim_text_to_pattern_preserves_unitary_semantics_across_ticks() -> None:
    initial = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    expected = np.asarray([1 + 1j, 1 - 1j], dtype=np.complex128) / 2

    for seed in range(8):
        pattern = stim_text_to_pattern("S 0\nTICK\nH 0\n").pattern
        simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
        simulator.state = StateVector(initial)
        simulator.simulate(rng=np.random.default_rng(seed))

        overlap = np.vdot(expected, simulator.state.state())
        assert np.isclose(abs(overlap), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    ("stim_text", "expected"),
    [
        ("SQRT_Y 0", [0.0, 1.0]),
        ("C_XYZ 0", [(1 - 1j) / 2, (1 + 1j) / 2]),
    ],
)
def test_stim_text_to_pattern_preserves_negative_measurement_gate_semantics(
    stim_text: str, expected: list[complex]
) -> None:
    """Test the X- and Y- measurement basis gates end to end on a |+> input."""
    initial = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    expected_state = np.asarray(expected, dtype=np.complex128)

    for seed in range(8):
        pattern = stim_text_to_pattern(stim_text).pattern
        simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
        simulator.state = StateVector(initial)
        simulator.simulate(rng=np.random.default_rng(seed))

        overlap = np.vdot(expected_state, simulator.state.state())
        assert np.isclose(abs(overlap), 1.0, atol=1e-9)


def test_stim_text_to_pattern_preserves_sparse_qubit_coordinates() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(1, 2) 10
        QUBIT_COORDS(3) 99
        QUBIT_COORDS(4) 99
        H 99
        """
    )

    assert result.stim_to_qubit == {10: 0, 99: 1}
    assert result.pattern.input_coordinates
    assert set(result.pattern.input_coordinates.values()) == {(1.0, 2.0, 1.0), (3.0, 4.0, 0.0)}


def test_stim_text_to_pattern_aligns_parallel_gate_outputs_with_different_depths() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        H 0
        S 1
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    output_coordinates = {qubit: graph.coordinates[node] for node, qubit in graph.output_node_indices.items()}
    lane_0_z = sorted(coord[2] for coord in graph.coordinates.values() if np.isclose(coord[0], 0.0))
    lane_1_z = sorted(coord[2] for coord in graph.coordinates.values() if np.isclose(coord[0], 1.0))

    assert output_coordinates == {0: (0.0, 0.0, 2.0), 1: (1.0, 0.0, 2.0)}
    assert lane_0_z == [0.0, 2.0]
    assert lane_1_z == [0.0, 1.0, 2.0]


def test_stim_text_to_pattern_relocates_idle_input_without_adding_a_wire_node() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        H 0
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    idle_input = next(node for node, qubit in graph.input_node_indices.items() if qubit == 1)
    idle_output = next(node for node, qubit in graph.output_node_indices.items() if qubit == 1)

    assert graph.number_of_nodes() == 3
    assert idle_input == idle_output
    assert graph.coordinates[idle_input] == (1.0, 0.0, 1.0)
    assert graph.neighbors(idle_input) == set()


@pytest.mark.parametrize(
    ("instruction", "initial", "expected"),
    [
        ("H 0", [1, 0], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        ("S 0", [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1j / np.sqrt(2)]),
        ("S_DAG 0", [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1j / np.sqrt(2)]),
        ("X 0", [1, 0], [0, 1]),
        ("Y 0", [1, 0], [0, 1j]),
        ("Z 0", [1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]),
    ],
)
def test_stim_text_to_pattern_preserves_supported_single_qubit_gates(
    instruction: str,
    initial: list[complex],
    expected: list[complex],
) -> None:
    pattern = stim_text_to_pattern(instruction).pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector(initial)

    simulator.simulate(rng=np.random.default_rng(3))

    assert np.isclose(abs(np.vdot(expected, simulator.state.state())), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    ("instruction", "initial", "expected"),
    [
        ("CX 0 1", [0, 0, 1, 0], [0, 0, 0, 1]),
        ("CZ 0 1", [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, -0.5]),
        ("SWAP 0 1", [0, 0, 1, 0], [0, 1, 0, 0]),
    ],
)
def test_stim_text_to_pattern_preserves_supported_two_qubit_gates(
    instruction: str,
    initial: list[complex],
    expected: list[complex],
) -> None:
    pattern = stim_text_to_pattern(instruction).pattern
    simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
    simulator.state = StateVector(initial)

    simulator.simulate(rng=np.random.default_rng(3))

    assert np.isclose(abs(np.vdot(expected, simulator.state.state())), 1.0, atol=1e-9)


def test_stim_text_to_pattern_imports_tick_separated_mpp_block() -> None:
    result = stim_text_to_pattern(
        """
        H 10
        TICK
        MPP X10*Z12
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(3) rec[-1]
        TICK
        CZ 10 12
        """
    )

    assert result.stim_to_qubit == {10: 0, 12: 1}
    assert len(result.mpp_extractions) == 1
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert set(result.pattern.pauli_frame.logical_observables) == {3}
    assert set(result.pattern.output_node_indices.values()) == {0, 1}


def test_stim_text_to_pattern_combines_commuting_mpp_instructions_in_one_tick_block() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        DETECTOR rec[-1]
        MPP Z1
        DETECTOR rec[-1]
        """
    )

    assert len(result.mpp_extractions) == 1
    assert result.mpp_extractions[0].supports == (((0, "X"),), ((1, "Z"),))
    assert len(result.pattern.pauli_frame.parity_check_group) == 2


def test_stim_text_to_pattern_rejects_anticommuting_mpp_in_one_tick_block() -> None:
    with pytest.raises(ValueError, match="must commute"):
        stim_text_to_pattern("MPP X0\nMPP Z0")


@pytest.mark.parametrize(
    ("y_foliation", "expected_node_count"),
    [(YFoliation.TYPE_I, 27), (YFoliation.TYPE_II, 28)],
)
def test_stim_text_to_pattern_builds_commuting_mpp_block_at_common_z(
    y_foliation: YFoliation,
    expected_node_count: int,
) -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 0) 3
        QUBIT_COORDS(4, 0) 4
        QUBIT_COORDS(5, 0) 5
        QUBIT_COORDS(6, 0) 6
        MPP X0*X1*X4*X5
        DETECTOR rec[-1]
        MPP Z0*Z1*Z2*Z3
        DETECTOR rec[-1]
        MPP Y0*X2*Z4*Z6
        DETECTOR rec[-1]
        MPP Z4*Z5
        DETECTOR rec[-1]
        MPP X1*X3
        DETECTOR rec[-1]
        MPP Z2*X6
        DETECTOR rec[-1]
        """,
        y_foliation=y_foliation,
    )
    graph = result.pattern.pauli_frame.graphstate
    z_coordinates = {coordinate[2] for coordinate in graph.coordinates.values()}

    assert len(result.mpp_extractions) == 1
    assert len(result.mpp_extractions[0].supports) == 6
    assert graph.number_of_nodes() == expected_node_count
    assert np.isclose(min(z_coordinates), 0.0)
    assert np.isclose(max(z_coordinates), 2.0)
    assert set(result.pattern.input_node_indices.values()) == set(range(7))
    assert set(result.pattern.output_node_indices.values()) == set(range(7))
    mixed_check_ancilla = next(iter(result.pattern.pauli_frame.parity_check_group[2]))
    mixed_loop_ancilla = next(iter(result.pattern.pauli_frame.parity_check_group[5]))
    assert graph.has_edge(mixed_check_ancilla, mixed_loop_ancilla)


def test_stim_text_to_pattern_advances_z_once_per_mpp_tick_block() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        MPP X0
        MPP X1
        TICK
        MPP X0*X1
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    z_coordinates = {coordinate[2] for coordinate in graph.coordinates.values()}

    assert len(result.mpp_extractions) == 2
    assert np.isclose(max(z_coordinates), 4.0)


def test_stim_text_to_pattern_relocates_idle_input_to_mpp_output_layer() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        MPP X0
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    idle_input = next(node for node, qubit in graph.input_node_indices.items() if qubit == 1)
    idle_output = next(node for node, qubit in graph.output_node_indices.items() if qubit == 1)
    active_output = next(node for node, qubit in graph.output_node_indices.items() if qubit == 0)

    assert idle_input == idle_output
    assert graph.coordinates[idle_input] == (1.0, 0.0, 2.0)
    assert graph.coordinates[active_output] == (0.0, 0.0, 2.0)
    assert graph.neighbors(idle_input) == set()


def test_stim_text_to_pattern_derives_complete_zflow_from_xflow() -> None:
    result = stim_text_to_pattern("H 0\nTICK\nMPP X0")
    frame = result.pattern.pauli_frame

    assert frame.zflow == {node: odd_neighbors(targets, frame.graphstate) for node, targets in frame.xflow.items()}


def test_stim_text_to_pattern_excludes_mpp_ancilla_from_xflow() -> None:
    result = stim_text_to_pattern("MPP X0\nDETECTOR rec[-1]")
    frame = result.pattern.pauli_frame

    ancilla_nodes = frame.parity_check_group[0]
    assert len(ancilla_nodes) == 1
    assert set(frame.xflow) == set(frame.graphstate.meas_bases) - ancilla_nodes
    assert all(targets.isdisjoint(ancilla_nodes) for targets in frame.xflow.values())


def test_stim_text_to_pattern_appends_output_after_type_i_mpp_measurements() -> None:
    result = stim_text_to_pattern("MPP X0")
    graph = result.pattern.pauli_frame.graphstate

    assert graph.number_of_nodes() == 4
    assert len(graph.meas_bases) == 3
    assert len(graph.output_node_indices) == 1
    assert next(iter(graph.output_node_indices)) not in graph.meas_bases


def test_stim_text_to_pattern_appends_output_after_type_ii_y_measurements() -> None:
    result = stim_text_to_pattern("MPP Y0", y_foliation=YFoliation.TYPE_II)
    graph = result.pattern.pauli_frame.graphstate
    y_measurements = [
        basis for basis in graph.meas_bases.values() if isinstance(basis, AxisMeasBasis) and basis.axis == Axis.Y
    ]

    assert graph.number_of_nodes() == 5
    assert len(y_measurements) == 3
    assert len(graph.output_node_indices) == 1
    assert next(iter(graph.output_node_indices)) not in graph.meas_bases


def test_stim_import_entry_points_accept_type_ii_foliation(tmp_path: Path) -> None:
    stim_path = tmp_path / "y_measurement.stim"
    stim_path.write_text("MPP Y0", encoding="utf-8")

    circuit_result = stim_circuit_to_pattern(stim.Circuit("MPP Y0"), y_foliation=YFoliation.TYPE_II)
    file_result = stim_file_to_pattern(stim_path, y_foliation=YFoliation.TYPE_II)

    for result in (circuit_result, file_result):
        axes = [
            basis.axis
            for basis in result.pattern.pauli_frame.graphstate.meas_bases.values()
            if isinstance(basis, AxisMeasBasis)
        ]
        assert axes.count(Axis.Y) == 3


@pytest.mark.parametrize(
    ("instruction", "expected_axis"),
    [
        ("M 10", Axis.Z),
        ("MZ 10", Axis.Z),
        ("MX 10", Axis.X),
        ("MY 10", Axis.Y),
    ],
)
def test_stim_text_to_pattern_imports_single_qubit_pauli_measurements(
    instruction: str,
    expected_axis: Axis,
) -> None:
    result = stim_text_to_pattern(instruction)

    measurements = [command for command in result.pattern.commands if isinstance(command, M)]

    assert result.mpp_extractions == ()
    assert result.stim_to_qubit == {10: 0}
    assert result.pattern.input_node_indices == {0: 0}
    assert result.pattern.output_node_indices == {0: 0}
    assert len(measurements) == 1
    assert measurements[0].node == 0
    assert isinstance(measurements[0].meas_basis, AxisMeasBasis)
    assert measurements[0].meas_basis.axis == expected_axis
    assert measurements[0].meas_basis.sign == Sign.PLUS


def test_stim_text_to_pattern_assigns_single_measurement_to_existing_wire_node() -> None:
    result = stim_text_to_pattern("QUBIT_COORDS(1, 2) 10\nH 10\nTICK\nMX 10")
    output_node = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 0)
    output_measurements = [
        command for command in result.pattern.commands if isinstance(command, M) and command.node == output_node
    ]

    assert result.mpp_extractions == ()
    assert len(output_measurements) == 1
    assert isinstance(output_measurements[0].meas_basis, AxisMeasBasis)
    assert output_measurements[0].meas_basis.axis == Axis.X
    assert output_measurements[0].meas_basis.sign == Sign.PLUS
    assert result.pattern.pauli_frame.graphstate.coordinates[output_node] == (1.0, 2.0, 1.0)


def test_stim_text_to_pattern_preserves_mpp_lane_coordinate_for_terminal_measurement() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(1, 2) 0
        MPP X0
        TICK
        MX 0
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    output_node = next(node for node, qubit in result.pattern.output_node_indices.items() if qubit == 0)
    output_basis = graph.meas_bases[output_node]

    assert graph.number_of_nodes() == 4
    assert graph.coordinates[output_node] == (1.0, 2.0, 2.0)
    assert isinstance(output_basis, AxisMeasBasis)
    assert output_basis.axis == Axis.X


def test_stim_text_to_pattern_places_gate_after_mpp_at_next_z_layer() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(1, 2) 0
        MPP X0
        TICK
        H 0
        TICK
        MX 0
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    output_node = next(node for node, qubit in graph.output_node_indices.items() if qubit == 0)

    assert graph.coordinates[output_node] == (1.0, 2.0, 3.0)
    assert all(len(coord) == 3 for coord in graph.coordinates.values())
    assert {coord[2] for coord in graph.coordinates.values()} == {0.0, 1.0, 2.0, 3.0}


def test_stim_text_to_pattern_composes_gate_output_with_mpp_input_at_same_z() -> None:
    result = stim_text_to_pattern(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        H 0
        TICK
        MPP X0*Z1
        """
    )
    graph = result.pattern.pauli_frame.graphstate
    input_coordinates = {qubit: graph.coordinates[node] for node, qubit in graph.input_node_indices.items()}
    output_coordinates = {qubit: graph.coordinates[node] for node, qubit in graph.output_node_indices.items()}

    assert input_coordinates == {0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 1.0)}
    assert output_coordinates == {0: (0.0, 0.0, 3.0), 1: (1.0, 0.0, 3.0)}
    assert {(coord[0], coord[2]) for coord in graph.coordinates.values()} >= {
        (0.0, 1.0),
        (1.0, 1.0),
    }


@pytest.mark.parametrize(
    ("instruction", "expected_axis"),
    [
        ("MXX 10 12", "X"),
        ("MYY 10 12", "Y"),
        ("MZZ 10 12", "Z"),
    ],
)
def test_stim_text_to_pattern_imports_pair_pauli_measurements(
    instruction: str,
    expected_axis: str,
) -> None:
    result = stim_text_to_pattern(instruction)

    assert result.mpp_extractions[0].supports == (((10, expected_axis), (12, expected_axis)),)
    assert result.stim_to_qubit == {10: 0, 12: 1}


def test_stim_text_to_pattern_preserves_multiple_measurement_results_in_target_order() -> None:
    result = stim_text_to_pattern(
        """
        M 0 2
        MXX 1 3 4 5
        DETECTOR rec[-4] rec[-1]
        """
    )

    assert result.mpp_extractions[0].supports == (
        ((1, "X"), (3, "X")),
        ((4, "X"), (5, "X")),
    )
    direct_measurements = [command for command in result.pattern.commands if isinstance(command, M)]
    assert (
        sum(
            isinstance(command.meas_basis, AxisMeasBasis) and command.meas_basis.axis == Axis.Z
            for command in direct_measurements
        )
        == 2
    )
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 2


def test_stim_text_to_pattern_maps_m_and_mpp_records_to_one_detector() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        X_ERROR(0.01) 7
        M 7
        DETECTOR rec[-2] rec[-1]
        """
    )

    assert result.mpp_extractions[0].supports == (((0, "X"),),)
    assert any(
        isinstance(command, M) and isinstance(command.meas_basis, AxisMeasBasis) and command.meas_basis.axis == Axis.Z
        for command in result.pattern.commands
    )
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 2


def test_stim_text_to_pattern_omits_noise_and_measurement_error_probabilities() -> None:
    result = stim_text_to_pattern(
        """
        DEPOLARIZE1(0.25) 0
        X_ERROR(0.125) 0
        MX(0.5) 0
        DETECTOR rec[-1]
        """
    )

    assert result.mpp_extractions == ()
    measurements = [command for command in result.pattern.commands if isinstance(command, M)]
    assert len(measurements) == 1
    assert isinstance(measurements[0].meas_basis, AxisMeasBasis)
    assert measurements[0].meas_basis.axis == Axis.X
    assert len(result.pattern.pauli_frame.parity_check_group) == 1


def test_stim_text_to_pattern_preserves_ideal_herald_records_as_zero() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        HERALDED_ERASE(0.25) 0
        DETECTOR rec[-2] rec[-1]
        """
    )

    assert result.mpp_extractions[0].supports == (((0, "X"),),)
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 1


def test_stim_text_to_pattern_preserves_cross_block_detector_records() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        TICK
        MPP Z0
        DETECTOR rec[-2]
        """
    )

    assert len(result.mpp_extractions) == 2
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 1


def test_stim_text_to_pattern_tracks_all_record_types_with_global_indices() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        HERALDED_ERASE(0.25) 2
        M 1
        TICK
        MPP Z0
        DETECTOR rec[-4] rec[-3] rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(5) rec[-4] rec[-1]
        """
    )

    assert result.stim_to_qubit == {0: 0, 1: 1}
    assert len(result.mpp_extractions) == 2
    for extraction in result.mpp_extractions:
        assert extraction.detector_record_indices == (frozenset({0, 1, 2, 3}),)
        assert extraction.logical_observable_record_indices == {5: frozenset({0, 3})}
    assert len(result.pattern.pauli_frame.parity_check_group) == 1
    assert len(result.pattern.pauli_frame.parity_check_group[0]) == 3
    assert len(result.pattern.pauli_frame.logical_observables[5]) == 2


@pytest.mark.parametrize(
    "text",
    [
        "MPP X0\nDETECTOR rec[-1]",
        "MPP X0\nTICK\nMPP X0\nDETECTOR rec[-1] rec[-2]",
    ],
)
def test_stim_text_to_pattern_preserves_deterministic_mpp_detectors(text: str) -> None:
    pattern = stim_text_to_pattern(text).pattern
    compiled = stim.Circuit(stim_compile(pattern, emit_qubit_coords=False))

    compiled.detector_error_model()


@pytest.mark.parametrize("y_foliation", [YFoliation.TYPE_I, YFoliation.TYPE_II])
def test_stim_text_to_pattern_preserves_detectors_for_twisted_stabilizer_orders(
    y_foliation: YFoliation,
) -> None:
    pattern = stim_text_to_pattern(
        """
        MPP X0*Z1
        MPP Z0*X1
        TICK
        MPP X0*Z1
        DETECTOR rec[-1] rec[-3]
        MPP Z0*X1
        DETECTOR rec[-1] rec[-3]
        """,
        y_foliation=y_foliation,
    ).pattern
    compiled = stim.Circuit(stim_compile(pattern, emit_qubit_coords=False))

    assert compiled.detector_error_model().num_detectors == 2


@pytest.mark.parametrize(
    ("text", "expected_ancilla_axis"),
    [
        ("RY 0\nTICK\nMPP Y0\nDETECTOR rec[-1]", Axis.Y),
        ("RY 0 1\nTICK\nMPP Y0*Y1\nDETECTOR rec[-1]", Axis.X),
    ],
)
def test_stim_text_to_pattern_preserves_deterministic_type_i_y_mpp_detector(
    text: str,
    expected_ancilla_axis: Axis,
) -> None:
    pattern = stim_text_to_pattern(text, y_foliation=YFoliation.TYPE_I).pattern
    ancilla_node = next(iter(pattern.pauli_frame.parity_check_group[0]))
    ancilla_basis = pattern.pauli_frame.graphstate.meas_bases[ancilla_node]
    compiled = stim.Circuit(stim_compile(pattern, emit_qubit_coords=False))

    assert isinstance(ancilla_basis, AxisMeasBasis)
    assert ancilla_basis.axis == expected_ancilla_axis
    compiled.detector_error_model()


def test_stim_text_to_pattern_composes_mpp_output_into_next_mpp_input() -> None:
    result = stim_text_to_pattern("MPP X0\nTICK\nMPP X0")
    graph = result.pattern.pauli_frame.graphstate

    assert graph.number_of_nodes() == 7
    assert len(graph.meas_bases) == 6
    assert len(graph.input_node_indices) == 1
    assert len(graph.output_node_indices) == 1


def test_stim_text_to_pattern_accepts_annotation_only_tick_block() -> None:
    result = stim_text_to_pattern(
        """
        MPP X0
        TICK
        DETECTOR rec[-1]
        """
    )

    assert len(result.pattern.pauli_frame.parity_check_group) == 1


@pytest.mark.parametrize("measurement", ["MPP X0", "M 0", "MX 0", "MY 0", "MXX 0 1"])
def test_stim_text_to_pattern_rejects_mixed_measurement_and_unitary_block(measurement: str) -> None:
    with pytest.raises(ValueError, match="separated from unitary gate instructions by TICK"):
        stim_text_to_pattern(f"H 0\n{measurement}\n")


@pytest.mark.parametrize(
    ("instruction", "expected_axis", "compiled_instruction"),
    [
        ("R 0", Axis.Z, "R 0"),
        ("RZ 0", Axis.Z, "R 0"),
        ("RX 0", Axis.X, "RX 0"),
        ("RY 0", Axis.Y, "RY 0"),
    ],
)
def test_stim_text_to_pattern_imports_initial_reset(
    instruction: str,
    expected_axis: Axis,
    compiled_instruction: str,
) -> None:
    result = stim_text_to_pattern(instruction)
    input_node = next(node for node, q_index in result.pattern.input_node_indices.items() if q_index == 0)

    assert result.pattern.input_initialization_axes[input_node] == expected_axis
    assert compiled_instruction in stim_compile(result.pattern, emit_qubit_coords=False).splitlines()


def test_stim_text_to_pattern_uses_last_leading_reset() -> None:
    result = stim_text_to_pattern("R 0\nRY 0\nH 0")
    input_node = next(node for node, q_index in result.pattern.input_node_indices.items() if q_index == 0)

    assert result.pattern.input_initialization_axes[input_node] == Axis.Y


def test_stim_text_to_pattern_allows_initial_reset_after_other_qubit_operation() -> None:
    result = stim_text_to_pattern("H 0\nR 1")
    input_node = next(node for node, q_index in result.pattern.input_node_indices.items() if q_index == 1)

    assert result.pattern.input_initialization_axes[input_node] == Axis.Z


@pytest.mark.parametrize("instruction", ["R", "RX", "RY"])
def test_stim_text_to_pattern_rejects_reset_after_quantum_operation(instruction: str) -> None:
    with pytest.raises(ValueError, match="only initial resets are supported"):
        stim_text_to_pattern(f"H 0\n{instruction} 0")


@pytest.mark.parametrize("instruction", ["MR 0", "MRX 0", "MRY 0"])
def test_stim_text_to_pattern_defers_measurement_reset_instructions(instruction: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Stim instruction"):
        stim_text_to_pattern(instruction)


def test_stim_text_to_pattern_rejects_duplicate_qubit_coordinates() -> None:
    with pytest.raises(ValueError, match="distinct XY projections"):
        stim_text_to_pattern("QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(0, 0) 1\nCZ 0 1")


def test_stim_text_to_pattern_rejects_coordinates_sharing_an_xy_projection() -> None:
    with pytest.raises(ValueError, match="distinct XY projections"):
        stim_text_to_pattern(
            "QUBIT_COORDS(0, 0, 1) 0\nQUBIT_COORDS(0, 0, 2) 1\nCZ 0 1",
            coord_dims=3,
        )


def test_stim_text_to_pattern_rejects_qubit_reuse_after_single_measurement() -> None:
    with pytest.raises(ValueError, match="terminate those qubit lifetimes"):
        stim_text_to_pattern("M 0\nTICK\nMPP X0")


def test_stim_text_to_pattern_rejects_unitary_reuse_after_single_measurement() -> None:
    with pytest.raises(ValueError, match="terminate those qubit lifetimes"):
        stim_text_to_pattern("M 0\nTICK\nH 0")


def test_stim_text_to_pattern_allows_disjoint_qubit_after_single_measurement() -> None:
    result = stim_text_to_pattern("M 0\nTICK\nH 1")
    measured_nodes = {command.node for command in result.pattern.commands if isinstance(command, M)}
    simulator = PatternSimulator(result.pattern, SimulatorBackend.StateVector)

    simulator.simulate(rng=np.random.default_rng(3))

    assert result.stim_to_qubit == {0: 0, 1: 1}
    assert set(result.pattern.output_node_indices.values()) == {0, 1}
    assert any(result.pattern.output_node_indices[node] == 0 for node in measured_nodes)
    assert set(simulator.output_results) == {0}
    assert np.isclose(abs(np.vdot([1, 0], simulator.state.state())), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    ("instruction", "expected_axis", "compiled_instruction"),
    [
        ("MX !0", Axis.X, "MX !0"),
        ("MY !0", Axis.Y, "MY !0"),
        ("M !0", Axis.Z, "MZ !0"),
    ],
)
def test_stim_text_to_pattern_preserves_inverted_single_measurement(
    instruction: str,
    expected_axis: Axis,
    compiled_instruction: str,
) -> None:
    result = stim_text_to_pattern(instruction)
    measurements = [command for command in result.pattern.commands if isinstance(command, M)]

    assert result.mpp_extractions == ()
    assert len(measurements) == 1
    assert isinstance(measurements[0].meas_basis, AxisMeasBasis)
    assert measurements[0].meas_basis.axis == expected_axis
    assert measurements[0].meas_basis.sign == Sign.MINUS
    assert compiled_instruction in stim_compile(result.pattern, emit_qubit_coords=False).splitlines()


def test_stim_text_to_pattern_rejects_inverted_pair_measurement_result() -> None:
    with pytest.raises(ValueError, match="Signed MPP products are not supported"):
        stim_text_to_pattern("MYY !0 1")


def test_stim_text_to_pattern_rejects_true_mpad_record() -> None:
    with pytest.raises(ValueError, match="MPAD 1 records are not supported"):
        stim_text_to_pattern("MPP X0\nMPAD 1")


def test_stim_text_to_pattern_rejects_record_before_beginning_of_time() -> None:
    with pytest.raises(ValueError, match="before the beginning of time"):
        stim_text_to_pattern("DETECTOR rec[-1]")
