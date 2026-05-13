# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **PTN Format**: Human-readable text format (`.ptn`) for pattern serialization
  - `ptn_format.dumps()` / `ptn_format.dump()`: Serialize patterns to text
  - `ptn_format.loads()` / `ptn_format.load()`: Deserialize patterns from text
  - Format separates quantum instructions and classical feedforward processing
  - Timeslice markers `[n]` indicate parallel execution groups
  - Pauli measurements use compact notation (`X +`, `Y -`, `Z +`)
  - Non-Pauli measurements use plane+angle format (`XY pi/4`)
  - Support for node coordinates and inline comments
- **Non-Unitary Parity Projection Example**: Added `examples/nonunitary_parity_projection.py` demonstrating measurement-induced entanglement via a 3-node star graph parity projector

### Fixed

- **Qompiler**: `qompile()` now validates a provided scheduler before pattern generation, so invalid manual schedules fail early with `ValueError`.

## [0.3.0] - 2026-04-08

### Added

- **Noise Model Module**: Added `graphqomb.noise_model` for event-driven noise injection during Stim compilation
  - Added `NoiseModel` hooks for `on_prepare`, `on_entangle`, `on_measure`, and `on_idle`
  - Added frozen, validated `NoiseOp` dataclasses: `PauliChannel1`, `PauliChannel2`, `HeraldedPauliChannel1`, `HeraldedErase`, `RawStimOp`, `MeasurementFlip`
  - Added event dataclasses `PrepareEvent`, `EntangleEvent`, `MeasureEvent`, `IdleEvent`, plus `NodeInfo` and `Coordinate`
  - Added `NoisePlacement`, `noise_op_to_stim()`, `depolarize1_probs()`, and `depolarize2_probs()`

- **Built-in Noise Models**: Ready-to-use noise model implementations
  - Added `DepolarizingNoiseModel` for single and two-qubit depolarizing noise
  - Added `MeasurementFlipNoiseModel` for measurement bit-flip errors using Stim's built-in `MX(p)` syntax

- **Stim Compiler Noise Integration**: Added noise-model-driven Stim compilation
  - Support for multiple noise models via `Sequence[NoiseModel]`
  - Added `tick_duration` parameter for idle noise calculations
  - Automatic measurement record tracking for heralded noise operations when emitting detectors and observables

- **Greedy Scheduler**: Fast greedy scheduling algorithms as an alternative to CP-SAT optimization
  - Added `greedy_minimize_time()` for minimal execution time scheduling with ALAP preparation optimization
  - Added `greedy_minimize_space()` for minimal qubit usage scheduling

- **Schedule Solver**: Added constraint that every non-input, non-output node must be prepared strictly before it is measured (`node2prep[node] < node2meas[node]`)

- **Circuit Conversion**: Added circuit-derived pre-scheduling support in `circuit2graph()`.
  - Added `CircuitScheduleStrategy` with `PARALLEL` and `MINIMIZE_SPACE`.
  - Added `schedule_strategy` argument to `circuit2graph()`.
  - `circuit2graph()` now returns `(graph, gflow, scheduler)` and pre-populates `Scheduler` via manual scheduling.
- **PyZX Integration**: Added optional `graphqomb.zx_util` utilities for importing strict graph-like PyZX diagrams into `GraphState`.
  - Added `from_pyzx()` to convert PyZX diagrams into a `GraphState`.
  - Added boundary rewriting and metadata import helpers to preserve graph structure, measurement bases, and coordinates during conversion.
  - Added optional phase-gadget recognition for supported lone-`Z` gadget patterns via `recognize_pg=True`, importing the adjacent node as a `YZ`-plane measurement.

- **Documentation**: Added comprehensive Sphinx documentation for the noise model module

### Changed

- **Stim Compiler API**: `stim_compile()` now has signature `stim_compile(pattern, *, emit_qubit_coords=True, noise_models=None, tick_duration=1.0)`
- **Stim Compiler**: Refactored internal structure to support event-driven noise model integration
- **Measurement Flip Semantics**: `MeasurementFlipNoiseModel` and custom `MeasurementFlip` ops now compile to Stim's native `MX(p)` / `MY(p)` / `MZ(p)` instructions instead of emitting separate Pauli error instructions
- **Noise Extension API**: `NoiseOp` values are now represented as plain frozen dataclasses collected under the `NoiseOp` union, improving type safety for custom `NoiseModel` implementations
- **Noise Validation**: Centralized noise parameter validation in `noise_model`
  - `NoiseOp` dataclasses now validate and normalize their inputs at construction time
  - `DepolarizingNoiseModel` and `MeasurementFlipNoiseModel` now reject invalid probabilities when instantiated
  - `MeasurementFlip` is now enforced as a measurement-only noise operation during Stim compilation
- **Graph State**: Made `meas_bases` read-only by returning `MappingProxyType` to avoid external mutation.
- **Graph State**: Added caching for `physical_nodes` snapshots and proper cache invalidation on node add/remove.
- **Docs/Examples**: Updated circuit conversion usage in README and `examples/pattern_from_circuit.py` for the new `circuit2graph()` return signature.
- **Packaging/Docs**: Added the optional `graphqomb[pyzx]` extra, documented PyZX installation in the README, and published Sphinx API reference pages for `graphqomb.zx_util`.
- **CI**: Split PyZX-marked tests into a dedicated GitHub Actions job and installed the optional dependency in coverage runs.

### Fixed

- **Stim Compiler**: Detector and observable record indices now stay aligned when noise models emit heralded instructions that add measurement records
- **Pattern Simulator**: Fixed adaptive measurement conjugation so non-Pauli measurements apply the missing angle sign flip from the Pauli frame during simulation ([#139](https://github.com/TeamGraphix/graphqomb/issues/139))
- **Feedforward**: Fixed operator precedence bug in `dag_from_flow` where self-loops were only removed from `zflow` but not from `xflow`. The expression `xflow | zflow - {node}` was evaluated as `xflow | (zflow - {node})` due to `-` binding tighter than `|`. Corrected to `(xflow | zflow) - {node}`.

### Tests

- **Noise Model / Stim Compiler**: Added comprehensive tests for `graphqomb.noise_model` and noise-aware `stim_compile()`, including heralded record tracking, `MeasurementFlip` validation, and removed legacy kwargs
- **Greedy Scheduler**: Added tests for greedy scheduling algorithms
- **Schedule Solver**: Added integration test verifying that CP-SAT MINIMIZE_SPACE strategy enforces node preparation before measurement
- **Circuit Conversion**: Expanded scheduling tests in `tests/test_circuit.py`, including scheduler return contract, J/CZ/phase-gadget timing behavior, schedule validation, and `MINIMIZE_SPACE` behavior.
- **Integration**: Added circuit-level integration tests for `signal_shifting()` and `pauli_simplification()` with circuit-vs-pattern statevector equivalence checks.
- **Pattern Simulator / Measurement Bases**: Added regression tests for measurement-basis conjugation semantics and adaptive simulation of non-Pauli measurements affected by the Pauli frame.
- **Stim Compiler / Pauli Frame**: Updated tests to explicitly pass parity-check groups where logical-observable and cache initialization paths are exercised.
- **PyZX Integration**: Added unit tests for vertex/edge collection, boundary rewrites, lone-spider phase-gadget recognition, and end-to-end `from_pyzx()` conversion behavior.

### Removed

- **Stim Compiler Legacy Noise Args**: Removed `p_depol_after_clifford` and `p_before_meas_flip` from `stim_compile()`
  - Use `noise_models=[DepolarizingNoiseModel(...), MeasurementFlipNoiseModel(...)]` instead

## [0.2.1] - 2026-01-16

### Added

- **Type Hints**: Added `py.typed` marker for PEP 561 compliance, enabling type checkers (mypy, pyright) to recognize the package as typed when installed from PyPI.

### Changed

- **Python Support**: Dropped Python 3.9 support, added Python 3.14 support. Now requires Python >=3.10, <3.15.

## [0.2.0] - 2025-12-26

### Added

- **TICK Command**: Time slice boundary marker for temporal scheduling in MBQC patterns
  - Added TICK command type to mark boundaries between time slices
  - Integrated TICK command handling in PatternSimulator
  - Integrated TICK command processing in Stim compiler

- **Edge Scheduler**: Automatic entanglement operation scheduling based on node preparation times ([#99](https://github.com/TeamGraphix/graphqomb/issues/99))
  - Added `entangle_time` attribute to Scheduler for tracking entanglement operation timing
  - Added `auto_schedule_entanglement()` method to automatically schedule CZ gates when both nodes are prepared
  - Extended the `timeline` property to include entanglement operations
  - Added entanglement time validation in schedule validation
  - Added `compress_schedule()` function to support entanglement time compression

- **Pattern**: Added the `depth` attribute into `Pattern`, which represents the depth of parallel execution.

- **Pattern**: Added pattern resource/throughput metrics (`active_volume`, `volume`, `idle_times`, `throughput`).

- **Scheduler Integration**: Enhanced qompile() to support temporal scheduling with TICK commands
  - Added `scheduler` parameter to qompile() for custom scheduling
  - Automatically inserts TICK commands between time slices

- **Examples**: Added entanglement_scheduling_demo.py demonstrating edge scheduler features

- **Feedforward Optimization**: Added a `signal_shifting` method as a feedforward optimization.
  - This optimization is equivalent to the operation of the same name in the measurement calculus, and makes the measurement pattern as parallel as possible.
  - The optimization is now self-contained within the feedforward module.

- **Feedforward Optimization**: Added `pauli_simplification()` to remove redundant Pauli corrections in correction maps when measuring in Pauli bases.

### Changed

- **Pattern**: Updated command sequence generation to support TICK commands
- **Command**: Extended Command type alias to include TICK
- The default strategy of `Scheduler.solve_schedule` is now `MINIMIZE_TIME` instead of `MINIMIZE_SPACE` for the compilation performance.

### Fixed

- **Scheduler**: Accept `entangle_time` edges in either order in `Scheduler.manual_schedule()`.

### Tests

- **Stim Compiler**: Add coverage that manual `entangle_time` determines CZ time slices in both Pattern and Stim output.

## [0.1.2] - 2025-10-31

### Added

- **Graph State**: Bulk initialization methods for GraphState ([#120](https://github.com/TeamGraphix/graphqomb/issues/120))
  - Added `from_graph()` class method for direct graph-based initialization
  - Added `from_base_graph_state()` class method for initialization from base GraphState objects
  - Improved initialization flexibility for diverse use cases

### Performance

- **Pauli Frame**: Optimized `_collect_dependent_chain` method with memoization and caching
  - Added Pauli axis cache to avoid redundant basis computations
  - Implemented chain memoization cache to prevent recalculating dependent chains
  - Optimized set operations for better performance in large graph states

### Tests

- **TICK Command**: Added comprehensive test suite for TICK command functionality
  - Added `test_simulator_with_tick_commands()` for TICK command handling in PatternSimulator
  - Added `test_stim_compile_with_tick_commands()` for TICK command compilation to Stim format
  - Extended scheduler integration tests with comprehensive edge scheduling validation

- **Pauli Frame**: Added comprehensive test suite for PauliFrame module
  - Added tests for basic methods (x_flip, z_flip, meas_flip, children, parents)
  - Added tests for Pauli axis cache initialization and chain cache memoization
  - Added tests for dependent chain collection across X, Y, Z measurement axes
  - Added tests for detector groups and logical observables
  - Improved test coverage from 77.78% to 97% for pauli_frame.py
- **Graph State**: Added comprehensive test suite for bulk initialization methods
  - Added tests for `from_graph()` initialization
  - Added tests for `from_base_graph_state()` initialization
  - Added tests for graph consistency and state equivalence

## [0.1.1] - 2025-10-23

### Added

- **Stim Compiler**: Pattern to Stim circuit compiler with detector and observable support for fault-tolerant quantum computing ([#67](https://github.com/TeamGraphix/graphqomb/issues/67))
  - Compile MBQC patterns into Stim format for error correction analysis
  - Support for detectors, observables, and error models
  - Configurable depolarization noise after Clifford gates and measurements

### Changed

- **Pauli Frame**: Extended with detector and syndrome analysis capabilities
  - Added `detector_groups` for detector grouping
  - Added `syndrome_parity_group` for syndrome extraction
  - Added parity check grouping for X and Z corrections

### Fixed

- Fixed inverse flow construction to avoid self-loops
- Fixed type hints in `graphstate.compose` for better type safety

## [0.1.0] - 2025-10-22

### Added

- **Core Infrastructure**: Initial repository setup with project structure, build system, and CI/CD workflows ([#12](https://github.com/TeamGraphix/graphqomb/pull/12))
- **Mathematical Foundations**: Euler angle computations for quantum operations ([#24](https://github.com/TeamGraphix/graphqomb/pull/24))
- **Graph State**: Graph state representation and manipulation ([#34](https://github.com/TeamGraphix/graphqomb/pull/34))
- **Pattern Module**: MBQC pattern data structures and operations ([#47](https://github.com/TeamGraphix/graphqomb/pull/47))
- **Feedforward System**: Feedforward strategy design and implementation for adaptive measurements ([#40](https://github.com/TeamGraphix/graphqomb/pull/40))
- **Command Module**: Measurement command definitions ([#43](https://github.com/TeamGraphix/graphqomb/pull/43))
- **Qompiler**: Pattern compiler with Pauli frame implementation ([#55](https://github.com/TeamGraphix/graphqomb/pull/55))
- **Scheduler**: Prepare time and measurement time scheduling for efficient execution ([#74](https://github.com/TeamGraphix/graphqomb/pull/74))
- **Circuit Framework**: Quantum gate definitions and circuit representation ([#73](https://github.com/TeamGraphix/graphqomb/pull/73))
- **Simulation Backend**: Statevector simulator backend for quantum state evolution ([#62](https://github.com/TeamGraphix/graphqomb/pull/62))
- **Pattern and Circuit Simulation**: Complete simulation support for both patterns and circuits ([#78](https://github.com/TeamGraphix/graphqomb/pull/78))
- **Visualization**: Basic visualizer for graph states and patterns ([#83](https://github.com/TeamGraphix/graphqomb/pull/83))
- **Documentation**: Comprehensive documentation on Read the Docs (https://graphqomb.readthedocs.io/)
