# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **TICK Command**: Time slice boundary marker for temporal scheduling in MBQC patterns
  - Added TICK command type to mark boundaries between time slices
  - Integrated TICK command handling in PatternSimulator
  - Integrated TICK command processing in Stim compiler

- **Edge Scheduler**: Automatic entanglement operation scheduling based on node preparation times ([#99](https://github.com/TeamGraphix/graphqomb/issues/99))
  - Added `entangle_time` attribute to Scheduler for tracking entanglement operation timing
  - Added `auto_schedule_entanglement()` method to automatically schedule CZ gates when both nodes are prepared
  - Added `detailed_timeline` property for comprehensive scheduling information including entanglement operations
  - Added entanglement time validation in schedule validation
  - Added `compress_schedule()` function to support entanglement time compression

- **Scheduler Integration**: Enhanced qompile() to support temporal scheduling with TICK commands
  - Added `scheduler` parameter to qompile() for custom scheduling
  - Added `insert_tick` parameter to enable TICK command insertion between time slices
  - Automatically inserts TICK commands between time slices when scheduler is provided

- **Examples**: Added entanglement_scheduling_demo.py demonstrating edge scheduler features

### Changed

- **Pattern**: Updated command sequence generation to support TICK commands
- **Command**: Extended Command type alias to include TICK
- **Qompiler**: Modified E command generation to explicitly unpack frozenset edges for type safety

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
