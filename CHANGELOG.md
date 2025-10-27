# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
