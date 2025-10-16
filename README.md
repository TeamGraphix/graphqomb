# GraphQOMB

![License](https://img.shields.io/github/license/TeamGraphix/graphqomb)
[![doc](https://github.com/TeamGraphix/graphqomb/actions/workflows/doc.yml/badge.svg)](https://github.com/TeamGraphix/graphqomb/actions/workflows/doc.yml)
[![pytest](https://github.com/TeamGraphix/graphqomb/actions/workflows/pytest.yml/badge.svg)](https://github.com/TeamGraphix/graphqomb/actions/workflows/pytest.yml)
[![typecheck](https://github.com/TeamGraphix/graphqomb/actions/workflows/typecheck.yml/badge.svg)](https://github.com/TeamGraphix/graphqomb/actions/workflows/typecheck.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**GraphQOMB** (Qompiler for Measurement-Based Quantum Computing) is a comprehensive reimplementation of the [graphix](https://github.com/TeamGraphix/graphix). Our package compiles ZX-diagrams with optional feedforward strategy into measurement patterns with Pauli frame tracking.

## Features

### Computation Design

- **ZX-Calculus Integration**: Use ZX-diagrams as an abstract expression of measurement pattern
- **Feedforward Strategy Design**: Our library accepts general feedforward strategy and optimization, eliminating the necessity of measurement calculus
- **Scheduler**: Scheduling the node preparation and measurement time

### Compilation

- **MBQC Pattern Generation**: Measurement pattern is treated as a quantum assembly
- **Pauli Frame Tracking**: Manage all the classical feedforward with Pauli Frame, enabling fault-tolerant computing as well

### Simulation

- **Pattern Simulation**: Simulate measurement patterns with statevector backend
- **Simulation in Stim circuit**: Generate a stim circuit implementing a fault-tolerant MBQC

### Others

- **Transpilation into Graphix Pattern**: Transpile generated pattern into `graphix.pattern.Pattern` object for variety of execution backend (WIP)
- **Visualization**: Visualize graph states

## Installation

Install GraphQOMB in development mode:

> NOTE: We will soon release the package on PyPI

```bash
cd graphqomb/
pip install -e .
```

Install with development dependencies:

```bash
pip install -e .[dev]
```

Install with documentation dependencies:

```bash
pip install -e .[doc]
```

## Quick Start

### Prepare Resource State and Feedforward

```python
from graphqomb.circuit import Circuit, circuit2graph
from graphqomb.gates import H, CNOT
from graphqomb.qompiler import qompile
from graphqomb.simulator import PatternSimulator, SimulatorBackend

# Create a quantum circuit
circuit = Circuit(2)
circuit.apply_macro_gate(H(0))
circuit.apply_macro_gate(CNOT((0, 1)))

graph, feedforward = circuit2graph(circuit)

# Compile into pattern
pattern = qompile(graph, feedforward)

# Simulate the pattern
simulator = PatternSimulator(pattern, SimulatorBackend.StateVector)
simulator.simulate()
print(simulator.state)
```

### Creating and Visualizing Graph States

```python
from graphqomb.graphstate import GraphState
from graphqomb.visualizer import visualize

# Create a graph state
graph = GraphState()
node1 = graph.add_physical_node()
node2 = graph.add_physical_node()
node3 = graph.add_physical_node()

# Register input/output nodes
q_index = 0
graph.register_input(node1, q_index)
graph.register_output(node3, q_index)

# Add edges
graph.add_physical_edge(node1, node2)
graph.add_physical_edge(node2, node3)

# Visualize the graph
visualize(graph)
```

## Documentation

- **Tutorial**: [WIP] for detailed usage guides
- **Examples**: See [WIP] for code demonstrations
- **API Reference**: Full API documentation is available [WIP]

## Development

### Running Tests

```bash
pytest                              # Run all tests
pytest tests/test_specific.py       # Run specific test file
```

### Code Quality

```bash
ruff check                          # Lint code
ruff format                         # Format code
mypy                                # Type checking
pyright                             # Type checking
```

### Building Documentation

```bash
cd docs/
make html                           # Build HTML documentation
# Output will be in docs/build/html/
```

## Project Structure

```
graphqomb/
├── graphqomb/               # Main source code
│   ├── circuit.py           # Quantum circuit implementation
│   ├── graphstate.py        # Graph state manipulation
|   ├── scheduler.py         # Scheduling computaional order
│   ├── qompiler.py          # Generate MBQC pattern
│   ├── simulator.py         # Pattern simulation
│   ├── visualizer.py        # Visualization tools
│   └── ...
├── tests/                   # Test suite
├── examples/                # Example scripts
├── docs/                    # Sphinx documentation
│   └── source/
│       ├── gallery/         # Example gallery
│       └── ...
└── pyproject.toml          # Project configuration
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass and code is properly formatted
5. Submit a pull request

## Related Projects

- [graphix](https://github.com/TeamGraphix/graphix): The original MBQC library
- [PyZX](https://github.com/Quantomatic/pyzx): ZX-calculus library for Python
- [swiflow](https://github.com/TeamGraphix/swiflow): Rust-based fast flow finding algorithms

## License

[MIT License](LICENSE)

## Citation

If you use GraphQOMB in your research, please cite:

```bibtex
@software{graphqomb,
  title = {GraphQOMB: A Modular Graph State Qompiler for Measurement-Based Quantum Computation},
  author = {Masato Fukushima, Yuki Watanabe, and Daichi Sasaki},
  year = {2025},
  url = {https://github.com/TeamGraphix/graphqomb}
}
```

## Acknowledgements

We acknowledge the [NICT Quantum Camp](https://nqc.nict.go.jp/) for supporting our development.

Special thanks to Fixstars Amplify:

<p><a href="https://amplify.fixstars.com/en/">
<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/imgs/fam_logo.png" alt="amplify" width="200"/>
</a></p>
