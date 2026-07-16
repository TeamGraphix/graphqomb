QEC graph-state builder
=======================

``graphqomb.qec.qeccode`` provides the stabilizer-code representation and the
Type I and Type II graph-state builders used by the QEC helper workflow.

Coordinate tuples are retained without restricting their dimensionality. The
graph-state builder uses the first two qubit-coordinate components for the data
plane and the first three explicitly supplied stabilizer-coordinate components
for ancilla placement.

With ``data_as_io=True``, the stabilizer-measurement unit has a separate,
unmeasured output layer. Type I therefore has two measured data layers followed
by an output layer. Type II uses three Y-measured layers for qubits with Y
support, followed by an output layer; qubits without Y support retain the Type I
two-measurement-layer layout. Ancilla support edges only touch measurement
layers, and the output of one composed unit becomes the input of the next.

.. automodule:: graphqomb.qec.qeccode
   :members:
   :show-inheritance:
