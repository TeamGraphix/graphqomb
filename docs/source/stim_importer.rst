Stim circuit import
===================

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

The circuit importer converts supported Stim circuits into GraphQOMB
measurement patterns. It accepts initial Pauli resets, Clifford unitary blocks,
and Pauli measurement blocks separated by ``TICK``. The supported Pauli
measurement instructions are ``M``/``MZ``, ``MX``, ``MY``, ``MXX``, ``MYY``,
``MZZ``, and ``MPP``.

Initial reset instructions
--------------------------

Leading reset instructions determine the positive Pauli eigenstate used for an
input: ``R``/``RZ`` initializes ``Z+``, ``RX`` initializes ``X+``, and ``RY``
initializes ``Y+``. A reset is initial when its target qubit has not previously
participated in a unitary or measurement operation. Operations on other qubits
do not prevent a later initial reset. Repeated leading resets are accepted, and
the last reset on each qubit determines its initialization state.

Mid-circuit resets are rejected because GraphQOMB patterns do not currently
represent multiple lifetimes for one logical qubit. Combined measurement-reset
instructions (``MR``/``MRZ``, ``MRX``, and ``MRY``) remain unsupported.

Single-qubit measurements assign an ``AxisMeasBasis`` directly to the measured
data-lane endpoint without replacing that node or its coordinate. They do not
create an ``MPP`` extraction or an ancillary parity measurement node. Inverted
single-qubit measurement targets select the minus sign of that node's basis. A
direct single-qubit measurement terminates that qubit's lifetime: a later
quantum operation on the same qubit is rejected, while operations on other
qubits may continue. A measured qubit cannot begin a new lifetime later in the
circuit.

The first two components of ``QUBIT_COORDS`` are used as the fixed spatial
``(x, y)`` position of each data lane. The importer supplies the temporal ``z``
component. Every unitary ``TICK`` block is transpiled before placement. Its
input layer starts at the preceding block's output ``z``, and all of its output
nodes share the maximum transpiled depth of the block. A shorter data-wire
chain is spread across that same interval. A live qubit with no operation in
the block remains a single input/output node and is relocated directly to the
common output layer; this adds no graph node or edge and does not change the
circuit semantics.

Two-qubit measurements are parity measurements and are lowered to equivalent
unsigned ``MPP`` products. Inverted targets in ``MXX``, ``MYY``, ``MZZ``, and
``MPP`` are rejected because GraphQOMB does not currently retain the
corresponding parity offset.

All ``MPP`` instructions within one ``TICK`` block are represented by one
combined extraction and are validated to commute. Anticommuting products in
the same block are rejected. Within a combined block, local stabilizer
interactions are ordered ``Z -> Y -> X`` on each shared data qubit. If an odd
number of shared-data-qubit pairs reverse the order of the same two
stabilizers, the graph-state builder adds the required CZ edge between their
ancillas. This rule is applied automatically for both Type I and Type II
foliation.

Only data-wire nodes contribute to the X-correction flow; MPP ancilla nodes do
not produce X corrections. After composing all graph fragments, the importer
derives the Z-correction flow from the odd neighborhood of the complete
X-correction flow. Both correction maps are passed directly to ``qompile()``
without Pauli simplification or an importer-specific fallback. Non-commuting
measurements must be separated by ``TICK`` in the source circuit. Each unit has
a distinct unmeasured output layer, which is composed with the next unitary or
measurement fragment by qubit index. Pass
``y_foliation=YFoliation.TYPE_II`` to any of the three import entry points to use
the three-layer Y-measurement construction; Type I is the default.

An MPP block starts at the preceding gate or MPP output layer and ends two
``z`` units later. Live lanes not used by that MPP block are relocated to the
same output layer without adding nodes. Consequently, a composed output and
the next active fragment input have the same ``z`` coordinate, and imported
patterns do not mix 2D spatial coordinates with 3D spacetime coordinates.

The flattened ideal circuit is analyzed once for measurement records. Records
from single-qubit measurements, pair measurements, ``MPP``, and ideal-zero
``MPAD`` replacements share one absolute index space. ``DETECTOR`` and
``OBSERVABLE_INCLUDE`` targets are resolved against that whole-circuit index
space, and each ``StimMppExtraction`` retains the corresponding absolute record
sets even when an annotation also references records outside that MPP unit.

Noise policy
------------

Circuit-level Stim noise is intentionally not imported. GraphQOMB applies
noise to the compiled MBQC pattern through its own noise-model API, where
preparation, entanglement, measurement, and idle events differ from the source
circuit operations.

Pure noise instructions are omitted during import. Error probabilities attached
to Pauli measurements are also omitted while retaining the ideal measurement.
Heralded noise records are retained as ideal zero-valued record positions so
that later ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` references remain aligned.

.. code-block:: python

   from graphqomb.qec.qeccode import YFoliation
   from graphqomb.stim_importer import stim_text_to_pattern

   result = stim_text_to_pattern(
       """
       RY 0
       R 1 2
       MX 0
       MYY 1 2
       DETECTOR rec[-2] rec[-1]
       """,
       y_foliation=YFoliation.TYPE_II,
   )
   pattern = result.pattern

API reference
-------------

.. automodule:: graphqomb.stim_importer
   :members:
   :show-inheritance:
