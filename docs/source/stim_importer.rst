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
graph node. They do not create an ``MPP`` extraction or an ancillary parity
measurement node. Inverted single-qubit measurement targets select the minus
sign of that node's basis. A direct single-qubit measurement terminates that
qubit's lifetime: a later quantum operation on the same qubit is rejected,
while operations on other qubits may continue. A measured qubit cannot begin a
new lifetime later in the circuit.

Two-qubit measurements are parity measurements and are lowered to equivalent
unsigned ``MPP`` products. Inverted targets in ``MXX``, ``MYY``, ``MZZ``, and
``MPP`` are rejected because GraphQOMB does not currently retain the
corresponding parity offset.

All ``MPP`` instructions within one ``TICK`` block are represented by one
combined extraction and are validated to commute. Anticommuting products in
the same block are rejected. The importer uses one compact
stabilizer-measurement unit when its correction flow is causal. If the compact
unit has a cyclic Pauli flow, the commuting products are lowered to equivalent
sequential units instead. Non-commuting measurements must be separated by
``TICK`` in the source circuit. Each unit has a distinct unmeasured output
layer, which is composed with the next unitary or measurement fragment by qubit
index. Pass ``y_foliation=YFoliation.TYPE_II`` to any of the three import entry
points to use the three-layer Y-measurement construction; Type I is the default.

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
