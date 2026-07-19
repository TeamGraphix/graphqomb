Stim Clifford parser
====================

Install the optional Stim integration before importing this module:

.. code-block:: console

   uv add "graphqomb[stim]"

The parser normalizes Stim Clifford circuits to GraphQOMB's unitary basis:
``H``, ``HS``, and ``CZ``. GraphQOMB defines
``HS = H S = J(pi / 2)`` in algebraic matrix order, so one ``HS`` becomes one
:class:`graphqomb.gates.J` primitive. Stim's native spelling for this Clifford
gate is ``C_XNYZ``.

.. code-block:: python

   from graphqomb.stim_parser import transpile

   normalized = transpile("S_DAG 0\nCX 0 1", optimize=True)

The parser supports all fixed one- and two-qubit Clifford gates exposed by
Stim, arbitrary-length ``SPP`` and ``SPP_DAG`` rotations, nested ``REPEAT``
blocks, Pauli reset and measurement boundaries, and coordinate/TICK
annotations. Noise, non-Clifford instructions, and classically controlled
targets are rejected.

The Stim importer applies this normalization independently to each unitary
``TICK`` block. Measurement records, detectors, observables, MPP operations,
and GraphQOMB's circuit-level noise policy remain the responsibility of the
importer.

API reference
-------------

.. automodule:: graphqomb.stim_parser
   :members:
   :show-inheritance:
