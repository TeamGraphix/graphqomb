Welcome to GraphQOMB's documentation!
======================================

GraphQOMB is a compiler framework for measurement-based quantum computation (MBQC). It keeps the resource-state structure, classical feedforward, and execution schedule as separate first-class objects, then lowers them into executable patterns with explicit schedule boundaries and Pauli-frame tracking.

Start here
----------

- :doc:`getting_started` for the main circuit-to-pattern workflow.
- :doc:`architecture` for the compiler model and lowering semantics.
- :doc:`gallery/index` for runnable examples.
- :doc:`references` for the full API reference.

Core workflow
-------------

1. Build or import a labelled graph state.
2. Provide explicit feedforward maps.
3. Optionally solve or inject a scheduler.
4. Lower the IR objects with :func:`graphqomb.qompiler.qompile`.
5. Simulate or export the resulting pattern.

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   getting_started
   architecture
   gallery/index
   references

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
