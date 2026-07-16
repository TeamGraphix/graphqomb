Pattern Text Format
===================

GraphQOMB writes pattern files using format version 2. Version 2 adds optional
per-input positive Pauli eigenstate initialization:

.. code-block:: text

   .version 2
   .input 0:0 1:1 2:2
   .input_basis 1:Y 2:Z

Each ``.input_basis`` entry has the form ``node:X``, ``node:Y``, or ``node:Z``
and must reference a node declared by ``.input``. ``X`` initialization is the
default and is omitted when serializing, so the example initializes nodes 0,
1, and 2 as ``X+``, ``Y+``, and ``Z+`` respectively.

Version 1 files remain readable. Inputs in a version 1 file, or in a version 2
file without a corresponding ``.input_basis`` entry, are initialized as
``X+``. The ``.input_basis`` directive is rejected in version 1 files.

:mod:`graphqomb.ptn_format` module
++++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.ptn_format
   :members:
   :member-order: bysource
