Common
========

:mod:`graphqomb.common` module
++++++++++++++++++++++++++++++++

.. automodule:: graphqomb.common

Enums
-----

.. class:: graphqomb.common.Plane

    Measurement planes for MBQC.

    We distinguish the axial measurements from the planar measurements.

    .. py:attribute:: XY

       Arbitrary-angle measurement on the XY plane.

    .. py:attribute:: YZ

       Arbitrary-angle measurement on the YZ plane.

    .. py:attribute:: XZ

       Arbitrary-angle measurement on the XZ plane.

.. class:: graphqomb.common.Axis

    Measurement axis for Pauli measurement.

    .. py:attribute:: X

        Pauli X-axis.

    .. py:attribute:: Y

        Pauli Y-axis.

    .. py:attribute:: Z

        Pauli Z-axis.

.. class:: graphqomb.common.Sign

    Measurement sign for Pauli measurement.

    .. py:attribute:: PLUS

      Positive sign.

    .. py:attribute:: MINUS

      Negative sign.

Abstract Base Classes
---------------------

.. autoclass:: graphqomb.common.MeasBasis
   :members:
   :undoc-members:
   :show-inheritance:

Measurement Basis Classes
-------------------------

.. autoclass:: graphqomb.common.PlannerMeasBasis
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: graphqomb.common.AxisMeasBasis
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: graphqomb.common.is_close_angle

.. autofunction:: graphqomb.common.is_clifford_angle

.. autofunction:: graphqomb.common.determine_pauli_axis

.. autofunction:: graphqomb.common.default_meas_basis

.. autofunction:: graphqomb.common.meas_basis
