Common
========

:mod:`graphix_zx.common` module
++++++++++++++++++++++++++++++++

.. automodule:: graphix_zx.common

Enums
-----

.. class:: graphix_zx.common.Plane

    Measurement planes for MBQC.

    We distinguish the axial measurements from the planar measurements.

    .. py:attribute:: XY

       Arbitrary-angle measurement on the XY plane.

    .. py:attribute:: YZ
        
       Arbitrary-angle measurement on the YZ plane.

    .. py:attribute:: XZ
        
       Arbitrary-angle measurement on the XZ plane.

.. class:: graphix_zx.common.Axis
    
    Measurement axis for Pauli measurement.

    .. py:attribute:: X

        Pauli X-axis.

    .. py:attribute:: Y
        
        Pauli Y-axis.

    .. py:attribute:: Z
        
        Pauli Z-axis.

.. class:: graphix_zx.common.Sign

    Measurement sign for Pauli measurement.

    .. py:attribute:: PLUS

      Positive sign.

    .. py:attribute:: MINUS
        
      Negative sign.

Abstract Base Classes
---------------------

.. autoclass:: graphix_zx.common.MeasBasis
   :members:
   :undoc-members:
   :show-inheritance:

Measurement Basis Classes
-------------------------

.. autoclass:: graphix_zx.common.PlannerMeasBasis
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: graphix_zx.common.AxisMeasBasis
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: graphix_zx.common.default_meas_basis

.. autofunction:: graphix_zx.common.meas_basis
