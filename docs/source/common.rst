Common
========

:mod:`graphix_zx.common` module
++++++++++++++++++++++++++++++++

.. automodule:: graphix_zx.common

Enums
-----

.. autoclass:: graphix_zx.common.Plane
    :show-inheritance:

    .. py:attribute:: XY

       Arbitrary-angle measurement on the XY plane.

    .. py:attribute:: YZ
        
       Arbitrary-angle measurement on the YZ plane.

    .. py:attribute:: XZ
        
       Arbitrary-angle measurement on the XZ plane.

.. autoclass:: graphix_zx.common.Axis
    :show-inheritance:

    .. py:attribute:: X

        Pauli X-axis.

    .. py:attribute:: Y
        
        Pauli Y-axis.

    .. py:attribute:: Z
        
        Pauli Z-axis.

.. autoclass:: graphix_zx.common.Sign
    :show-inheritance:

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
