Feedforward
===========

:mod:`graphix_zx.feedforward` module
++++++++++++++++++++++++++++++++++++

.. automodule:: graphix_zx.feedforward

Type Aliases
------------

.. py:data:: Flow
   :type: Mapping[int, int]

.. py:data:: GFlow
   :type: Mapping[int, AbstractSet[int]]

.. py:data:: FlowLike
   :type: Flow | GFlow

Functions
---------

.. autofunction:: graphix_zx.feedforward.is_flow

.. autofunction:: graphix_zx.feedforward.is_gflow

.. autofunction:: graphix_zx.feedforward.dag_from_flow

.. autofunction:: graphix_zx.feedforward.check_causality
