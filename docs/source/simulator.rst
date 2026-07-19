Statevector Simulator
=====================

``PatternSimulator`` samples measurement results from their exact Born
probabilities by default. This is required when inputs use ``Y+`` or ``Z+``
initialization, because non-output measurements are not necessarily uniformly
random.

For compatibility with the previous faster approximation, pass
``calc_prob=False``. In that mode, non-output measurements are sampled 50/50;
output measurements still use their exact probabilities. This approximation is
only appropriate when the pattern guarantees uniformly random non-output
measurements.

.. automodule:: graphqomb.simulator
   :members:
   :show-inheritance:
