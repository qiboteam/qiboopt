Amplitude estimation
====================

Iterative (IQAE) and maximum-likelihood (MLAE) quantum amplitude estimation. Both estimate
the probability :math:`a` of measuring a *good* state of :math:`\mathcal A|0\rangle` using
powers of the Grover operator :math:`\mathcal Q`, without phase estimation.

A typical workflow combines a loader, a
:class:`~qiboopt.circuit_library.LinearAmplitudeFunction` and an
:class:`~qiboopt.amplitude_estimation.EstimationProblem`:

.. code-block:: python

    from qibo import Circuit
    from qiboopt.amplitude_estimation import EstimationProblem, IterativeAmplitudeEstimation
    from qiboopt.circuit_library import LinearAmplitudeFunction, NormalDistribution

    n = 3
    loader = NormalDistribution(n, mu=0.5, sigma=0.3, bounds=(-1, 2))
    F = LinearAmplitudeFunction(
        n, slope=[0, 1], offset=[0, 0], domain=(-1, 2), image=(0, 1.6),
        rescaling_factor=0.05, breakpoints=[-1, 0.4],
    )
    circ = Circuit(F.num_qubits)
    circ.add(loader.on_qubits(*range(n)))
    circ.add(F.on_qubits(*range(F.num_qubits)))

    problem = EstimationProblem(circ, objective_qubits=[F.objective_qubit],
                                post_processing=F.post_processing)
    result = IterativeAmplitudeEstimation(epsilon_target=0.005, alpha=0.05).estimate(problem)
    print(result.estimation_processed, result.confidence_interval_processed)

Problem definition
------------------

.. autoclass:: qiboopt.amplitude_estimation.EstimationProblem
    :members:
    :member-order: bysource

Grover operator
---------------

.. autofunction:: qiboopt.amplitude_estimation.grover_operator

.. autofunction:: qiboopt.amplitude_estimation.construct_amplitude_circuit

.. autofunction:: qiboopt.amplitude_estimation.oracle_gates

.. autofunction:: qiboopt.amplitude_estimation.zero_reflection_gates

Iterative amplitude estimation
------------------------------

.. autoclass:: qiboopt.amplitude_estimation.IterativeAmplitudeEstimation
    :members:
    :member-order: bysource

.. autoclass:: qiboopt.amplitude_estimation.IterativeAmplitudeEstimationResult
    :members:

Maximum likelihood amplitude estimation
---------------------------------------

.. autoclass:: qiboopt.amplitude_estimation.MaximumLikelihoodAmplitudeEstimation
    :members:
    :member-order: bysource

.. autoclass:: qiboopt.amplitude_estimation.MaximumLikelihoodAmplitudeEstimationResult
    :members:
