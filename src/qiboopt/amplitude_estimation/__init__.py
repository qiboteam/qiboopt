"""Amplitude estimation: problem container, Grover operator, IQAE and MLAE.

These are Qibo implementations of the Qiskit ``EstimationProblem``,
``IterativeAmplitudeEstimation`` and ``MaximumLikelihoodAmplitudeEstimation`` primitives.
"""

from qiboopt.amplitude_estimation.estimation_problem import EstimationProblem
from qiboopt.amplitude_estimation.grover import (
    construct_amplitude_circuit,
    grover_operator,
    oracle_gates,
    zero_reflection_gates,
)
from qiboopt.amplitude_estimation.iqae import (
    IterativeAmplitudeEstimation,
    IterativeAmplitudeEstimationResult,
)
from qiboopt.amplitude_estimation.mlae import (
    MaximumLikelihoodAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimationResult,
)

__all__ = [
    "EstimationProblem",
    "IterativeAmplitudeEstimation",
    "IterativeAmplitudeEstimationResult",
    "MaximumLikelihoodAmplitudeEstimation",
    "MaximumLikelihoodAmplitudeEstimationResult",
    "construct_amplitude_circuit",
    "grover_operator",
    "oracle_gates",
    "zero_reflection_gates",
]
