"""Quantum circuit primitives: arithmetic, amplitude encoders, ansatzes and loaders.

These are Qibo implementations of the Qiskit circuit-library primitives used in
quantum-enhanced simulation-based optimisation (QSBO) workflows:

* :class:`IntegerComparator`, :class:`CDKMRippleCarryAdder`
* :class:`LinearPauliRotations`, :class:`PiecewiseLinearPauliRotations`
* :class:`LinearAmplitudeFunction`
* :func:`real_amplitudes`
* :class:`ProbabilityLoader`, :class:`NormalDistribution`
"""

from qiboopt.circuit_library._base import CircuitBlock
from qiboopt.circuit_library.adders import CDKMRippleCarryAdder, cdkm_adder_gates
from qiboopt.circuit_library.distributions import (
    NormalDistribution,
    ProbabilityLoader,
    probability_loader_gates,
    uniformly_controlled_ry_gates,
)
from qiboopt.circuit_library.integer_comparator import (
    IntegerComparator,
    integer_comparator_gates,
    num_comparator_ancillas,
)
from qiboopt.circuit_library.linear_amplitude_function import LinearAmplitudeFunction
from qiboopt.circuit_library.pauli_rotations import (
    LinearPauliRotations,
    PiecewiseLinearPauliRotations,
    linear_rotation_gates,
)
from qiboopt.circuit_library.real_amplitudes import (
    entanglement_pairs,
    num_real_amplitudes_parameters,
    real_amplitudes,
)

__all__ = [
    "CircuitBlock",
    "CDKMRippleCarryAdder",
    "IntegerComparator",
    "LinearAmplitudeFunction",
    "LinearPauliRotations",
    "NormalDistribution",
    "PiecewiseLinearPauliRotations",
    "ProbabilityLoader",
    "cdkm_adder_gates",
    "entanglement_pairs",
    "integer_comparator_gates",
    "linear_rotation_gates",
    "num_comparator_ancillas",
    "num_real_amplitudes_parameters",
    "probability_loader_gates",
    "real_amplitudes",
    "uniformly_controlled_ry_gates",
]
