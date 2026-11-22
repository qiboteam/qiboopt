Circuit library
===============

Qibo implementations of the Qiskit circuit-library primitives used in quantum-enhanced
simulation-based optimisation: arithmetic sub-circuits, amplitude encoders, the
``RealAmplitudes`` ansatz and probability-distribution loaders.

.. note::
    Wherever a block encodes an integer :math:`x` on a register, the first qubit of the
    register carries the *least significant* bit (the Qiskit convention). Blocks are
    embedded into a larger circuit with ``circuit.add(block.on_qubits(*qubits))``.

Arithmetic
----------

.. autoclass:: qiboopt.circuit_library.IntegerComparator
    :members:
    :member-order: bysource

.. autofunction:: qiboopt.circuit_library.integer_comparator_gates

.. autoclass:: qiboopt.circuit_library.CDKMRippleCarryAdder
    :members:
    :member-order: bysource

.. autofunction:: qiboopt.circuit_library.cdkm_adder_gates

Rotations and amplitude encoding
--------------------------------

.. autoclass:: qiboopt.circuit_library.LinearPauliRotations
    :members:
    :member-order: bysource

.. autoclass:: qiboopt.circuit_library.PiecewiseLinearPauliRotations
    :members:
    :member-order: bysource

.. autofunction:: qiboopt.circuit_library.linear_rotation_gates

.. autoclass:: qiboopt.circuit_library.LinearAmplitudeFunction
    :members:
    :member-order: bysource

Ansatz
------

.. autofunction:: qiboopt.circuit_library.real_amplitudes

.. autofunction:: qiboopt.circuit_library.entanglement_pairs

.. autofunction:: qiboopt.circuit_library.num_real_amplitudes_parameters

Distribution loaders
--------------------

.. autoclass:: qiboopt.circuit_library.ProbabilityLoader
    :members:
    :member-order: bysource

.. autoclass:: qiboopt.circuit_library.NormalDistribution
    :members:
    :member-order: bysource

.. autofunction:: qiboopt.circuit_library.probability_loader_gates

.. autofunction:: qiboopt.circuit_library.uniformly_controlled_ry_gates

Base class
----------

.. autoclass:: qiboopt.circuit_library.CircuitBlock
    :members:
    :member-order: bysource
