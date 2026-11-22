"""Grover operator and circuits :math:`\\mathcal Q^k\\mathcal A|0\\rangle`."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

from qibo import Circuit, gates
from qibo.config import raise_error

MAX_ORACLE_OBJECTIVE_QUBITS = 16


def _phase_flip_basis_state(qubits: Sequence[int], bits: Sequence[int]) -> list:
    """Gates flipping the sign of the basis state ``bits`` of ``qubits`` (open controls via X)."""
    zeros = [q for q, b in zip(qubits, bits) if not b]
    flips = [gates.X(q) for q in zeros]
    if len(qubits) == 1:
        core = [gates.Z(qubits[0])]
    else:
        core = [gates.Z(qubits[-1]).controlled_by(*qubits[:-1])]
    return flips + core + [gates.X(q) for q in zeros]


def oracle_gates(
    objective_qubits: Sequence[int], is_good_state: Callable[[str], bool] | None = None
) -> list:
    r"""Gates of the oracle :math:`\mathcal S_\chi`: a phase flip on the good states.

    The oracle flips the sign of the (computational-basis) states of the objective qubits
    for which ``is_good_state(bitstring)`` is true. If more than half of the states are good,
    the bad states are flipped instead; this differs only by a global phase.

    Args:
        objective_qubits (list[int]): measured qubits.
        is_good_state (Callable[[str], bool]): predicate; ``None`` means all ones.
    """
    objective_qubits = [int(q) for q in objective_qubits]
    m = len(objective_qubits)
    if m > MAX_ORACLE_OBJECTIVE_QUBITS:
        raise_error(
            ValueError,
            f"At most {MAX_ORACLE_OBJECTIVE_QUBITS} objective qubits are supported.",
        )
    if is_good_state is None:
        return _phase_flip_basis_state(objective_qubits, [1] * m)
    good, bad = [], []
    for bits in itertools.product((0, 1), repeat=m):
        key = "".join(map(str, bits))
        (good if is_good_state(key) else bad).append(bits)
    flipped = good if len(good) <= len(bad) else bad
    out = []
    for bits in flipped:
        out += _phase_flip_basis_state(objective_qubits, bits)
    return out


def zero_reflection_gates(nqubits: int) -> list:
    r"""Gates of :math:`\mathbb 1 - 2|0\rangle\langle 0|` on ``nqubits`` qubits."""
    if nqubits == 1:
        return [gates.X(0), gates.Z(0), gates.X(0)]
    flips = [gates.X(q) for q in range(nqubits)]
    return (
        flips
        + [gates.Z(nqubits - 1).controlled_by(*range(nqubits - 1))]
        + [gates.X(q) for q in range(nqubits)]
    )


def grover_operator(
    state_preparation: Circuit,
    objective_qubits: int | Sequence[int],
    is_good_state: Callable[[str], bool] | None = None,
) -> Circuit:
    r"""Build the Grover operator :math:`\mathcal Q = \mathcal A\,\mathcal S_0\,\mathcal A^\dagger\,\mathcal S_\chi`.

    :math:`\mathcal S_\chi` is the oracle flipping the phase of the good states and
    :math:`\mathcal S_0 = \mathbb 1 - 2|0\rangle\langle 0|`. As in Qiskit's ``GroverOperator``
    the overall sign :math:`-1` of the textbook :math:`\mathcal Q` is dropped; it is a global
    phase, irrelevant for the uncontrolled powers :math:`\mathcal Q^k` used by IQAE and MLAE.
    If :math:`\mathcal A|0\rangle = \sin\theta\,|\psi_1\rangle + \cos\theta\,|\psi_0\rangle`
    then :math:`\mathcal Q^k\mathcal A|0\rangle` has good-state probability
    :math:`\sin^2((2k+1)\theta)`.

    Args:
        state_preparation (Circuit): the :math:`\mathcal A` circuit.
        objective_qubits (int | list[int]): qubits defining the good states.
        is_good_state (Callable[[str], bool]): predicate on the objective bitstring; ``None`` means all ones.

    Returns:
        Circuit: :math:`\mathcal Q` on the same qubits as ``state_preparation``.
    """
    if isinstance(objective_qubits, int) or hasattr(objective_qubits, "__index__"):
        objective_qubits = [int(objective_qubits)]
    nqubits = state_preparation.nqubits
    inverse = state_preparation.invert()
    circuit = Circuit(nqubits)
    circuit.add(oracle_gates(objective_qubits, is_good_state))
    circuit.add(inverse.on_qubits(*range(nqubits)))
    circuit.add(zero_reflection_gates(nqubits))
    circuit.add(state_preparation.on_qubits(*range(nqubits)))
    return circuit


def construct_amplitude_circuit(
    state_preparation: Circuit,
    grover: Circuit,
    objective_qubits: Sequence[int],
    k: int = 0,
    measurement: bool = False,
) -> Circuit:
    r"""Circuit :math:`\mathcal Q^k\mathcal A|0\rangle`, optionally measuring the objective qubits.

    Args:
        state_preparation (Circuit): :math:`\mathcal A`.
        grover (Circuit): :math:`\mathcal Q` (same qubits as :math:`\mathcal A`).
        objective_qubits (list[int]): qubits to measure.
        k (int): number of Grover iterations (``k >= 0``).
        measurement (bool): add a measurement of ``objective_qubits`` (in the given order).
    """
    if k < 0 or int(k) != k:
        raise_error(ValueError, f"`k` must be a non-negative integer, got {k}.")
    nqubits = state_preparation.nqubits
    circuit = Circuit(nqubits)
    circuit.add(state_preparation.on_qubits(*range(nqubits)))
    for _ in range(int(k)):
        circuit.add(grover.on_qubits(*range(nqubits)))
    if measurement:
        circuit.add(gates.M(*objective_qubits))
    return circuit
