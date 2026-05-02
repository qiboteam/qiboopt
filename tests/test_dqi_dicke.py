"""Tests for qiboopt.dqi.dicke."""

import math

import numpy as np
import pytest

from qiboopt.dqi.dicke import (
    _unitary_with_first_column,
    dicke_circuit,
    weighted_dicke_amplitudes,
)
from qiboopt.dqi.weights import optimal_weights


def test_amplitudes_normalised():
    w = optimal_weights(4, 2)
    amps = weighted_dicke_amplitudes(4, 2, w)
    assert amps.shape == (16,)
    assert abs(np.linalg.norm(amps) - 1.0) < 1e-12


@pytest.mark.parametrize("m,ell", [(3, 2), (4, 2), (5, 3)])
def test_amplitudes_match_analytic(m, ell):
    """Acceptance criterion 1.2: prepared amplitudes match analytic vector to 1e-10."""
    w = optimal_weights(m, ell)
    amps = weighted_dicke_amplitudes(m, ell, w)

    # Recompute the analytic amplitude for every basis state.
    expected = np.zeros(2**m, dtype=complex)
    for state in range(2**m):
        k = bin(state).count("1")
        if k <= ell:
            expected[state] = w[k] / math.sqrt(math.comb(m, k))
    expected /= np.linalg.norm(expected)
    assert np.allclose(amps, expected, atol=1e-10)


def test_unitary_first_column():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    v = v / np.linalg.norm(v)
    U = _unitary_with_first_column(v)
    assert np.allclose(U @ U.conj().T, np.eye(8), atol=1e-12)
    assert np.allclose(U[:, 0], v, atol=1e-12)


def test_dicke_circuit_state():
    """Run the Dicke prep circuit and confirm its state matches the analytic amplitudes."""
    m, ell = 4, 2
    w = optimal_weights(m, ell)
    amps = weighted_dicke_amplitudes(m, ell, w)
    circuit = dicke_circuit(m, ell, w)
    state = circuit().state()
    assert np.allclose(state, amps, atol=1e-10)


def test_dicke_circuit_qubit_count():
    m, ell = 5, 3
    w = optimal_weights(m, ell)
    circuit = dicke_circuit(m, ell, w)
    assert circuit.nqubits == m
