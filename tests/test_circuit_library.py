"""Tests for :mod:`qiboopt.circuit_library`."""

import itertools

import numpy as np
import pytest
from qibo import Circuit, gates

from qiboopt.circuit_library import (
    CDKMRippleCarryAdder,
    IntegerComparator,
    LinearAmplitudeFunction,
    LinearPauliRotations,
    NormalDistribution,
    PiecewiseLinearPauliRotations,
    ProbabilityLoader,
    entanglement_pairs,
    integer_comparator_gates,
    num_real_amplitudes_parameters,
    probability_loader_gates,
    real_amplitudes,
    uniformly_controlled_ry_gates,
)

# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _index(nqubits, bits):
    """Qibo basis-state index (qubit 0 = most significant bit) from ``{qubit: bit}``."""
    return sum(bit << (nqubits - 1 - q) for q, bit in bits.items())


def _bits(nqubits, index):
    return {q: (index >> (nqubits - 1 - q)) & 1 for q in range(nqubits)}


def _permutation(circuit):
    """Map input basis index -> output basis index for a classical reversible circuit."""
    unitary = circuit.unitary()
    out = {}
    for idx in range(unitary.shape[1]):
        column = np.abs(unitary[:, idx])
        j = int(np.argmax(column))
        assert np.isclose(column[j], 1.0), "circuit is not a basis-state permutation"
        out[idx] = j
    return out


def _run_on_basis(circuit, bits):
    state = circuit(
        initial_state=np.eye(2**circuit.nqubits)[_index(circuit.nqubits, bits)]
    ).state()
    return np.asarray(state)


# --------------------------------------------------------------------------------------------
# IntegerComparator
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ripple", "greedy"])
@pytest.mark.parametrize("geq", [True, False])
@pytest.mark.parametrize("nstate", [1, 2, 3, 4])
def test_integer_comparator_truth_table(nstate, geq, method):
    values = [-3, 0, 1, 2, 2.5, 2**nstate // 2, 2**nstate - 1, 2**nstate, 2**nstate + 5]
    for value in values:
        comp = IntegerComparator(nstate, value, geq=geq, method=method)
        N = comp.nqubits
        perm = _permutation(comp.circuit)
        threshold = int(np.ceil(value))
        for x in range(2**nstate):
            for t in (0, 1):
                bits = {q: (x >> q) & 1 for q in comp.state_qubits}
                bits[comp.compare_qubit] = t
                bits.update({q: 0 for q in comp.ancilla_qubits})
                out = _bits(N, perm[_index(N, bits)])
                flag = (x >= threshold) if geq else (x < threshold)
                assert [out[q] for q in comp.state_qubits] == [
                    bits[q] for q in comp.state_qubits
                ]
                assert out[comp.compare_qubit] == t ^ int(flag), (
                    nstate,
                    value,
                    geq,
                    x,
                    t,
                )
                assert all(
                    out[q] == 0 for q in comp.ancilla_qubits
                )  # ancillas uncomputed


def test_integer_comparator_float_noise_and_involution():
    # a breakpoint that is 3 up to round-off must not be rounded up to 4
    value = 3 + 4e-16 * 3
    comp = IntegerComparator(3, value, method="greedy")
    perm = _permutation(comp.circuit)
    assert (
        _bits(comp.nqubits, perm[_index(comp.nqubits, {0: 1, 1: 1})])[
            comp.compare_qubit
        ]
        == 1
    )
    # applying the gate list twice is the identity
    gate_list = integer_comparator_gates([0, 1, 2], 3, 5, True, [4, 5], "ripple")
    c = Circuit(6)
    c.add(gate_list)
    c.add(integer_comparator_gates([0, 1, 2], 3, 5, True, [4, 5], "ripple"))
    assert np.allclose(c.unitary(), np.eye(64))


def test_integer_comparator_resources_and_errors():
    n = 5
    assert IntegerComparator(n, 7).nqubits == 2 * n
    assert IntegerComparator(n, 7, method="greedy").nqubits == n + 1
    with pytest.raises(ValueError):
        IntegerComparator(0, 1)
    with pytest.raises(ValueError):
        IntegerComparator(3, 1, method="nope")
    with pytest.raises(ValueError):
        integer_comparator_gates(
            [0, 1, 2], 3, 5, True, [4], "ripple"
        )  # too few ancillas
    with pytest.raises(ValueError):
        integer_comparator_gates(
            [0, 1, 2], 2, 5, True, [4, 5], "ripple"
        )  # repeated qubit


def test_integer_comparator_on_qubits_embedding():
    comp = IntegerComparator(2, 2, method="ripple")  # 4 qubits
    big = Circuit(6)
    big.add(comp.on_qubits(5, 3, 1, 0))  # state = (5, 3), compare = 1, ancilla = 0
    state = np.asarray(
        big(initial_state=np.eye(64)[_index(6, {5: 0, 3: 1})]).state()
    )  # x = 2 (bit 1 on state qubit index 1 -> qubit 3)
    assert np.isclose(abs(state[_index(6, {5: 0, 3: 1, 1: 1})]), 1.0)


# --------------------------------------------------------------------------------------------
# CDKM adder
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["full", "half", "fixed"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_cdkm_adder(n, kind):
    adder = CDKMRippleCarryAdder(n, kind)
    N = adder.nqubits
    assert N == {"full": 2 * n + 2, "half": 2 * n + 2, "fixed": 2 * n + 1}[kind]
    perm = _permutation(adder.circuit)
    carries = (0, 1) if kind == "full" else (0,)
    for a, b, cin, cout0 in itertools.product(
        range(2**n), range(2**n), carries, (0, 1)
    ):
        if kind == "fixed" and cout0:
            continue
        bits = {q: 0 for q in range(N)}
        for j in range(n):
            bits[adder.a_qubits[j]] = (a >> j) & 1
            bits[adder.b_qubits[j]] = (b >> j) & 1
        if kind == "full":
            bits[adder.carry_in] = cin
        if adder.carry_out is not None:
            bits[adder.carry_out] = cout0
        out = _bits(N, perm[_index(N, bits)])
        total = a + b + cin
        assert sum(out[adder.a_qubits[j]] << j for j in range(n)) == a  # a unchanged
        assert sum(out[adder.b_qubits[j]] << j for j in range(n)) == total % 2**n
        if adder.carry_out is not None:
            assert out[adder.carry_out] == cout0 ^ (total >> n)
        if kind == "full":
            assert out[adder.carry_in] == cin
        if adder.helper_qubit is not None:
            assert out[adder.helper_qubit] == 0


def test_cdkm_adder_errors():
    with pytest.raises(ValueError):
        CDKMRippleCarryAdder(0)
    with pytest.raises(ValueError):
        CDKMRippleCarryAdder(2, kind="bogus")


# --------------------------------------------------------------------------------------------
# Pauli rotations
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["X", "Y", "Z", "y"])
def test_linear_pauli_rotations_matches_matrices(basis):
    n, slope, offset = 3, 0.37, -0.21
    block = LinearPauliRotations(n, slope, offset, basis)
    U = block.unitary()
    N = n + 1
    pauli = {
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }[basis.upper()]
    for x in range(2**n):
        angle = slope * x + offset
        rot = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * pauli
        for t in (0, 1):
            bits = {q: (x >> q) & 1 for q in range(n)}
            bits[n] = t
            col = U[:, _index(N, bits)]
            expected = np.zeros(2**N, dtype=complex)
            for t_out in (0, 1):
                b2 = dict(bits)
                b2[n] = t_out
                expected[_index(N, b2)] = rot[t_out, t]
            assert np.allclose(col, expected)


def test_linear_pauli_rotations_errors():
    with pytest.raises(ValueError):
        LinearPauliRotations(0)
    with pytest.raises(ValueError):
        LinearPauliRotations(2, basis="W")


@pytest.mark.parametrize("method", ["ripple", "greedy"])
def test_piecewise_linear_rotations(method):
    n = 3
    breakpoints = [0, 2.5, 5]
    slopes = [0.1, -0.2, 0.3]
    offsets = [0.05, 0.4, -0.3]
    pwl = PiecewiseLinearPauliRotations(
        n, breakpoints, slopes, offsets, comparator_method=method
    )
    N = pwl.nqubits
    for x in range(2**n):
        bits = {q: 0 for q in range(N)}
        bits.update({q: (x >> q) & 1 for q in range(n)})
        probs = np.abs(_run_on_basis(pwl.circuit, bits)) ** 2
        probs = probs.reshape([2] * N)
        index = [slice(None)] * N
        for q in range(n):
            index[q] = bits[q]
        marginal = probs[tuple(index)]  # over (target, ancillas)
        p_target1 = (
            marginal.sum(axis=tuple(range(1, marginal.ndim)))[1]
            if marginal.ndim > 1
            else marginal[1]
        )
        assert np.isclose(p_target1, np.sin(pwl.evaluate(x) / 2) ** 2)
        # all ancillas are back in |0>
        ancilla_zero = marginal[(slice(None),) + (0,) * (marginal.ndim - 1)].sum()
        assert np.isclose(ancilla_zero, 1.0)
    # evaluate() is the piecewise function a_j (x - x_j) + b_j
    assert np.isclose(pwl.evaluate(1), 0.1 * 1 + 0.05)
    assert np.isclose(pwl.evaluate(3), -0.2 * (3 - 2.5) + 0.4)
    assert np.isclose(pwl.evaluate(6), 0.3 * (6 - 5) - 0.3)


def test_piecewise_linear_rotations_errors():
    with pytest.raises(ValueError):
        PiecewiseLinearPauliRotations(2, [0, 1], [1.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        PiecewiseLinearPauliRotations(2, [0], comparator_method="x")
    with pytest.raises(ValueError):
        PiecewiseLinearPauliRotations(0)


# --------------------------------------------------------------------------------------------
# LinearAmplitudeFunction
# --------------------------------------------------------------------------------------------

LAF_CASES = [
    dict(n=3, slope=1.0, offset=0.0, domain=(0, 7), image=(0, 7), c=1.0, bps=None),
    dict(
        n=3,
        slope=[0.5, -1.0],
        offset=[0.0, 3.0],
        domain=(0, 7),
        image=(0, 4),
        c=0.3,
        bps=[0, 3],
    ),
    dict(
        n=4,
        slope=[0.5, 0, 2.0],
        offset=[0.2, 1.0, -3.0],
        domain=(-2, 5),
        image=(-4, 6),
        c=0.1,
        bps=[-2, 0.3, 3.1],
    ),
    dict(
        n=2,
        slope=[1, 2, 3],
        offset=[0, 0, 0.2],
        domain=(0, 3),
        image=(0, 10),
        c=0.5,
        bps=[0, 1, 2],
    ),
]


@pytest.mark.parametrize("method", ["ripple", "greedy"])
@pytest.mark.parametrize("case", LAF_CASES)
def test_linear_amplitude_function_amplitudes(case, method):
    n = case["n"]
    F = LinearAmplitudeFunction(
        n,
        case["slope"],
        case["offset"],
        case["domain"],
        case["image"],
        case["c"],
        case["bps"],
        comparator_method=method,
    )
    assert F.objective_qubit == n
    N = F.num_qubits
    circ = Circuit(N)
    circ.add(gates.H(q) for q in range(n))
    circ.add(F.on_qubits(*range(N)))
    probs = (np.abs(np.asarray(circ().state())) ** 2).reshape([2] * N)
    for x in range(2**n):
        idx = [slice(None)] * N
        for q in range(n):
            idx[q] = (x >> q) & 1
        block = probs[tuple(idx)] * 2**n  # conditional on x
        p_obj1 = block.sum(axis=tuple(i for i in range(block.ndim) if i != 0))[1]
        assert np.isclose(p_obj1, F.amplitude(x))
        # ancillas return to |0>
        if block.ndim > 2:
            assert np.isclose(
                block[(slice(None), slice(None)) + (0,) * (block.ndim - 2)].sum(), 1.0
            )


def test_linear_amplitude_function_ancilla_counts():
    n = 4
    single = LinearAmplitudeFunction(n, 1.0, 0.0, (0, 15), (0, 15))
    assert single.num_qubits == n + 1
    multi = LinearAmplitudeFunction(
        n, [1.0, 2.0], [0.0, 1.0], (0, 15), (0, 40), breakpoints=[0, 5]
    )
    assert multi.num_qubits == 2 * n + 1  # n ancillas, as in Qiskit's circuit
    multi_greedy = LinearAmplitudeFunction(
        n,
        [1.0, 2.0],
        [0.0, 1.0],
        (0, 15),
        (0, 40),
        breakpoints=[0, 5],
        comparator_method="greedy",
    )
    assert multi_greedy.num_qubits == n + 2


@pytest.mark.parametrize("case", LAF_CASES[1:3])
def test_linear_amplitude_function_post_processing(case):
    """post_processing(amplitude(x)) recovers f(x) as the Taylor parameter goes to zero."""
    n = case["n"]
    a, b = case["domain"]
    lo, hi = case["image"]
    slopes, offsets = np.atleast_1d(case["slope"]), np.atleast_1d(case["offset"])
    bps = list(case["bps"])

    def f(t):  # offset_j is the value at the left end of segment j
        j = max(i for i, p in enumerate(bps) if t >= p - 1e-12) if t >= bps[0] else 0
        return offsets[j] + slopes[j] * (t - bps[j])

    # compare with the (clipped) target: only meaningful where f lies in the image
    errors = []
    for c in (0.2, 0.02):
        F = LinearAmplitudeFunction(
            n,
            case["slope"],
            case["offset"],
            case["domain"],
            case["image"],
            c,
            case["bps"],
        )
        err, compared = 0.0, 0
        for x in range(2**n):
            t = a + (b - a) / (2**n - 1) * x
            # the comparator uses integer thresholds: use the same breakpoint rule
            mapped = [(p - a) / (b - a) * (2**n - 1) for p in bps]
            j = max([i for i, m in enumerate(mapped) if x >= np.ceil(m - 1e-9)] or [0])
            fx = offsets[j] + slopes[j] * (t - bps[j])
            if lo <= fx <= hi:
                compared += 1
                err = max(err, abs(F.post_processing(F.amplitude(x)) - fx))
        assert compared >= 3
        errors.append(err)
    assert 0 < errors[1] < errors[0] and errors[1] < 5e-3 * (hi - lo)


def test_linear_amplitude_function_validation():
    ok = dict(slope=1.0, offset=0.0, domain=(0, 7), image=(0, 7))
    with pytest.raises(ValueError):
        LinearAmplitudeFunction(0, **ok)
    with pytest.raises(ValueError):
        LinearAmplitudeFunction(3, 1.0, 0.0, (1, 1), (0, 7))
    with pytest.raises(ValueError):
        LinearAmplitudeFunction(3, 1.0, 0.0, (0, 7), (3, 3))
    with pytest.raises(ValueError):
        LinearAmplitudeFunction(3, rescaling_factor=0.0, **ok)
    with pytest.raises(ValueError):  # slope/offset size mismatch
        LinearAmplitudeFunction(
            3, [1.0, 2.0], [0.0], (0, 7), (0, 7), breakpoints=[0, 3]
        )
    with pytest.raises(ValueError):  # breakpoints/slope mismatch
        LinearAmplitudeFunction(
            3, [1.0, 2.0], [0.0, 1.0], (0, 7), (0, 7), breakpoints=[0, 3, 5]
        )
    with pytest.raises(ValueError):  # unsorted
        LinearAmplitudeFunction(
            3, [1.0, 2.0, 3.0], [0.0, 1.0, 2.0], (0, 7), (0, 9), breakpoints=[0, 5, 3]
        )
    with pytest.raises(ValueError):  # outside the domain
        LinearAmplitudeFunction(
            3, [1.0, 2.0], [0.0, 1.0], (0, 7), (0, 9), breakpoints=[0, 9]
        )
    with pytest.raises(ValueError):
        LinearAmplitudeFunction(3, comparator_method="foo", **ok)
    F = LinearAmplitudeFunction(3, **ok)
    assert (
        F.domain == (0.0, 7.0) and F.image == (0.0, 7.0) and F.rescaling_factor == 1.0
    )


def test_domain_breakpoint_is_prepended():
    # breakpoints without the domain start get it prepended (slope/offset must then match)
    F = LinearAmplitudeFunction(
        3, [1.0, 2.0], [0.0, 1.0], (0, 7), (0, 15), breakpoints=[4]
    )
    assert F.breakpoints == [0.0, 4.0]


# --------------------------------------------------------------------------------------------
# real_amplitudes
# --------------------------------------------------------------------------------------------


def test_real_amplitudes_structure_and_parameters():
    n, reps = 4, 3
    ansatz = real_amplitudes(n, reps)
    assert num_real_amplitudes_parameters(n, reps) == n * (reps + 1)
    assert len(ansatz.get_parameters()) == n * (reps + 1)
    assert ansatz.depth > 0
    assert sum(isinstance(g, gates.RY) for g in ansatz.queue) == n * (reps + 1)
    assert sum(isinstance(g, gates.CNOT) for g in ansatz.queue) == reps * (n - 1)
    skip = real_amplitudes(n, reps, skip_final_rotation_layer=True)
    assert sum(isinstance(g, gates.RY) for g in skip.queue) == n * reps
    assert num_real_amplitudes_parameters(n, reps, True) == n * reps
    # zero angles leave |0...0> invariant
    state = np.asarray(ansatz().state())
    assert np.isclose(abs(state[0]), 1.0)


def test_real_amplitudes_real_state_and_parameter_order():
    rng = np.random.default_rng(0)
    # reps = 0: product state of RY(theta_q)|0>, so the order is qubit 0, 1, ...
    theta = rng.uniform(-3, 3, 3)
    state = np.asarray(real_amplitudes(3, 0, parameters=theta)().state())
    expected = np.array([1.0])
    for t in theta:  # qubit 0 is the most significant tensor factor
        expected = np.kron(expected, [np.cos(t / 2), np.sin(t / 2)])
    assert np.allclose(state, expected)
    # general: amplitudes stay real
    theta = rng.uniform(-3, 3, num_real_amplitudes_parameters(4, 2))
    ansatz = real_amplitudes(4, 2, entanglement="full")
    ansatz.set_parameters(theta)
    state = np.asarray(ansatz().state())
    assert np.allclose(state.imag, 0, atol=1e-12) and np.isclose(
        np.linalg.norm(state), 1.0
    )
    # layer-major order: the second layer starts at index n
    ref = real_amplitudes(2, 1, parameters=[0.1, 0.2, 0.3, 0.4])
    manual = Circuit(2)
    manual.add(gates.RY(0, 0.1))
    manual.add(gates.RY(1, 0.2))
    manual.add(gates.CNOT(0, 1))
    manual.add(gates.RY(0, 0.3))
    manual.add(gates.RY(1, 0.4))
    assert np.allclose(ref().state(), manual().state())


def test_entanglement_pairs_and_errors():
    assert entanglement_pairs(4, "linear") == [(0, 1), (1, 2), (2, 3)]
    assert entanglement_pairs(4, "reverse_linear") == [(2, 3), (1, 2), (0, 1)]
    assert entanglement_pairs(3, "full") == [(0, 1), (0, 2), (1, 2)]
    assert entanglement_pairs(4, "circular") == [(3, 0), (0, 1), (1, 2), (2, 3)]
    assert entanglement_pairs(3, [(2, 0)]) == [(2, 0)]
    with pytest.raises(ValueError):
        entanglement_pairs(3, "zigzag")
    with pytest.raises(ValueError):
        entanglement_pairs(3, [(0, 3)])
    with pytest.raises(ValueError):
        real_amplitudes(0)
    with pytest.raises(ValueError):
        real_amplitudes(2, reps=-1)
    with pytest.raises(ValueError):
        real_amplitudes(2, 1, parameters=[0.1])
    assert len(real_amplitudes(1, 2).queue) == 3  # single qubit: rotations only


# --------------------------------------------------------------------------------------------
# distribution loaders
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_uniformly_controlled_ry(k):
    rng = np.random.default_rng(k)
    angles = rng.uniform(-3, 3, 2**k)
    controls, target = list(range(k)), k
    circ = Circuit(k + 1)
    circ.add(uniformly_controlled_ry_gates(controls, target, angles))
    U = circ.unitary()
    for m in range(2**k):
        a = angles[m]
        ry = np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]])
        for t in (0, 1):
            bits = {q: (m >> q) & 1 for q in controls}
            bits[target] = t
            expected = np.zeros(2 ** (k + 1))
            for t_out in (0, 1):
                b2 = dict(bits)
                b2[target] = t_out
                expected[_index(k + 1, b2)] = ry[t_out, t]
            assert np.allclose(U[:, _index(k + 1, bits)], expected)
    if k:
        assert len(circ.queue) == 2 * 2**k


def test_uniformly_controlled_ry_errors():
    with pytest.raises(ValueError):
        uniformly_controlled_ry_gates([0, 1], 2, [0.1, 0.2])
    with pytest.raises(ValueError):
        uniformly_controlled_ry_gates([0], 0, [0.1, 0.2])


def _amplitudes_by_integer(circuit, nqubits):
    """Statevector re-indexed by the integer x whose bit j is qubit j."""
    state = np.asarray(circuit().state())
    out = np.zeros_like(state)
    for idx in range(2**nqubits):
        x = sum(((idx >> (nqubits - 1 - q)) & 1) << q for q in range(nqubits))
        out[x] = state[idx]
    return out


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_probability_loader_exact(n):
    rng = np.random.default_rng(n)
    p = rng.random(2**n)
    p[rng.integers(2**n)] = 0.0  # include a zero-probability outcome
    loader = ProbabilityLoader(p)
    circ = Circuit(n)
    circ.add(loader.on_qubits(*range(n)))
    amps = _amplitudes_by_integer(circ, n)
    assert np.allclose(amps, np.sqrt(p / p.sum()), atol=1e-12)
    assert np.allclose(loader.probabilities, p / p.sum())
    assert {type(g) for g in loader.queue} <= {gates.RY, gates.CNOT}


def test_probability_loader_degenerate_and_errors():
    # point mass
    p = np.zeros(8)
    p[5] = 1.0
    circ = Circuit(3)
    circ.add(ProbabilityLoader(p).on_qubits(0, 1, 2))
    assert np.allclose(_amplitudes_by_integer(circ, 3), p, atol=1e-12)
    with pytest.raises(ValueError):
        ProbabilityLoader([0.2, 0.3, 0.5])  # not a power of two
    with pytest.raises(ValueError):
        ProbabilityLoader([1.0])
    with pytest.raises(ValueError):
        probability_loader_gates([0.5, -0.5], [0])
    with pytest.raises(ValueError):
        probability_loader_gates([0.0, 0.0], [0])
    with pytest.raises(ValueError):
        probability_loader_gates([0.5, 0.5, 0.0, 0.0], [0])


def test_normal_distribution_1d():
    from scipy.stats import norm

    n, mu, var, bounds = 4, 0.7, 0.8, (-0.5, 2.0)
    dist = NormalDistribution(n, mu, var, bounds)  # sigma is the VARIANCE
    x = np.linspace(*bounds, 2**n)
    p = norm.pdf(x, mu, np.sqrt(var))
    p /= p.sum()
    assert np.allclose(dist.values, x) and np.allclose(dist.probabilities, p)
    assert dist.bounds == bounds and dist.num_qubits == n
    circ = Circuit(n)
    circ.add(dist.on_qubits(*range(n)))
    assert np.allclose(_amplitudes_by_integer(circ, n), np.sqrt(p), atol=1e-12)
    # defaults: standard normal on (-1, 1)
    default = NormalDistribution(3)
    assert default.bounds == (-1.0, 1.0)


def test_normal_distribution_multivariate():
    from scipy.stats import multivariate_normal

    mu, sigma, bounds = [1, 0.9], [[1, -0.2], [-0.2, 1]], [(0, 1), (-1, 1)]
    dist = NormalDistribution([2, 3], mu, sigma, bounds)
    assert dist.num_qubits == 5
    g0, g1 = np.linspace(0, 1, 4), np.linspace(-1, 1, 8)
    pts = [(a, b) for a in g0 for b in g1]  # dimension 0 = most significant index part
    p = multivariate_normal.pdf(pts, mu, sigma)
    p /= p.sum()
    assert np.allclose(dist.probabilities, p)
    circ = Circuit(5)
    circ.add(dist.on_qubits(*range(5)))
    assert np.allclose(_amplitudes_by_integer(circ, 5), np.sqrt(p), atol=1e-12)


def test_normal_distribution_errors():
    with pytest.raises(ValueError):
        NormalDistribution(0)
    with pytest.raises(ValueError):
        NormalDistribution(3, sigma=-1.0)
    with pytest.raises(ValueError):
        NormalDistribution(3, bounds=(1.0, 1.0))
    with pytest.raises(ValueError):
        NormalDistribution(3, mu=1e6, sigma=1e-3)  # density underflows
    with pytest.raises(ValueError):
        NormalDistribution([2, 2], mu=[0.0], sigma=np.eye(2))
    with pytest.raises(ValueError):
        NormalDistribution([2, 0])
    with pytest.raises(ValueError):
        NormalDistribution([2, 2], bounds=[(0, 1), (1, 1)])


def test_circuit_block_utilities():
    comp = IntegerComparator(2, 1)
    inverse = comp.invert()
    c = Circuit(comp.nqubits)
    c.add(comp.on_qubits(*range(comp.nqubits)))
    c.add(inverse.on_qubits(*range(comp.nqubits)))
    assert np.allclose(c.unitary(), np.eye(2**comp.nqubits))
    assert "IntegerComparator" in repr(comp)
    assert comp.num_qubits == comp.nqubits == comp.circuit.nqubits
    assert len(comp.queue) == len(comp.circuit.queue)
    with pytest.raises(ValueError):
        list(comp.on_qubits(0, 1))


# --------------------------------------------------------------------------------------------
# input validation of the gate-list builders
# --------------------------------------------------------------------------------------------


def test_cdkm_adder_gates_validation():
    from qiboopt.circuit_library import cdkm_adder_gates

    with pytest.raises(ValueError, match="at least 1"):
        cdkm_adder_gates([], [], 0)
    with pytest.raises(ValueError, match="same size"):
        cdkm_adder_gates([0, 1], [2], 3)
    with pytest.raises(ValueError, match="distinct"):
        cdkm_adder_gates([0, 1], [2, 3], 4, carry_out=0)  # carry-out reuses a qubit
    with pytest.raises(ValueError, match="distinct"):
        cdkm_adder_gates([0, 1], [2, 3], 3)  # carry-in reuses a qubit
    # a valid call still works, with and without carry-out
    assert cdkm_adder_gates([0, 1], [2, 3], 4)
    assert cdkm_adder_gates([0, 1], [2, 3], 4, carry_out=5)


def test_check_qubit_list_validation():
    from qiboopt.circuit_library._base import check_qubit_list

    assert check_qubit_list((2, 0, 1), "q") == [2, 0, 1]
    with pytest.raises(ValueError, match="repeated"):
        check_qubit_list([0, 1, 1], "q")
    with pytest.raises(ValueError, match="negative"):
        check_qubit_list([0, -1], "q")


def test_integer_comparator_gates_validation():
    with pytest.raises(ValueError, match="At least one"):
        integer_comparator_gates([], 0, 1)
    with pytest.raises(ValueError, match="Unknown comparator method"):
        integer_comparator_gates([0, 1], 2, 1, method="nope")
    from qiboopt.circuit_library import num_comparator_ancillas

    with pytest.raises(ValueError, match="Unknown comparator method"):
        num_comparator_ancillas(3, "nope")
    assert num_comparator_ancillas(3, "greedy") == 0
    assert num_comparator_ancillas(3, "ripple") == 2


def test_linear_rotation_gates_validation():
    from qiboopt.circuit_library import linear_rotation_gates

    with pytest.raises(ValueError, match="distinct"):
        linear_rotation_gates([0, 1], 1, 0.1, 0.2)  # target is also a state qubit
    with pytest.raises(ValueError, match="distinct"):
        linear_rotation_gates(
            [0, 1], 2, 0.1, 0.2, controls=[0]
        )  # control is a state qubit
    with pytest.raises(ValueError, match="repeated"):
        linear_rotation_gates([0, 0], 2, 0.1, 0.2)
    # several extra controls use the multi-controlled form
    assert linear_rotation_gates([0, 1], 4, 0.1, 0.2, controls=[2, 3])


def test_piecewise_linear_rotations_empty_breakpoints():
    with pytest.raises(ValueError, match="At least one breakpoint"):
        PiecewiseLinearPauliRotations(2, [], [], [])
