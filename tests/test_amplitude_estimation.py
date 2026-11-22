"""Tests for :mod:`qiboopt.amplitude_estimation`."""

import numpy as np
import pytest
from qibo import Circuit, gates

from qiboopt.amplitude_estimation import (
    EstimationProblem,
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
    construct_amplitude_circuit,
    grover_operator,
    oracle_gates,
    zero_reflection_gates,
)
from qiboopt.circuit_library import LinearAmplitudeFunction, NormalDistribution

# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def bernoulli_problem(a, **kwargs):
    """A = RY(2 theta) on one qubit, so that the good-state amplitude is a = sin^2(theta)."""
    theta = np.arcsin(np.sqrt(a))
    circuit = Circuit(1)
    circuit.add(gates.RY(0, 2 * theta))
    return EstimationProblem(circuit, 0, **kwargs), theta


def good_probability(circuit, qubit):
    """Probability of ``qubit`` being 1 in the final state of ``circuit``."""
    probs = (np.abs(np.asarray(circuit().state())) ** 2).reshape([2] * circuit.nqubits)
    return float(np.take(probs, 1, axis=qubit).sum())


def random_circuit(nqubits, seed, depth=8):
    rng = np.random.default_rng(seed)
    circuit = Circuit(nqubits)
    for _ in range(depth):
        circuit.add(gates.RY(int(rng.integers(nqubits)), float(rng.uniform(-3, 3))))
        if nqubits > 1:
            c, t = rng.choice(nqubits, 2, replace=False)
            circuit.add(gates.CNOT(int(c), int(t)))
            circuit.add(gates.RZ(int(t), float(rng.uniform(-3, 3))))
    return circuit


def counting_sampler(a, rng_seed, nqueries=None):
    """Sampler of the Bernoulli problem that infers k from the circuit length."""
    theta = np.arcsin(np.sqrt(a))
    rng = np.random.default_rng(rng_seed)
    q_len = len(bernoulli_problem(a)[0].grover_operator.queue)

    def sampler(circuit, shots):
        k = (len(circuit.queue) - 2) // q_len  # A + k * Q + measurement
        if nqueries is not None:
            nqueries.append(k)
        ones = int(rng.binomial(shots, np.sin((2 * k + 1) * theta) ** 2))
        return {"1": ones, "0": shots - ones}

    return sampler


# --------------------------------------------------------------------------------------------
# EstimationProblem
# --------------------------------------------------------------------------------------------


def test_estimation_problem_properties():
    circuit = random_circuit(3, 0)
    problem = EstimationProblem(circuit, 2)
    assert problem.objective_qubits == [2]
    assert problem.state_preparation is circuit
    assert problem.post_processing(0.3) == 0.3  # identity
    assert problem.is_good_state("11") and not problem.is_good_state("10")
    assert problem.has_good_state is False
    assert problem.grover_operator is problem.grover_operator  # cached
    assert problem.grover_operator.nqubits == 3

    post = lambda x: 2 * x
    custom = EstimationProblem(
        circuit, [0, 2], post_processing=post, is_good_state=lambda b: b == "10"
    )
    assert custom.objective_qubits == [0, 2] and custom.has_good_state
    assert custom.post_processing(0.25) == 0.5 and custom.is_good_state("10")
    custom.post_processing = None
    assert custom.post_processing(0.25) == 0.25


def test_estimation_problem_alias_setters_and_errors():
    circuit = random_circuit(2, 1)
    assert EstimationProblem(circuit, objective_qubit=1).objective_qubits == [1]
    assert EstimationProblem(circuit, np.int64(1)).objective_qubits == [1]
    with pytest.raises(ValueError):
        EstimationProblem(circuit, 0, objective_qubit=1)
    with pytest.raises(ValueError):
        EstimationProblem(circuit)
    with pytest.raises(ValueError):
        EstimationProblem(circuit, [])
    with pytest.raises(ValueError):
        EstimationProblem(circuit, [0, 0])
    with pytest.raises(ValueError):
        EstimationProblem(circuit, 2)  # out of range
    with pytest.raises(TypeError):
        EstimationProblem("not a circuit", 0)
    with pytest.raises(ValueError):  # grover operator on a different number of qubits
        EstimationProblem(circuit, 0, grover_operator=Circuit(3))
    with pytest.raises(TypeError):
        EstimationProblem(circuit, 0, grover_operator="Q")

    problem = EstimationProblem(circuit, 0)
    first = problem.grover_operator
    problem.objective_qubits = 1  # invalidates the cache
    assert problem.grover_operator is not first
    problem.state_preparation = random_circuit(2, 5)
    problem.is_good_state = lambda b: b == "0"
    assert problem.has_good_state
    custom_q = Circuit(2)
    problem.grover_operator = custom_q
    assert problem.grover_operator is custom_q
    problem.grover_operator = None
    assert problem.grover_operator is not custom_q


# --------------------------------------------------------------------------------------------
# Grover operator
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2, 5])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4])
def test_grover_operator_amplification(nqubits, k):
    """Pr[good] after Q^k A|0> is sin^2((2k+1) theta)."""
    A = random_circuit(nqubits, 10 + nqubits)
    obj = nqubits - 1
    a = good_probability(A, obj)
    theta = np.arcsin(np.sqrt(a))
    circuit = construct_amplitude_circuit(A, grover_operator(A, obj), [obj], k)
    assert np.isclose(good_probability(circuit, obj), np.sin((2 * k + 1) * theta) ** 2)


def test_grover_operator_is_unitary_and_oracle_phases():
    A = random_circuit(3, 4)
    Q = grover_operator(A, [0, 2])
    U = Q.unitary()
    assert np.allclose(U @ U.conj().T, np.eye(8))
    # oracle: phase flip on the all-ones state of the objective qubits (default)
    c = Circuit(3)
    c.add(oracle_gates([0, 2]))
    diag = np.diag(c.unitary())
    expected = np.ones(8)
    for idx in range(8):
        if (idx >> 2) & 1 and idx & 1:  # qubit 0 and qubit 2 are 1
            expected[idx] = -1
    assert np.allclose(diag, expected)
    # custom predicate; character i refers to objective_qubits[i]
    for good in (
        lambda b: b == "10",
        lambda b: b != "00",
    ):  # second: more good than bad states
        c = Circuit(2)
        c.add(oracle_gates([1, 0], good))
        diag = np.real(np.diag(c.unitary()))
        sign = np.array(
            [1 if not good(f"{q1}{q0}") else -1 for q1 in (0, 1) for q0 in (0, 1)]
        )
        # basis index = 2 * bit(qubit 0) + bit(qubit 1); relate to the bitstring "q1 q0"
        by_qubits = np.array([sign[2 * q1 + q0] for q0 in (0, 1) for q1 in (0, 1)])
        # the oracle may differ from the textbook sign by a global phase (-1)
        assert np.allclose(diag, by_qubits) or np.allclose(diag, -by_qubits)


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_zero_reflection(nqubits):
    c = Circuit(nqubits)
    c.add(zero_reflection_gates(nqubits))
    expected = np.eye(2**nqubits)
    expected[0, 0] = -1
    assert np.allclose(c.unitary(), expected)


def test_multi_objective_custom_good_state_amplification():
    A = random_circuit(4, 7)
    objective = [3, 1]
    is_good = lambda bits: bits in (
        "10",
        "01",
    )  # exactly one of the two objective qubits is 1
    state = np.asarray(A().state())
    probs = (np.abs(state) ** 2).reshape([2] * 4)
    a = sum(probs.sum(axis=(0, 2))[b3, b1] for b3, b1 in [(1, 0), (0, 1)])
    theta = np.arcsin(np.sqrt(a))
    Q = grover_operator(A, objective, is_good)
    for k in (1, 3):
        circ = construct_amplitude_circuit(A, Q, objective, k)
        probs_k = (np.abs(np.asarray(circ().state())) ** 2).reshape([2] * 4)
        p_good = sum(probs_k.sum(axis=(0, 2))[b3, b1] for b3, b1 in [(1, 0), (0, 1)])
        assert np.isclose(p_good, np.sin((2 * k + 1) * theta) ** 2)


def test_construct_amplitude_circuit_measurement_and_errors():
    problem, _ = bernoulli_problem(0.3)
    c = construct_amplitude_circuit(
        problem.state_preparation, problem.grover_operator, [0], 2, measurement=True
    )
    assert isinstance(c.queue[-1], gates.M)
    assert len(c.queue) == 1 + 2 * len(problem.grover_operator.queue) + 1
    with pytest.raises(ValueError):
        construct_amplitude_circuit(
            problem.state_preparation, problem.grover_operator, [0], -1
        )
    with pytest.raises(ValueError):
        oracle_gates(list(range(20)), lambda b: True)


# --------------------------------------------------------------------------------------------
# IQAE
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, expected",
    [  # reference values computed with qiskit_algorithms.IterativeAmplitudeEstimation
        ((0, True, (0, 0.25)), (0, True)),
        ((0, True, (0.0, 0.05)), (2, True)),
        ((2, True, (0.10, 0.11)), (12, True)),
        ((3, False, (0.7, 0.7004)), (307, True)),
        ((5, True, (0.2, 0.2005)), (247, True)),
        ((0, True, (0.0, 0.01)), (12, True)),
    ],
)
def test_find_next_k_reference(args, expected):
    assert IterativeAmplitudeEstimation.find_next_k(*args) == expected
    assert IterativeAmplitudeEstimation.find_next_k(
        0, True, (0.0, 0.003), min_ratio=1.5
    ) == (41, True)
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation.find_next_k(0, True, (0.0, 0.1), min_ratio=1.0)


@pytest.mark.parametrize(
    "method, shots, epsilon", [("beta", 100, 0.01), ("chernoff", 3000, 0.03)]
)
@pytest.mark.parametrize("a", [0.23, 0.0, 1.0, 0.5])
def test_iqae_accuracy(a, method, shots, epsilon):
    problem, _ = bernoulli_problem(a)
    np.random.seed(11)
    iae = IterativeAmplitudeEstimation(
        epsilon, 0.05, confint_method=method, shots=shots
    )
    result = iae.estimate(problem)
    lower, upper = result.confidence_interval
    assert lower - 1e-9 <= a <= upper + 1e-9
    assert abs(result.estimation - a) <= epsilon + 1e-9
    assert result.epsilon_estimated <= epsilon + 1e-12
    assert np.isclose(result.estimation, (lower + upper) / 2)
    assert np.isclose(result.epsilon_estimated, (upper - lower) / 2)
    # bookkeeping
    assert (
        result.alpha == 0.05
        and result.epsilon_target == epsilon
        and result.shots == shots
    )
    assert result.num_oracle_queries == shots * sum(result.powers[1:])
    assert (
        len(result.ratios) == len(result.powers) - 1 == len(result.theta_intervals) - 1
    )
    assert len(result.estimate_intervals) == len(result.theta_intervals)
    assert result.estimation_processed == result.estimation  # identity post-processing
    assert result.confidence_interval_processed == result.confidence_interval


def test_iqae_quadratic_speedup_in_oracle_queries():
    """Halving epsilon should (roughly) not more than double the oracle calls: O(1/eps)."""
    problem, _ = bernoulli_problem(0.3)
    queries = []
    for eps in (0.01, 0.0025):
        np.random.seed(2)
        res = IterativeAmplitudeEstimation(eps, 0.05, shots=100).estimate(problem)
        assert abs(res.estimation - 0.3) <= eps
        queries.append(res.num_oracle_queries)
    assert (
        queries[1] < 16 * queries[0]
    )  # 4x smaller epsilon: far fewer than 16x queries ...
    assert queries[1] < (1 / 0.0025) * 100 * 20  # ... and bounded by O(shots / eps)


def test_iqae_post_processing_custom_sampler_and_alias():
    problem, _ = bernoulli_problem(0.4, post_processing=lambda x: 10 * x + 1)
    ks = []
    iae = IterativeAmplitudeEstimation(
        epsilon=0.02, alpha=0.1, shots=200, sampler=counting_sampler(0.4, 3, ks)
    )
    result = iae.estimate(problem)
    assert max(ks) == max(result.powers)
    assert np.isclose(result.estimation_processed, 10 * result.estimation + 1)
    lo, hi = result.confidence_interval
    assert np.allclose(result.confidence_interval_processed, (10 * lo + 1, 10 * hi + 1))
    assert np.isclose(result.epsilon_estimated_processed, 10 * result.epsilon_estimated)
    assert abs(result.estimation - 0.4) < 0.02
    # explicit backend instance
    from qibo.backends import construct_backend

    np.random.seed(0)
    r2 = IterativeAmplitudeEstimation(
        0.05, 0.1, shots=100, backend=construct_backend("numpy")
    ).estimate(problem)
    assert abs(r2.estimation - 0.4) < 0.05


def test_iqae_reproducible_with_seed_and_construct_circuit():
    problem, _ = bernoulli_problem(0.3)
    iae = IterativeAmplitudeEstimation(0.02, 0.05, shots=100)
    np.random.seed(5)
    r1 = iae.estimate(problem)
    np.random.seed(5)
    r2 = iae.estimate(problem)
    assert r1.estimation == r2.estimation and r1.powers == r2.powers
    circ = iae.construct_circuit(problem, k=3, measurement=True)
    assert isinstance(circ.queue[-1], gates.M)
    assert len(iae.construct_circuit(problem, k=0).queue) == 1


def test_iqae_argument_validation():
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(0.0, 0.05)
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(0.6, 0.05)
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(0.01, 1.0)
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(0.01, 0.0)
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(0.01)  # alpha missing
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(alpha=0.05)  # epsilon missing
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(0.01, 0.05, epsilon=0.01)
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(0.01, 0.05, confint_method="wald")
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(0.01, 0.05, min_ratio=1.0)
    with pytest.raises(ValueError):
        IterativeAmplitudeEstimation(0.01, 0.05, shots=0)


def test_iqae_empty_sampler_output():
    problem, _ = bernoulli_problem(0.3)
    iae = IterativeAmplitudeEstimation(0.05, 0.05, sampler=lambda circuit, shots: {})
    with pytest.raises(RuntimeError):
        iae.estimate(problem)


# --------------------------------------------------------------------------------------------
# MLAE
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("a", [0.23, 0.5, 0.9])
def test_mlae_accuracy(a):
    problem, theta = bernoulli_problem(a)
    np.random.seed(3)
    mlae = MaximumLikelihoodAmplitudeEstimation(evaluation_schedule=5, shots=300)
    assert mlae.evaluation_schedule == [0, 1, 2, 4, 8, 16]
    result = mlae.estimate(problem)
    assert abs(result.estimation - a) < 5e-3
    assert abs(np.sin(result.theta) ** 2 - result.estimation) < 1e-15
    lo, hi = result.confidence_interval
    assert lo < a < hi
    assert result.num_oracle_queries == 300 * sum(result.evaluation_schedule)
    assert result.shots == 300 and len(result.good_counts) == 6
    assert len(result.circuit_results) == 6
    assert result.fisher_information > 0
    assert result.estimation_processed == result.estimation


def test_mlae_edge_amplitudes_and_zero_schedule():
    for a in (0.0, 1.0):
        problem, _ = bernoulli_problem(a)
        np.random.seed(0)
        result = MaximumLikelihoodAmplitudeEstimation([0, 1, 2], shots=100).estimate(
            problem
        )
        assert abs(result.estimation - a) < 1e-6
        assert (
            np.isfinite(result.estimation) and result.fisher_information > 1e12
        )  # inf or huge
        assert result.confidence_interval[1] - result.confidence_interval[0] < 1e-6
    # a schedule with only Q^0 reduces to the empirical frequency
    problem, _ = bernoulli_problem(0.3)
    np.random.seed(1)
    result = MaximumLikelihoodAmplitudeEstimation(0, shots=2000).estimate(problem)
    assert result.evaluation_schedule == [0]
    assert np.isclose(result.estimation, result.good_counts[0] / 2000, atol=1e-6)


def test_mlae_confidence_intervals_and_post_processing():
    post = lambda x: 3 * x - 1
    problem, _ = bernoulli_problem(0.35, post_processing=post)
    np.random.seed(8)
    result = MaximumLikelihoodAmplitudeEstimation([0, 1, 3, 7], shots=500).estimate(
        problem
    )
    fi = result.confidence_interval
    assert np.allclose(
        result.confidence_interval_processed, (post(fi[0]), post(fi[1]))
    )  # applied once
    assert np.isclose(result.estimation_processed, post(result.estimation))
    mlae = MaximumLikelihoodAmplitudeEstimation
    for kind in (
        "fisher",
        "fi",
        "observed_fisher",
        "observed_information",
        "oi",
        "likelihood_ratio",
        "lr",
    ):
        lo, hi = mlae.compute_confidence_interval(result, 0.05, kind)
        assert lo <= result.estimation <= hi and hi - lo < 0.1, kind
        plo, phi = mlae.compute_confidence_interval(
            result, 0.05, kind, apply_post_processing=True
        )
        assert np.isclose(plo, post(lo)) and np.isclose(phi, post(hi))
    # a smaller alpha gives a wider interval
    narrow = mlae.compute_confidence_interval(result, 0.5, "fisher")
    wide = mlae.compute_confidence_interval(result, 0.01, "fisher")
    assert wide[1] - wide[0] > narrow[1] - narrow[0]
    with pytest.raises(NotImplementedError):
        mlae.compute_confidence_interval(result, 0.05, "bootstrap")
    with pytest.raises(ValueError):
        mlae.compute_confidence_interval(result, 0.9)


def test_mlae_custom_minimizer_sampler_and_compute_mle():
    problem, theta = bernoulli_problem(0.3)
    calls = []

    def minimizer(fn, bounds):
        calls.append(bounds)
        grid = np.linspace(bounds[0][0], bounds[0][1], 20001)
        return float(grid[np.argmin([fn(t) for t in grid])])

    ks = []

    # sampler that returns exact expectations (rounded) so that the MLE is essentially exact
    def sampler(circuit, shots):
        k = (len(circuit.queue) - 2) // len(problem.grover_operator.queue)
        ks.append(k)
        ones = int(round(shots * np.sin((2 * k + 1) * theta) ** 2))
        return {"1": ones, "0": shots - ones}

    mlae = MaximumLikelihoodAmplitudeEstimation(
        [0, 1, 2, 4], minimizer, shots=10000, sampler=sampler
    )
    result = mlae.estimate(problem)
    assert ks == [0, 1, 2, 4] and len(calls) == 1
    assert abs(result.estimation - 0.3) < 1e-4
    assert result.minimizer is minimizer
    # compute_mle directly, with and without counts
    counts = [{"1": 30, "0": 70}] * 4
    est, good = mlae.compute_mle(counts, problem, return_counts=True)
    assert good == [30] * 4 and 0 < est < np.pi / 2
    assert isinstance(mlae.compute_mle(counts, problem), float)
    with pytest.raises(ValueError):
        mlae.compute_mle(counts[:2], problem)


def test_mlae_argument_validation_and_circuits():
    with pytest.raises(ValueError):
        MaximumLikelihoodAmplitudeEstimation(-1)
    with pytest.raises(ValueError):
        MaximumLikelihoodAmplitudeEstimation([0, -2])
    with pytest.raises(ValueError):
        MaximumLikelihoodAmplitudeEstimation([])
    with pytest.raises(ValueError):
        MaximumLikelihoodAmplitudeEstimation(3, shots=0)
    problem, _ = bernoulli_problem(0.3)
    circuits = MaximumLikelihoodAmplitudeEstimation([0, 2, 5]).construct_circuits(
        problem, measurement=True
    )
    q_len = len(problem.grover_operator.queue)
    assert [len(c.queue) for c in circuits] == [2, 2 + 2 * q_len, 2 + 5 * q_len]


# --------------------------------------------------------------------------------------------
# end-to-end: the CVaR-type tail expectation of the Qiskit -> Qibo mapping document
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tail_problem():
    n = 3
    loader = NormalDistribution(n, mu=0.5, sigma=0.3, bounds=(-1, 2))
    lo, hi, strike = -1.0, 2.0, 0.4
    payoff = lambda v: np.maximum(v - strike, 0.0)
    exact = float(np.sum(loader.probabilities * payoff(loader.values)))
    F = LinearAmplitudeFunction(
        n,
        slope=[0, 1],
        offset=[0, 0],
        domain=(lo, hi),
        image=(0, float(payoff(hi))),
        rescaling_factor=0.05,
        breakpoints=[lo, strike],
    )
    circ = Circuit(F.num_qubits)
    circ.add(loader.on_qubits(*range(n)))
    circ.add(F.on_qubits(*range(F.num_qubits)))
    problem = EstimationProblem(
        circ, objective_qubit=n, post_processing=F.post_processing
    )
    return problem, F, loader, exact


def test_tail_expectation_pipeline(tail_problem):
    problem, F, loader, exact = tail_problem
    n = F.num_state_qubits
    # amplitude of the circuit equals the classical expectation of the amplitude function
    a_exact = good_probability(problem.state_preparation, n)
    a_classical = float(np.sum(loader.probabilities * F.amplitude(np.arange(2**n))))
    assert np.isclose(a_exact, a_classical)
    # ... and post-processing recovers E[(X - K)+] up to the (tiny) Taylor error
    assert abs(F.post_processing(a_exact) - exact) < 2e-3

    np.random.seed(21)
    iqae = IterativeAmplitudeEstimation(0.002, 0.05, shots=100).estimate(problem)
    assert abs(iqae.estimation - a_exact) <= 0.002
    assert abs(iqae.estimation_processed - exact) < 0.02
    lo, hi = iqae.confidence_interval_processed
    assert lo <= F.post_processing(a_exact) <= hi

    np.random.seed(22)
    mlae = MaximumLikelihoodAmplitudeEstimation(4, shots=500).estimate(problem)
    assert abs(mlae.estimation - a_exact) < 0.01
    assert abs(mlae.estimation_processed - exact) < 0.06


def test_mlae_default_minimizer_scalar_only_objective():
    """The default minimiser falls back to point-wise evaluation for non-vectorised functions."""
    mlae = MaximumLikelihoodAmplitudeEstimation([0, 1], shots=10)
    bounds = [(0.0, np.pi / 2)]

    # float(array) raises TypeError for arrays with several elements
    def scalar_only(t):
        return (float(t) - 1.0) ** 2

    assert abs(mlae._default_minimizer(scalar_only, bounds) - 1.0) < 1e-6

    # returns a scalar even for array input: wrong shape -> point-wise fallback
    def wrong_shape(t):
        return 0.5 if np.ndim(t) else (t - 0.7) ** 2

    assert abs(mlae._default_minimizer(wrong_shape, bounds) - 0.7) < 1e-6
