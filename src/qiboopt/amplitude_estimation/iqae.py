"""Iterative Quantum Amplitude Estimation (IQAE).

Qibo implementation of the algorithm of D. Grinko, J. Gacon, C. Zoufal and S. Woerner,
*Iterative Quantum Amplitude Estimation*, npj Quantum Information 7, 52 (2021),
arXiv:1912.05559. The control flow (``FindNextK``, confidence-interval bookkeeping and
the merging of repeated rounds) follows ``qiskit_algorithms.IterativeAmplitudeEstimation``
(Apache-2.0, https://github.com/qiskit-community/qiskit-algorithms).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from qibo.config import raise_error
from scipy.stats import beta

from qiboopt.amplitude_estimation._sampling import Sampler, good_counts, run_counts
from qiboopt.amplitude_estimation.estimation_problem import EstimationProblem
from qiboopt.amplitude_estimation.grover import construct_amplitude_circuit


@dataclass
class IterativeAmplitudeEstimationResult:
    r"""Result of :class:`IterativeAmplitudeEstimation`.

    Attributes:
        alpha (float): confidence level; the interval holds with probability :math:`\ge 1-\alpha`.
        epsilon_target (float): requested half-width of the confidence interval.
        shots (int): shots per circuit execution.
        estimation (float): estimate of the amplitude :math:`a \in [0, 1]` (interval midpoint).
        confidence_interval (tuple[float, float]): confidence interval of :math:`a`.
        epsilon_estimated (float): achieved half-width of the interval of :math:`a`.
        estimation_processed (float): ``post_processing(estimation)``.
        confidence_interval_processed (tuple[float, float]): post-processed interval.
        epsilon_estimated_processed (float): half-width of the processed interval.
        num_oracle_queries (int): total number of Grover operator applications, :math:`\sum_i N_{shots} k_i`.
        estimate_intervals (list): interval of :math:`a` after every round.
        theta_intervals (list): interval of :math:`\theta/2\pi` after every round.
        powers (list[int]): Grover powers :math:`k_i` (``powers[0] = 0`` is the initial value).
        ratios (list[float]): growth :math:`(2k_{i+1}+1)/(2k_i+1) = K_{i+1}/K_i` (with :math:`K_i = 4k_i+2`).
        post_processing (Callable): the post-processing used.
    """

    alpha: float | None = None
    epsilon_target: float | None = None
    shots: int | None = None
    estimation: float | None = None
    confidence_interval: tuple[float, float] | None = None
    epsilon_estimated: float | None = None
    estimation_processed: float | None = None
    confidence_interval_processed: tuple[float, float] | None = None
    epsilon_estimated_processed: float | None = None
    num_oracle_queries: int | None = None
    estimate_intervals: list = field(default_factory=list)
    theta_intervals: list = field(default_factory=list)
    powers: list = field(default_factory=list)
    ratios: list = field(default_factory=list)
    post_processing: Callable | None = None


class IterativeAmplitudeEstimation:
    r"""Iterative Quantum Amplitude Estimation (Grinko et al., arXiv:1912.05559).

    IQAE estimates the amplitude :math:`a = \sin^2\theta` of the good states of
    :math:`\mathcal A|0\rangle` to additive error ``epsilon_target`` with probability at least
    :math:`1-\alpha`, using only Grover powers :math:`\mathcal Q^k` and measurements (no phase
    estimation, no extra qubits). In every round the largest power :math:`k` is chosen for
    which the angle :math:`(4k+2)\theta` can still be unambiguously resolved from the current
    confidence interval; the measured good-state frequency of
    :math:`\mathcal Q^k\mathcal A|0\rangle`, whose probability is :math:`\sin^2((2k+1)\theta)`,
    then shrinks the interval for :math:`\theta`. The number of oracle calls scales as
    :math:`O(\epsilon^{-1}\log(\log(1/\epsilon)/\alpha))`.

    Args:
        epsilon_target (float): target half-width :math:`\epsilon \in (0, 0.5]`.
        alpha (float): confidence level :math:`\alpha \in (0, 1)`.
        confint_method (str): ``"beta"`` (Clopper-Pearson, default) or ``"chernoff"``.
        min_ratio (float): minimal growth :math:`K_{i+1}/K_i` of ``FindNextK`` (> 1).
        shots (int): shots per round (default 1024).
        backend: Qibo backend (global backend if ``None``).
        sampler (Callable): optional ``sampler(circuit, shots) -> {bitstring: counts}``
            replacing the default noiseless execution, e.g. to apply a noise model.
        epsilon (float): alias of ``epsilon_target`` (keyword only).

    Example:

        .. code-block:: python

            iae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, shots=100)
            result = iae.estimate(problem)
            E_tail = F.post_processing(result.estimation)   # == result.estimation_processed

    Note:
        Reproducibility: the default sampler draws from the backend's random number
        generator; seed it (for the numpy backend ``numpy.random.seed``) before ``estimate``.
    """

    def __init__(
        self,
        epsilon_target: float | None = None,
        alpha: float | None = None,
        confint_method: str = "beta",
        min_ratio: float = 2.0,
        shots: int = 1024,
        backend=None,
        sampler: Sampler | None = None,
        *,
        epsilon: float | None = None,
    ):
        if epsilon is not None:
            if epsilon_target is not None:
                raise_error(
                    ValueError, "Pass either `epsilon_target` or `epsilon`, not both."
                )
            epsilon_target = epsilon
        if epsilon_target is None or alpha is None:
            raise_error(ValueError, "`epsilon_target` and `alpha` are required.")
        if not 0 < epsilon_target <= 0.5:
            raise_error(
                ValueError,
                f"The target epsilon must be in (0, 0.5], but is {epsilon_target}.",
            )
        if not 0 < alpha < 1:
            raise_error(
                ValueError,
                f"The confidence level alpha must be in (0, 1), got {alpha}.",
            )
        if confint_method not in {"chernoff", "beta"}:
            raise_error(
                ValueError,
                f"The confidence interval method must be 'chernoff' or 'beta', got {confint_method!r}.",
            )
        if not min_ratio > 1:
            raise_error(
                ValueError, "`min_ratio` must be larger than 1 to ensure convergence."
            )
        if int(shots) != shots or shots < 1:
            raise_error(ValueError, "`shots` must be a positive integer.")
        self.epsilon_target = float(epsilon_target)
        self.alpha = float(alpha)
        self.confint_method = confint_method
        self.min_ratio = float(min_ratio)
        self.shots = int(shots)
        self.backend = backend
        self.sampler = sampler

    @staticmethod
    def find_next_k(
        k: int,
        upper_half_circle: bool,
        theta_interval: tuple[float, float],
        min_ratio: float = 2.0,
    ) -> tuple[int, bool]:
        r"""Largest :math:`k_{next}` such that :math:`(4k_{next}+2)\,\theta`-interval stays in one half-circle.

        Args:
            k (int): current power of :math:`\mathcal Q`.
            upper_half_circle (bool): whether the scaled angle currently lies in :math:`[0, \pi]`
                (``True``) or in :math:`[\pi, 2\pi]`.
            theta_interval (tuple[float, float]): current interval of :math:`\theta/2\pi`.
            min_ratio (float): minimal ratio :math:`K_{next}/K`.

        Returns:
            tuple[int, bool]: the next power and the half-circle flag of the extrapolated interval.
            If no admissible larger power exists, the old ``k`` is returned.
        """
        if min_ratio <= 1:
            raise_error(
                ValueError, "min_ratio must be larger than 1 to ensure convergence."
            )
        theta_l, theta_u = theta_interval
        old_scaling = 4 * k + 2  # current scaling factor K := 4k + 2
        # the largest feasible K is bounded by the length of the current confidence interval
        max_scaling = int(1 / (2 * (theta_u - theta_l)))
        scaling = max_scaling - (max_scaling - 2) % 4  # bring into the form 4 k_max + 2

        while scaling >= min_ratio * old_scaling:
            theta_min = scaling * theta_l - int(scaling * theta_l)
            theta_max = scaling * theta_u - int(scaling * theta_u)
            if theta_min <= theta_max <= 0.5 and theta_min <= 0.5:
                return (
                    int((scaling - 2) / 4),
                    True,
                )  # extrapolated interval in upper half-circle
            if theta_max >= 0.5 and theta_max >= theta_min >= 0.5:
                return (
                    int((scaling - 2) / 4),
                    False,
                )  # extrapolated interval in lower half-circle
            scaling -= 4
        return int(k), upper_half_circle  # no feasible k found: keep the old one

    def construct_circuit(
        self,
        estimation_problem: EstimationProblem,
        k: int = 0,
        measurement: bool = False,
    ):
        r"""The circuit :math:`\mathcal Q^k\mathcal A|0\rangle` (objective qubits measured if requested)."""
        return construct_amplitude_circuit(
            estimation_problem.state_preparation,
            estimation_problem.grover_operator,
            estimation_problem.objective_qubits,
            k,
            measurement,
        )

    def estimate(
        self, estimation_problem: EstimationProblem
    ) -> IterativeAmplitudeEstimationResult:
        """Run IQAE on ``estimation_problem``.

        Args:
            estimation_problem (EstimationProblem): the problem to solve.

        Returns:
            IterativeAmplitudeEstimationResult: estimate and confidence intervals.
        """
        epsilon, alpha, shots = self.epsilon_target, self.alpha, self.shots
        is_good_state = estimation_problem.is_good_state

        powers = [0]  # powers k of Q^k (index 0 is the a-priori value)
        ratios = []
        theta_intervals = [[0, 1 / 4]]  # a-priori knowledge of theta / (2 pi)
        a_intervals = [[0.0, 1.0]]  # a-priori knowledge of the confidence interval of a
        num_oracle_queries = 0
        num_one_shots = []

        # maximum number of rounds
        max_rounds = (
            int(np.log(self.min_ratio * np.pi / 8 / epsilon) / np.log(self.min_ratio))
            + 1
        )
        upper_half_circle = True  # initially theta is in the upper half-circle

        num_iterations = 0
        # theta is scaled mod 2 pi such that it lies in [0, 1]
        while theta_intervals[-1][1] - theta_intervals[-1][0] > epsilon / np.pi:
            num_iterations += 1
            k, upper_half_circle = self.find_next_k(
                powers[-1],
                upper_half_circle,
                theta_intervals[-1],
                min_ratio=self.min_ratio,
            )
            powers.append(k)
            ratios.append((2 * powers[-1] + 1) / (2 * powers[-2] + 1))

            circuit = self.construct_circuit(estimation_problem, k, measurement=True)
            counts = run_counts(circuit, shots, self.backend, self.sampler)
            round_shots_taken = sum(counts.values())
            one_counts = good_counts(counts, is_good_state)
            prob = one_counts / round_shots_taken
            num_one_shots.append(one_counts)
            num_oracle_queries += round_shots_taken * k

            # if K_{i-1} == K_i the samples of the consecutive rounds are merged
            j = 1  # number of rounds that stayed at the same K
            round_shots = round_shots_taken
            round_one_counts = one_counts
            if num_iterations > 1:
                while (
                    powers[num_iterations - j] == powers[num_iterations]
                    and num_iterations >= j + 1
                ):
                    j += 1
                    round_shots += round_shots_taken
                    round_one_counts += num_one_shots[-j]

            if self.confint_method == "chernoff":
                a_i_min, a_i_max = _chernoff_confint(
                    prob, round_shots, max_rounds, alpha
                )
            else:
                a_i_min, a_i_max = _clopper_pearson_confint(
                    round_one_counts, round_shots, alpha / max_rounds
                )

            if upper_half_circle:
                theta_min_i = np.arccos(1 - 2 * a_i_min) / 2 / np.pi
                theta_max_i = np.arccos(1 - 2 * a_i_max) / 2 / np.pi
            else:
                theta_min_i = 1 - np.arccos(1 - 2 * a_i_max) / 2 / np.pi
                theta_max_i = 1 - np.arccos(1 - 2 * a_i_min) / 2 / np.pi

            scaling = 4 * k + 2  # current K_i factor
            theta_u = (int(scaling * theta_intervals[-1][1]) + theta_max_i) / scaling
            theta_l = (int(scaling * theta_intervals[-1][0]) + theta_min_i) / scaling
            theta_intervals.append([theta_l, theta_u])

            a_u = float(np.sin(2 * np.pi * theta_u) ** 2)
            a_l = float(np.sin(2 * np.pi * theta_l) ** 2)
            a_intervals.append([a_l, a_u])

        confidence_interval = (float(a_intervals[-1][0]), float(a_intervals[-1][1]))
        estimation = float(np.mean(confidence_interval))  # midpoint of the interval
        post = estimation_problem.post_processing
        processed_interval = (
            float(post(confidence_interval[0])),
            float(post(confidence_interval[1])),
        )

        return IterativeAmplitudeEstimationResult(
            alpha=alpha,
            epsilon_target=epsilon,
            shots=shots,
            estimation=estimation,
            confidence_interval=confidence_interval,
            epsilon_estimated=(confidence_interval[1] - confidence_interval[0]) / 2,
            estimation_processed=float(post(estimation)),
            confidence_interval_processed=processed_interval,
            epsilon_estimated_processed=(processed_interval[1] - processed_interval[0])
            / 2,
            num_oracle_queries=int(num_oracle_queries),
            estimate_intervals=a_intervals,
            theta_intervals=theta_intervals,
            powers=powers,
            ratios=ratios,
            post_processing=post,
        )


def _chernoff_confint(
    value: float, shots: int, max_rounds: int, alpha: float
) -> tuple[float, float]:
    r"""Chernoff confidence interval for ``shots`` i.i.d. Bernoulli trials.

    :math:`[\text{value} - \varepsilon, \text{value} + \varepsilon]` with
    :math:`\varepsilon = \sqrt{3\ln(2 \cdot max\_rounds / \alpha)/shots}`, clipped to :math:`[0, 1]`.
    """
    eps = np.sqrt(3 * np.log(2 * max_rounds / alpha) / shots)
    return float(np.maximum(0, value - eps)), float(np.minimum(1, value + eps))


def _clopper_pearson_confint(
    counts: int, shots: int, alpha: float
) -> tuple[float, float]:
    """Clopper-Pearson confidence interval for ``shots`` i.i.d. Bernoulli trials."""
    lower, upper = 0.0, 1.0
    if counts != 0:  # the beta quantile would return nan
        lower = float(beta.ppf(alpha / 2, counts, shots - counts + 1))
    if counts != shots:
        upper = float(beta.ppf(1 - alpha / 2, counts + 1, shots - counts))
    return lower, upper
