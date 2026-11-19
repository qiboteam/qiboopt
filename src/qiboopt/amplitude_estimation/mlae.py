"""Maximum Likelihood Amplitude Estimation (MLAE).

Qibo implementation of Y. Suzuki, S. Uno, R. Raymond, T. Tanaka, T. Onodera and
N. Yamamoto, *Amplitude estimation without phase estimation*, Quantum Information
Processing 19, 75 (2020), arXiv:1904.10246. The structure follows
``qiskit_algorithms.MaximumLikelihoodAmplitudeEstimation``
(Apache-2.0, https://github.com/qiskit-community/qiskit-algorithms).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from qibo.config import raise_error
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

from qiboopt.amplitude_estimation._sampling import Sampler, good_counts, run_counts
from qiboopt.amplitude_estimation.estimation_problem import EstimationProblem
from qiboopt.amplitude_estimation.grover import construct_amplitude_circuit

Minimizer = Callable[[Callable[[float], float], list[tuple[float, float]]], float]
_EPS = 1e-15  # keeps the search interval away from log(0)


@dataclass
class MaximumLikelihoodAmplitudeEstimationResult:
    r"""Result of :class:`MaximumLikelihoodAmplitudeEstimation`.

    Attributes:
        theta (float): maximum likelihood estimate of the angle :math:`\theta`, :math:`a = \sin^2\theta`.
        estimation (float): estimate of the amplitude :math:`a`.
        estimation_processed (float): ``post_processing(estimation)``.
        confidence_interval (tuple[float, float]): 95% Fisher-information interval of :math:`a`.
        confidence_interval_processed (tuple[float, float]): the post-processed interval.
        fisher_information (float): Fisher information of the estimate (``inf`` if :math:`a \in \{0, 1\}`).
        evaluation_schedule (list[int]): Grover powers :math:`m_j` used.
        good_counts (list[int]): number of good shots at every power.
        shots (int): shots per power.
        circuit_results (list[dict]): measured counts per power.
        num_oracle_queries (int): ``shots * sum(evaluation_schedule)``.
        minimizer (Callable): minimiser used for the likelihood.
        post_processing (Callable): the post-processing used.
    """

    theta: float | None = None
    estimation: float | None = None
    estimation_processed: float | None = None
    confidence_interval: tuple[float, float] | None = None
    confidence_interval_processed: tuple[float, float] | None = None
    fisher_information: float | None = None
    evaluation_schedule: list = field(default_factory=list)
    good_counts: list = field(default_factory=list)
    shots: int | None = None
    circuit_results: list = field(default_factory=list)
    num_oracle_queries: int | None = None
    minimizer: Callable | None = None
    post_processing: Callable | None = None


def _log_likelihood(theta, schedule, good, total):
    """Log-likelihood (vectorised in ``theta``) of the observed counts."""
    theta = np.asarray(theta, dtype=float)
    out = np.zeros_like(theta)
    tiny = np.finfo(float).tiny
    for m, g, n in zip(schedule, good, total):
        angle = (2 * m + 1) * theta
        if g > 0:
            out = out + g * np.log(np.maximum(np.sin(angle) ** 2, tiny))
        if n - g > 0:
            out = out + (n - g) * np.log(np.maximum(np.cos(angle) ** 2, tiny))
    return out


class MaximumLikelihoodAmplitudeEstimation:
    r"""Maximum Likelihood Amplitude Estimation (Suzuki et al., arXiv:1904.10246).

    The circuits :math:`\mathcal Q^{m_j}\mathcal A|0\rangle` for a fixed schedule of powers
    :math:`m_j` are each sampled ``shots`` times. The probability of a good outcome at power
    :math:`m` is :math:`\sin^2((2m+1)\theta)`, so the likelihood of the observed counts can be
    maximised over :math:`\theta \in (0, \pi/2)`; the estimate is :math:`a = \sin^2\theta`. With an
    exponential schedule :math:`(0, 1, 2, 4, \dots, 2^{M-1})` the error decays as :math:`\sim 1/(N_{shots}2^M)`
    (Heisenberg-like), without phase estimation or extra qubits.

    Args:
        evaluation_schedule (list[int] | int): either the list of non-negative powers, or a
            non-negative integer :math:`M` giving the exponential schedule
            ``[0, 1, 2, 4, ..., 2**(M-1)]``.
        minimizer (Callable): ``minimizer(fn, [(low, high)]) -> theta`` minimising the negative
            log-likelihood ``fn``. Default: a fine grid search over :math:`(0, \pi/2)` with
            ``max(10000, pi/2 * 1000 * 2 * max(schedule))`` points, polished by bounded Brent
            minimisation around the best grid point.
        shots (int): shots per circuit (default 1024).
        backend: Qibo backend (global backend if ``None``).
        sampler (Callable): optional ``sampler(circuit, shots) -> {bitstring: counts}``.

    Example:

        .. code-block:: python

            mlae = MaximumLikelihoodAmplitudeEstimation(evaluation_schedule=5, shots=200)
            result = mlae.estimate(problem)
            print(result.estimation_processed, result.confidence_interval_processed)
    """

    def __init__(
        self,
        evaluation_schedule: Sequence[int] | int,
        minimizer: Minimizer | None = None,
        shots: int = 1024,
        backend=None,
        sampler: Sampler | None = None,
    ):
        if isinstance(evaluation_schedule, (int, np.integer)):
            if evaluation_schedule < 0:
                raise_error(ValueError, "The evaluation schedule cannot be < 0.")
            schedule = [0] + [2**j for j in range(int(evaluation_schedule))]
        else:
            schedule = [int(m) for m in evaluation_schedule]
            if len(schedule) == 0:
                raise_error(ValueError, "The evaluation schedule must not be empty.")
            if any(m < 0 for m in schedule):
                raise_error(
                    ValueError, "The elements of the evaluation schedule cannot be < 0."
                )
        if int(shots) != shots or shots < 1:
            raise_error(ValueError, "`shots` must be a positive integer.")
        self.evaluation_schedule = schedule
        self.shots = int(shots)
        self.backend = backend
        self.sampler = sampler
        self._custom_minimizer = minimizer is not None
        self.minimizer = minimizer if minimizer is not None else self._default_minimizer

    def _grid_size(self) -> int:
        return max(10000, int(np.pi / 2 * 1000 * 2 * max(self.evaluation_schedule)))

    def _default_minimizer(self, objective_fn, bounds) -> float:
        """Vectorised grid search followed by a bounded local refinement."""
        low, high = bounds[0]
        grid = np.linspace(low, high, self._grid_size())
        try:
            values = np.asarray(objective_fn(grid), dtype=float)
            if values.shape != grid.shape:
                raise TypeError
        except (TypeError, ValueError):  # not vectorised: evaluate point by point
            values = np.array([objective_fn(t) for t in grid], dtype=float)
        best = int(np.nanargmin(values))
        lo = grid[max(best - 1, 0)]
        hi = grid[min(best + 1, len(grid) - 1)]
        refined = minimize_scalar(
            lambda t: float(objective_fn(t)),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-14},
        )
        if refined.success and float(refined.fun) <= float(values[best]):
            return float(refined.x)
        return float(grid[best])

    def construct_circuits(
        self, estimation_problem: EstimationProblem, measurement: bool = False
    ):
        r"""One circuit :math:`\mathcal Q^{m}\mathcal A|0\rangle` for every power of the schedule."""
        return [
            construct_amplitude_circuit(
                estimation_problem.state_preparation,
                estimation_problem.grover_operator,
                estimation_problem.objective_qubits,
                m,
                measurement,
            )
            for m in self.evaluation_schedule
        ]

    def compute_mle(
        self,
        circuit_results: Sequence[dict[str, int]],
        estimation_problem: EstimationProblem,
        return_counts: bool = False,
    ):
        r"""Maximum likelihood estimate of :math:`\theta` from measured counts.

        Args:
            circuit_results (list[dict]): counts for every power of the schedule.
            estimation_problem (EstimationProblem): provides ``is_good_state``.
            return_counts (bool): also return the list of good counts.
        """
        if len(circuit_results) != len(self.evaluation_schedule):
            raise_error(
                ValueError, "One counts dictionary per schedule entry is required."
            )
        good = [
            good_counts(c, estimation_problem.is_good_state) for c in circuit_results
        ]
        total = [int(sum(c.values())) for c in circuit_results]
        schedule = self.evaluation_schedule

        def negative_log_likelihood(theta):
            return -_log_likelihood(theta, schedule, good, total)

        theta = float(
            self.minimizer(negative_log_likelihood, [(_EPS, np.pi / 2 - _EPS)])
        )
        if return_counts:
            return theta, good
        return theta

    def estimate(
        self, estimation_problem: EstimationProblem
    ) -> MaximumLikelihoodAmplitudeEstimationResult:
        """Run MLAE on ``estimation_problem``.

        Args:
            estimation_problem (EstimationProblem): the problem to solve.

        Returns:
            MaximumLikelihoodAmplitudeEstimationResult: the estimate, its 95% Fisher
            confidence interval and the raw data.
        """
        circuits = self.construct_circuits(estimation_problem, measurement=True)
        results = [
            run_counts(c, self.shots, self.backend, self.sampler) for c in circuits
        ]
        theta, good = self.compute_mle(results, estimation_problem, return_counts=True)
        post = estimation_problem.post_processing

        result = MaximumLikelihoodAmplitudeEstimationResult(
            theta=theta,
            estimation=float(np.sin(theta) ** 2),
            good_counts=good,
            evaluation_schedule=list(self.evaluation_schedule),
            shots=self.shots,
            circuit_results=results,
            minimizer=self.minimizer,
            post_processing=post,
        )
        result.estimation_processed = float(post(result.estimation))
        result.fisher_information = _compute_fisher_information(
            result, [sum(r.values()) for r in results]
        )
        result.num_oracle_queries = int(
            sum(sum(r.values()) * m for r, m in zip(results, self.evaluation_schedule))
        )
        interval = self.compute_confidence_interval(result, alpha=0.05, kind="fisher")
        result.confidence_interval = interval
        result.confidence_interval_processed = (
            float(post(interval[0])),
            float(post(interval[1])),
        )
        return result

    @staticmethod
    def compute_confidence_interval(
        result: MaximumLikelihoodAmplitudeEstimationResult,
        alpha: float,
        kind: str = "fisher",
        apply_post_processing: bool = False,
    ) -> tuple[float, float]:
        r"""Confidence interval of the amplitude with confidence level :math:`1 - \alpha`.

        Args:
            result: a result returned by :meth:`estimate`.
            alpha (float): :math:`\alpha`, at most 0.5.
            kind (str): ``"fisher"`` / ``"fi"`` (theoretical Fisher information; default),
                ``"observed_fisher"`` / ``"observed_information"`` / ``"oi"``, or
                ``"likelihood_ratio"`` / ``"lr"``.
            apply_post_processing (bool): map the interval through ``post_processing``.

        Returns:
            tuple[float, float]: the interval of :math:`a` (not clipped to :math:`[0, 1]`).

        Note:
            Unlike Qiskit, which post-processes inside its Fisher/likelihood routines and
            post-processes again in ``estimate``, the raw amplitude interval is returned here and
            the post-processing is applied exactly once when requested.
        """
        if not 0 < alpha <= 0.5:
            raise_error(ValueError, "`alpha` must be in (0, 0.5].")
        if kind in ("likelihood_ratio", "lr"):
            interval = _likelihood_ratio_confint(result, alpha)
        elif kind in ("fisher", "fi"):
            interval = _fisher_confint(result, alpha, observed=False)
        elif kind in ("observed_fisher", "observed_information", "oi"):
            interval = _fisher_confint(result, alpha, observed=True)
        else:
            raise NotImplementedError(f"CI `{kind}` is not implemented.")
        if apply_post_processing:
            return (
                float(result.post_processing(interval[0])),
                float(result.post_processing(interval[1])),
            )
        return interval


def _compute_fisher_information(result, all_hits, observed: bool = False) -> float:
    """(Observed) Fisher information of the amplitude estimate."""
    a = result.estimation
    if observed:
        theta_a = np.arcsin(np.sqrt(np.real(a)))
        d_loglik = 0.0
        for shots_k, h_k, m_k in zip(
            all_hits, result.good_counts, result.evaluation_schedule
        ):
            tan = np.tan((2 * m_k + 1) * theta_a)
            d_loglik += (2 * m_k + 1) * (h_k / tan + (shots_k - h_k) * tan)
        d_loglik /= np.sqrt(a * (1 - a))
        return float(d_loglik**2 / len(all_hits))
    if not 0 < a < 1:
        return float("inf")
    info = sum(
        shots_k * (2 * m_k + 1) ** 2
        for shots_k, m_k in zip(all_hits, result.evaluation_schedule)
    )
    return float(info / (a * (1 - a)))


def _fisher_confint(
    result, alpha: float = 0.05, observed: bool = False
) -> tuple[float, float]:
    """Fisher-information (Wald) confidence interval of the amplitude."""
    all_hits = [sum(r.values()) for r in result.circuit_results]
    info = result.fisher_information
    if observed:
        with np.errstate(all="ignore"):
            info = _compute_fisher_information(result, all_hits, observed=True)
    quantile = norm.ppf(1 - alpha / 2)
    if not np.isfinite(info) or info <= 0:
        half_width = 0.0 if np.isinf(info) else float("inf")
    else:
        half_width = quantile / np.sqrt(info)
    a = float(np.real(result.estimation))
    return a - half_width, a + half_width


def _likelihood_ratio_confint(
    result, alpha: float = 0.05, nevals: int | None = None
) -> tuple[float, float]:
    """Likelihood-ratio confidence interval of the amplitude."""
    schedule = result.evaluation_schedule
    if nevals is None:
        nevals = max(10000, int(np.pi / 2 * 1000 * 2 * schedule[-1]))
    all_counts = [sum(r.values()) for r in result.circuit_results]
    thetas = np.linspace(_EPS, np.pi / 2 - _EPS, nevals)
    values = _log_likelihood(thetas, schedule, result.good_counts, all_counts)
    loglik_mle = float(
        _log_likelihood(result.theta, schedule, result.good_counts, all_counts)
    )
    threshold = loglik_mle - chi2.ppf(1 - alpha, df=1) / 2
    above = thetas[values >= threshold]
    low = float(above.min()) if above.size else 0.0
    high = float(above.max()) if above.size else np.pi / 2
    return float(np.sin(low) ** 2), float(np.sin(high) ** 2)
