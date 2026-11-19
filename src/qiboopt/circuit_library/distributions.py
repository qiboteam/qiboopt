"""Loading of (discretised) probability distributions into qubit amplitudes.

The loader is the Grover-Rudolph state preparation (L. Grover and T. Rudolph, *Creating
superpositions that correspond to efficiently integrable probability distributions*,
arXiv:quant-ph/0208112) with every level realised by a uniformly controlled
:math:`R_y` rotation in its CNOT/:math:`R_y` (Gray-code) form (Mottonen et al.,
arXiv:quant-ph/0404089). It plays the role of Qiskit's ``NormalDistribution`` /
``Initialize`` based loaders, and it is exactly the structure a qGAN-trained generator
approximates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qibo import gates
from qibo.config import raise_error
from scipy.stats import multivariate_normal, norm

from qiboopt.circuit_library._base import CircuitBlock, check_qubit_list


def _fwht(values: np.ndarray) -> np.ndarray:
    """Unnormalised fast Walsh-Hadamard transform (length must be a power of two)."""
    a = np.array(values, dtype=float)
    h = 1
    while h < len(a):
        blocks = np.reshape(a, (-1, 2, h))
        a = np.reshape(
            np.stack(
                [blocks[:, 0] + blocks[:, 1], blocks[:, 0] - blocks[:, 1]], axis=1
            ),
            -1,
        )
        h *= 2
    return a


def uniformly_controlled_ry_gates(
    controls: Sequence[int], target: int, angles: Sequence[float]
) -> list:
    r"""Gates of a uniformly controlled :math:`R_y` rotation (a multiplexer).

    Applies :math:`R_y(\alpha_m)` to ``target`` when the controls encode the integer
    :math:`m`, where ``controls[l]`` carries the bit of weight :math:`2^l`. For :math:`k`
    controls the circuit consists of :math:`2^k` rotations and :math:`2^k` CNOTs.

    Args:
        controls (list[int]): control qubits, least significant bit first.
        target (int): target qubit.
        angles (list[float]): the :math:`2^k` angles :math:`\alpha_m`.

    Returns:
        list: Qibo gates.
    """
    controls = check_qubit_list(controls, "controls")
    if target in controls:
        raise_error(ValueError, "`target` must not be one of the controls.")
    k = len(controls)
    angles = np.asarray(angles, dtype=float).ravel()
    if angles.size != 2**k:
        raise_error(
            ValueError, f"Expected {2 ** k} angles for {k} controls, got {angles.size}."
        )
    if k == 0:
        return [gates.RY(target, float(angles[0]))]
    size = 2**k
    gray = [i ^ (i >> 1) for i in range(size)]
    # alpha_m = sum_i (-1)^{<m, gray(i)>} theta_i  =>  theta_i = WHT(alpha)[gray(i)] / 2^k
    transformed = _fwht(angles) / size
    out = []
    for i in range(size):
        out.append(gates.RY(target, float(transformed[gray[i]])))
        if i < size - 1:
            flipped = (
                (i + 1) & -(i + 1)
            ).bit_length() - 1  # bit changing gray(i) -> gray(i+1)
        else:
            flipped = k - 1  # gray(2^k - 1) -> gray(0)
        out.append(gates.CNOT(controls[flipped], target))
    return out


def probability_loader_gates(
    probabilities: Sequence[float], qubits: Sequence[int]
) -> list:
    r"""Gates preparing :math:`\sum_x \sqrt{p_x}\,|x\rangle` from :math:`|0\ldots0\rangle`.

    Args:
        probabilities (list[float]): :math:`2^n` non-negative weights (normalised internally).
        qubits (list[int]): the :math:`n` qubits, least significant bit first, i.e. the state
            :math:`|x\rangle` has qubit ``qubits[j]`` in the state given by bit :math:`j` of
            :math:`x`.

    Returns:
        list: Qibo gates (:math:`R_y` and CNOT only).
    """
    qubits = check_qubit_list(qubits, "qubits")
    p = np.asarray(probabilities, dtype=float).ravel()
    n = len(qubits)
    if p.size != 2**n:
        raise_error(
            ValueError, f"Expected {2 ** n} probabilities for {n} qubits, got {p.size}."
        )
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise_error(ValueError, "Probabilities must be finite and non-negative.")
    total = p.sum()
    if not total > 0:
        raise_error(
            ValueError, "The probabilities sum to zero and cannot be normalised."
        )
    p = p / total
    out = []
    for j in reversed(range(n)):  # most significant bit first
        blocks = p.reshape(2 ** (n - 1 - j), 2, 2**j)  # [prefix h, bit j, lower bits]
        p0 = blocks[:, 0, :].sum(axis=1)
        p1 = blocks[:, 1, :].sum(axis=1)
        alphas = 2 * np.arctan2(
            np.sqrt(p1), np.sqrt(p0)
        )  # P(bit_j = 1 | prefix) = sin^2(a/2)
        out += uniformly_controlled_ry_gates(qubits[j + 1 :], qubits[j], alphas)
    return out


class ProbabilityLoader(CircuitBlock):
    r"""Circuit loading a discrete probability distribution into amplitudes.

    .. math::

        \mathcal P |0\rangle^{n} = \sum_{x=0}^{2^n - 1} \sqrt{p_x}\,|x\rangle,

    exactly (up to floating point), using only :math:`R_y` and CNOT gates and no ancillas.
    The qubit ``j`` carries bit :math:`j` of :math:`x` (least significant bit first).

    Args:
        probabilities (list[float]): :math:`2^n` non-negative weights; normalised internally.

    Example:

        .. code-block:: python

            from qibo import Circuit
            from qiboopt.circuit_library import ProbabilityLoader

            loader = ProbabilityLoader([0.1, 0.2, 0.3, 0.4])
            circ = Circuit(loader.num_qubits)
            circ.add(loader.on_qubits(*range(loader.num_qubits)))
    """

    def __init__(self, probabilities: Sequence[float], name: str = "P(X)"):
        p = np.asarray(probabilities, dtype=float).ravel()
        if p.size < 2 or p.size & (p.size - 1):
            raise_error(
                ValueError,
                f"The number of probabilities must be a power of two >= 2, got {p.size}.",
            )
        n = int(np.log2(p.size))
        super().__init__(n, name)
        self.num_state_qubits = n
        gate_list = probability_loader_gates(p, list(range(n)))
        self._probabilities = p / p.sum()
        self._add(gate_list)

    @property
    def probabilities(self) -> np.ndarray:
        """The normalised probabilities :math:`p_x` that are loaded."""
        return self._probabilities


class NormalDistribution(ProbabilityLoader):
    r"""Discretised, truncated (multivariate) normal distribution.

    Qibo counterpart of ``qiskit_finance.circuit.library.NormalDistribution``. The density is
    evaluated on :math:`2^n` equidistant points of ``bounds`` and renormalised (truncation),
    then loaded with :class:`ProbabilityLoader`:

    .. math::

        \mathcal P_X|0\rangle^n = \sum_{i=0}^{2^n-1}\sqrt{\mathbb P(x_i)}\,|i\rangle .

    .. note::

        As in Qiskit, ``sigma`` is the **variance** :math:`\sigma^2` (the covariance matrix
        :math:`\Sigma` in several dimensions), *not* the standard deviation.

    For several dimensions ``num_qubits`` is a list with the number of qubits of each
    dimension; the grid point with per-dimension indices :math:`(i_0, i_1, \dots)` is
    loaded into the basis state :math:`i_0 2^{n_1 + \dots} + i_1 2^{n_2+\dots} + \dots`
    (the same flattening as Qiskit's), so dimension 0 occupies the most significant qubits.

    Args:
        num_qubits (int | list[int]): qubits per dimension.
        mu (float | list[float]): mean(s); default 0.
        sigma (float | array): variance / covariance matrix; default identity.
        bounds (tuple | list[tuple]): truncation interval(s); default ``(-1, 1)``.

    Attributes:
        values: grid points (1-D: array of shape ``(2**n,)``; otherwise list of tuples).
        probabilities: normalised probabilities of the grid points.
        bounds: truncation bounds.
    """

    def __init__(
        self,
        num_qubits: int | Sequence[int],
        mu: float | Sequence[float] | None = None,
        sigma: float | Sequence | None = None,
        bounds: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
        name: str = "P(X)",
    ):
        if isinstance(num_qubits, (int, np.integer)):
            if num_qubits < 1:
                raise_error(ValueError, "`num_qubits` must be at least 1.")
            mu = 0.0 if mu is None else float(mu)
            sigma = 1.0 if sigma is None else float(sigma)
            bounds = (
                (-1.0, 1.0) if bounds is None else (float(bounds[0]), float(bounds[1]))
            )
            if not sigma > 0:
                raise_error(ValueError, "`sigma` (the variance) must be positive.")
            if not bounds[1] > bounds[0]:
                raise_error(ValueError, f"Invalid bounds {bounds}.")
            x = np.linspace(bounds[0], bounds[1], num=2**num_qubits)
            density = norm.pdf(x, loc=mu, scale=np.sqrt(sigma))
            values = x
        else:
            dims = [int(q) for q in num_qubits]
            dim = len(dims)
            if dim < 1 or any(q < 1 for q in dims):
                raise_error(
                    ValueError, "`num_qubits` must be a list of positive integers."
                )
            mu = np.zeros(dim) if mu is None else np.asarray(mu, dtype=float)
            sigma = np.eye(dim) if sigma is None else np.asarray(sigma, dtype=float)
            bounds = (
                [(-1.0, 1.0)] * dim
                if bounds is None
                else [tuple(map(float, b)) for b in bounds]
            )
            if mu.shape != (dim,) or sigma.shape != (dim, dim) or len(bounds) != dim:
                raise_error(
                    ValueError,
                    "Dimensions of `num_qubits`, `mu`, `sigma` and `bounds` differ.",
                )
            if any(not hi > lo for lo, hi in bounds):
                raise_error(ValueError, f"Invalid bounds {bounds}.")
            grids = np.meshgrid(
                *[np.linspace(lo, hi, num=2**q) for q, (lo, hi) in zip(dims, bounds)],
                indexing="ij",
            )
            values = list(zip(*[g.flatten() for g in grids]))
            density = multivariate_normal.pdf(values, mu, sigma)
            num_qubits = sum(dims)
        density = np.asarray(density, dtype=float).ravel()
        if not density.sum() > 0:
            raise_error(
                ValueError,
                "The density underflows to zero on the whole grid; check `mu`, `sigma`, `bounds`.",
            )
        super().__init__(density, name)
        if (
            self.num_state_qubits != num_qubits
        ):  # pragma: no cover - internal consistency
            raise_error(RuntimeError, "Inconsistent number of qubits.")
        self.values = values
        self.bounds = bounds
