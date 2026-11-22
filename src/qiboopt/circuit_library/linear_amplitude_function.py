"""(Piecewise-) linear function encoded in a qubit amplitude."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qibo.config import raise_error

from qiboopt.circuit_library._base import CircuitBlock
from qiboopt.circuit_library.integer_comparator import COMPARATOR_METHODS
from qiboopt.circuit_library.pauli_rotations import PiecewiseLinearPauliRotations


class LinearAmplitudeFunction(CircuitBlock):
    r"""Circuit encoding a (piecewise) linear function in the amplitude of an objective qubit.

    This is the Qibo counterpart of Qiskit's ``LinearAmplitudeFunction``. It implements the
    amplitude function

    .. math::

        F|x\rangle|0\rangle = \sqrt{1 - \hat f(x)}\,|x\rangle|0\rangle
            + \sqrt{\hat f(x)}\,|x\rangle|1\rangle

    for :math:`\hat f : \{0, \dots, 2^n - 1\} \to [0, 1]` *approximately*, using the
    first-order (Taylor) expansion

    .. math::

        \sin^2\!\Big(\tfrac{\pi}{4} c + \tfrac{\pi c}{2}\big(\hat f(x) - \tfrac12\big)\Big)
        \approx \hat f(x), \qquad c = \text{rescaling\_factor},

    realised through (piecewise-)linearly controlled :math:`R_y` rotations (Woerner and Egger,
    arXiv:1806.06893; Gacon et al., arXiv:2005.10780; see also Holmes and Matsuura,
    arXiv:2005.04351). The accuracy improves as ``rescaling_factor`` :math:`\to 0` at the price
    of a smaller signal, which amplitude estimation has to resolve.

    A general function :math:`f` from the ``domain`` :math:`[a, b]` to the ``image``
    :math:`[c', d']` is mapped to :math:`\hat f(x) = (f(\phi(x)) - c')/(d' - c')` with
    :math:`\phi(x) = a + \frac{b - a}{2^n - 1}x`. On the segment that starts at
    ``breakpoints[j]`` (domain units) the function is

    .. math::

        f(t) = \text{offset}_j + \text{slope}_j\,(t - \text{breakpoints}_j),

    i.e. ``offset[j]`` is the **value of** :math:`f` **at the left end of segment** ``j`` (for a
    single, non-piecewise function with ``breakpoints=None`` this is :math:`f(a)`) and
    ``slope[j]`` is its slope in domain units. This is the convention of Qiskit's
    implementation (its docstring writes :math:`\alpha_i x + \beta_i`, but the code, which
    is reproduced here, uses values at the breakpoints). For a continuous function pass the
    value of :math:`f` at every breakpoint as ``offset``; a discontinuity is encoded by an
    offset that differs from the value reached by the previous segment.

    Qubit layout (identical to Qiskit's, ``objective_qubit == num_state_qubits``):

    * ``0 .. n-1``: state register :math:`x`, least significant bit first,
    * ``n``: objective qubit,
    * ``n+1 ..``: ancillas, only if there is more than one segment (``n`` for the default
      ``"ripple"`` comparator, ``1`` for ``"greedy"``).

    Args:
        num_state_qubits (int): number of qubits :math:`n` encoding :math:`x`.
        slope (float | list[float]): slope(s) of the linear piece(s), in domain units.
        offset (float | list[float]): value(s) of :math:`f` at the left end of each piece.
        domain (tuple[float, float]): :math:`(x_{\min}, x_{\max})`.
        image (tuple[float, float]): :math:`(f_{\min}, f_{\max})`, the range onto which
            :math:`f` is rescaled to :math:`[0, 1]`.
        rescaling_factor (float): :math:`c` of the Taylor approximation (default 1).
        breakpoints (list[float]): left ends of the segments in domain units. ``domain[0]`` is
            prepended if missing. ``None`` means a single (non-piecewise) function.
        comparator_method (str): ``"ripple"`` (default) or ``"greedy"``.

    Example:

        .. code-block:: python

            import numpy as np
            from qibo import Circuit
            from qiboopt.circuit_library import LinearAmplitudeFunction

            n = 3
            # f(t) = max(t - 4, 0) on t in [0, 7]: zero up to the breakpoint 4, then slope 1
            F = LinearAmplitudeFunction(
                n, slope=[0, 1], offset=[0, 0], domain=(0, 7), image=(0, 3),
                breakpoints=[0, 4], rescaling_factor=0.25,
            )
            circ = Circuit(F.num_qubits)
            circ.add(F.on_qubits(*range(F.num_qubits)))
            # probability of the objective qubit being 1 -> F.amplitude(x), and
            # F.post_processing(prob) recovers f(x) up to the Taylor error.
    """

    def __init__(
        self,
        num_state_qubits: int,
        slope: float | Sequence[float],
        offset: float | Sequence[float],
        domain: tuple[float, float],
        image: tuple[float, float],
        rescaling_factor: float = 1.0,
        breakpoints: Sequence[float] | None = None,
        comparator_method: str = "ripple",
        name: str = "F",
    ):
        if num_state_qubits < 1:
            raise_error(ValueError, "`num_state_qubits` must be at least 1.")
        if comparator_method not in COMPARATOR_METHODS:
            raise_error(
                ValueError,
                f"Unknown comparator method {comparator_method!r}; use {COMPARATOR_METHODS}.",
            )
        slope = [float(s) for s in np.atleast_1d(np.asarray(slope, dtype=float))]
        offset = [float(o) for o in np.atleast_1d(np.asarray(offset, dtype=float))]
        a, b = float(domain[0]), float(domain[1])
        c, d = float(image[0]), float(image[1])
        if not b > a:
            raise_error(
                ValueError,
                f"`domain` must satisfy domain[0] < domain[1], got {domain}.",
            )
        if not d > c:
            raise_error(
                ValueError, f"`image` must satisfy image[0] < image[1], got {image}."
            )
        if not rescaling_factor > 0:
            raise_error(ValueError, "`rescaling_factor` must be positive.")

        # the breakpoints must include the first point of the domain
        if breakpoints is None:
            breakpoints = [a]
        else:
            breakpoints = [float(p) for p in breakpoints]
            if not np.isclose(breakpoints[0], a):
                breakpoints = [a] + breakpoints
        if len(slope) != len(offset):
            raise_error(
                ValueError,
                f"Size mismatch of slope ({len(slope)}) and offset ({len(offset)}).",
            )
        if len(breakpoints) != len(slope):
            raise_error(
                ValueError,
                f"Size mismatch of slope ({len(slope)}) and breakpoints ({len(breakpoints)}).",
            )
        if not np.all(np.diff(breakpoints) > 0):
            raise_error(ValueError, "Breakpoints must be unique and sorted.")
        if breakpoints[0] < a or breakpoints[-1] > b:
            raise_error(ValueError, "Breakpoints must be included in the domain.")

        self._domain = (a, b)
        self._image = (c, d)
        self._rescaling_factor = float(rescaling_factor)
        self.slope = slope
        self.offset = offset
        self.breakpoints = breakpoints
        n = num_state_qubits
        scale = 2**n - 1

        mapped_breakpoints, mapped_slope, mapped_offset = [], [], []
        for i, point in enumerate(breakpoints):
            mapped_breakpoints.append((point - a) / (b - a) * scale)
            # (b - a) / (2^n - 1) rescales x to the domain; the shift by `a` is part of `offset`
            mapped_slope.append(slope[i] * (b - a) / scale)
            mapped_offset.append(offset[i])

        # approximate linear behaviour by scaling and contracting around pi/4
        slope_angles = np.zeros(len(breakpoints))
        offset_angles = np.pi / 4 * (1 - rescaling_factor) * np.ones(len(breakpoints))
        for i in range(len(breakpoints)):
            slope_angles[i] = np.pi * rescaling_factor * mapped_slope[i] / 2 / (d - c)
            offset_angles[i] += (
                np.pi * rescaling_factor * (mapped_offset[i] - c) / 2 / (d - c)
            )

        pwl = PiecewiseLinearPauliRotations(
            n,
            mapped_breakpoints,
            2 * slope_angles,
            2 * offset_angles,
            comparator_method=comparator_method,
        )
        super().__init__(pwl.nqubits, name)
        self.num_state_qubits = n
        self.objective_qubit = n
        self.state_qubits = list(range(n))
        self.ancilla_qubits = list(range(n + 1, pwl.nqubits))
        self.comparator_method = comparator_method
        self._pwl = pwl
        self._add(pwl.on_qubits(*range(pwl.nqubits)))

    @property
    def domain(self) -> tuple[float, float]:
        """Domain :math:`(x_{\\min}, x_{\\max})` of :math:`f`."""
        return self._domain

    @property
    def image(self) -> tuple[float, float]:
        """Image :math:`(f_{\\min}, f_{\\max})` used for the rescaling of :math:`f`."""
        return self._image

    @property
    def rescaling_factor(self) -> float:
        """Rescaling factor of the Taylor approximation."""
        return self._rescaling_factor

    def amplitude(self, x):
        r"""Exact probability of measuring the objective qubit in :math:`|1\rangle` for ``|x>``.

        Args:
            x (int | array): integer value(s) encoded in the state register.

        Returns:
            float | array: :math:`\sin^2` of half the total :math:`R_y` rotation angle. This is
            the quantity amplitude estimation measures; :meth:`post_processing` maps it back
            to :math:`f` (up to the Taylor error).
        """
        angle = self._pwl.evaluate(np.asarray(x, dtype=float))
        return np.sin(angle / 2) ** 2

    def post_processing(self, scaled_value):
        r"""Map an estimated amplitude back to the function value :math:`f`.

        Inverts the Taylor-expansion mapping and then the rescaling of :math:`f` to
        :math:`[0, 1]`.

        Args:
            scaled_value (float): amplitude (probability of the objective qubit being 1), e.g.
                ``result.estimation``.

        Returns:
            float: the corresponding value in the original image of :math:`f`.
        """
        value = scaled_value - 1 / 2 + np.pi / 4 * self._rescaling_factor
        value = value * 2 / np.pi / self._rescaling_factor
        value = value * (self._image[1] - self._image[0])
        value = value + self._image[0]
        return value
