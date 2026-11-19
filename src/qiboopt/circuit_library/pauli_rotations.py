"""Linearly-controlled and piecewise-linearly-controlled Pauli rotations.

Ports of ``qiskit.circuit.library.LinearPauliRotations`` and
``qiskit.circuit.library.PiecewiseLinearPauliRotations`` to Qibo (gate sequences follow the
Apache-2.0 Qiskit implementation, https://github.com/Qiskit/qiskit). Background: Woerner and
Egger, arXiv:1806.06893; Gacon, Zoufal and Woerner, arXiv:2005.10780.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qibo import gates
from qibo.config import raise_error

from qiboopt.circuit_library._base import CircuitBlock, check_qubit_list
from qiboopt.circuit_library.integer_comparator import (
    COMPARATOR_METHODS,
    integer_comparator_gates,
    num_comparator_ancillas,
)

_ROTATIONS = {"X": gates.RX, "Y": gates.RY, "Z": gates.RZ}
_CONTROLLED_ROTATIONS = {"X": gates.CRX, "Y": gates.CRY, "Z": gates.CRZ}


def _check_basis(basis: str) -> str:
    basis = str(basis).upper()
    if basis not in _ROTATIONS:
        raise_error(ValueError, f"`basis` must be 'X', 'Y' or 'Z', got {basis!r}.")
    return basis


def _rotation(basis: str, target: int, theta: float, controls: Sequence[int]):
    controls = list(controls)
    if not controls:
        return _ROTATIONS[basis](target, theta)
    if len(controls) == 1:
        return _CONTROLLED_ROTATIONS[basis](controls[0], target, theta)
    return _ROTATIONS[basis](target, theta).controlled_by(*controls)


def linear_rotation_gates(
    state_qubits: Sequence[int],
    target: int,
    slope: float,
    offset: float,
    basis: str = "Y",
    controls: Sequence[int] = (),
) -> list:
    r"""Gates of a linearly-controlled Pauli rotation.

    Implements, for the basis ``Y``,

    .. math::

        |x\rangle|0\rangle \mapsto \cos(ax + b)|x\rangle|0\rangle + \sin(ax + b)|x\rangle|1\rangle,
        \qquad a = \text{slope}/2,\; b = \text{offset}/2,

    by one rotation by ``offset`` followed by a rotation by ``slope * 2**i`` controlled on
    the ``i``-th state qubit. Optional extra ``controls`` control every rotation.

    Args:
        state_qubits (list[int]): qubits encoding :math:`x` (least significant bit first).
        target (int): rotated qubit.
        slope (float): slope (in rotation-angle units).
        offset (float): offset (in rotation-angle units).
        basis (str): ``"X"``, ``"Y"`` (default) or ``"Z"``.
        controls (list[int]): additional control qubits.

    Returns:
        list: Qibo gates.
    """
    basis = _check_basis(basis)
    state_qubits = check_qubit_list(state_qubits, "state_qubits")
    controls = check_qubit_list(controls, "controls")
    everything = state_qubits + controls + [target]
    if len(set(everything)) != len(everything):
        raise_error(ValueError, "State, control and target qubits must be distinct.")
    out = [_rotation(basis, target, float(offset), controls)]
    for i, q in enumerate(state_qubits):
        out.append(_rotation(basis, target, float(slope) * 2**i, controls + [q]))
    return out


class LinearPauliRotations(CircuitBlock):
    r"""Linearly-controlled X, Y or Z rotation (Qibo version of Qiskit's class).

    For the basis ``"Y"`` the block acts as

    .. math::

        |x\rangle|0\rangle \mapsto \cos(ax + b)|x\rangle|0\rangle + \sin(ax + b)|x\rangle|1\rangle,

    with :math:`a =` ``slope``:math:`/2` and :math:`b =` ``offset``:math:`/2`.

    Qubit layout: ``[x_0 .. x_{n-1}, target]`` (least significant bit first).

    Args:
        num_state_qubits (int): number of qubits encoding :math:`x`.
        slope (float): slope of the controlled rotation.
        offset (float): offset of the rotation.
        basis (str): ``"X"``, ``"Y"`` (default) or ``"Z"``.
    """

    def __init__(
        self,
        num_state_qubits: int,
        slope: float = 1.0,
        offset: float = 0.0,
        basis: str = "Y",
        name: str = "LinPauliRot",
    ):
        if num_state_qubits < 1:
            raise_error(ValueError, "`num_state_qubits` must be at least 1.")
        super().__init__(num_state_qubits + 1, name)
        self.num_state_qubits = num_state_qubits
        self.slope = slope
        self.offset = offset
        self.basis = _check_basis(basis)
        self.target_qubit = num_state_qubits
        self._add(
            linear_rotation_gates(
                list(range(num_state_qubits)),
                num_state_qubits,
                slope,
                offset,
                self.basis,
            )
        )


class PiecewiseLinearPauliRotations(CircuitBlock):
    r"""Piecewise-linearly-controlled Pauli rotations.

    For breakpoints :math:`(x_0, \dots, x_J) \subset [0, 2^n-1]`, slopes :math:`a_j` and
    offsets :math:`b_j` the rotation angle on the target is

    .. math::

        f(x) = \begin{cases} 0, & x < x_0 \\ a_j (x - x_j) + b_j, & x_j \le x < x_{j+1} \end{cases}

    with :math:`x_{J+1} = 2^n`. Each segment is realised by an :class:`IntegerComparator`
    that raises a flag when :math:`x \ge x_j`, a rotation controlled on the flag, and the
    uncomputation of the flag.

    Qubit layout: ``[x_0 .. x_{n-1}, target]`` followed, if there is more than one
    breakpoint, by ``[compare, helpers...]`` (``n`` ancillas for the ``"ripple"``
    comparator, 1 for ``"greedy"``). All ancillas are returned to :math:`|0\rangle`.

    Args:
        num_state_qubits (int): number of qubits encoding :math:`x`.
        breakpoints (list[float]): breakpoints; defaults to ``[0]``.
        slopes (list[float]): slope of each segment; defaults to ``[1]``.
        offsets (list[float]): offset of each segment; defaults to ``[0]``.
        basis (str): ``"X"``, ``"Y"`` (default) or ``"Z"``.
        comparator_method (str): ``"ripple"`` (default) or ``"greedy"``, see
            :class:`~qiboopt.circuit_library.IntegerComparator`.
    """

    def __init__(
        self,
        num_state_qubits: int,
        breakpoints: Sequence[float] | None = None,
        slopes: Sequence[float] | None = None,
        offsets: Sequence[float] | None = None,
        basis: str = "Y",
        comparator_method: str = "ripple",
        name: str = "pw_lin",
    ):
        if num_state_qubits < 1:
            raise_error(ValueError, "`num_state_qubits` must be at least 1.")
        if comparator_method not in COMPARATOR_METHODS:
            raise_error(
                ValueError,
                f"Unknown comparator method {comparator_method!r}; use {COMPARATOR_METHODS}.",
            )
        self.breakpoints = (
            [0.0] if breakpoints is None else [float(p) for p in breakpoints]
        )
        self.slopes = [1.0] if slopes is None else [float(s) for s in slopes]
        self.offsets = [0.0] if offsets is None else [float(o) for o in offsets]
        if not (len(self.breakpoints) == len(self.slopes) == len(self.offsets)):
            raise_error(
                ValueError, "Mismatching sizes of breakpoints, slopes and offsets."
            )
        if len(self.breakpoints) == 0:
            raise_error(ValueError, "At least one breakpoint is required.")
        self.basis = _check_basis(basis)
        self.comparator_method = comparator_method
        n = num_state_qubits
        self.num_state_qubits = n

        multiple = len(self.breakpoints) > 1
        n_helpers = num_comparator_ancillas(n, comparator_method) if multiple else 0
        total = n + 1 + (1 + n_helpers if multiple else 0)
        super().__init__(total, name)
        self.state_qubits = list(range(n))
        self.target_qubit = n
        self.compare_qubit = n + 1 if multiple else None
        self.helper_qubits = list(range(n + 2, total)) if multiple else []

        mapped_slopes = self.mapped_slopes
        mapped_offsets = self.mapped_offsets
        for i, point in enumerate(self.breakpoints):
            if i == 0 and self.contains_zero_breakpoint:
                self._add(
                    linear_rotation_gates(
                        self.state_qubits,
                        self.target_qubit,
                        mapped_slopes[i],
                        mapped_offsets[i],
                        self.basis,
                    )
                )
            else:
                comparator = integer_comparator_gates(
                    self.state_qubits,
                    self.compare_qubit,
                    point,
                    True,
                    self.helper_qubits,
                    comparator_method,
                )
                self._add(comparator)
                self._add(
                    linear_rotation_gates(
                        self.state_qubits,
                        self.target_qubit,
                        mapped_slopes[i],
                        mapped_offsets[i],
                        self.basis,
                        controls=[self.compare_qubit],
                    )
                )
                # the comparator is an involution: applying it again uncomputes the flag
                self._add(
                    integer_comparator_gates(
                        self.state_qubits,
                        self.compare_qubit,
                        point,
                        True,
                        self.helper_qubits,
                        comparator_method,
                    )
                )

    @property
    def mapped_slopes(self) -> np.ndarray:
        """Slopes of the internal (cumulative) representation."""
        mapped = np.zeros(len(self.slopes), dtype=float)
        for i, slope in enumerate(self.slopes):
            mapped[i] = slope - np.sum(mapped[:i])
        return mapped

    @property
    def mapped_offsets(self) -> np.ndarray:
        """Offsets of the internal (cumulative) representation."""
        mapped = np.zeros(len(self.offsets), dtype=float)
        for i, (offset, slope, point) in enumerate(
            zip(self.offsets, self.slopes, self.breakpoints)
        ):
            mapped[i] = offset - slope * point - np.sum(mapped[:i])
        return mapped

    @property
    def contains_zero_breakpoint(self) -> bool:
        """Whether 0 is the first breakpoint."""
        return bool(np.isclose(0, self.breakpoints[0]))

    def evaluate(self, x: float) -> float:
        """Classically evaluate the rotation angle function at ``x``."""
        ms, mo = self.mapped_slopes, self.mapped_offsets
        y = (x >= self.breakpoints[0]) * (x * ms[0] + mo[0])
        for i in range(1, len(self.breakpoints)):
            y = y + (x >= self.breakpoints[i]) * (x * ms[i] + mo[i])
        return y
