"""Hardware-efficient ``RealAmplitudes`` ansatz."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qibo import Circuit, gates
from qibo.config import raise_error

ENTANGLEMENTS = ("full", "linear", "reverse_linear", "circular")


def entanglement_pairs(
    nqubits: int, entanglement: str | Sequence[Sequence[int]] = "reverse_linear"
) -> list[tuple[int, int]]:
    """CNOT (control, target) pairs of one entanglement block.

    Args:
        nqubits (int): number of qubits.
        entanglement (str | list[tuple[int, int]]): ``"full"`` (all pairs ``i < j``),
            ``"linear"`` (``(i, i+1)``), ``"reverse_linear"`` (``(i, i+1)`` in reversed order;
            equivalent to ``"full"`` for :func:`real_amplitudes`, with fewer gates),
            ``"circular"`` (``(n-1, 0)`` followed by the linear pairs), or an explicit list.
    """
    n = nqubits
    if not isinstance(entanglement, str):
        pairs = [(int(c), int(t)) for c, t in entanglement]
        for c, t in pairs:
            if not (0 <= c < n and 0 <= t < n) or c == t:
                raise_error(
                    ValueError, f"Invalid entangling pair ({c}, {t}) for {n} qubits."
                )
        return pairs
    if entanglement == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "reverse_linear":
        return [(i, i + 1) for i in reversed(range(n - 1))]
    if entanglement == "circular":
        linear = [(i, i + 1) for i in range(n - 1)]
        return ([(n - 1, 0)] + linear) if n > 2 else linear
    raise_error(
        ValueError,
        f"Unknown entanglement {entanglement!r}; use one of {ENTANGLEMENTS} or a list of pairs.",
    )


def num_real_amplitudes_parameters(
    nqubits: int, reps: int = 3, skip_final_rotation_layer: bool = False
) -> int:
    """Number of trainable parameters of :func:`real_amplitudes`."""
    return nqubits * (reps + (0 if skip_final_rotation_layer else 1))


def real_amplitudes(
    nqubits: int,
    reps: int = 3,
    entanglement: str | Sequence[Sequence[int]] = "reverse_linear",
    skip_final_rotation_layer: bool = False,
    parameters: Sequence[float] | None = None,
) -> Circuit:
    r"""Hardware-efficient :math:`R_y` + CNOT ansatz producing real amplitudes.

    Qibo counterpart of Qiskit's ``RealAmplitudes`` (a ``TwoLocal`` circuit with :math:`R_y`
    rotation layers and CNOT entanglement layers): ``reps`` times a layer of :math:`R_y`
    gates on every qubit followed by an entanglement block of CNOTs, and a final
    :math:`R_y` layer (unless ``skip_final_rotation_layer``). With all-real parameters the
    unitary, hence the prepared state, is real.

    The trainable parameters are the :math:`R_y` angles in layer-major order
    (``theta[layer * nqubits + qubit]``), the same order as Qiskit's parameter vector, so
    values can be passed to :meth:`qibo.models.circuit.Circuit.set_parameters`.

    Args:
        nqubits (int): number of qubits.
        reps (int): number of entanglement repetitions (default 3).
        entanglement (str | list): entanglement pattern, see :func:`entanglement_pairs`.
        skip_final_rotation_layer (bool): omit the last :math:`R_y` layer.
        parameters (list[float]): initial angles; zeros if ``None``.

    Returns:
        :class:`qibo.models.circuit.Circuit`: the ansatz.

    Example:

        .. code-block:: python

            import numpy as np
            from qiboopt.circuit_library import real_amplitudes

            ansatz = real_amplitudes(4, reps=2)
            ansatz.set_parameters(np.random.uniform(0, 2 * np.pi, 12))
            state = ansatz().state()
    """
    if nqubits < 1:
        raise_error(ValueError, "`nqubits` must be at least 1.")
    if reps < 0:
        raise_error(ValueError, "`reps` must be non-negative.")
    nparams = num_real_amplitudes_parameters(nqubits, reps, skip_final_rotation_layer)
    if parameters is None:
        parameters = np.zeros(nparams)
    parameters = np.asarray(parameters, dtype=float).ravel()
    if parameters.size != nparams:
        raise_error(
            ValueError, f"Expected {nparams} parameters, got {parameters.size}."
        )
    pairs = entanglement_pairs(nqubits, entanglement) if nqubits > 1 else []

    circuit = Circuit(nqubits)
    idx = 0
    for layer in range(reps + 1):
        if layer == reps and skip_final_rotation_layer:
            break
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=float(parameters[idx])))
            idx += 1
        if layer < reps:
            for control, target in pairs:
                circuit.add(gates.CNOT(control, target))
    return circuit
