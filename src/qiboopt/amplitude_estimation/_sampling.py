"""Execution of measured circuits to obtain bitstring counts."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from qibo import Circuit
from qibo.backends import _check_backend
from qibo.config import raise_error

Sampler = Callable[[Circuit, int], Mapping[str, int]]


def run_counts(
    circuit: Circuit, shots: int, backend=None, sampler: Sampler | None = None
) -> dict[str, int]:
    """Execute ``circuit`` for ``shots`` shots and return ``{bitstring: counts}``.

    Args:
        circuit (Circuit): circuit with a measurement of the objective qubits.
        shots (int): number of shots.
        backend: Qibo backend; the global backend if ``None``. Ignored if ``sampler`` is set.
        sampler (Callable): optional ``sampler(circuit, shots) -> {bitstring: counts}``,
            e.g. to execute noisy circuits or to run on hardware. The bitstring characters
            follow the order of the measured objective qubits.
    """
    if sampler is not None:
        counts = sampler(circuit, shots)
    else:
        backend = _check_backend(backend)
        result = backend.execute_circuit(circuit, nshots=shots)
        counts = result.frequencies(binary=True)
    counts = {str(key): int(value) for key, value in dict(counts).items()}
    if sum(counts.values()) <= 0:
        raise_error(RuntimeError, "The sampler returned no counts.")
    return counts


def good_counts(counts: Mapping[str, int], is_good_state: Callable[[str], bool]) -> int:
    """Number of shots whose bitstring is a good state."""
    return int(sum(c for bitstring, c in counts.items() if is_good_state(bitstring)))
