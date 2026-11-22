"""Shared helpers for the circuit library."""

from __future__ import annotations

import math
from collections.abc import Iterable

from qibo import Circuit
from qibo.config import raise_error


def ceil_tol(value: float, tol: float = 1e-9) -> int:
    """Ceiling of ``value`` that is robust to floating point round-off.

    ``ceil(3.0000000000000004)`` would be ``4``, although ``3`` is clearly meant when
    ``value`` is a breakpoint obtained from an affine rescaling. Values within ``tol``
    above an integer are therefore treated as that integer.
    """
    return int(math.ceil(value - tol))


def check_qubit_list(qubits: Iterable[int], name: str) -> list[int]:
    """Validate a list of distinct, non-negative integer qubit indices."""
    qubits = [int(q) for q in qubits]
    if len(set(qubits)) != len(qubits):
        raise_error(ValueError, f"`{name}` contains repeated qubits: {qubits}.")
    if any(q < 0 for q in qubits):
        raise_error(ValueError, f"`{name}` contains negative qubit indices: {qubits}.")
    return qubits


class CircuitBlock:
    """Base class of the reusable sub-circuits of :mod:`qiboopt.circuit_library`.

    A block is a thin wrapper around a :class:`qibo.models.circuit.Circuit` built from a
    fixed qubit layout. It is meant to be embedded in a larger circuit with
    :meth:`on_qubits`, exactly like any other Qibo sub-circuit:

    .. code-block:: python

        block = IntegerComparator(3, 5)
        big = Circuit(8)
        big.add(block.on_qubits(0, 1, 2, 3, 4, 5))

    **Qubit-order convention.** Wherever a block represents an integer :math:`x` on a
    register of qubits, the qubit with the *smallest position in the register* carries the
    *least significant bit* (weight :math:`2^0`), the next one weight :math:`2^1`, and so
    on. This is the convention of the Qiskit primitives these blocks replace. It is
    independent of Qibo's own state-vector convention (qubit 0 is the most significant
    bit of the basis-state index).
    """

    def __init__(self, nqubits: int, name: str):
        self._circuit = Circuit(nqubits)
        self.name = name

    @property
    def circuit(self) -> Circuit:
        """The underlying :class:`qibo.models.circuit.Circuit`."""
        return self._circuit

    @property
    def nqubits(self) -> int:
        """Total number of qubits (registers + ancillas) of the block."""
        return self._circuit.nqubits

    @property
    def num_qubits(self) -> int:
        """Alias of :attr:`nqubits` (Qiskit naming)."""
        return self._circuit.nqubits

    @property
    def queue(self):
        """The gates of the block, in application order."""
        return self._circuit.queue

    def _add(self, gate_list) -> None:
        self._circuit.add(gate_list)

    def on_qubits(self, *qubits: int):
        """Generator of the block's gates mapped onto ``qubits``.

        Args:
            qubits (int): one target qubit per block qubit (``len(qubits) == nqubits``).
        """
        return self._circuit.on_qubits(*qubits)

    def invert(self) -> Circuit:
        """Return the inverse (dagger) of the block as a new circuit."""
        return self._circuit.invert()

    def unitary(self, backend=None):
        """Dense unitary matrix of the block (Qibo big-endian basis ordering)."""
        return self._circuit.unitary(backend)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, nqubits={self.nqubits})"
