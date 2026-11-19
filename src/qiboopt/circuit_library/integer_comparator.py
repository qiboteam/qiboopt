"""Integer comparator: ``|i>|t> -> |i>|t XOR (i >= L)>``.

Port of ``qiskit.circuit.library.IntegerComparator`` to Qibo. The ripple (two's complement
carry) construction follows Appendix B of Gacon, Zoufal and Woerner, *Quantum-Enhanced
Simulation-Based Optimization*, arXiv:2005.10780, which is built on the carry chain of
Cuccaro et al., arXiv:quant-ph/0410184. The ancilla-free ("greedy") construction places one
multi-controlled X per accepted value. Gate sequences follow the Qiskit implementation
(Apache-2.0, https://github.com/Qiskit/qiskit).
"""

from __future__ import annotations

from collections.abc import Sequence

from qibo import gates
from qibo.config import raise_error

from qiboopt.circuit_library._base import CircuitBlock, ceil_tol, check_qubit_list

COMPARATOR_METHODS = ("ripple", "greedy")


def num_comparator_ancillas(num_state_qubits: int, method: str = "ripple") -> int:
    """Number of helper (ancilla) qubits needed by a comparator, excluding the target."""
    if method not in COMPARATOR_METHODS:
        raise_error(
            ValueError,
            f"Unknown comparator method {method!r}; use {COMPARATOR_METHODS}.",
        )
    if method == "greedy":
        return 0
    return max(num_state_qubits - 1, 0)


def _or_gates(a: int, b: int, target: int) -> list:
    """Gates implementing ``target ^= (a OR b)``."""
    return [
        gates.X(a),
        gates.X(b),
        gates.TOFFOLI(a, b, target),
        gates.X(a),
        gates.X(b),
        gates.X(target),
    ]


def integer_comparator_gates(
    state_qubits: Sequence[int],
    target: int,
    value: float,
    geq: bool = True,
    ancillas: Sequence[int] = (),
    method: str = "ripple",
) -> list:
    """Gate list of an integer comparator acting on the given qubits.

    The returned gates map ``|i>|t>|0...0> -> |i>|t XOR (i >= L)>|0...0>`` (or ``i < L`` if
    ``geq`` is ``False``), where ``L = ceil(value)`` and :math:`i` is the integer encoded in
    ``state_qubits`` (first qubit = least significant bit). The target is *XOR-ed*, so it need
    not be in :math:`|0\\rangle`. The operation is an involution, hence applying the same
    gate list twice uncomputes it.

    Args:
        state_qubits (list[int]): qubits encoding :math:`i`, least significant bit first.
        target (int): qubit receiving the comparison result.
        value (float): the classical value :math:`L`; rounded up to the next integer.
        geq (bool): ``True`` for :math:`i \\geq L`, ``False`` for :math:`i < L`.
        ancillas (list[int]): ``num_state_qubits - 1`` clean ancillas, only for ``"ripple"``.
            They are returned to :math:`|0\\rangle`.
        method (str): ``"ripple"`` (:math:`O(n)` gates, :math:`n-1` ancillas) or
            ``"greedy"`` (no ancillas, up to :math:`2^{n-1}` multi-controlled X gates).

    Returns:
        list: Qibo gates.
    """
    state_qubits = check_qubit_list(state_qubits, "state_qubits")
    n = len(state_qubits)
    if n < 1:
        raise_error(ValueError, "At least one state qubit is required.")
    if method not in COMPARATOR_METHODS:
        raise_error(
            ValueError,
            f"Unknown comparator method {method!r}; use {COMPARATOR_METHODS}.",
        )
    ancillas = list(ancillas)
    used = state_qubits + [target]
    if method == "ripple":
        needed = num_comparator_ancillas(n, "ripple")
        if len(ancillas) < needed:
            raise_error(
                ValueError,
                f"The ripple comparator needs {needed} ancillas, got {len(ancillas)}.",
            )
        ancillas = ancillas[:needed]
        used = used + ancillas
    if len(set(used)) != len(used):
        raise_error(
            ValueError, "State, target and ancilla qubits must all be distinct."
        )

    threshold = ceil_tol(value)
    if threshold <= 0:  # i >= L always true, i < L never true
        return [gates.X(target)] if geq else []
    if threshold >= 2**n:  # i >= L never true, i < L always true
        return [] if geq else [gates.X(target)]

    if method == "greedy":
        return _greedy_gates(state_qubits, target, threshold, geq)

    # --- ripple: carry-out of  i + (2^n - L)  is 1  <=>  i >= L -----------------------------
    if n == 1:  # here L == 1
        out = [gates.CNOT(state_qubits[0], target)]
        if not geq:
            out.append(gates.X(target))
        return out

    twos = 2**n - threshold
    bits = [(twos >> j) & 1 for j in range(n)]  # least significant bit first

    def carry_step(i: int, out_qubit: int) -> list:
        """Compute carry c_{i+1} onto ``out_qubit`` (XOR) from x_i and c_i."""
        if i == 0:  # c_0 = 0: c_1 = x_0 if bit else 0
            return [gates.CNOT(state_qubits[0], out_qubit)] if bits[0] else []
        carry_in = ancillas[i - 1]
        if bits[i]:  # c_{i+1} = x_i OR c_i
            return _or_gates(state_qubits[i], carry_in, out_qubit)
        return [gates.TOFFOLI(state_qubits[i], carry_in, out_qubit)]  # x_i AND c_i

    forward = []
    for i in range(n - 1):
        forward += carry_step(i, ancillas[i])
    out = list(forward)
    out += carry_step(n - 1, target)
    if not geq:
        out.append(gates.X(target))
    # uncompute the carries in reverse order (every gate is self-inverse)
    for i in reversed(range(n - 1)):
        out += carry_step(i, ancillas[i])
    return out


def _greedy_gates(
    state_qubits: list[int], target: int, threshold: int, geq: bool
) -> list:
    n = len(state_qubits)
    out = []
    # always use the side with fewer than 2^n / 2 multi-controlled gates
    if (threshold < 2 ** (n - 1) and geq) or (threshold > 2 ** (n - 1) and not geq):
        geq = not geq
        out.append(gates.X(target))
    accepted = range(threshold, 2**n) if geq else range(threshold)
    for v in accepted:
        zeros = [q for j, q in enumerate(state_qubits) if not (v >> j) & 1]
        out += [gates.X(q) for q in zeros]
        out.append(gates.X(target).controlled_by(*state_qubits))
        out += [gates.X(q) for q in zeros]
    return out


class IntegerComparator(CircuitBlock):
    r"""Compare a qubit register with a classical integer.

    Implements

    .. math::

        |i\rangle_n |t\rangle |0\rangle \mapsto |i\rangle_n |t \oplus (i \geq L)\rangle |0\rangle

    (or :math:`i < L` when ``geq=False``). It is the Qibo counterpart of Qiskit's
    ``IntegerComparator``.

    Qubit layout (``nqubits = num_state_qubits + 1 + num_ancillas``):

    * ``0 .. n-1``: state register, least significant bit first,
    * ``n``: compare (target) qubit,
    * ``n+1 ..``: ancillas (``n-1`` for ``method="ripple"``, none for ``"greedy"``).

    Args:
        num_state_qubits (int): number of qubits :math:`n` of the register.
        value (float): classical value :math:`L` (rounded up).
        geq (bool): compute :math:`i \geq L` (default) or :math:`i < L`.
        method (str): ``"ripple"`` (default; linear gate count) or ``"greedy"`` (no ancillas).

    Example:

        .. code-block:: python

            from qibo import Circuit
            from qiboopt.circuit_library import IntegerComparator

            comp = IntegerComparator(3, 5)  # flips qubit 3 if the register holds i >= 5
            c = Circuit(comp.nqubits)
            c.add(comp.on_qubits(*range(comp.nqubits)))
    """

    def __init__(
        self,
        num_state_qubits: int,
        value: float,
        geq: bool = True,
        method: str = "ripple",
        name: str = "IntComp",
    ):
        if num_state_qubits < 1:
            raise_error(ValueError, "`num_state_qubits` must be at least 1.")
        n_anc = num_comparator_ancillas(num_state_qubits, method)
        super().__init__(num_state_qubits + 1 + n_anc, name)
        self.num_state_qubits = num_state_qubits
        self.value = value
        self.geq = geq
        self.method = method
        self.num_ancillas = n_anc
        self.state_qubits = list(range(num_state_qubits))
        self.compare_qubit = num_state_qubits
        self.ancilla_qubits = list(range(num_state_qubits + 1, self.nqubits))
        self._add(
            integer_comparator_gates(
                self.state_qubits,
                self.compare_qubit,
                value,
                geq,
                self.ancilla_qubits,
                method,
            )
        )
