"""Cuccaro-Draper-Kutin-Moulton (CDKM) ripple-carry adder.

Port of ``qiskit.circuit.library.CDKMRippleCarryAdder`` /
``qiskit.synthesis.adder_ripple_c04`` to Qibo. Reference: S. A. Cuccaro, T. G. Draper,
S. A. Kutin and D. P. Moulton, *A new quantum ripple-carry addition circuit*,
arXiv:quant-ph/0410184. The gate wiring follows the Qiskit implementation
(Apache-2.0, https://github.com/Qiskit/qiskit).
"""

from __future__ import annotations

from collections.abc import Sequence

from qibo import gates
from qibo.config import raise_error

from qiboopt.circuit_library._base import CircuitBlock, check_qubit_list

ADDER_KINDS = ("full", "half", "fixed")


def _maj(x0: int, x1: int, x2: int) -> list:
    """MAJ gate of [CDKM04] (``x0`` holds the a-bit, ``x1`` the b-bit, ``x2`` the carry)."""
    return [gates.CNOT(x0, x1), gates.CNOT(x0, x2), gates.TOFFOLI(x2, x1, x0)]


def _uma(x0: int, x1: int, x2: int) -> list:
    """UMA gate of [CDKM04] (2-CNOT version)."""
    return [gates.TOFFOLI(x2, x1, x0), gates.CNOT(x0, x2), gates.CNOT(x2, x1)]


def cdkm_adder_gates(
    a: Sequence[int],
    b: Sequence[int],
    carry_in: int,
    carry_out: int | None = None,
) -> list:
    r"""Gate list of the CDKM in-place adder :math:`|a\rangle|b\rangle \to |a\rangle|a+b\rangle`.

    Args:
        a (list[int]): addend register, least significant bit first. Left unchanged.
        b (list[int]): register overwritten with the sum, least significant bit first.
        carry_in (int): carry-in qubit (must be :math:`|0\rangle` unless it is an intentional
            carry-in bit; returned unchanged).
        carry_out (int): optional qubit that is XOR-ed with the carry-out. If ``None`` the sum
            is taken modulo :math:`2^n`.

    Returns:
        list: Qibo gates.
    """
    a = check_qubit_list(a, "a")
    b = check_qubit_list(b, "b")
    n = len(a)
    if n < 1:
        raise_error(ValueError, "The number of qubits must be at least 1.")
    if len(b) != n:
        raise_error(ValueError, "Registers `a` and `b` must have the same size.")
    everything = a + b + [carry_in] + ([] if carry_out is None else [carry_out])
    if len(set(everything)) != len(everything):
        raise_error(ValueError, "All qubits of the adder must be distinct.")

    out = _maj(a[0], b[0], carry_in)
    for i in range(n - 1):
        out += _maj(a[i + 1], b[i + 1], a[i])
    if carry_out is not None:
        out.append(gates.CNOT(a[-1], carry_out))
    for i in reversed(range(n - 1)):
        out += _uma(a[i + 1], b[i + 1], a[i])
    out += _uma(a[0], b[0], carry_in)
    return out


class CDKMRippleCarryAdder(CircuitBlock):
    r"""In-place ripple-carry adder of Cuccaro et al.

    Computes :math:`|a\rangle|b\rangle \mapsto |a\rangle|(a + b) \bmod 2^n\rangle` (plus the
    carry-out on a separate qubit for ``"full"``/``"half"``). Uses :math:`2n + O(1)` Toffoli
    and :math:`5n + O(1)` CNOT gates and one helper qubit, as in the Qiskit original.
    All registers are least significant bit first.

    Qubit layout depends on ``kind``:

    * ``"full"``  (carry-in and carry-out): ``[cin, a_0..a_{n-1}, b_0..b_{n-1}, cout]``,
      :math:`2n+2` qubits. The sum :math:`a+b+cin` is written to ``b`` and ``cout``.
    * ``"half"``  (carry-out only): ``[a, b, cout, helper]``, :math:`2n+2` qubits.
    * ``"fixed"`` (neither): ``[a, b, helper]``, :math:`2n+1` qubits; result modulo :math:`2^n`.

    ``helper`` (``half``/``fixed``) must be :math:`|0\rangle`; it is returned to :math:`|0\rangle`.

    Args:
        num_state_qubits (int): size :math:`n` of each input register.
        kind (str): ``"full"`` (default), ``"half"`` or ``"fixed"``.
    """

    def __init__(
        self,
        num_state_qubits: int,
        kind: str = "full",
        name: str = "CDKMRippleCarryAdder",
    ):
        if num_state_qubits < 1:
            raise_error(ValueError, "The number of qubits must be at least 1.")
        if kind not in ADDER_KINDS:
            raise_error(
                ValueError, f"`kind` must be one of {ADDER_KINDS}, got {kind!r}."
            )
        n = num_state_qubits
        if kind == "full":
            total = 2 * n + 2
            cin, a = 0, list(range(1, n + 1))
            b = list(range(n + 1, 2 * n + 1))
            cout, helper = 2 * n + 1, None
        elif kind == "half":
            total = 2 * n + 2
            a, b = list(range(n)), list(range(n, 2 * n))
            cout, helper = 2 * n, 2 * n + 1
            cin = helper
        else:
            total = 2 * n + 1
            a, b = list(range(n)), list(range(n, 2 * n))
            cout, helper = None, 2 * n
            cin = helper
        super().__init__(total, name)
        self.num_state_qubits = n
        self.kind = kind
        self.carry_in = 0 if kind == "full" else None
        self.a_qubits = a
        self.b_qubits = b
        self.carry_out = cout
        self.helper_qubit = helper
        self._add(cdkm_adder_gates(a, b, cin, cout))
