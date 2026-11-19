"""Container describing an amplitude estimation problem."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from qibo import Circuit
from qibo.config import raise_error


class EstimationProblem:
    r"""Input of the amplitude estimation algorithms.

    It bundles everything an algorithm needs: the state preparation :math:`\mathcal A`, the
    qubits whose measurement defines the *good* states, and optionally a custom Grover operator
    and a classical post-processing. The amplitude :math:`a = \|\Pi_{good}\mathcal A|0\rangle\|^2`
    is what the algorithms estimate.

    Args:
        state_preparation (:class:`qibo.models.circuit.Circuit`): the circuit :math:`\mathcal A`
            (without measurements).
        objective_qubits (int | list[int]): qubit(s) measured to decide whether a state is good.
        grover_operator (:class:`qibo.models.circuit.Circuit`): optional custom
            :math:`\mathcal Q` on the same qubits as ``state_preparation``. By default it is
            built with :func:`~qiboopt.amplitude_estimation.grover_operator`.
        post_processing (Callable): map applied to the estimated amplitude (and to the ends of
            its confidence interval); identity by default. Typically
            ``LinearAmplitudeFunction.post_processing``.
        is_good_state (Callable[[str], bool]): predicate on the bitstring of the objective
            qubits. **Character** ``i`` **of the bitstring is the outcome of**
            ``objective_qubits[i]`` (the order of the Qibo measurement keys). Defaults to "all
            objective qubits are 1".
        objective_qubit (int): alias of a single ``objective_qubits`` (singular spelling).

    Example:

        .. code-block:: python

            problem = EstimationProblem(
                state_preparation=circ,           # A, e.g. loader + LinearAmplitudeFunction
                objective_qubits=[F.objective_qubit],
                post_processing=F.post_processing,
            )
    """

    def __init__(
        self,
        state_preparation: Circuit,
        objective_qubits: int | Sequence[int] | None = None,
        grover_operator: Circuit | None = None,
        post_processing: Callable | None = None,
        is_good_state: Callable[[str], bool] | None = None,
        *,
        objective_qubit: int | None = None,
    ):
        if objective_qubit is not None:
            if objective_qubits is not None:
                raise_error(
                    ValueError,
                    "Pass either `objective_qubits` or `objective_qubit`, not both.",
                )
            objective_qubits = objective_qubit
        if objective_qubits is None:
            raise_error(ValueError, "`objective_qubits` must be given.")
        if not isinstance(state_preparation, Circuit):
            raise_error(TypeError, "`state_preparation` must be a qibo Circuit.")
        self._objective_qubits = None
        self._state_preparation = None
        self._grover_operator = None
        self.state_preparation = state_preparation
        self.objective_qubits = objective_qubits
        self.grover_operator = grover_operator
        self._post_processing = post_processing
        self._is_good_state = is_good_state

    # -- state preparation -------------------------------------------------------------------
    @property
    def state_preparation(self) -> Circuit:
        """The :math:`\\mathcal A` operator."""
        return self._state_preparation

    @state_preparation.setter
    def state_preparation(self, circuit: Circuit) -> None:
        self._state_preparation = circuit
        self._cached_grover = None

    # -- objective qubits --------------------------------------------------------------------
    @property
    def objective_qubits(self) -> list[int]:
        """Qubits measured to decide whether a state is good."""
        return list(self._objective_qubits)

    @objective_qubits.setter
    def objective_qubits(self, qubits: int | Sequence[int]) -> None:
        if hasattr(qubits, "__index__"):  # a single (python or numpy) integer
            qubits = [int(qubits)]
        else:
            qubits = [int(q) for q in qubits]
        if len(qubits) == 0:
            raise_error(ValueError, "At least one objective qubit is required.")
        if len(set(qubits)) != len(qubits):
            raise_error(ValueError, f"Repeated objective qubits: {qubits}.")
        nqubits = self._state_preparation.nqubits
        if any(q < 0 or q >= nqubits for q in qubits):
            raise_error(
                ValueError,
                f"Objective qubits {qubits} out of range for a {nqubits}-qubit circuit.",
            )
        self._objective_qubits = qubits
        self._cached_grover = None

    # -- post processing ---------------------------------------------------------------------
    @property
    def post_processing(self) -> Callable:
        """Post-processing of the estimated amplitude (identity by default)."""
        if self._post_processing is None:
            return lambda x: x
        return self._post_processing

    @post_processing.setter
    def post_processing(self, post_processing: Callable | None) -> None:
        self._post_processing = post_processing

    # -- good states -------------------------------------------------------------------------
    @property
    def has_good_state(self) -> bool:
        """Whether a custom ``is_good_state`` has been set."""
        return self._is_good_state is not None

    @property
    def is_good_state(self) -> Callable[[str], bool]:
        """Predicate on the measured bitstring of the objective qubits."""
        if self._is_good_state is None:
            return lambda bitstring: all(bit == "1" for bit in bitstring)
        return self._is_good_state

    @is_good_state.setter
    def is_good_state(self, is_good_state: Callable[[str], bool] | None) -> None:
        self._is_good_state = is_good_state
        self._cached_grover = None

    # -- Grover operator ---------------------------------------------------------------------
    @property
    def grover_operator(self) -> Circuit:
        r"""The Grover operator :math:`\mathcal Q`, built on demand if not given."""
        if self._grover_operator is not None:
            return self._grover_operator
        if self._cached_grover is None:
            # imported here to avoid a circular import
            from qiboopt.amplitude_estimation.grover import grover_operator

            self._cached_grover = grover_operator(
                self.state_preparation,
                self.objective_qubits,
                is_good_state=self._is_good_state,
            )
        return self._cached_grover

    @grover_operator.setter
    def grover_operator(self, grover_operator: Circuit | None) -> None:
        if grover_operator is not None:
            if not isinstance(grover_operator, Circuit):
                raise_error(TypeError, "`grover_operator` must be a qibo Circuit.")
            if grover_operator.nqubits != self._state_preparation.nqubits:
                raise_error(
                    ValueError,
                    "`grover_operator` and `state_preparation` must act on the same number of "
                    f"qubits ({grover_operator.nqubits} != {self._state_preparation.nqubits}).",
                )
        self._grover_operator = grover_operator
