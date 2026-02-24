"""qiboml integration helpers for QAOA training."""

from __future__ import annotations

from typing import Any

import numpy as np


def _energy_shift(qubo) -> float:
    """Constant shift between Ising expectation and QUBO objective value."""
    _h, _J, constant = qubo.qubo_to_ising()
    return float(constant)


def _get_differentiation_class(name: str | None):
    if name is None or name == "torch":
        return None
    from qiboml.operations.differentiation import PSR, Adjoint, Jax

    mapping = {
        "psr": PSR,
        "jax": Jax,
        "adjoint": Adjoint,
        "torch": None,
    }
    diff = mapping.get(name.lower())
    if diff is None:
        raise ValueError(
            "Unknown qiboml differentiation method. "
            "Supported values are: None, 'psr', 'jax', 'adjoint', 'torch'."
        )
    return diff


def optimize_qaoa_with_qiboml(
    *,
    qubo,
    parameters,
    p: int,
    nshots: int | None,
    noise_model,
    custom_mixer,
    has_alphas: bool,
    optimizer: str,
    lr: float,
    epochs: int,
    differentiation: str | None,
    backend,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Optimize QAOA parameters using qiboml's pytorch interface."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "engine='qiboml' requires torch. Install optional dependencies, "
            "for example with `poetry install --with qiboml`."
        ) from exc

    try:
        from qiboml.interfaces.pytorch import QuantumModel
        from qiboml.models.decoding import Expectation
    except ImportError as exc:
        raise ImportError(
            "engine='qiboml' requires qiboml. Install optional dependencies, "
            "for example with `poetry install --with qiboml`."
        ) from exc

    # Reuse qiboopt's own QAOA-object construction path for the Hamiltonian.
    hamiltonian = qubo.qubo_to_qaoa_object().hamiltonian
    if not hasattr(hamiltonian, "expectation_from_circuit"):
        # qiboml 0.1.0 calls this method on SymbolicHamiltonian.
        hamiltonian.expectation_from_circuit = hamiltonian.expectation
    circuit_builder = qubo.make_qaoa_circuit_callable(
        p=p,
        custom_mixer=custom_mixer,
        has_alphas=has_alphas,
        include_measurements=False,
    )
    decoder = Expectation(
        nqubits=qubo.n,
        observable=hamiltonian,
        nshots=nshots,
        noise_model=noise_model,
        backend=backend,
    )
    energy_shift = _energy_shift(qubo)

    diff_class = _get_differentiation_class(differentiation)
    model = QuantumModel(
        circuit_structure=[circuit_builder],
        decoding=decoder,
        parameters_initialization=np.asarray(parameters, dtype=np.float64),
        differentiation=diff_class,
    )
    model = model.to(dtype=torch.float64)

    optimizer_name = optimizer.lower()
    if optimizer_name == "adam":
        torch_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(
            "Unsupported optimizer for qiboml engine. Use 'adam' or 'sgd'."
        )

    losses = []
    best = float("inf")
    best_params = np.asarray(parameters, dtype=np.float64)
    for _ in range(epochs):
        torch_optimizer.zero_grad()
        loss = model()
        if loss.ndim > 0:
            loss = loss.reshape(-1)[0]
        loss.backward()
        torch_optimizer.step()
        loss_value = float(loss.detach().cpu().item()) + energy_shift
        losses.append(loss_value)
        if loss_value < best:
            best = loss_value
            best_params = model.circuit_parameters.detach().cpu().numpy().copy()

    extra = {
        "engine": "qiboml",
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "epochs": epochs,
        "loss_history": losses,
    }
    return best, best_params, extra
