from __future__ import annotations

from dataclasses import dataclass
from logging import Logger

import numpy as np
import torch

from neural_orbital_maps.io.config import RunConfig
from neural_orbital_maps.models.basis import BasisNet
from neural_orbital_maps.numerics.grid import Grid, make_uniform_grid
from neural_orbital_maps.numerics.ritz import assemble_ritz, solve_generalized
from neural_orbital_maps.physics.potentials import evaluate_potential


@dataclass(frozen=True)
class OneElectronResult:
    grid: Grid
    potential: np.ndarray
    orbitals: np.ndarray
    energies: np.ndarray
    overlap: np.ndarray
    hamiltonian: np.ndarray
    coefficients: np.ndarray
    projected_residuals: np.ndarray
    metrics: list[dict]
    model_state: dict


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def train_one_electron(config: RunConfig, logger: Logger | None = None) -> OneElectronResult:
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    dtype = _dtype(config.training.dtype)
    grid = make_uniform_grid(config.domain, dtype=dtype, device=config.training.device)
    model = BasisNet(
        out_dim=config.solver.basis_size,
        hidden_dim=config.solver.hidden_dim,
        hidden_layers=config.solver.hidden_layers,
        kind=config.solver.network,
        envelope_alpha=config.solver.envelope_alpha,
    ).to(device=config.training.device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    metrics: list[dict] = []
    best_loss = float("inf")
    stale = 0

    for step in range(1, config.training.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        points = grid.points.detach().clone().requires_grad_(True)
        potential = evaluate_potential(points, config.potential)
        basis = model(points)
        overlap, hamiltonian = assemble_ritz(points, grid.weights, potential, basis)
        vals, _ = solve_generalized(hamiltonian, overlap)
        loss = vals[: config.solver.num_states].sum()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if step % config.training.log_every == 0 or step == 1 or step == config.training.steps:
            item = {"step": step, "eigsum": loss_value, "E0": float(vals[0].detach().cpu())}
            metrics.append(item)
            if logger:
                logger.info("step=%s eigsum=%.8f E0=%.8f", step, item["eigsum"], item["E0"])

        if loss_value + config.training.early_stop_min_delta < best_loss:
            best_loss = loss_value
            stale = 0
        else:
            stale += 1
        if stale >= config.training.early_stop_patience and step >= config.training.log_every:
            if logger:
                logger.info("early stop at step=%s best_eigsum=%.8f", step, best_loss)
            break

    points = grid.points.detach().clone().requires_grad_(True)
    potential_t = evaluate_potential(points, config.potential)
    basis = model(points)
    overlap, hamiltonian = assemble_ritz(points, grid.weights, potential_t, basis)
    vals, coeffs = solve_generalized(hamiltonian, overlap)
    psi = basis @ coeffs[:, : config.solver.num_states]
    projected_residuals = hamiltonian @ coeffs[:, : config.solver.num_states] - overlap @ coeffs[:, : config.solver.num_states] @ torch.diag(vals[: config.solver.num_states])
    shape = config.domain.num_points
    orbitals = psi.detach().cpu().numpy().T.reshape(config.solver.num_states, shape, shape)
    return OneElectronResult(
        grid=grid,
        potential=potential_t.detach().cpu().numpy().reshape(shape, shape),
        orbitals=orbitals,
        energies=vals[: config.solver.num_states].detach().cpu().numpy(),
        overlap=overlap.detach().cpu().numpy(),
        hamiltonian=hamiltonian.detach().cpu().numpy(),
        coefficients=coeffs[:, : config.solver.num_states].detach().cpu().numpy(),
        projected_residuals=torch.linalg.norm(projected_residuals, dim=0).detach().cpu().numpy(),
        metrics=metrics,
        model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
    )
