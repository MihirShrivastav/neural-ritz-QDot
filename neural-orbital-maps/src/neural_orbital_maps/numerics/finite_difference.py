from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh

from neural_orbital_maps.io.config import RunConfig
from neural_orbital_maps.numerics.grid import Grid, make_uniform_grid
from neural_orbital_maps.physics.potentials import evaluate_potential


@dataclass(frozen=True)
class FiniteDifferenceResult:
    grid: Grid
    potential: np.ndarray
    orbitals: np.ndarray
    energies: np.ndarray


def _one_dimensional_dirichlet_laplacian(num_points: int, spacing: float) -> sparse.csr_matrix:
    main = np.full(num_points, 2.0 / (spacing * spacing), dtype=float)
    off = np.full(num_points - 1, -1.0 / (spacing * spacing), dtype=float)
    return sparse.diags([off, main, off], [-1, 0, 1], format="csr")


def _normalize_orbitals(vectors: np.ndarray, num_points: int, cell_area: float) -> np.ndarray:
    orbitals = vectors.T.reshape(vectors.shape[1], num_points, num_points)
    for idx in range(orbitals.shape[0]):
        norm = float(np.sum(orbitals[idx] * orbitals[idx]) * cell_area)
        if norm > 0:
            orbitals[idx] = orbitals[idx] / np.sqrt(norm)
    return orbitals


def solve_finite_difference(config: RunConfig, num_states: int | None = None) -> FiniteDifferenceResult:
    """Solve the one-electron Hamiltonian with a conventional five-point stencil.

    The operator matches the neural Block-Ritz weak form: H = -Delta + V in the
    same dimensionless coordinates, with homogeneous Dirichlet values outside
    the rectangular grid.
    """
    k = num_states or config.solver.num_states
    if k >= config.domain.num_points * config.domain.num_points:
        raise ValueError("finite-difference num_states must be smaller than the number of grid nodes")
    grid = make_uniform_grid(config.domain, dtype=torch.float64, device="cpu")
    potential = evaluate_potential(grid.points, config.potential).detach().cpu().numpy()
    nx = config.domain.num_points
    ny = config.domain.num_points
    lx = _one_dimensional_dirichlet_laplacian(nx, grid.dx)
    ly = _one_dimensional_dirichlet_laplacian(ny, grid.dy)
    identity_x = sparse.identity(nx, format="csr")
    identity_y = sparse.identity(ny, format="csr")
    kinetic = sparse.kron(identity_y, lx, format="csr") + sparse.kron(ly, identity_x, format="csr")
    hamiltonian = kinetic + sparse.diags(potential, format="csr")
    vals, vecs = eigsh(hamiltonian, k=k, which="SA")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    orbitals = _normalize_orbitals(vecs, config.domain.num_points, grid.cell_area)
    return FiniteDifferenceResult(
        grid=grid,
        potential=potential.reshape(config.domain.num_points, config.domain.num_points),
        orbitals=orbitals,
        energies=vals,
    )
