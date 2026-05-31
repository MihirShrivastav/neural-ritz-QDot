from __future__ import annotations

import torch


def assemble_ritz(points: torch.Tensor, weights: torch.Tensor, potential: torch.Tensor, basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble overlap and weak-form Hamiltonian matrices for basis columns."""
    m = basis.shape[1]
    gradients = []
    for j in range(m):
        grad_j = torch.autograd.grad(basis[:, j].sum(), points, create_graph=True, retain_graph=True)[0]
        gradients.append(grad_j)
    grad = torch.stack(gradients, dim=2)
    weighted_basis = basis * weights[:, None]
    overlap = basis.T @ weighted_basis
    kinetic = torch.zeros((m, m), dtype=basis.dtype, device=basis.device)
    for i in range(m):
        for j in range(m):
            kinetic[i, j] = torch.sum(weights * torch.sum(grad[:, :, i] * grad[:, :, j], dim=1))
    potential_matrix = basis.T @ (basis * (weights * potential)[:, None])
    hamiltonian = kinetic + potential_matrix
    return 0.5 * (overlap + overlap.T), 0.5 * (hamiltonian + hamiltonian.T)


def solve_generalized(hamiltonian: torch.Tensor, overlap: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve H c = S c e using Cholesky reduction."""
    n = overlap.shape[0]
    eye = torch.eye(n, dtype=overlap.dtype, device=overlap.device)
    chol = torch.linalg.cholesky(overlap + eps * eye)
    inv_chol = torch.linalg.inv(chol)
    standard = inv_chol @ hamiltonian @ inv_chol.T
    vals, vecs = torch.linalg.eigh(0.5 * (standard + standard.T))
    coeffs = inv_chol.T @ vecs
    return vals, coeffs
