"""Autodiff utilities for basis gradients."""

from __future__ import annotations

import torch


def basis_gradients(coords: torch.Tensor, phi: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Return gradients dphi_i/dx_j as [N, M, 2]."""
    n, m = phi.shape
    grads = []
    for i in range(m):
        gi = torch.autograd.grad(
            phi[:, i].sum(),
            coords,
            create_graph=create_graph,
            retain_graph=create_graph or i < (m - 1),
        )[0]
        grads.append(gi)
    return torch.stack(grads, dim=1).reshape(n, m, 2)
