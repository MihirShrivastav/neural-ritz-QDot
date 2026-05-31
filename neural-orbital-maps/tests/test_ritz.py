import torch

from neural_orbital_maps.io.config import DomainConfig
from neural_orbital_maps.numerics.grid import make_uniform_grid
from neural_orbital_maps.numerics.ritz import assemble_ritz, solve_generalized


def test_ritz_shapes_and_symmetry():
    grid = make_uniform_grid(DomainConfig(num_points=10), dtype=torch.float64)
    points = grid.points.detach().clone().requires_grad_(True)
    x = points[:, 0]
    y = points[:, 1]
    basis = torch.stack([torch.exp(-(x**2 + y**2)), x * torch.exp(-(x**2 + y**2))], dim=1)
    potential = x * x + y * y
    s, h = assemble_ritz(points, grid.weights, potential, basis)
    vals, coeffs = solve_generalized(h, s)
    assert s.shape == (2, 2)
    assert h.shape == (2, 2)
    assert coeffs.shape == (2, 2)
    assert torch.allclose(s, s.T)
    assert torch.allclose(h, h.T)
    assert torch.all(vals > 0)
