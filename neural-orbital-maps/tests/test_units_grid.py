import torch

from neural_orbital_maps.io.config import DomainConfig
from neural_orbital_maps.numerics.grid import make_uniform_grid
from neural_orbital_maps.physics.units import coulomb_scale_meV, energy_scale_meV


def test_material_unit_scales_are_positive():
    assert energy_scale_meV(0.067, 30.0) > 0
    assert coulomb_scale_meV(12.9, 30.0) > 0


def test_grid_weights_match_domain_area_approximately():
    cfg = DomainConfig(x_extent=2.0, y_extent=3.0, num_points=21)
    grid = make_uniform_grid(cfg, dtype=torch.float64)
    expected = (2 * cfg.x_extent) * (2 * cfg.y_extent)
    assert abs(float(grid.weights.sum()) - expected) / expected < 0.15
