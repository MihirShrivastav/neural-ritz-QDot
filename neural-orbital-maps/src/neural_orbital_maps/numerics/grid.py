from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neural_orbital_maps.io.config import DomainConfig


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    points: torch.Tensor
    weights: torch.Tensor
    dx: float
    dy: float

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy


def make_uniform_grid(config: DomainConfig, dtype: torch.dtype = torch.float64, device: str = "cpu") -> Grid:
    xs = np.linspace(-config.x_extent, config.x_extent, config.num_points)
    ys = np.linspace(-config.y_extent, config.y_extent, config.num_points)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    weights = np.full((pts.shape[0],), dx * dy)
    return Grid(
        x=xx,
        y=yy,
        points=torch.tensor(pts, dtype=dtype, device=device),
        weights=torch.tensor(weights, dtype=dtype, device=device),
        dx=dx,
        dy=dy,
    )
