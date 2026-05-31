from __future__ import annotations

import torch

from neural_orbital_maps.io.config import DoubleDotConfig


def biquadratic_double_dot(points: torch.Tensor, config: DoubleDotConfig) -> torch.Tensor:
    """Analytic 2D double-dot potential in dimensionless units."""
    x = points[:, 0]
    y = points[:, 1]
    a = config.separation
    return config.barrier * (x * x - a * a) ** 2 + config.y_confinement * y * y + config.detuning * x


def evaluate_potential(points: torch.Tensor, config: DoubleDotConfig) -> torch.Tensor:
    if config.kind == "biquadratic":
        return biquadratic_double_dot(points, config)
    raise ValueError(f"unsupported potential kind: {config.kind}")
