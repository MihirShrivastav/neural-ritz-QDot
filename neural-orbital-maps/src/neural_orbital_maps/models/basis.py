from __future__ import annotations

import math

import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega: float = 20.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega = omega
        bound = 1.0 / in_dim
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class BasisNet(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        kind: str = "siren",
        envelope_alpha: float = 0.12,
    ) -> None:
        super().__init__()
        self.envelope_alpha = envelope_alpha
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(hidden_layers):
            if kind == "siren":
                layers.append(SineLayer(in_dim, hidden_dim))
            elif kind == "mlp":
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            else:
                raise ValueError(f"unsupported network kind: {kind}")
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        if isinstance(self.net[-1], nn.Linear):
            nn.init.uniform_(self.net[-1].weight, -1.0 / math.sqrt(in_dim), 1.0 / math.sqrt(in_dim))
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        raw = self.net(points)
        if self.envelope_alpha == 0:
            return raw
        radius2 = torch.sum(points * points, dim=1, keepdim=True)
        return raw * torch.exp(-self.envelope_alpha * radius2)
