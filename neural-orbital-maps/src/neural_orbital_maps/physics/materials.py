from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    m_eff: float
    epsilon_r: float
    L0_nm: float


GAAS = Material(name="GaAs", m_eff=0.067, epsilon_r=12.9, L0_nm=30.0)
