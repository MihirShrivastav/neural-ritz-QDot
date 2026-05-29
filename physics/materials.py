"""Material constants and unit conversion helpers for quantum dot simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    m_eff: float
    epsilon_r: float


GAAS = Material(name="GaAs", m_eff=0.067, epsilon_r=12.9)


def energy_scale_meV(m_eff: float, L0_nm: float) -> float:
    """Return E0 = hbar^2 / (2 m* L0^2) in meV.

    Uses the common constant hbar^2 / (2 m_e) = 3.809981944 eV*Angstrom^2.
    """
    if m_eff <= 0:
        raise ValueError("m_eff must be positive")
    if L0_nm <= 0:
        raise ValueError("L0_nm must be positive")

    hbar2_over_2me_eVA2 = 3.809981944
    L0_A = L0_nm * 10.0
    e0_eV = hbar2_over_2me_eVA2 / (m_eff * (L0_A**2))
    return e0_eV * 1_000.0


def coulomb_scale_meV(epsilon_r: float, L0_nm: float) -> float:
    """Return e^2 / (4 pi eps0 eps_r L0) in meV."""
    if epsilon_r <= 0:
        raise ValueError("epsilon_r must be positive")
    if L0_nm <= 0:
        raise ValueError("L0_nm must be positive")

    e2_over_4pi_eps0_meV_nm = 1_439.964547
    return e2_over_4pi_eps0_meV_nm / (epsilon_r * L0_nm)
