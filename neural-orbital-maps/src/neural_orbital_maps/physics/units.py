from __future__ import annotations


def energy_scale_meV(m_eff: float, L0_nm: float) -> float:
    """Return hbar^2/(2 m* L0^2) in meV."""
    if m_eff <= 0 or L0_nm <= 0:
        raise ValueError("m_eff and L0_nm must be positive")
    hbar2_over_2me_eVA2 = 3.80998212
    L0_angstrom = L0_nm * 10.0
    return 1_000.0 * hbar2_over_2me_eVA2 / (m_eff * L0_angstrom**2)


def coulomb_scale_meV(epsilon_r: float, L0_nm: float) -> float:
    """Return e^2/(4 pi eps0 eps_r L0) in meV."""
    if epsilon_r <= 0 or L0_nm <= 0:
        raise ValueError("epsilon_r and L0_nm must be positive")
    return 1_439.964547 / (epsilon_r * L0_nm)
