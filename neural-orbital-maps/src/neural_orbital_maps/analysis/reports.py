from __future__ import annotations

import numpy as np

from neural_orbital_maps.numerics.pair_ci import PairResult
from neural_orbital_maps.physics.units import energy_scale_meV


def orthonormality_report(orbitals: np.ndarray, cell_area: float) -> dict:
    flat = orbitals.reshape(orbitals.shape[0], -1)
    p = flat.shape[0]
    overlap = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            overlap[i, j] = float(np.sum(flat[i] * flat[j]) * cell_area)
    return {
        "weighted_overlap_matrix": overlap.tolist(),
        "max_abs_offdiag": float(np.max(np.abs(overlap - np.diag(np.diag(overlap))))),
        "max_diag_deviation_from_1": float(np.max(np.abs(np.diag(overlap) - 1.0))),
    }


def density_checks(pair: PairResult, cell_area: float) -> dict:
    return {
        sector: {
            "one_body_integral": float(rho.sum() * cell_area),
            "min": float(rho.min()),
            "max": float(rho.max()),
        }
        for sector, rho in pair.one_body_densities.items()
    }


def exchange_report(pair: PairResult, material: dict) -> dict:
    e0 = energy_scale_meV(float(material["m_eff"]), float(material["L0_nm"]))
    singlet = pair.sector_energies.get("singlet")
    triplet = pair.sector_energies.get("triplet")
    s0 = float(singlet[0]) if singlet is not None else None
    t0 = float(triplet[0]) if triplet is not None else None
    j = t0 - s0 if s0 is not None and t0 is not None else None
    return {
        "singlet_ground_E_dimless": s0,
        "triplet_ground_E_dimless": t0,
        "J_dimless": j,
        "J_meV": j * e0 if j is not None else None,
        "J_GHz": j * e0 * 241.7989348 if j is not None else None,
        "definition": "J = E_triplet_0 - E_singlet_0",
    }


def ci_weights(pair: PairResult, top_k: int = 8) -> dict:
    payload = {}
    for sector, coeffs in pair.sector_coeffs.items():
        weights = coeffs[:, 0] ** 2
        pairs = pair.sector_bases[sector].orbital_pairs
        order = np.argsort(weights)[::-1][:top_k]
        payload[sector] = {
            "norm": float(weights.sum()),
            "top": [{"pair": list(pairs[i]), "weight": float(weights[i])} for i in order],
        }
    return payload
