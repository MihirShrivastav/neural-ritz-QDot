from __future__ import annotations

import math

import numpy as np
import torch


def _localized_transform(sign: float) -> np.ndarray:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return np.array(
        [
            [inv_sqrt2, sign * inv_sqrt2],
            [inv_sqrt2, -sign * inv_sqrt2],
        ],
        dtype=float,
    )


def _transform_coulomb(two_orbital_tensor: np.ndarray, transform: np.ndarray) -> np.ndarray:
    localized = np.zeros((2, 2, 2, 2), dtype=float)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    total = 0.0
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                for l in range(2):
                                    total += transform[a, i] * transform[b, j] * transform[c, k] * transform[d, l] * two_orbital_tensor[i, j, k, l]
                    localized[a, b, c, d] = total
    return localized


def two_site_hubbard_report(orbital_energies: np.ndarray, coulomb_tensor: np.ndarray, localization_sign: float = 1.0) -> dict:
    """Estimate two-site Hubbard parameters from the lowest two orbital states.

    This is a diagnostic reduction: it is useful for interpreting CI results,
    but it is not a substitute for the full symmetry-resolved CI calculation.
    """
    if orbital_energies.shape[0] < 2 or coulomb_tensor.shape[0] < 2:
        return {"available": False, "reason": "at least two orbitals are required"}
    transform = _localized_transform(localization_sign)
    h_eigen = np.diag(orbital_energies[:2])
    h_loc = transform @ h_eigen @ transform.T
    v_loc = _transform_coulomb(coulomb_tensor[:2, :2, :2, :2], transform)
    eps_l = float(h_loc[0, 0])
    eps_r = float(h_loc[1, 1])
    tunnel_t = float(-h_loc[0, 1])
    u_l = float(v_loc[0, 0, 0, 0])
    u_r = float(v_loc[1, 1, 1, 1])
    direct_v = float(v_loc[0, 1, 0, 1])
    exchange_k = float(v_loc[0, 1, 1, 0])

    singlet_h = np.array(
        [
            [2.0 * eps_l + u_l, -math.sqrt(2.0) * tunnel_t, 0.0],
            [-math.sqrt(2.0) * tunnel_t, eps_l + eps_r + direct_v + exchange_k, -math.sqrt(2.0) * tunnel_t],
            [0.0, -math.sqrt(2.0) * tunnel_t, 2.0 * eps_r + u_r],
        ],
        dtype=float,
    )
    singlet_vals = torch.linalg.eigvalsh(torch.tensor(0.5 * (singlet_h + singlet_h.T), dtype=torch.float64)).numpy()
    triplet_energy = eps_l + eps_r + direct_v - exchange_k
    return {
        "available": True,
        "basis": "localized orbitals from lowest two eigenstates",
        "limitations": [
            "Two-site reduction ignores higher orbitals retained by full CI.",
            "Parameters are diagnostics for interpretation, not calibrated device-model fits.",
        ],
        "parameters_dimless": {
            "epsilon_left": eps_l,
            "epsilon_right": eps_r,
            "detuning": eps_r - eps_l,
            "t": tunnel_t,
            "U_left": u_l,
            "U_right": u_r,
            "V_direct": direct_v,
            "K_exchange": exchange_k,
        },
        "energies_dimless": {
            "singlet": singlet_vals.tolist(),
            "triplet": [float(triplet_energy)],
            "J_hubbard": float(triplet_energy - singlet_vals[0]),
        },
        "matrices": {
            "one_body_localized": h_loc.tolist(),
            "singlet_hamiltonian": singlet_h.tolist(),
        },
    }
