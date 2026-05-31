from __future__ import annotations

import math

import numpy as np
import torch

from neural_orbital_maps.numerics.pair_ci import PairResult, SectorBasis


def expand_sector_coefficients(basis: SectorBasis, coeff: np.ndarray) -> np.ndarray:
    expanded = np.zeros(basis.transform.shape[1], dtype=float)
    for product_idx in range(basis.transform.shape[1]):
        total = 0.0
        for sector_idx in range(basis.transform.shape[0]):
            total += basis.transform[sector_idx, product_idx] * coeff[sector_idx]
        expanded[product_idx] = total
    return expanded


def one_body_rdm(pair: PairResult, sector: str, state_index: int = 0) -> np.ndarray:
    basis = pair.sector_bases[sector]
    coeff = pair.sector_coeffs[sector][:, state_index]
    amplitude = expand_sector_coefficients(basis, coeff)
    p = int(round(math.sqrt(amplitude.size)))
    gamma = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            total = 0.0
            for k in range(p):
                total += amplitude[i * p + k] * amplitude[j * p + k]
                total += amplitude[k * p + i] * amplitude[k * p + j]
            gamma[i, j] = total
    return 0.5 * (gamma + gamma.T)


def natural_occupations(pair: PairResult, sector: str, state_index: int = 0) -> np.ndarray:
    gamma = one_body_rdm(pair, sector, state_index)
    vals = torch.linalg.eigvalsh(torch.tensor(gamma, dtype=torch.float64)).numpy()
    return np.sort(vals)[::-1]


def orbital_entropy(occupations: np.ndarray) -> float:
    total = float(np.sum(occupations))
    if total <= 0:
        return 0.0
    probs = np.clip(occupations / total, 1e-15, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def ci_participation_ratio(pair: PairResult, sector: str, state_index: int = 0) -> float:
    coeff = pair.sector_coeffs[sector][:, state_index]
    weights = coeff * coeff
    denom = float(np.sum(weights * weights))
    return 1.0 / denom if denom > 0 else 0.0


def left_right_density_report(density: np.ndarray, x: np.ndarray, cell_area: float, cut: float = 0.0) -> dict:
    left_mask = x < cut
    right_mask = ~left_mask
    left = float(np.sum(density[left_mask]) * cell_area)
    right = float(np.sum(density[right_mask]) * cell_area)
    total = left + right
    return {
        "left_integral": left,
        "right_integral": right,
        "total_integral": total,
        "imbalance": (right - left) / total if total > 0 else 0.0,
        "cut": cut,
    }


def _projector_matrix(orbitals: np.ndarray, mask: np.ndarray, cell_area: float) -> np.ndarray:
    p = orbitals.shape[0]
    projected = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            projected[i, j] = float(np.sum(orbitals[i][mask] * orbitals[j][mask]) * cell_area)
    return 0.5 * (projected + projected.T)


def _two_body_projector_expectation(amplitude: np.ndarray, p: int, first: np.ndarray, second: np.ndarray) -> float:
    total = 0.0
    for i in range(p):
        for j in range(p):
            a_ij = amplitude[i * p + j]
            if a_ij == 0.0:
                continue
            for k in range(p):
                for l in range(p):
                    total += a_ij * amplitude[k * p + l] * first[i, k] * second[j, l]
    return float(total)


def charge_sector_probabilities(
    orbitals: np.ndarray,
    pair: PairResult,
    sector: str,
    x: np.ndarray,
    cell_area: float,
    cut: float = 0.0,
    state_index: int = 0,
) -> dict:
    """Estimate (2,0), (1,1), (0,2) weights with spatial left/right projectors."""
    p = orbitals.shape[0]
    left = _projector_matrix(orbitals, x < cut, cell_area)
    right = _projector_matrix(orbitals, x >= cut, cell_area)
    basis = pair.sector_bases[sector]
    coeff = pair.sector_coeffs[sector][:, state_index]
    amplitude = expand_sector_coefficients(basis, coeff)
    p20 = max(0.0, _two_body_projector_expectation(amplitude, p, left, left))
    p02 = max(0.0, _two_body_projector_expectation(amplitude, p, right, right))
    p11_lr = max(0.0, _two_body_projector_expectation(amplitude, p, left, right))
    p11_rl = max(0.0, _two_body_projector_expectation(amplitude, p, right, left))
    total = p20 + p02 + p11_lr + p11_rl
    if total > 0:
        p20, p02, p11_lr, p11_rl = (p20 / total, p02 / total, p11_lr / total, p11_rl / total)
    p11 = p11_lr + p11_rl
    return {
        "P_20": p20,
        "P_11": p11,
        "P_02": p02,
        "P_11_left_right": p11_lr,
        "P_11_right_left": p11_rl,
        "double_occupancy": p20 + p02,
        "charge_imbalance": p02 - p20,
        "normalization": p20 + p11 + p02,
        "cut": cut,
    }


def _normalize_field(field: np.ndarray, cell_area: float) -> np.ndarray:
    norm = float(np.sqrt(np.sum(field * field) * cell_area))
    return field / norm if norm > 0 else field


def localized_orbitals_from_lowest_pair(orbitals: np.ndarray, x: np.ndarray, cell_area: float, cut: float = 0.0) -> tuple[dict[str, np.ndarray], dict]:
    """Construct left/right orbitals from the two lowest delocalized states."""
    if orbitals.shape[0] < 2:
        return {}, {"available": False, "reason": "at least two orbitals are required"}
    best = None
    for sign in (1.0, -1.0):
        left = _normalize_field((orbitals[0] + sign * orbitals[1]) / np.sqrt(2.0), cell_area)
        right = _normalize_field((orbitals[0] - sign * orbitals[1]) / np.sqrt(2.0), cell_area)
        left_report = left_right_density_report(left * left, x, cell_area, cut)
        right_report = left_right_density_report(right * right, x, cell_area, cut)
        score = left_report["left_integral"] + right_report["right_integral"]
        candidate = (score, sign, left, right, left_report, right_report)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    _, sign, left, right, left_report, right_report = best
    overlap = float(np.sum(left * right) * cell_area)
    report = {
        "available": True,
        "construction": "left/right = normalized (psi0 +/- sign psi1) / sqrt(2)",
        "sign": sign,
        "cut": cut,
        "overlap": overlap,
        "left_orbital": left_report,
        "right_orbital": right_report,
        "localization_score": left_report["left_integral"] + right_report["right_integral"],
    }
    return {"left": left, "right": right}, report


def conditional_density(
    orbitals: np.ndarray,
    pair: PairResult,
    sector: str,
    anchor_index: tuple[int, int] | None = None,
    state_index: int = 0,
) -> np.ndarray:
    p, n, _ = orbitals.shape
    if anchor_index is None:
        anchor_index = (n // 2, n // 4)
    ay, ax = anchor_index
    basis = pair.sector_bases[sector]
    coeff = pair.sector_coeffs[sector][:, state_index]
    amplitude = expand_sector_coefficients(basis, coeff)
    density = np.zeros((n, n), dtype=float)
    for i in range(p):
        for j in range(p):
            amp = amplitude[i * p + j] * orbitals[i, ay, ax]
            density += (amp * orbitals[j]) ** 2
    norm = float(np.sum(density))
    return density / norm if norm > 0 else density


def correlation_report(pair: PairResult, x: np.ndarray, cell_area: float, orbitals: np.ndarray | None = None) -> dict:
    report = {}
    for sector, density in pair.one_body_densities.items():
        occ = natural_occupations(pair, sector)
        sector_report = {
            "natural_occupations": occ.tolist(),
            "orbital_entropy": orbital_entropy(occ),
            "ci_participation_ratio": ci_participation_ratio(pair, sector),
            "left_right_density": left_right_density_report(density, x, cell_area),
        }
        if orbitals is not None:
            sector_report["charge_sectors"] = charge_sector_probabilities(orbitals, pair, sector, x, cell_area)
        report[sector] = sector_report
    return report
