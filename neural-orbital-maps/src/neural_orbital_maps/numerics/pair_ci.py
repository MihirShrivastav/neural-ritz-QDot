from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neural_orbital_maps.physics.units import coulomb_scale_meV, energy_scale_meV


@dataclass(frozen=True)
class SectorBasis:
    sector: str
    orbital_pairs: list[tuple[int, int]]
    transform: np.ndarray


@dataclass(frozen=True)
class PairResult:
    sector_bases: dict[str, SectorBasis]
    sector_energies: dict[str, np.ndarray]
    sector_coeffs: dict[str, np.ndarray]
    coulomb_tensor: np.ndarray
    one_body_densities: dict[str, np.ndarray]


def build_sector_basis(num_orbitals: int, sector: str) -> SectorBasis:
    if sector not in {"singlet", "triplet"}:
        raise ValueError("sector must be singlet or triplet")
    product_dim = num_orbitals * num_orbitals
    pairs: list[tuple[int, int]] = []
    rows: list[np.ndarray] = []
    for i in range(num_orbitals):
        start = i if sector == "singlet" else i + 1
        for j in range(start, num_orbitals):
            row = np.zeros(product_dim)
            if i == j:
                row[i * num_orbitals + j] = 1.0
            else:
                sign = 1.0 if sector == "singlet" else -1.0
                row[i * num_orbitals + j] = 1.0 / np.sqrt(2.0)
                row[j * num_orbitals + i] = sign / np.sqrt(2.0)
            pairs.append((i, j))
            rows.append(row)
    return SectorBasis(sector=sector, orbital_pairs=pairs, transform=np.stack(rows, axis=0))


def coulomb_strength_dimless(material: dict, strength: str | float) -> float:
    if strength == "zero":
        return 0.0
    if isinstance(strength, (int, float)):
        if float(strength) < 0:
            raise ValueError("coulomb strength must be non-negative")
        return float(strength)
    e0 = energy_scale_meV(float(material["m_eff"]), float(material["L0_nm"]))
    vc = coulomb_scale_meV(float(material["epsilon_r"]), float(material["L0_nm"]))
    return vc / e0


def build_coulomb_tensor(orbitals: np.ndarray, x: np.ndarray, y: np.ndarray, strength: float, softening: float) -> np.ndarray:
    p, n, _ = orbitals.shape
    tensor = np.zeros((p, p, p, p), dtype=float)
    if strength == 0:
        return tensor
    dx = float(abs(x[0, 1] - x[0, 0]))
    dy = float(abs(y[1, 0] - y[0, 0]))
    area = dx * dy
    coords = np.stack([x.ravel(), y.ravel()], axis=1)
    dist = coords[:, None, :] - coords[None, :, :]
    kernel = strength / np.sqrt(np.sum(dist * dist, axis=2) + softening**2)
    flat = orbitals.reshape(p, n * n)
    for a in range(p):
        for c in range(p):
            rho_ac = flat[a] * flat[c]
            conv = np.zeros_like(rho_ac)
            for r in range(kernel.shape[0]):
                conv[r] = float(np.sum(kernel[r] * rho_ac) * area)
            for b in range(p):
                for d in range(p):
                    tensor[a, b, c, d] = float(np.sum(flat[b] * flat[d] * conv) * area)
    return tensor


def product_hamiltonian(energies: np.ndarray, coulomb: np.ndarray) -> np.ndarray:
    p = len(energies)
    h = np.zeros((p * p, p * p), dtype=float)
    for i in range(p):
        for j in range(p):
            row = i * p + j
            for k in range(p):
                for l in range(p):
                    col = k * p + l
                    val = coulomb[i, j, k, l]
                    if i == k and j == l:
                        val += energies[i] + energies[j]
                    h[row, col] = val
    return 0.5 * (h + h.T)


def _eigh_symmetric(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.tensor(matrix, dtype=torch.float64)
    vals, vecs = torch.linalg.eigh(tensor)
    return vals.numpy(), vecs.numpy()


def _project_sector_hamiltonian(h_product: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rows, cols = transform.shape
    projected = np.zeros((rows, rows), dtype=float)
    for a in range(rows):
        for b in range(rows):
            total = 0.0
            for i in range(cols):
                if transform[a, i] == 0.0:
                    continue
                for j in range(cols):
                    if transform[b, j] != 0.0:
                        total += transform[a, i] * h_product[i, j] * transform[b, j]
            projected[a, b] = total
    return projected


def _expand_coeff(transform: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    expanded = np.zeros(transform.shape[1], dtype=float)
    for i in range(transform.shape[1]):
        total = 0.0
        for a in range(transform.shape[0]):
            total += transform[a, i] * coeff[a]
        expanded[i] = total
    return expanded


def one_body_density(orbitals: np.ndarray, basis: SectorBasis, coeff: np.ndarray) -> np.ndarray:
    p, n, _ = orbitals.shape
    amplitude = _expand_coeff(basis.transform, coeff)
    gamma = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            total = 0.0
            for k in range(p):
                total += amplitude[i * p + k] * amplitude[j * p + k]
                total += amplitude[k * p + i] * amplitude[k * p + j]
            gamma[i, j] = total
    rho = np.zeros((n, n), dtype=float)
    for i in range(p):
        for j in range(p):
            rho += gamma[i, j] * orbitals[i] * orbitals[j]
    return np.maximum(rho, 0.0)


def solve_pair_ci(
    orbitals: np.ndarray,
    energies: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    material: dict,
    num_orbitals: int,
    sectors: list[str],
    coulomb_strength: str | float,
    softening: float,
) -> PairResult:
    orbitals = orbitals[:num_orbitals]
    energies = energies[:num_orbitals]
    strength = coulomb_strength_dimless(material, coulomb_strength)
    coulomb = build_coulomb_tensor(orbitals, x, y, strength, softening)
    h_product = product_hamiltonian(energies, coulomb)
    bases: dict[str, SectorBasis] = {}
    sector_energies: dict[str, np.ndarray] = {}
    sector_coeffs: dict[str, np.ndarray] = {}
    densities: dict[str, np.ndarray] = {}
    for sector in sectors:
        basis = build_sector_basis(num_orbitals, sector)
        h_sector = _project_sector_hamiltonian(h_product, basis.transform)
        vals, vecs = _eigh_symmetric(0.5 * (h_sector + h_sector.T))
        bases[sector] = basis
        sector_energies[sector] = vals
        sector_coeffs[sector] = vecs
        densities[sector] = one_body_density(orbitals, basis, vecs[:, 0])
    return PairResult(bases, sector_energies, sector_coeffs, coulomb, densities)
