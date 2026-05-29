"""Two-electron configuration interaction utilities.

The CI basis is built from real one-electron orbitals produced by the
single-particle Block-Ritz solver. Spin is handled by solving separate spatial
singlet and triplet symmetry sectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from physics.materials import coulomb_scale_meV, energy_scale_meV


@dataclass(frozen=True)
class SectorBasis:
    sector: str
    transform: np.ndarray
    orbital_pairs: list[tuple[int, int]]


@dataclass(frozen=True)
class PairCIResult:
    sector_energies: dict[str, np.ndarray]
    sector_coeffs: dict[str, np.ndarray]
    sector_product_coeffs: dict[str, np.ndarray]
    sector_bases: dict[str, SectorBasis]
    one_body_densities: dict[str, np.ndarray]
    conditional_densities: dict[str, np.ndarray]
    coulomb_tensor: np.ndarray
    product_hamiltonian: np.ndarray
    coulomb_report: dict


def cell_area(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        raise ValueError("x and y grids must each contain at least two points")
    return float(abs((x[1] - x[0]) * (y[1] - y[0])))


def normalize_orbitals(psi_grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    orbitals = np.asarray(psi_grid, dtype=float).copy()
    area = cell_area(x, y)
    norms = np.sqrt(np.sum(orbitals * orbitals, axis=(1, 2)) * area)
    if np.any(norms <= 0):
        raise ValueError("all orbitals must have positive norm")
    return orbitals / norms[:, None, None]


def coulomb_strength_dimless(physics_cfg: dict, coulomb_cfg: dict) -> float:
    strength = coulomb_cfg.get("strength", "material")
    if isinstance(strength, str):
        key = strength.lower()
        if key == "zero":
            return 0.0
        if key != "material":
            raise ValueError("coulomb strength must be 'material', 'zero', or a non-negative number")
        epsilon_r = float(coulomb_cfg.get("epsilon_r", physics_cfg.get("epsilon_r", 12.9)))
        L0_nm = float(physics_cfg["L0_nm"])
        e0 = energy_scale_meV(m_eff=float(physics_cfg["m_eff"]), L0_nm=L0_nm)
        return coulomb_scale_meV(epsilon_r=epsilon_r, L0_nm=L0_nm) / e0
    value = float(strength)
    if value < 0:
        raise ValueError("coulomb strength must be non-negative")
    return value


def build_sector_basis(num_orbitals: int, sector: str) -> SectorBasis:
    sector_key = sector.lower()
    if sector_key not in {"singlet", "triplet"}:
        raise ValueError("sector must be 'singlet' or 'triplet'")

    rows = []
    pairs: list[tuple[int, int]] = []
    product_dim = num_orbitals * num_orbitals
    for i in range(num_orbitals):
        start_j = i if sector_key == "singlet" else i + 1
        for j in range(start_j, num_orbitals):
            row = np.zeros(product_dim, dtype=float)
            if i == j:
                row[_product_index(i, j, num_orbitals)] = 1.0
            else:
                scale = 1.0 / np.sqrt(2.0)
                row[_product_index(i, j, num_orbitals)] = scale
                sign = 1.0 if sector_key == "singlet" else -1.0
                row[_product_index(j, i, num_orbitals)] = sign * scale
            rows.append(row)
            pairs.append((i, j))

    return SectorBasis(sector=sector_key, transform=np.vstack(rows), orbital_pairs=pairs)


def build_coulomb_tensor(
    orbitals: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    strength_dimless: float,
    softening: float,
    integration: str = "fft_convolution",
) -> tuple[np.ndarray, dict]:
    if strength_dimless == 0.0:
        p = int(orbitals.shape[0])
        tensor = np.zeros((p, p, p, p), dtype=float)
        return tensor, _coulomb_report(tensor, strength_dimless, softening, integration, skipped=True)

    mode = integration.lower()
    if mode == "direct":
        tensor = _coulomb_tensor_direct(orbitals, x, y, strength_dimless, softening)
    elif mode == "fft_convolution":
        tensor = _coulomb_tensor_fft(orbitals, x, y, strength_dimless, softening)
    else:
        raise ValueError("integration must be 'fft_convolution' or 'direct'")
    return tensor, _coulomb_report(tensor, strength_dimless, softening, mode, skipped=False)


def build_product_hamiltonian(orbital_energies: np.ndarray, coulomb_tensor: np.ndarray) -> np.ndarray:
    eps = np.asarray(orbital_energies, dtype=float)
    p = int(eps.shape[0])
    h = np.zeros((p * p, p * p), dtype=float)
    for i in range(p):
        for j in range(p):
            row = _product_index(i, j, p)
            for k in range(p):
                for l in range(p):
                    col = _product_index(k, l, p)
                    value = coulomb_tensor[i, j, k, l]
                    if i == k and j == l:
                        value += eps[i] + eps[j]
                    h[row, col] = value
    return 0.5 * (h + h.T)


def solve_pair_ci(
    psi_grid: np.ndarray,
    orbital_energies: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    physics_cfg: dict,
    pair_cfg: dict,
) -> PairCIResult:
    num_orbitals = int(pair_cfg["num_orbitals"])
    if psi_grid.shape[0] < num_orbitals:
        raise ValueError("not enough one-electron orbitals were saved for the pair solve")
    if len(orbital_energies) < num_orbitals:
        raise ValueError("not enough one-electron energies were saved for the pair solve")

    orbitals = normalize_orbitals(psi_grid[:num_orbitals], x, y)
    energies = np.asarray(orbital_energies[:num_orbitals], dtype=float)
    coulomb_cfg = dict(pair_cfg.get("coulomb", {}))
    strength = coulomb_strength_dimless(physics_cfg, coulomb_cfg)
    softening = float(coulomb_cfg.get("softening", 0.05))
    integration = str(coulomb_cfg.get("integration", "fft_convolution"))

    coulomb_tensor, report = build_coulomb_tensor(orbitals, x, y, strength, softening, integration)
    product_h = build_product_hamiltonian(energies, coulomb_tensor)

    sector_energies: dict[str, np.ndarray] = {}
    sector_coeffs: dict[str, np.ndarray] = {}
    sector_product_coeffs: dict[str, np.ndarray] = {}
    sector_bases: dict[str, SectorBasis] = {}
    one_body_densities: dict[str, np.ndarray] = {}
    conditional_densities: dict[str, np.ndarray] = {}

    for sector in [str(s).lower() for s in pair_cfg.get("sectors", ["singlet", "triplet"])]:
        basis = build_sector_basis(num_orbitals, sector)
        h_sector = basis.transform @ product_h @ basis.transform.T
        vals, vecs = _symmetric_eigh(0.5 * (h_sector + h_sector.T))
        sector_energies[sector] = vals
        sector_coeffs[sector] = vecs
        ground_product = expand_sector_coefficients(vecs[:, 0], basis, num_orbitals)
        sector_product_coeffs[sector] = ground_product
        sector_bases[sector] = basis
        one_body_densities[sector] = one_body_density(ground_product, orbitals)
        conditional_densities[sector] = conditional_density(ground_product, orbitals, x, y)

    return PairCIResult(
        sector_energies=sector_energies,
        sector_coeffs=sector_coeffs,
        sector_product_coeffs=sector_product_coeffs,
        sector_bases=sector_bases,
        one_body_densities=one_body_densities,
        conditional_densities=conditional_densities,
        coulomb_tensor=coulomb_tensor,
        product_hamiltonian=product_h,
        coulomb_report=report,
    )


def expand_sector_coefficients(coeffs: np.ndarray, basis: SectorBasis, num_orbitals: int) -> np.ndarray:
    product_vec = basis.transform.T @ np.asarray(coeffs, dtype=float)
    return product_vec.reshape(num_orbitals, num_orbitals)


def one_body_density(product_coeffs: np.ndarray, orbitals: np.ndarray) -> np.ndarray:
    p = product_coeffs.shape[0]
    reduced = np.zeros((p, p), dtype=float)
    for i in range(p):
        for k in range(p):
            reduced[i, k] = float(np.sum(product_coeffs[i, :] * product_coeffs[k, :]))

    density = np.zeros_like(orbitals[0], dtype=float)
    for i in range(p):
        for k in range(p):
            density += 2.0 * reduced[i, k] * orbitals[i] * orbitals[k]
    return density


def conditional_density(product_coeffs: np.ndarray, orbitals: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    density = one_body_density(product_coeffs, orbitals)
    iy, ix = np.unravel_index(int(np.argmax(density)), density.shape)
    anchor_values = orbitals[:, iy, ix]
    amplitude = np.zeros_like(orbitals[0], dtype=float)
    for i in range(product_coeffs.shape[0]):
        for j in range(product_coeffs.shape[1]):
            amplitude += product_coeffs[i, j] * anchor_values[i] * orbitals[j]
    cond = amplitude * amplitude
    norm = np.sum(cond) * cell_area(x, y)
    if norm > 0:
        cond = cond / norm
    return cond


def ci_weight_spectrum(coeffs: np.ndarray) -> np.ndarray:
    weights = np.asarray(coeffs, dtype=float) ** 2
    return np.sort(weights)[::-1]


def _product_index(i: int, j: int, num_orbitals: int) -> int:
    return i * num_orbitals + j


def _symmetric_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = torch.as_tensor(np.asarray(matrix, dtype=np.float64), dtype=torch.float64)
    vals, vecs = torch.linalg.eigh(mat)
    return vals.detach().cpu().numpy(), vecs.detach().cpu().numpy()


def _coulomb_tensor_fft(
    orbitals: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    strength_dimless: float,
    softening: float,
) -> np.ndarray:
    p, ny, nx = orbitals.shape
    area = cell_area(x, y)
    rho = np.empty((p, p, ny, nx), dtype=float)
    for a in range(p):
        for c in range(p):
            rho[a, c] = orbitals[a] * orbitals[c]
    kernel = _displacement_kernel(x, y, softening)
    conv = np.empty_like(rho)
    for a in range(p):
        for c in range(p):
            conv[a, c] = _fft_convolve_same(rho[a, c], kernel)
    tensor = np.empty((p, p, p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            for k in range(p):
                for l in range(p):
                    tensor[i, j, k, l] = strength_dimless * area * area * float(np.sum(rho[i, k] * conv[j, l]))
    return tensor


def _coulomb_tensor_direct(
    orbitals: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    strength_dimless: float,
    softening: float,
) -> np.ndarray:
    p, ny, nx = orbitals.shape
    area = cell_area(x, y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    diff = coords[:, None, :] - coords[None, :, :]
    kernel = 1.0 / np.sqrt(np.sum(diff * diff, axis=-1) + softening * softening)
    flat_orbs = orbitals.reshape(p, ny * nx)
    rho = np.empty((p, p, ny * nx), dtype=float)
    for a in range(p):
        for c in range(p):
            rho[a, c] = flat_orbs[a] * flat_orbs[c]

    tensor = np.empty((p, p, p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            for k in range(p):
                for l in range(p):
                    integrand = rho[i, k][:, None] * kernel * rho[j, l][None, :]
                    tensor[i, j, k, l] = strength_dimless * area * area * float(np.sum(integrand))
    return tensor


def _displacement_kernel(x: np.ndarray, y: np.ndarray, softening: float) -> np.ndarray:
    dx = float(abs(x[1] - x[0]))
    dy = float(abs(y[1] - y[0]))
    x_disp = (np.arange(-(len(x) - 1), len(x), dtype=float)) * dx
    y_disp = (np.arange(-(len(y) - 1), len(y), dtype=float)) * dy
    xx, yy = np.meshgrid(x_disp, y_disp, indexing="xy")
    return 1.0 / np.sqrt(xx * xx + yy * yy + softening * softening)


def _fft_convolve_same(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    full_shape = (field.shape[0] + kernel.shape[0] - 1, field.shape[1] + kernel.shape[1] - 1)
    out = np.fft.ifft2(np.fft.fft2(field, full_shape) * np.fft.fft2(kernel, full_shape)).real
    start_y = kernel.shape[0] // 2
    start_x = kernel.shape[1] // 2
    return out[start_y : start_y + field.shape[0], start_x : start_x + field.shape[1]]


def _coulomb_report(
    tensor: np.ndarray,
    strength_dimless: float,
    softening: float,
    integration: str,
    skipped: bool,
) -> dict:
    exchange_swapped = np.transpose(tensor, (1, 0, 3, 2))
    bra_ket_swapped = np.transpose(tensor, (2, 3, 0, 1))
    pair_exchange_sym = float(np.max(np.abs(tensor - exchange_swapped))) if tensor.size else 0.0
    bra_ket_sym = float(np.max(np.abs(tensor - bra_ket_swapped))) if tensor.size else 0.0
    return {
        "strength_dimless": float(strength_dimless),
        "softening": float(softening),
        "integration": integration,
        "skipped": bool(skipped),
        "tensor_shape": list(tensor.shape),
        "max_pair_exchange_symmetry_error": pair_exchange_sym,
        "max_bra_ket_symmetry_error": bra_ket_sym,
    }
