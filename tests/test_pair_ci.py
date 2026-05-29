import numpy as np

from numerics.pair_ci import (
    build_coulomb_tensor,
    build_product_hamiltonian,
    build_sector_basis,
    cell_area,
    one_body_density,
    solve_pair_ci,
)


def _toy_orbitals():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    orbitals = np.stack(
        [
            np.exp(-(xx * xx + yy * yy)),
            xx * np.exp(-(xx * xx + yy * yy)),
            yy * np.exp(-(xx * xx + yy * yy)),
        ],
        axis=0,
    )
    area = cell_area(x, y)
    orbitals = orbitals / np.sqrt(np.sum(orbitals * orbitals, axis=(1, 2), keepdims=True) * area)
    return orbitals, x, y


def test_sector_basis_dimensions_and_symmetry():
    singlet = build_sector_basis(4, "singlet")
    triplet = build_sector_basis(4, "triplet")

    assert singlet.transform.shape == (10, 16)
    assert triplet.transform.shape == (6, 16)

    first_mixed_singlet = singlet.transform[1].reshape(4, 4)
    first_triplet = triplet.transform[0].reshape(4, 4)
    assert np.allclose(first_mixed_singlet, first_mixed_singlet.T)
    assert np.allclose(first_triplet, -first_triplet.T)


def test_non_interacting_pair_energies_reduce_to_orbital_sums():
    eps = np.array([1.0, 2.0, 4.0])
    coulomb = np.zeros((3, 3, 3, 3))
    product_h = build_product_hamiltonian(eps, coulomb)

    singlet = build_sector_basis(3, "singlet")
    triplet = build_sector_basis(3, "triplet")
    singlet_vals = np.sort(np.diag(singlet.transform @ product_h @ singlet.transform.T))
    triplet_vals = np.sort(np.diag(triplet.transform @ product_h @ triplet.transform.T))

    assert np.isclose(singlet_vals[0], 2.0)
    assert np.isclose(triplet_vals[0], 3.0)


def test_direct_coulomb_tensor_symmetries_and_repulsion():
    orbitals, x, y = _toy_orbitals()
    tensor, report = build_coulomb_tensor(
        orbitals,
        x,
        y,
        strength_dimless=0.5,
        softening=0.1,
        integration="direct",
    )

    assert tensor.shape == (3, 3, 3, 3)
    assert report["max_pair_exchange_symmetry_error"] < 1e-10
    assert report["max_bra_ket_symmetry_error"] < 1e-10
    assert tensor[0, 0, 0, 0] > 0


def test_solve_pair_ci_density_normalization_and_exchange_consistency():
    orbitals, x, y = _toy_orbitals()
    result = solve_pair_ci(
        psi_grid=orbitals,
        orbital_energies=np.array([1.0, 2.0, 4.0]),
        x=x,
        y=y,
        physics_cfg={"m_eff": 0.067, "L0_nm": 30.0, "epsilon_r": 12.9},
        pair_cfg={
            "num_orbitals": 3,
            "sectors": ["singlet", "triplet"],
            "coulomb": {
                "epsilon_r": 12.9,
                "softening": 0.1,
                "strength": "zero",
                "integration": "direct",
            },
        },
    )
    area = cell_area(x, y)
    singlet_density = result.one_body_densities["singlet"]
    coeffs = result.sector_product_coeffs["singlet"]

    assert np.isclose(np.sum(singlet_density) * area, 2.0)
    assert np.isclose(np.sum(coeffs * coeffs), 1.0)
    assert result.sector_energies["triplet"][0] > result.sector_energies["singlet"][0]
