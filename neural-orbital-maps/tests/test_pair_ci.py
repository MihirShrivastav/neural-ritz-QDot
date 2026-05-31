import numpy as np

from neural_orbital_maps.analysis.observables import (
    charge_sector_probabilities,
    correlation_report,
    entanglement_summary,
    interpret_correlation_sector,
    localized_orbitals_from_lowest_pair,
    natural_occupations,
    pair_correlation_map,
)
from neural_orbital_maps.numerics.pair_ci import build_coulomb_tensor, build_sector_basis, solve_pair_ci


def _toy_orbitals():
    xs = np.linspace(-2, 2, 12)
    ys = np.linspace(-2, 2, 12)
    x, y = np.meshgrid(xs, ys, indexing="xy")
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    g0 = np.exp(-((x + 0.7) ** 2 + y**2))
    g1 = np.exp(-((x - 0.7) ** 2 + y**2))
    g2 = x * np.exp(-(x**2 + y**2))
    orbitals = np.stack([g0, g1, g2])
    for i in range(orbitals.shape[0]):
        orbitals[i] /= np.sqrt((orbitals[i] ** 2).sum() * dx * dy)
    return orbitals, x, y, dx * dy


def test_sector_basis_dimensions():
    assert len(build_sector_basis(4, "singlet").orbital_pairs) == 10
    assert len(build_sector_basis(4, "triplet").orbital_pairs) == 6


def test_zero_coulomb_pair_energies_are_orbital_sums():
    orbitals, x, y, _ = _toy_orbitals()
    energies = np.array([1.0, 2.0, 4.0])
    result = solve_pair_ci(orbitals, energies, x, y, {"m_eff": 0.067, "epsilon_r": 12.9, "L0_nm": 30.0}, 3, ["singlet", "triplet"], "zero", 0.05)
    assert np.isclose(result.sector_energies["singlet"][0], 2.0)
    assert np.isclose(result.sector_energies["triplet"][0], 3.0)


def test_coulomb_tensor_symmetry():
    orbitals, x, y, _ = _toy_orbitals()
    tensor = build_coulomb_tensor(orbitals[:2], x, y, strength=1.0, softening=0.1)
    assert tensor.shape == (2, 2, 2, 2)
    assert np.allclose(tensor, tensor.transpose(2, 3, 0, 1))
    assert tensor[0, 0, 0, 0] > 0


def test_pair_density_integrates_to_two():
    orbitals, x, y, area = _toy_orbitals()
    result = solve_pair_ci(orbitals, np.array([1.0, 2.0, 4.0]), x, y, {"m_eff": 0.067, "epsilon_r": 12.9, "L0_nm": 30.0}, 3, ["singlet"], "zero", 0.05)
    assert np.isclose(result.one_body_densities["singlet"].sum() * area, 2.0, atol=1e-5)


def test_correlation_observables_are_normalized():
    orbitals, x, y, area = _toy_orbitals()
    result = solve_pair_ci(orbitals, np.array([1.0, 2.0, 4.0]), x, y, {"m_eff": 0.067, "epsilon_r": 12.9, "L0_nm": 30.0}, 3, ["singlet"], "zero", 0.05)
    occupations = natural_occupations(result, "singlet")
    report = correlation_report(result, x, area)
    assert np.isclose(occupations.sum(), 2.0)
    assert report["singlet"]["ci_participation_ratio"] >= 1.0
    assert 0.0 <= report["singlet"]["entanglement"]["normalized_entropy"] <= 1.0
    assert np.isclose(report["singlet"]["left_right_density"]["total_integral"], 2.0, atol=1e-5)
    assert "interpretation" in report["singlet"]


def test_entanglement_summary_bounds():
    summary = entanglement_summary(np.array([1.0, 1.0, 0.0]))
    assert 0.0 <= summary["normalized_entropy"] <= 1.0
    assert summary["effective_orbital_count"] >= 1.0
    assert 0.0 <= summary["dominant_occupation_fraction"] <= 1.0


def test_charge_sector_probabilities_sum_to_one():
    orbitals, x, y, area = _toy_orbitals()
    result = solve_pair_ci(orbitals, np.array([1.0, 2.0, 4.0]), x, y, {"m_eff": 0.067, "epsilon_r": 12.9, "L0_nm": 30.0}, 3, ["singlet"], "zero", 0.05)
    charge = charge_sector_probabilities(orbitals, result, "singlet", x, area)
    assert np.isclose(charge["normalization"], 1.0)
    assert np.isclose(charge["P_20"] + charge["P_11"] + charge["P_02"], 1.0)


def test_localized_orbitals_are_reported_and_normalized():
    orbitals, x, y, area = _toy_orbitals()
    localized, report = localized_orbitals_from_lowest_pair(orbitals, x, area)
    assert report["available"] is True
    assert set(localized) == {"left", "right"}
    assert np.isclose(np.sum(localized["left"] ** 2) * area, 1.0)
    assert np.isclose(np.sum(localized["right"] ** 2) * area, 1.0)
    assert report["localization_score"] >= 0.0
    assert np.isfinite(report["overlap"])


def test_pair_correlation_map_has_normalized_conditional_density():
    orbitals, x, y, area = _toy_orbitals()
    result = solve_pair_ci(orbitals, np.array([1.0, 2.0, 4.0]), x, y, {"m_eff": 0.067, "epsilon_r": 12.9, "L0_nm": 30.0}, 3, ["singlet"], "zero", 0.05)
    ratio, report = pair_correlation_map(orbitals, result, "singlet", result.one_body_densities["singlet"], area)
    assert ratio.shape == orbitals.shape[1:]
    assert np.isclose(report["conditional_integral"], 1.0)
    assert report["ratio_max"] >= report["ratio_min"]


def test_correlation_interpretation_labels_regimes():
    report = {
        "entanglement": {"normalized_entropy": 0.8, "effective_orbital_count": 3.0},
        "ci_participation_ratio": 4.5,
        "charge_sectors": {"P_11": 0.8, "double_occupancy": 0.2},
    }
    interpretation = interpret_correlation_sector(report)
    assert interpretation["correlation_regime"] == "strongly-correlated"
    assert interpretation["ci_regime"] == "multi-configuration-mixture"
    assert interpretation["charge_regime"] == "separated-one-electron-per-dot"
