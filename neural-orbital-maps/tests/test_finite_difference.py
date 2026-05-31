from pathlib import Path

import numpy as np

from neural_orbital_maps.io.config import load_config
from neural_orbital_maps.numerics.finite_difference import solve_finite_difference
from neural_orbital_maps.workflows import run_finite_difference_baseline


def test_finite_difference_baseline_is_sorted_and_normalized():
    cfg = load_config("configs/smoke_pair.yaml")
    cfg.domain.num_points = 10
    cfg.solver.num_states = 3
    result = solve_finite_difference(cfg)
    assert result.energies.shape == (3,)
    assert np.all(np.diff(result.energies) > 0.0)
    assert result.orbitals.shape == (3, 10, 10)
    norms = np.sum(result.orbitals * result.orbitals, axis=(1, 2)) * result.grid.cell_area
    assert np.allclose(norms, 1.0)


def test_finite_difference_workflow_creates_expected_artifacts(tmp_path):
    cfg = load_config("configs/smoke_pair.yaml")
    cfg.results_root = str(tmp_path)
    cfg.domain.num_points = 10
    cfg.solver.num_states = 2
    run_dir = run_finite_difference_baseline(cfg)
    run_path = Path(run_dir)
    assert (run_path / "reports" / "finite_difference_energies.json").exists()
    assert (run_path / "reports" / "orthonormality.json").exists()
    assert (run_path / "arrays" / "fd_orbitals.npy").exists()
    assert (run_path / "plots" / "fd_density_state_0.png").exists()
