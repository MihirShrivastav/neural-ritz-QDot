import pytest

from neural_orbital_maps.io.config import ConvergenceStudyConfig, RunConfig, load_config, load_convergence_study_config


def test_smoke_config_loads():
    cfg = load_config("configs/smoke_pair.yaml")
    assert cfg.solver.num_states == 3
    assert cfg.solver.basis_size == 4
    assert cfg.pair.num_orbitals == 3
    assert cfg.solver.renormalize_final_orbitals is True


def test_reject_pair_orbitals_above_states():
    with pytest.raises(ValueError):
        RunConfig.model_validate({"solver": {"K": 2, "M": 2}, "pair": {"num_orbitals": 3}})


def test_reject_negative_min_stop_steps():
    with pytest.raises(ValueError):
        RunConfig.model_validate({"training": {"min_steps_before_early_stop": -1}})


def test_smoke_convergence_config_loads():
    cfg = load_convergence_study_config("configs/smoke_convergence.yaml")
    assert cfg.grid_points == [8, 10]
    assert cfg.base_config.pair.enabled is False
    assert cfg.include_finite_difference is True


def test_reject_duplicate_convergence_grid_points():
    with pytest.raises(ValueError):
        ConvergenceStudyConfig.model_validate({"grid_points": [8, 8]})
