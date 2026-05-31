from pathlib import Path

from neural_orbital_maps.io.artifacts import load_json
from neural_orbital_maps.io.config import load_convergence_study_config
from neural_orbital_maps.studies.convergence import convergence_points, run_convergence_study


def test_convergence_study_creates_expected_artifacts(tmp_path):
    cfg = load_convergence_study_config("configs/smoke_convergence.yaml")
    cfg.results_root = str(tmp_path)
    cfg.base_config.results_root = str(tmp_path)
    study_dir = run_convergence_study(cfg, study_dir=tmp_path / "study")
    assert Path(study_dir, "points.csv").exists()
    assert Path(study_dir, "summary.json").exists()
    assert Path(study_dir, "plots", "one_electron_convergence.png").exists()
    summary = load_json(Path(study_dir, "summary.json"))
    assert summary["finite_difference_enabled"] is True
    assert "max_abs_E0_minus_fd_dimless" in summary


def test_convergence_axes_expand_to_named_points():
    cfg = load_convergence_study_config("configs/smoke_convergence_axes.yaml")
    points = convergence_points(cfg)
    assert [point.axis for point in points] == ["basis_size", "basis_size"]
    assert [point.value for point in points] == [3, 4]


def test_convergence_axis_study_creates_axis_plot(tmp_path):
    cfg = load_convergence_study_config("configs/smoke_convergence_axes.yaml")
    cfg.results_root = str(tmp_path)
    cfg.base_config.results_root = str(tmp_path)
    study_dir = run_convergence_study(cfg, study_dir=tmp_path / "axis_study")
    summary = load_json(Path(study_dir, "summary.json"))
    assert summary["axes"] == ["basis_size"]
    assert summary["by_axis"]["basis_size"]["values"] == [3, 4]
    assert Path(study_dir, "plots", "basis_size_convergence.png").exists()
