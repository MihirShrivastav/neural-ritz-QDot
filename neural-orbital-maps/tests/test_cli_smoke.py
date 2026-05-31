from pathlib import Path
import json

from neural_orbital_maps.io.config import load_config
from neural_orbital_maps.workflows import run_pair_ci


def test_cli_workflow_creates_pair_artifacts(tmp_path):
    cfg = load_config("configs/smoke_pair.yaml")
    cfg.results_root = str(tmp_path)
    cfg.training.steps = 2
    cfg.domain.num_points = 10
    run_dir = run_pair_ci(cfg)
    run_path = Path(run_dir)
    assert (run_path / "reports" / "pair_exchange.json").exists()
    assert (run_path / "reports" / "one_electron_quality.json").exists()
    assert (run_path / "reports" / "localized_orbitals.json").exists()
    assert (run_path / "reports" / "density_checks.json").exists()
    assert (run_path / "reports" / "pair_correlation_report.json").exists()
    assert (run_path / "reports" / "hubbard_report.json").exists()
    quality = (run_path / "reports" / "one_electron_quality.json").read_text(encoding="utf-8")
    assert "final_orbital_norms_after" in quality
    final_summary = json.loads((run_path / "reports" / "final_summary.json").read_text(encoding="utf-8"))
    assert final_summary["training_status"]["steps_completed"] >= 1
    assert final_summary["training_status"]["stop_reason"] in {"max_steps", "early_stop_patience"}
    assert (run_path / "arrays" / "orbitals.npy").exists()
    assert (run_path / "arrays" / "localized_orbital_left.npy").exists()
    assert (run_path / "arrays" / "pair_correlation_singlet.npy").exists()
    assert (run_path / "plots" / "exchange_summary.png").exists()
    assert (run_path / "plots" / "pair_summary_dashboard.png").exists()
