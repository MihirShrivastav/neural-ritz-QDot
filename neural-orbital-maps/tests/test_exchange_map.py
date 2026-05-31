from pathlib import Path

from neural_orbital_maps.io.config import load_exchange_map_config
from neural_orbital_maps.studies.exchange_map import run_exchange_map


def test_exchange_map_smoke_creates_maps(tmp_path):
    cfg = load_exchange_map_config("configs/smoke_exchange_map.yaml")
    cfg.results_root = str(tmp_path)
    study_dir = run_exchange_map(cfg)
    study_path = Path(study_dir)
    assert (study_path / "points.csv").exists()
    assert (study_path / "summary.json").exists()
    assert (study_path / "manifest.json").exists()
    assert (study_path / "maps" / "J_meV.npy").exists()
    assert (study_path / "maps" / "sensitivity_norm.npy").exists()
    assert (study_path / "plots" / "exchange_map_meV.png").exists()


def test_exchange_map_resume_reuses_completed_points(tmp_path):
    cfg = load_exchange_map_config("configs/smoke_exchange_map.yaml")
    cfg.results_root = str(tmp_path)
    study_dir = run_exchange_map(cfg)
    points_csv = Path(study_dir) / "points.csv"
    before = points_csv.read_text(encoding="utf-8")
    resumed_dir = run_exchange_map(cfg, study_dir=study_dir, resume=True)
    after = points_csv.read_text(encoding="utf-8")
    assert resumed_dir == study_dir
    assert before == after
