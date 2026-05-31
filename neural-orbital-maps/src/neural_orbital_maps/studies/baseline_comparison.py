from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from neural_orbital_maps.analysis.baselines import write_pair_baseline_comparison
from neural_orbital_maps.io.artifacts import save_json
from neural_orbital_maps.io.config import RunConfig, save_config
from neural_orbital_maps.plotting.figures import pair_baseline_comparison_plot
from neural_orbital_maps.workflows import run_finite_difference_pair_ci, run_pair_ci


def _study_dir(config: RunConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(config.results_root) / "pair_baseline_benchmark" / f"{stamp}_{config.experiment_name}_baseline"


def run_pair_baseline_benchmark(config: RunConfig, study_dir: str | Path | None = None) -> Path:
    start = time.time()
    study_dir = Path(study_dir) if study_dir is not None else _study_dir(config)
    runs_dir = study_dir / "runs"
    comparison_dir = study_dir / "comparison"
    runs_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, study_dir / "base_config.yaml")
    save_json(
        study_dir / "manifest.json",
        {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "experiment_name": config.experiment_name,
        },
    )
    try:
        neural_cfg = config.model_copy(deep=True)
        neural_cfg.results_root = str(runs_dir)
        neural_cfg.experiment_name = f"{config.experiment_name}_neural_pair"
        fd_cfg = config.model_copy(deep=True)
        fd_cfg.results_root = str(runs_dir)
        fd_cfg.experiment_name = f"{config.experiment_name}_fd_pair"
        neural_run = run_pair_ci(neural_cfg)
        fd_run = run_finite_difference_pair_ci(fd_cfg)
        report = write_pair_baseline_comparison(neural_run, fd_run, comparison_dir)
        pair_baseline_comparison_plot(report, comparison_dir / "pair_baseline_comparison.png")
        save_json(
            study_dir / "manifest.json",
            {
                "status": "completed",
                "finished_at": datetime.now().isoformat(),
                "duration_sec": time.time() - start,
                "neural_run": str(neural_run),
                "finite_difference_run": str(fd_run),
                "comparison_dir": str(comparison_dir),
            },
        )
    except Exception as exc:
        save_json(
            study_dir / "manifest.json",
            {
                "status": "failed",
                "finished_at": datetime.now().isoformat(),
                "duration_sec": time.time() - start,
                "last_error": str(exc),
            },
        )
        raise
    return study_dir
