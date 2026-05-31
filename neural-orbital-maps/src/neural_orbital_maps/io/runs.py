from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from neural_orbital_maps.io.artifacts import save_json
from neural_orbital_maps.io.config import RunConfig, save_config


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    logs: Path
    arrays: Path
    reports: Path
    plots: Path
    checkpoints: Path


def create_run(config: RunConfig) -> RunPaths:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{config.experiment_name}_k{config.solver.num_states}_m{config.solver.basis_size}_s{config.training.seed}"
    run_dir = Path(config.results_root) / config.experiment_name / run_id
    paths = RunPaths(
        run_dir=run_dir,
        logs=run_dir / "logs",
        arrays=run_dir / "arrays",
        reports=run_dir / "reports",
        plots=run_dir / "plots",
        checkpoints=run_dir / "checkpoints",
    )
    for path in [paths.logs, paths.arrays, paths.reports, paths.plots, paths.checkpoints]:
        path.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    save_json(
        run_dir / "run_meta.json",
        {
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    return paths


def finalize_run(paths: RunPaths, status: str, extra: dict | None = None) -> None:
    payload = {
        "status": status,
        "finished_at": datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    save_json(paths.run_dir / "run_meta.json", payload)
