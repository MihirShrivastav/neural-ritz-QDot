from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from neural_orbital_maps.io.artifacts import load_json, save_json
from neural_orbital_maps.io.config import ConvergenceStudyConfig, save_config
from neural_orbital_maps.plotting.figures import convergence_plot
from neural_orbital_maps.workflows import run_one_electron


def _study_dir(config: ConvergenceStudyConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(config.results_root) / config.study_name / f"{stamp}_{config.study_name}"


def _point_id(num_points: int) -> str:
    return f"grid_n{num_points:04d}"


def _write_points(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "point_id",
        "status",
        "num_points",
        "E0_dimless",
        "energy_sum_dimless",
        "max_projected_residual",
        "overlap_condition_number",
        "max_final_norm_deviation_after",
        "run_dir",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict], duration_sec: float) -> dict:
    completed = [row for row in rows if row.get("status") == "completed"]
    summary = {
        "status": "completed" if len(completed) == len(rows) else "partial",
        "duration_sec": duration_sec,
        "total_points": len(rows),
        "completed_points": len(completed),
        "failed_points": len(rows) - len(completed),
    }
    if completed:
        ordered = sorted(completed, key=lambda row: int(row["num_points"]))
        e0 = np.array([float(row["E0_dimless"]) for row in ordered], dtype=float)
        residual = np.array([float(row["max_projected_residual"]) for row in ordered], dtype=float)
        summary.update(
            {
                "grid_points": [int(row["num_points"]) for row in ordered],
                "E0_dimless": e0.tolist(),
                "E0_delta_last_first": float(e0[-1] - e0[0]),
                "E0_delta_last_previous": float(e0[-1] - e0[-2]) if len(e0) > 1 else 0.0,
                "max_projected_residual": residual.tolist(),
                "best_residual": float(np.min(residual)),
            }
        )
    return summary


def run_convergence_study(config: ConvergenceStudyConfig, study_dir: str | Path | None = None) -> Path:
    start = time.time()
    study_dir = Path(study_dir) if study_dir is not None else _study_dir(config)
    points_dir = study_dir / "points"
    plots_dir = study_dir / "plots"
    points_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        study_dir / "study_config.json",
        {
            "study_name": config.study_name,
            "grid_points": config.grid_points,
        },
    )
    save_config(config.base_config, study_dir / "base_config.yaml")
    save_json(
        study_dir / "manifest.json",
        {
            "status": "running",
            "study_name": config.study_name,
            "started_at": datetime.now().isoformat(),
            "total_points": len(config.grid_points),
            "completed_points": 0,
            "failed_points": 0,
        },
    )

    rows: list[dict] = []
    points_csv = study_dir / "points.csv"
    try:
        for num_points in config.grid_points:
            point_cfg = config.base_config.model_copy(deep=True)
            point_cfg.experiment_name = _point_id(num_points)
            point_cfg.results_root = str(points_dir)
            point_cfg.domain.num_points = num_points
            point_cfg.pair.enabled = False
            try:
                run_dir = run_one_electron(point_cfg)
                energies = load_json(Path(run_dir) / "reports" / "one_electron_energies.json")
                quality = load_json(Path(run_dir) / "reports" / "one_electron_quality.json")
                residuals = np.array(quality["projected_residual_norms"], dtype=float)
                rows.append(
                    {
                        "point_id": _point_id(num_points),
                        "status": "completed",
                        "num_points": num_points,
                        "E0_dimless": energies["E_dimless"][0],
                        "energy_sum_dimless": float(sum(energies["E_dimless"])),
                        "max_projected_residual": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
                        "overlap_condition_number": quality["basis_overlap_condition"],
                        "max_final_norm_deviation_after": quality["final_orbital_max_norm_deviation_after"],
                        "run_dir": str(run_dir),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "point_id": _point_id(num_points),
                        "status": "failed",
                        "num_points": num_points,
                        "E0_dimless": "",
                        "energy_sum_dimless": "",
                        "max_projected_residual": "",
                        "overlap_condition_number": "",
                        "max_final_norm_deviation_after": "",
                        "run_dir": "",
                        "error": str(exc),
                    }
                )
                _write_points(points_csv, rows)
                raise
            _write_points(points_csv, rows)

        summary = _summarize(rows, time.time() - start)
        save_json(study_dir / "summary.json", summary)
        convergence_plot(rows, plots_dir / "one_electron_convergence.png")
        save_json(
            study_dir / "manifest.json",
            {
                "status": summary["status"],
                "study_name": config.study_name,
                "finished_at": datetime.now().isoformat(),
                "duration_sec": summary["duration_sec"],
                "total_points": summary["total_points"],
                "completed_points": summary["completed_points"],
                "failed_points": summary["failed_points"],
            },
        )
    except Exception as exc:
        save_json(
            study_dir / "manifest.json",
            {
                "status": "failed",
                "study_name": config.study_name,
                "finished_at": datetime.now().isoformat(),
                "duration_sec": time.time() - start,
                "total_points": len(config.grid_points),
                "completed_points": len([row for row in rows if row.get("status") == "completed"]),
                "failed_points": len([row for row in rows if row.get("status") == "failed"]),
                "last_error": str(exc),
            },
        )
        raise
    return study_dir
