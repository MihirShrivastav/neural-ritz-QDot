from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from neural_orbital_maps.io.artifacts import load_json, save_json
from neural_orbital_maps.io.config import ExchangeMapConfig, save_config
from neural_orbital_maps.plotting.figures import exchange_map
from neural_orbital_maps.workflows import run_pair_ci


def _study_dir(config: ExchangeMapConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(config.results_root) / config.study_name / f"{stamp}_{config.study_name}"


def _point_id(ix: int, iy: int, detuning: float, barrier: float) -> str:
    return f"b{iy:04d}_d{ix:04d}_det{detuning:+.6g}_bar{barrier:+.6g}".replace("+", "p").replace("-", "m").replace(".", "p")


def _load_completed_points(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    completed: dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "completed":
                completed[str(row["point_id"])] = row
    return completed


def _central_gradient(values: np.ndarray, xs: list[float], ys: list[float]) -> tuple[np.ndarray, np.ndarray]:
    grad_x = np.zeros_like(values)
    grad_y = np.zeros_like(values)
    for iy in range(values.shape[0]):
        for ix in range(values.shape[1]):
            if len(xs) == 1:
                grad_x[iy, ix] = 0.0
            elif ix == 0:
                grad_x[iy, ix] = (values[iy, ix + 1] - values[iy, ix]) / (xs[ix + 1] - xs[ix])
            elif ix == len(xs) - 1:
                grad_x[iy, ix] = (values[iy, ix] - values[iy, ix - 1]) / (xs[ix] - xs[ix - 1])
            else:
                grad_x[iy, ix] = (values[iy, ix + 1] - values[iy, ix - 1]) / (xs[ix + 1] - xs[ix - 1])

            if len(ys) == 1:
                grad_y[iy, ix] = 0.0
            elif iy == 0:
                grad_y[iy, ix] = (values[iy + 1, ix] - values[iy, ix]) / (ys[iy + 1] - ys[iy])
            elif iy == len(ys) - 1:
                grad_y[iy, ix] = (values[iy, ix] - values[iy - 1, ix]) / (ys[iy] - ys[iy - 1])
            else:
                grad_y[iy, ix] = (values[iy + 1, ix] - values[iy - 1, ix]) / (ys[iy + 1] - ys[iy - 1])
    return grad_x, grad_y


def run_exchange_map(config: ExchangeMapConfig, study_dir: str | Path | None = None, resume: bool = True) -> Path:
    start = time.time()
    study_dir = Path(study_dir) if study_dir is not None else _study_dir(config)
    maps_dir = study_dir / "maps"
    plots_dir = study_dir / "plots"
    points_dir = study_dir / "points"
    maps_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)

    detuning_values = config.detuning.values()
    barrier_values = config.barrier.values()
    j_mev = np.zeros((len(barrier_values), len(detuning_values)), dtype=float)
    j_ghz = np.zeros_like(j_mev)
    points_csv = study_dir / "points.csv"
    completed = _load_completed_points(points_csv) if resume else {}
    rows = list(completed.values())
    failures = []

    save_json(
        study_dir / "study_config.json",
        {
            "study_name": config.study_name,
            "detuning": detuning_values,
            "barrier": barrier_values,
        },
    )
    save_config(config.base_config, study_dir / "base_config.yaml")
    save_json(
        study_dir / "manifest.json",
        {
            "status": "running",
            "study_name": config.study_name,
            "started_at": datetime.now().isoformat(),
            "resume": resume,
            "total_points": len(detuning_values) * len(barrier_values),
            "completed_points": len(completed),
            "failed_points": 0,
        },
    )

    for iy, barrier in enumerate(barrier_values):
        for ix, detuning in enumerate(detuning_values):
            point_id = _point_id(ix, iy, detuning, barrier)
            if point_id in completed:
                row = completed[point_id]
                j_mev[iy, ix] = float(row["J_meV"])
                j_ghz[iy, ix] = float(row["J_GHz"])
                continue
            try:
                point_cfg = config.base_config.model_copy(deep=True)
                point_cfg.experiment_name = point_id
                point_cfg.results_root = str(points_dir)
                point_cfg.potential.detuning = detuning
                point_cfg.potential.barrier = barrier
                run_dir = run_pair_ci(point_cfg)
                exchange = load_json(Path(run_dir) / "reports" / "pair_exchange.json")
                j_mev[iy, ix] = float(exchange["J_meV"])
                j_ghz[iy, ix] = float(exchange["J_GHz"])
                rows.append(
                    {
                        "point_id": point_id,
                        "status": "completed",
                        "ix": ix,
                        "iy": iy,
                        "detuning": detuning,
                        "barrier": barrier,
                        "J_meV": j_mev[iy, ix],
                        "J_GHz": j_ghz[iy, ix],
                        "run_dir": str(run_dir),
                        "error": "",
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "point_id": point_id,
                        "status": "failed",
                        "ix": ix,
                        "iy": iy,
                        "detuning": detuning,
                        "barrier": barrier,
                        "J_meV": "",
                        "J_GHz": "",
                        "run_dir": "",
                        "error": str(exc),
                    }
                )
                rows.extend(failures)
                with points_csv.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["point_id", "status", "ix", "iy", "detuning", "barrier", "J_meV", "J_GHz", "run_dir", "error"])
                    writer.writeheader()
                    writer.writerows(rows)
                save_json(
                    study_dir / "manifest.json",
                    {
                        "status": "failed",
                        "study_name": config.study_name,
                        "duration_sec": time.time() - start,
                        "resume": resume,
                        "total_points": len(detuning_values) * len(barrier_values),
                        "completed_points": len([r for r in rows if r.get("status") == "completed"]),
                        "failed_points": len(failures),
                        "last_error": str(exc),
                    },
                )
                raise

    dJ_d_detuning, dJ_d_barrier = _central_gradient(j_mev, detuning_values, barrier_values)
    sensitivity = np.sqrt(dJ_d_detuning * dJ_d_detuning + dJ_d_barrier * dJ_d_barrier)
    np.save(maps_dir / "J_meV.npy", j_mev)
    np.save(maps_dir / "J_GHz.npy", j_ghz)
    np.save(maps_dir / "dJ_d_detuning.npy", dJ_d_detuning)
    np.save(maps_dir / "dJ_d_barrier.npy", dJ_d_barrier)
    np.save(maps_dir / "sensitivity_norm.npy", sensitivity)

    completed_rows = [row for row in rows if row.get("status") == "completed"]
    with points_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["point_id", "status", "ix", "iy", "detuning", "barrier", "J_meV", "J_GHz", "run_dir", "error"])
        writer.writeheader()
        writer.writerows(completed_rows)

    detuning_array = np.asarray(detuning_values)
    barrier_array = np.asarray(barrier_values)
    exchange_map(detuning_array, barrier_array, j_mev, "Exchange Map J", plots_dir / "exchange_map_meV.png")
    exchange_map(detuning_array, barrier_array, np.log10(np.maximum(np.abs(j_mev), 1e-12)), "log10 |J meV|", plots_dir / "log_exchange_map.png")
    exchange_map(detuning_array, barrier_array, sensitivity, "Exchange Sensitivity Norm", plots_dir / "sensitivity_map.png")

    save_json(
        study_dir / "summary.json",
        {
            "status": "completed",
            "duration_sec": time.time() - start,
            "num_points": len(completed_rows),
            "J_meV_min": float(j_mev.min()),
            "J_meV_max": float(j_mev.max()),
            "J_meV_mean": float(j_mev.mean()),
            "max_sensitivity_norm": float(sensitivity.max()),
            "points_csv": "points.csv",
            "maps": ["J_meV.npy", "J_GHz.npy", "dJ_d_detuning.npy", "dJ_d_barrier.npy", "sensitivity_norm.npy"],
        },
    )
    save_json(
        study_dir / "manifest.json",
        {
            "status": "completed",
            "study_name": config.study_name,
            "duration_sec": time.time() - start,
            "resume": resume,
            "total_points": len(detuning_values) * len(barrier_values),
            "completed_points": len(completed_rows),
            "failed_points": 0,
            "points_csv": "points.csv",
            "summary": "summary.json",
        },
    )
    return study_dir
