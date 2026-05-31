from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from neural_orbital_maps.io.artifacts import load_json, save_json
from neural_orbital_maps.io.config import ConvergenceStudyConfig, save_config
from neural_orbital_maps.numerics.finite_difference import solve_finite_difference
from neural_orbital_maps.plotting.figures import convergence_plot
from neural_orbital_maps.workflows import run_one_electron


@dataclass(frozen=True)
class ConvergencePoint:
    axis: str
    value: int
    point_id: str


def _study_dir(config: ConvergenceStudyConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(config.results_root) / config.study_name / f"{stamp}_{config.study_name}"


def _point_id(num_points: int) -> str:
    return f"grid_n{num_points:04d}"


def convergence_points(config: ConvergenceStudyConfig) -> list[ConvergencePoint]:
    if not config.axes:
        return [ConvergencePoint("grid_points", num_points, _point_id(num_points)) for num_points in config.grid_points]
    points: list[ConvergencePoint] = []
    for axis in config.axes:
        for value in axis.values:
            if axis.name == "grid_points" and value < 8:
                raise ValueError("grid_points convergence axis values must be >= 8")
            points.append(ConvergencePoint(axis.name, value, f"{axis.name}_{value:04d}"))
    return points


def _apply_point_config(config: ConvergenceStudyConfig, point: ConvergencePoint, points_dir: Path):
    point_cfg = config.base_config.model_copy(deep=True)
    point_cfg.experiment_name = point.point_id
    point_cfg.results_root = str(points_dir)
    point_cfg.pair.enabled = False
    if point.axis == "grid_points":
        point_cfg.domain.num_points = point.value
    elif point.axis == "basis_size":
        point_cfg.solver.basis_size = point.value
        if point_cfg.solver.num_states > point_cfg.solver.basis_size:
            point_cfg.solver.num_states = point_cfg.solver.basis_size
        if point_cfg.pair.num_orbitals > point_cfg.solver.num_states:
            point_cfg.pair.num_orbitals = point_cfg.solver.num_states
    elif point.axis == "hidden_dim":
        point_cfg.solver.hidden_dim = point.value
    elif point.axis == "training_seed":
        point_cfg.training.seed = point.value
    else:
        raise ValueError(f"unsupported convergence axis: {point.axis}")
    return point_cfg


def _write_points(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "point_id",
        "status",
        "axis",
        "axis_value",
        "num_points",
        "basis_size",
        "hidden_dim",
        "training_seed",
        "E0_dimless",
        "energy_sum_dimless",
        "fd_E0_dimless",
        "E0_minus_fd_dimless",
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
        ordered = sorted(completed, key=lambda row: (str(row.get("axis", "")), int(row["axis_value"])))
        e0 = np.array([float(row["E0_dimless"]) for row in ordered], dtype=float)
        residual = np.array([float(row["max_projected_residual"]) for row in ordered], dtype=float)
        fd_errors = [row.get("E0_minus_fd_dimless", "") for row in ordered]
        numeric_fd_errors = [float(value) for value in fd_errors if value != ""]
        axes = sorted(set(str(row.get("axis", "grid_points")) for row in completed))
        by_axis = {}
        for axis in axes:
            axis_rows = sorted([row for row in completed if row.get("axis") == axis], key=lambda row: int(row["axis_value"]))
            by_axis[axis] = {
                "values": [int(row["axis_value"]) for row in axis_rows],
                "E0_dimless": [float(row["E0_dimless"]) for row in axis_rows],
                "max_projected_residual": [float(row["max_projected_residual"]) for row in axis_rows],
                "E0_minus_fd_dimless": [float(row["E0_minus_fd_dimless"]) for row in axis_rows if row.get("E0_minus_fd_dimless") != ""],
            }
        summary.update(
            {
                "axes": axes,
                "by_axis": by_axis,
                "grid_points": [int(row["num_points"]) for row in ordered],
                "E0_dimless": e0.tolist(),
                "E0_delta_last_first": float(e0[-1] - e0[0]),
                "E0_delta_last_previous": float(e0[-1] - e0[-2]) if len(e0) > 1 else 0.0,
                "max_projected_residual": residual.tolist(),
                "best_residual": float(np.min(residual)),
                "finite_difference_enabled": bool(numeric_fd_errors),
                "E0_minus_fd_dimless": numeric_fd_errors,
                "max_abs_E0_minus_fd_dimless": float(np.max(np.abs(numeric_fd_errors))) if numeric_fd_errors else None,
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
            "axes": [axis.model_dump() for axis in config.axes],
            "include_finite_difference": config.include_finite_difference,
        },
    )
    save_config(config.base_config, study_dir / "base_config.yaml")
    save_json(
        study_dir / "manifest.json",
        {
            "status": "running",
            "study_name": config.study_name,
            "started_at": datetime.now().isoformat(),
            "total_points": len(convergence_points(config)),
            "completed_points": 0,
            "failed_points": 0,
        },
    )

    rows: list[dict] = []
    points_csv = study_dir / "points.csv"
    try:
        points = convergence_points(config)
        for point in points:
            point_cfg = _apply_point_config(config, point, points_dir)
            try:
                run_dir = run_one_electron(point_cfg)
                energies = load_json(Path(run_dir) / "reports" / "one_electron_energies.json")
                quality = load_json(Path(run_dir) / "reports" / "one_electron_quality.json")
                residuals = np.array(quality["projected_residual_norms"], dtype=float)
                fd_e0 = ""
                e0_minus_fd = ""
                if config.include_finite_difference:
                    fd = solve_finite_difference(point_cfg, num_states=1)
                    fd_e0 = float(fd.energies[0])
                    e0_minus_fd = float(energies["E_dimless"][0] - fd_e0)
                rows.append(
                    {
                        "point_id": point.point_id,
                        "status": "completed",
                        "axis": point.axis,
                        "axis_value": point.value,
                        "num_points": point_cfg.domain.num_points,
                        "basis_size": point_cfg.solver.basis_size,
                        "hidden_dim": point_cfg.solver.hidden_dim,
                        "training_seed": point_cfg.training.seed,
                        "E0_dimless": energies["E_dimless"][0],
                        "energy_sum_dimless": float(sum(energies["E_dimless"])),
                        "fd_E0_dimless": fd_e0,
                        "E0_minus_fd_dimless": e0_minus_fd,
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
                        "point_id": point.point_id,
                        "status": "failed",
                        "axis": point.axis,
                        "axis_value": point.value,
                        "num_points": point_cfg.domain.num_points,
                        "basis_size": point_cfg.solver.basis_size,
                        "hidden_dim": point_cfg.solver.hidden_dim,
                        "training_seed": point_cfg.training.seed,
                        "E0_dimless": "",
                        "energy_sum_dimless": "",
                        "fd_E0_dimless": "",
                        "E0_minus_fd_dimless": "",
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
        for axis in summary.get("axes", []):
            axis_rows = [row for row in rows if row.get("axis") == axis]
            convergence_plot(axis_rows, plots_dir / f"{axis}_convergence.png", x_key="axis_value", x_label=axis)
        if any(row.get("axis") == "grid_points" for row in rows):
            convergence_plot([row for row in rows if row.get("axis") == "grid_points"], plots_dir / "one_electron_convergence.png", x_key="axis_value", x_label="grid points per axis")
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
                "total_points": len(convergence_points(config)),
                "completed_points": len([row for row in rows if row.get("status") == "completed"]),
                "failed_points": len([row for row in rows if row.get("status") == "failed"]),
                "last_error": str(exc),
            },
        )
        raise
    return study_dir
