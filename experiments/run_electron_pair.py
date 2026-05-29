"""Run neural-orbital two-electron configuration-interaction experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from numerics.pair_ci import ci_weight_spectrum, solve_pair_ci
from physics.materials import energy_scale_meV
from utils.artifacts import save_arrays, save_json
from utils.config import load_config, save_config, validate_config
from utils.logging import build_logger
from utils.plotting import (
    plot_ci_weight_spectrum,
    plot_pair_conditional_maps,
    plot_pair_density_maps,
    plot_pair_exchange_summary,
)
from utils.run_manager import finalize_run


def _latest_new_run(results_root: Path, experiment_name: str, previous: set[Path]) -> Path:
    exp_dir = results_root / experiment_name
    candidates = [p for p in exp_dir.glob("*") if p.is_dir() and p not in previous]
    if not candidates:
        raise RuntimeError(f"no new run directory found under {exp_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_one_electron_stage(cfg: dict, args: argparse.Namespace) -> Path:
    results_root = Path(args.results_root)
    exp_dir = results_root / args.experiment_name
    previous = set(exp_dir.glob("*")) if exp_dir.exists() else set()

    pair_orbitals = int(cfg["pair"]["num_orbitals"])
    cfg["solver"]["K"] = pair_orbitals
    cfg["solver"]["M"] = max(int(cfg["solver"]["M"]), pair_orbitals)
    validate_config(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "electron_pair_one_electron_config.json"
        save_config(cfg, cfg_path)
        cmd = [
            sys.executable,
            "-m",
            "experiments.run_single_config",
            "--config",
            str(cfg_path),
            "--results-root",
            str(results_root),
            "--experiment-name",
            args.experiment_name,
            "--seed",
            str(args.seed),
        ]
        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        subprocess.run(cmd, check=True, env=env)

    return _latest_new_run(results_root, args.experiment_name, previous)


def _load_one_electron_artifacts(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays_dir = run_dir / "arrays"
    reports_dir = run_dir / "reports"
    psi_grid = np.load(arrays_dir / "psi_grid.npy")
    grid_x = np.load(arrays_dir / "grid_x.npy")
    grid_y = np.load(arrays_dir / "grid_y.npy")
    with (reports_dir / "energies.json").open("r", encoding="utf-8") as f:
        energies = np.asarray(json.load(f)["E_dimless"], dtype=float)
    return psi_grid, grid_x, grid_y, energies


def _pair_exchange_payload(pair_result, physics_cfg: dict) -> dict:
    e0 = energy_scale_meV(m_eff=float(physics_cfg["m_eff"]), L0_nm=float(physics_cfg["L0_nm"]))
    singlet = pair_result.sector_energies.get("singlet")
    triplet = pair_result.sector_energies.get("triplet")
    s0 = float(singlet[0]) if singlet is not None and len(singlet) else None
    t0 = float(triplet[0]) if triplet is not None and len(triplet) else None
    j_dimless = (t0 - s0) if s0 is not None and t0 is not None else None
    return {
        "singlet_ground_E_dimless": s0,
        "triplet_ground_E_dimless": t0,
        "J_dimless": j_dimless,
        "J_meV": j_dimless * e0 if j_dimless is not None else None,
        "J_GHz": j_dimless * e0 * 241.7989348 if j_dimless is not None else None,
        "definition": "J = E_triplet_0 - E_singlet_0",
    }


def _save_pair_outputs(run_dir: Path, cfg: dict, pair_result, x: np.ndarray, y: np.ndarray) -> None:
    arrays_dir = run_dir / "arrays"
    reports_dir = run_dir / "reports"
    plots_dir = run_dir / "plots"

    arrays = {
        "pair_coulomb_tensor": pair_result.coulomb_tensor,
        "pair_product_hamiltonian": pair_result.product_hamiltonian,
    }
    for sector, coeffs in pair_result.sector_coeffs.items():
        arrays[f"pair_ci_coeffs_{sector}"] = coeffs
    for sector, density in pair_result.one_body_densities.items():
        arrays[f"pair_one_body_density_{sector}"] = density
    for sector, density in pair_result.conditional_densities.items():
        arrays[f"pair_conditional_density_{sector}"] = density
    save_arrays(arrays_dir, **arrays)

    e0 = energy_scale_meV(m_eff=float(cfg["physics"]["m_eff"]), L0_nm=float(cfg["physics"]["L0_nm"]))
    pair_energies = {
        "E0_meV": e0,
        "sectors": {
            sector: {
                "E_dimless": vals.tolist(),
                "E_meV": (vals * e0).tolist(),
                "basis_dim": len(pair_result.sector_bases[sector].orbital_pairs),
                "orbital_pairs": [list(pair) for pair in pair_result.sector_bases[sector].orbital_pairs],
            }
            for sector, vals in pair_result.sector_energies.items()
        },
    }
    exchange = _pair_exchange_payload(pair_result, cfg["physics"])
    save_json(reports_dir / "pair_energies.json", pair_energies)
    save_json(reports_dir / "pair_exchange.json", exchange)
    save_json(reports_dir / "pair_coulomb_report.json", pair_result.coulomb_report)

    plot_pair_density_maps(pair_result.one_body_densities, x=x, y=y, out_dir=plots_dir)
    plot_pair_conditional_maps(pair_result.conditional_densities, x=x, y=y, out_dir=plots_dir)
    plot_pair_exchange_summary(exchange, out_dir=plots_dir)
    weights = {
        sector: ci_weight_spectrum(coeffs[:, 0])
        for sector, coeffs in pair_result.sector_coeffs.items()
        if coeffs.size
    }
    plot_ci_weight_spectrum(weights, out_dir=plots_dir)

    final_summary_path = reports_dir / "final_summary.json"
    if final_summary_path.exists():
        with final_summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["electron_pair"] = {
            "enabled": True,
            "num_orbitals": int(cfg["pair"]["num_orbitals"]),
            "sectors": [str(s).lower() for s in cfg["pair"]["sectors"]],
            "exchange": exchange,
            "coulomb": pair_result.coulomb_report,
            "plots": {
                "pair_exchange_summary": "pair_exchange_summary.png",
                "pair_ci_weight_spectrum": "pair_ci_weight_spectrum.png",
            },
        }
        save_json(final_summary_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="electron_pair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--one-electron-run", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not bool(cfg.get("pair", {}).get("enabled", False)):
        raise ValueError("pair.enabled must be true for experiments.run_electron_pair")

    run_dir = Path(args.one_electron_run) if args.one_electron_run else _run_one_electron_stage(cfg, args)
    logger = build_logger(run_dir / "logs" / "pair.log")
    try:
        logger.info("Starting electron-pair CI solve for %s", str(run_dir))
        psi_grid, x, y, orbital_energies = _load_one_electron_artifacts(run_dir)
        pair_result = solve_pair_ci(
            psi_grid=psi_grid,
            orbital_energies=orbital_energies,
            x=x,
            y=y,
            physics_cfg=cfg["physics"],
            pair_cfg=cfg["pair"],
        )
        _save_pair_outputs(run_dir, cfg, pair_result, x, y)
        logger.info("Electron-pair CI solve completed for %s", str(run_dir))
    except Exception as exc:
        tb = traceback.format_exc()
        (run_dir / "reports" / "pair_error_traceback.txt").write_text(tb, encoding="utf-8")
        finalize_run_from_pair = args.one_electron_run is None
        if finalize_run_from_pair:
            finalize_run(SimpleNamespace(run_dir=run_dir), status="failed", error_message=str(exc))
        raise


if __name__ == "__main__":
    main()
