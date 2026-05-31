from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from neural_orbital_maps.analysis.observables import conditional_density, correlation_report
from neural_orbital_maps.analysis.reports import ci_weights, density_checks, exchange_report, orthonormality_report
from neural_orbital_maps.io.artifacts import load_json, save_arrays, save_json
from neural_orbital_maps.io.config import RunConfig
from neural_orbital_maps.io.logging import build_logger
from neural_orbital_maps.io.runs import RunPaths, create_run, finalize_run
from neural_orbital_maps.numerics.pair_ci import solve_pair_ci
from neural_orbital_maps.physics.units import energy_scale_meV
from neural_orbital_maps.plotting.figures import difference_plot, exchange_bar, field_plot
from neural_orbital_maps.training.block_ritz import OneElectronResult, train_one_electron


def _material_dict(config: RunConfig) -> dict:
    return config.material.model_dump()


def _save_one_electron(paths: RunPaths, config: RunConfig, result: OneElectronResult) -> None:
    x = result.grid.x
    y = result.grid.y
    e0 = energy_scale_meV(config.material.m_eff, config.material.L0_nm)
    save_arrays(
        paths.arrays,
        grid_x=x,
        grid_y=y,
        potential=result.potential,
        orbitals=result.orbitals,
        orbital_energies=result.energies,
        overlap=result.overlap,
        hamiltonian=result.hamiltonian,
    )
    save_json(
        paths.reports / "one_electron_energies.json",
        {"E0_meV": e0, "E_dimless": result.energies.tolist(), "E_meV": (result.energies * e0).tolist()},
    )
    save_json(paths.reports / "orthonormality.json", orthonormality_report(result.orbitals, result.grid.cell_area))
    save_json(paths.reports / "training_metrics.json", {"metrics": result.metrics})
    torch.save(result.model_state, paths.checkpoints / "model_final.pt")
    field_plot(result.potential, x, y, "Potential", "V(x,y)", paths.plots / "potential.png")
    for idx, orbital in enumerate(result.orbitals):
        field_plot(orbital**2, x, y, f"State {idx} Density", "|psi|^2", paths.plots / f"density_state_{idx}.png")


def run_one_electron(config: RunConfig) -> Path:
    start = time.time()
    paths = create_run(config)
    logger = build_logger(paths.logs / "run.log")
    logger.info("starting one-electron run")
    try:
        result = train_one_electron(config, logger)
        _save_one_electron(paths, config, result)
        save_json(
            paths.reports / "final_summary.json",
            {
                "status": "completed",
                "mode": "one_electron",
                "run_dir": str(paths.run_dir),
                "duration_sec": time.time() - start,
                "num_states": config.solver.num_states,
            },
        )
        finalize_run(paths, "completed")
        logger.info("completed one-electron run at %s", paths.run_dir)
    except Exception as exc:
        finalize_run(paths, "failed", {"error": str(exc)})
        logger.exception("one-electron run failed")
        raise
    return paths.run_dir


def run_pair_ci(config: RunConfig) -> Path:
    start = time.time()
    paths = create_run(config)
    logger = build_logger(paths.logs / "run.log")
    logger.info("starting pair CI run")
    try:
        result = train_one_electron(config, logger)
        _save_one_electron(paths, config, result)
        pair = solve_pair_ci(
            orbitals=result.orbitals,
            energies=result.energies,
            x=result.grid.x,
            y=result.grid.y,
            material=_material_dict(config),
            num_orbitals=config.pair.num_orbitals,
            sectors=list(config.pair.sectors),
            coulomb_strength=config.pair.coulomb.strength,
            softening=config.pair.coulomb.softening,
        )
        e0 = energy_scale_meV(config.material.m_eff, config.material.L0_nm)
        pair_energy_payload = {
            "E0_meV": e0,
            "sectors": {
                sector: {
                    "E_dimless": vals.tolist(),
                    "E_meV": (vals * e0).tolist(),
                    "basis_dim": len(pair.sector_bases[sector].orbital_pairs),
                    "orbital_pairs": [list(p) for p in pair.sector_bases[sector].orbital_pairs],
                }
                for sector, vals in pair.sector_energies.items()
            },
        }
        exchange = exchange_report(pair, _material_dict(config))
        save_arrays(
            paths.arrays,
            coulomb_tensor=pair.coulomb_tensor,
            pair_ci_coeffs_singlet=pair.sector_coeffs.get("singlet", np.empty((0, 0))),
            pair_ci_coeffs_triplet=pair.sector_coeffs.get("triplet", np.empty((0, 0))),
            one_body_density_singlet=pair.one_body_densities.get("singlet", np.empty((0, 0))),
            one_body_density_triplet=pair.one_body_densities.get("triplet", np.empty((0, 0))),
        )
        conditional_arrays = {}
        for sector in pair.sector_energies:
            conditional_arrays[f"conditional_density_{sector}"] = conditional_density(result.orbitals[: config.pair.num_orbitals], pair, sector)
        save_arrays(paths.arrays, **conditional_arrays)
        save_json(paths.reports / "pair_energies.json", pair_energy_payload)
        save_json(paths.reports / "pair_exchange.json", exchange)
        save_json(paths.reports / "density_checks.json", density_checks(pair, result.grid.cell_area))
        save_json(paths.reports / "ci_weights.json", ci_weights(pair))
        save_json(
            paths.reports / "correlation_report.json",
            correlation_report(pair, result.grid.x, result.grid.cell_area, result.orbitals[: config.pair.num_orbitals]),
        )
        for sector, density in pair.one_body_densities.items():
            field_plot(density, result.grid.x, result.grid.y, f"{sector.title()} One-Body Density", "rho(x,y)", paths.plots / f"one_body_density_{sector}.png")
        for sector, density in conditional_arrays.items():
            field_plot(density, result.grid.x, result.grid.y, sector.replace("_", " ").title(), "P(r2 | r1)", paths.plots / f"{sector}.png")
        if "singlet" in pair.one_body_densities and "triplet" in pair.one_body_densities:
            difference_plot(
                pair.one_body_densities["singlet"],
                pair.one_body_densities["triplet"],
                result.grid.x,
                result.grid.y,
                "Singlet - Triplet One-Body Density",
                paths.plots / "one_body_density_difference.png",
            )
        exchange_bar(exchange, paths.plots / "exchange_summary.png")
        save_json(
            paths.reports / "final_summary.json",
            {
                "status": "completed",
                "mode": "pair_ci",
                "run_dir": str(paths.run_dir),
                "duration_sec": time.time() - start,
                "exchange": exchange,
            },
        )
        finalize_run(paths, "completed")
        logger.info("completed pair CI run at %s", paths.run_dir)
    except Exception as exc:
        finalize_run(paths, "failed", {"error": str(exc)})
        logger.exception("pair CI run failed")
        raise
    return paths.run_dir


def summarize_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    summary = {}
    for name in ["final_summary.json", "one_electron_energies.json", "pair_exchange.json", "density_checks.json", "ci_weights.json", "correlation_report.json"]:
        path = run_dir / "reports" / name
        if path.exists():
            summary[name] = load_json(path)
    return summary
