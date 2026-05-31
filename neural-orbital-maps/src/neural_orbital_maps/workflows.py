from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from neural_orbital_maps.analysis.hubbard import two_site_hubbard_report
from neural_orbital_maps.analysis.observables import conditional_density, correlation_report, localized_orbitals_from_lowest_pair, pair_correlation_map
from neural_orbital_maps.analysis.reports import ci_weights, density_checks, exchange_report, one_electron_quality_report, orthonormality_report
from neural_orbital_maps.io.artifacts import load_json, save_arrays, save_json
from neural_orbital_maps.io.config import RunConfig
from neural_orbital_maps.io.logging import build_logger
from neural_orbital_maps.io.runs import RunPaths, create_run, finalize_run
from neural_orbital_maps.numerics.finite_difference import FiniteDifferenceResult, solve_finite_difference
from neural_orbital_maps.numerics.pair_ci import solve_pair_ci
from neural_orbital_maps.physics.units import energy_scale_meV
from neural_orbital_maps.plotting.figures import difference_plot, exchange_bar, field_plot, hubbard_exchange_comparison, pair_summary_dashboard
from neural_orbital_maps.training.block_ritz import OneElectronResult, train_one_electron


def _material_dict(config: RunConfig) -> dict:
    return config.material.model_dump()


def _training_status(result: OneElectronResult, configured_steps: int) -> dict:
    return {
        "steps_completed": result.steps_completed,
        "configured_steps": configured_steps,
        "early_stopped": result.early_stopped,
        "stop_reason": result.stop_reason,
        "best_eigsum": result.best_eigsum,
    }


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
        ritz_coefficients=result.coefficients,
    )
    save_json(
        paths.reports / "one_electron_energies.json",
        {"E0_meV": e0, "E_dimless": result.energies.tolist(), "E_meV": (result.energies * e0).tolist()},
    )
    save_json(paths.reports / "orthonormality.json", orthonormality_report(result.orbitals, result.grid.cell_area))
    save_json(
        paths.reports / "one_electron_quality.json",
        one_electron_quality_report(
            result.overlap,
            result.hamiltonian,
            result.coefficients,
            result.energies,
            result.projected_residuals,
            result.final_norms_before,
            result.final_norms_after,
        ),
    )
    localized, localized_report = localized_orbitals_from_lowest_pair(result.orbitals, x, result.grid.cell_area)
    save_json(paths.reports / "localized_orbitals.json", localized_report)
    if localized:
        save_arrays(paths.arrays, localized_orbital_left=localized["left"], localized_orbital_right=localized["right"])
        field_plot(localized["left"] ** 2, x, y, "Localized Left Orbital Density", "|phi_L|^2", paths.plots / "localized_orbital_left.png")
        field_plot(localized["right"] ** 2, x, y, "Localized Right Orbital Density", "|phi_R|^2", paths.plots / "localized_orbital_right.png")
    save_json(paths.reports / "training_metrics.json", {"metrics": result.metrics, "training_status": _training_status(result, config.training.steps)})
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
                "training_status": _training_status(result, config.training.steps),
            },
        )
        finalize_run(paths, "completed")
        logger.info("completed one-electron run at %s", paths.run_dir)
    except Exception as exc:
        finalize_run(paths, "failed", {"error": str(exc)})
        logger.exception("one-electron run failed")
        raise
    return paths.run_dir


def _save_finite_difference(paths: RunPaths, config: RunConfig, result: FiniteDifferenceResult) -> None:
    x = result.grid.x
    y = result.grid.y
    e0 = energy_scale_meV(config.material.m_eff, config.material.L0_nm)
    save_arrays(
        paths.arrays,
        grid_x=x,
        grid_y=y,
        potential=result.potential,
        fd_orbitals=result.orbitals,
        fd_energies=result.energies,
    )
    save_json(
        paths.reports / "finite_difference_energies.json",
        {
            "E0_meV": e0,
            "E_dimless": result.energies.tolist(),
            "E_meV": (result.energies * e0).tolist(),
            "operator": "-Delta + V",
            "boundary_condition": "homogeneous Dirichlet outside the grid",
            "stencil": "five-point second-order finite difference",
        },
    )
    save_json(paths.reports / "orthonormality.json", orthonormality_report(result.orbitals, result.grid.cell_area))
    field_plot(result.potential, x, y, "Finite-Difference Potential", "V(x,y)", paths.plots / "potential.png")
    for idx, orbital in enumerate(result.orbitals):
        field_plot(orbital**2, x, y, f"FD State {idx} Density", "|psi|^2", paths.plots / f"fd_density_state_{idx}.png")


def run_finite_difference_baseline(config: RunConfig) -> Path:
    start = time.time()
    paths = create_run(config)
    logger = build_logger(paths.logs / "run.log")
    logger.info("starting finite-difference baseline run")
    try:
        result = solve_finite_difference(config)
        _save_finite_difference(paths, config, result)
        save_json(
            paths.reports / "final_summary.json",
            {
                "status": "completed",
                "mode": "finite_difference_baseline",
                "run_dir": str(paths.run_dir),
                "duration_sec": time.time() - start,
                "num_states": config.solver.num_states,
                "ground_energy_dimless": float(result.energies[0]),
            },
        )
        finalize_run(paths, "completed")
        logger.info("completed finite-difference baseline run at %s", paths.run_dir)
    except Exception as exc:
        finalize_run(paths, "failed", {"error": str(exc)})
        logger.exception("finite-difference baseline run failed")
        raise
    return paths.run_dir


def _pair_energy_payload(pair, e0: float) -> dict:
    return {
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


def _save_pair_outputs(paths: RunPaths, config: RunConfig, pair, orbitals: np.ndarray, energies: np.ndarray, x: np.ndarray, y: np.ndarray, potential: np.ndarray) -> dict:
    e0 = energy_scale_meV(config.material.m_eff, config.material.L0_nm)
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
        conditional_arrays[f"conditional_density_{sector}"] = conditional_density(orbitals[: config.pair.num_orbitals], pair, sector)
    save_arrays(paths.arrays, **conditional_arrays)
    pair_correlation_arrays = {}
    pair_correlation_reports = {}
    for sector, density in pair.one_body_densities.items():
        ratio, report = pair_correlation_map(orbitals[: config.pair.num_orbitals], pair, sector, density, float(abs(x[0, 1] - x[0, 0]) * abs(y[1, 0] - y[0, 0])))
        pair_correlation_arrays[f"pair_correlation_{sector}"] = ratio
        pair_correlation_reports[sector] = report
    save_arrays(paths.arrays, **pair_correlation_arrays)
    save_json(paths.reports / "pair_energies.json", _pair_energy_payload(pair, e0))
    save_json(paths.reports / "pair_exchange.json", exchange)
    cell_area = float(abs(x[0, 1] - x[0, 0]) * abs(y[1, 0] - y[0, 0]))
    save_json(paths.reports / "density_checks.json", density_checks(pair, cell_area))
    save_json(paths.reports / "ci_weights.json", ci_weights(pair))
    _, localized_report = localized_orbitals_from_lowest_pair(orbitals[: config.pair.num_orbitals], x, cell_area)
    hubbard = two_site_hubbard_report(energies[: config.pair.num_orbitals], pair.coulomb_tensor, localized_report.get("sign", 1.0))
    save_json(paths.reports / "hubbard_report.json", hubbard)
    save_json(paths.reports / "correlation_report.json", correlation_report(pair, x, cell_area, orbitals[: config.pair.num_orbitals]))
    save_json(paths.reports / "pair_correlation_report.json", pair_correlation_reports)
    for sector, density in pair.one_body_densities.items():
        field_plot(density, x, y, f"{sector.title()} One-Body Density", "rho(x,y)", paths.plots / f"one_body_density_{sector}.png")
    for sector, density in conditional_arrays.items():
        field_plot(density, x, y, sector.replace("_", " ").title(), "P(r2 | r1)", paths.plots / f"{sector}.png")
    for sector, ratio in pair_correlation_arrays.items():
        field_plot(ratio, x, y, sector.replace("_", " ").title(), "g(r2 | r1)", paths.plots / f"{sector}.png")
    if "singlet" in pair.one_body_densities and "triplet" in pair.one_body_densities:
        difference_plot(
            pair.one_body_densities["singlet"],
            pair.one_body_densities["triplet"],
            x,
            y,
            "Singlet - Triplet One-Body Density",
            paths.plots / "one_body_density_difference.png",
        )
    exchange_bar(exchange, paths.plots / "exchange_summary.png")
    hubbard_exchange_comparison(exchange, hubbard, e0, paths.plots / "hubbard_exchange_comparison.png")
    pair_summary_dashboard(
        potential,
        pair.one_body_densities.get("singlet"),
        pair.one_body_densities.get("triplet"),
        pair_correlation_arrays.get("pair_correlation_singlet"),
        x,
        y,
        exchange,
        paths.plots / "pair_summary_dashboard.png",
    )
    return exchange


def run_finite_difference_pair_ci(config: RunConfig) -> Path:
    start = time.time()
    paths = create_run(config)
    logger = build_logger(paths.logs / "run.log")
    logger.info("starting finite-difference pair CI baseline run")
    try:
        fd = solve_finite_difference(config)
        _save_finite_difference(paths, config, fd)
        pair = solve_pair_ci(
            orbitals=fd.orbitals,
            energies=fd.energies,
            x=fd.grid.x,
            y=fd.grid.y,
            material=_material_dict(config),
            num_orbitals=config.pair.num_orbitals,
            sectors=list(config.pair.sectors),
            coulomb_strength=config.pair.coulomb.strength,
            softening=config.pair.coulomb.softening,
        )
        exchange = _save_pair_outputs(paths, config, pair, fd.orbitals, fd.energies, fd.grid.x, fd.grid.y, fd.potential)
        save_json(
            paths.reports / "final_summary.json",
            {
                "status": "completed",
                "mode": "finite_difference_pair_ci",
                "run_dir": str(paths.run_dir),
                "duration_sec": time.time() - start,
                "exchange": exchange,
                "one_electron_source": "finite_difference",
            },
        )
        finalize_run(paths, "completed")
        logger.info("completed finite-difference pair CI baseline run at %s", paths.run_dir)
    except Exception as exc:
        finalize_run(paths, "failed", {"error": str(exc)})
        logger.exception("finite-difference pair CI baseline run failed")
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
        pair_correlation_arrays = {}
        pair_correlation_reports = {}
        for sector, density in pair.one_body_densities.items():
            ratio, report = pair_correlation_map(result.orbitals[: config.pair.num_orbitals], pair, sector, density, result.grid.cell_area)
            pair_correlation_arrays[f"pair_correlation_{sector}"] = ratio
            pair_correlation_reports[sector] = report
        save_arrays(paths.arrays, **pair_correlation_arrays)
        save_json(paths.reports / "pair_energies.json", pair_energy_payload)
        save_json(paths.reports / "pair_exchange.json", exchange)
        save_json(paths.reports / "density_checks.json", density_checks(pair, result.grid.cell_area))
        save_json(paths.reports / "ci_weights.json", ci_weights(pair))
        _, localized_report = localized_orbitals_from_lowest_pair(result.orbitals[: config.pair.num_orbitals], result.grid.x, result.grid.cell_area)
        hubbard = two_site_hubbard_report(result.energies[: config.pair.num_orbitals], pair.coulomb_tensor, localized_report.get("sign", 1.0))
        save_json(
            paths.reports / "hubbard_report.json",
            hubbard,
        )
        save_json(
            paths.reports / "correlation_report.json",
            correlation_report(pair, result.grid.x, result.grid.cell_area, result.orbitals[: config.pair.num_orbitals]),
        )
        save_json(paths.reports / "pair_correlation_report.json", pair_correlation_reports)
        for sector, density in pair.one_body_densities.items():
            field_plot(density, result.grid.x, result.grid.y, f"{sector.title()} One-Body Density", "rho(x,y)", paths.plots / f"one_body_density_{sector}.png")
        for sector, density in conditional_arrays.items():
            field_plot(density, result.grid.x, result.grid.y, sector.replace("_", " ").title(), "P(r2 | r1)", paths.plots / f"{sector}.png")
        for sector, ratio in pair_correlation_arrays.items():
            field_plot(ratio, result.grid.x, result.grid.y, sector.replace("_", " ").title(), "g(r2 | r1)", paths.plots / f"{sector}.png")
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
        hubbard_exchange_comparison(exchange, hubbard, e0, paths.plots / "hubbard_exchange_comparison.png")
        pair_summary_dashboard(
            result.potential,
            pair.one_body_densities.get("singlet"),
            pair.one_body_densities.get("triplet"),
            pair_correlation_arrays.get("pair_correlation_singlet"),
            result.grid.x,
            result.grid.y,
            exchange,
            paths.plots / "pair_summary_dashboard.png",
        )
        save_json(
            paths.reports / "final_summary.json",
            {
                "status": "completed",
                "mode": "pair_ci",
                "run_dir": str(paths.run_dir),
                "duration_sec": time.time() - start,
                "exchange": exchange,
                "training_status": _training_status(result, config.training.steps),
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
    for name in [
        "final_summary.json",
        "one_electron_energies.json",
        "finite_difference_energies.json",
        "one_electron_quality.json",
        "localized_orbitals.json",
        "pair_exchange.json",
        "density_checks.json",
        "ci_weights.json",
        "correlation_report.json",
        "pair_correlation_report.json",
        "hubbard_report.json",
    ]:
        path = run_dir / "reports" / name
        if path.exists():
            summary[name] = load_json(path)
    return summary
