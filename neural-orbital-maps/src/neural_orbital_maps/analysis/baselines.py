from __future__ import annotations

from pathlib import Path

from neural_orbital_maps.io.artifacts import load_json, save_json


def pair_baseline_comparison(neural_run: str | Path, finite_difference_run: str | Path) -> dict:
    neural_run = Path(neural_run)
    finite_difference_run = Path(finite_difference_run)
    neural_exchange = load_json(neural_run / "reports" / "pair_exchange.json")
    fd_exchange = load_json(finite_difference_run / "reports" / "pair_exchange.json")
    neural_j_mev = float(neural_exchange["J_meV"])
    fd_j_mev = float(fd_exchange["J_meV"])
    delta = neural_j_mev - fd_j_mev
    denominator = abs(fd_j_mev) if abs(fd_j_mev) > 1e-15 else None
    return {
        "neural_run": str(neural_run),
        "finite_difference_run": str(finite_difference_run),
        "observable": "J = E_triplet_0 - E_singlet_0",
        "neural": {
            "J_dimless": neural_exchange["J_dimless"],
            "J_meV": neural_exchange["J_meV"],
            "J_GHz": neural_exchange["J_GHz"],
        },
        "finite_difference": {
            "J_dimless": fd_exchange["J_dimless"],
            "J_meV": fd_exchange["J_meV"],
            "J_GHz": fd_exchange["J_GHz"],
        },
        "difference": {
            "neural_minus_fd_J_meV": delta,
            "abs_neural_minus_fd_J_meV": abs(delta),
            "relative_to_fd": abs(delta) / denominator if denominator is not None else None,
        },
        "notes": [
            "This compares two one-electron orbital sources passed through the same pair-CI machinery.",
            "Differences include neural orbital approximation error, finite-difference discretization error, and any config mismatch between runs.",
        ],
    }


def write_pair_baseline_comparison(neural_run: str | Path, finite_difference_run: str | Path, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = pair_baseline_comparison(neural_run, finite_difference_run)
    save_json(output_dir / "pair_baseline_comparison.json", report)
    return report
