from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def field_plot(field: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, label: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(field, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def exchange_bar(exchange: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [exchange.get("singlet_ground_E_dimless"), exchange.get("triplet_ground_E_dimless")]
    ax.bar(["S0", "T0"], vals, color=["#315c9f", "#b34b3f"])
    ax.set_ylabel("Energy (dimensionless)")
    ax.set_title(f"Exchange J = {exchange.get('J_meV', 0.0):.4g} meV")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def difference_plot(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, path: str | Path) -> None:
    diff = a - b
    vmax = float(np.max(np.abs(diff))) if diff.size else 1.0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(diff, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="difference")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def exchange_map(detuning: np.ndarray, barrier: np.ndarray, values: np.ndarray, title: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        values,
        extent=[detuning.min(), detuning.max(), barrier.min(), barrier.max()],
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    ax.set_title(title)
    ax.set_xlabel("detuning")
    ax.set_ylabel("barrier")
    fig.colorbar(im, ax=ax, label="J (meV)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def convergence_plot(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row.get("status") == "completed"]
    completed.sort(key=lambda row: int(row["num_points"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if not completed:
        for ax in axes:
            ax.axis("off")
            ax.set_title("No completed points")
    else:
        xs = np.array([int(row["num_points"]) for row in completed], dtype=float)
        e0 = np.array([float(row["E0_dimless"]) for row in completed], dtype=float)
        residual = np.array([float(row["max_projected_residual"]) for row in completed], dtype=float)
        axes[0].plot(xs, e0, marker="o", color="#315c9f", label="neural Ritz")
        fd_rows = [row for row in completed if row.get("fd_E0_dimless") not in ("", None)]
        if fd_rows:
            fd_xs = np.array([int(row["num_points"]) for row in fd_rows], dtype=float)
            fd_e0 = np.array([float(row["fd_E0_dimless"]) for row in fd_rows], dtype=float)
            axes[0].plot(fd_xs, fd_e0, marker="s", color="#4b8f4a", label="finite difference")
            axes[0].legend()
        axes[0].set_title("Ground Energy Convergence")
        axes[0].set_xlabel("grid points per axis")
        axes[0].set_ylabel("E0 (dimensionless)")
        axes[0].grid(alpha=0.25)
        axes[1].semilogy(xs, np.maximum(residual, 1e-16), marker="o", color="#b34b3f")
        axes[1].set_title("Projected Residual")
        axes[1].set_xlabel("grid points per axis")
        axes[1].set_ylabel("max residual")
        axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pair_summary_dashboard(
    potential: np.ndarray,
    singlet_density: np.ndarray | None,
    triplet_density: np.ndarray | None,
    correlation: np.ndarray | None,
    x: np.ndarray,
    y: np.ndarray,
    exchange: dict,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (potential, "Potential", "viridis"),
        (singlet_density, "Singlet density", "viridis"),
        (triplet_density, "Triplet density", "viridis"),
        (correlation, "Pair correlation ratio", "magma"),
    ]
    for ax, (field, title, cmap) in zip(axes.ravel(), panels):
        if field is None or field.size == 0:
            ax.axis("off")
            ax.set_title(f"{title}: unavailable")
            continue
        im = ax.imshow(field, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Pair CI Summary: J = {exchange.get('J_meV', 0.0):.4g} meV")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
