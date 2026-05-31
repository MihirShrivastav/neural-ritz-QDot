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
