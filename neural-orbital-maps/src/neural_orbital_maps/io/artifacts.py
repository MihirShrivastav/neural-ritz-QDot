from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_arrays(directory: str | Path, **arrays: np.ndarray) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(directory / f"{name}.npy", array)
