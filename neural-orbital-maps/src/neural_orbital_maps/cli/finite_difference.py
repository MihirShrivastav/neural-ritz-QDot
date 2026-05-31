from __future__ import annotations

import argparse

from neural_orbital_maps.io.config import load_config
from neural_orbital_maps.workflows import run_finite_difference_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a conventional one-electron finite-difference baseline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_finite_difference_baseline(load_config(args.config)))


if __name__ == "__main__":
    main()
