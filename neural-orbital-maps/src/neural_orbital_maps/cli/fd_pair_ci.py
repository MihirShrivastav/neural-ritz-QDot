from __future__ import annotations

import argparse

from neural_orbital_maps.io.config import load_config
from neural_orbital_maps.workflows import run_finite_difference_pair_ci


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pair CI using conventional finite-difference one-electron orbitals.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_finite_difference_pair_ci(load_config(args.config)))


if __name__ == "__main__":
    main()
