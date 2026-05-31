from __future__ import annotations

import argparse

from neural_orbital_maps.io.config import load_convergence_study_config
from neural_orbital_maps.studies.convergence import run_convergence_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-electron grid convergence study.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-dir", default=None, help="Explicit output directory for the convergence study.")
    args = parser.parse_args()
    print(run_convergence_study(load_convergence_study_config(args.config), study_dir=args.study_dir))


if __name__ == "__main__":
    main()
