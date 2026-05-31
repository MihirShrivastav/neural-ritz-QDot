from __future__ import annotations

import argparse

from neural_orbital_maps.io.config import load_config
from neural_orbital_maps.studies.baseline_comparison import run_pair_baseline_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run neural pair CI, finite-difference pair CI, and compare exchange baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-dir", default=None)
    args = parser.parse_args()
    print(run_pair_baseline_benchmark(load_config(args.config), study_dir=args.study_dir))


if __name__ == "__main__":
    main()
