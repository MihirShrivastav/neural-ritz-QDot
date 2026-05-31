from __future__ import annotations

import argparse

from neural_orbital_maps.io.config import load_exchange_map_config
from neural_orbital_maps.studies.exchange_map import run_exchange_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a detuning/barrier exchange-map study.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-dir", default=None, help="Existing study directory to resume or explicit output directory for a new study.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing points.csv and recompute all points.")
    args = parser.parse_args()
    print(run_exchange_map(load_exchange_map_config(args.config), study_dir=args.study_dir, resume=not args.no_resume))


if __name__ == "__main__":
    main()
