from __future__ import annotations

import argparse

from neural_orbital_maps.io.config import load_config
from neural_orbital_maps.workflows import run_one_electron


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-electron neural Block-Ritz training.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_dir = run_one_electron(load_config(args.config))
    print(run_dir)


if __name__ == "__main__":
    main()
