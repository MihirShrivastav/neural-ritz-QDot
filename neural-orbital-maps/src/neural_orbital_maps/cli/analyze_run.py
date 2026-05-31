from __future__ import annotations

import argparse
import json

from neural_orbital_maps.workflows import summarize_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a Neural-Orbital Maps run.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
