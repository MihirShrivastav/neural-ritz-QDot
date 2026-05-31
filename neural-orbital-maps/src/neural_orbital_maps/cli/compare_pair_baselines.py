from __future__ import annotations

import argparse
from pathlib import Path

from neural_orbital_maps.analysis.baselines import write_pair_baseline_comparison
from neural_orbital_maps.plotting.figures import pair_baseline_comparison_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare neural pair-CI exchange against finite-difference pair-CI exchange.")
    parser.add_argument("--neural-run", required=True)
    parser.add_argument("--fd-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = write_pair_baseline_comparison(args.neural_run, args.fd_run, args.output_dir)
    pair_baseline_comparison_plot(report, Path(args.output_dir) / "pair_baseline_comparison.png")
    print(Path(args.output_dir))


if __name__ == "__main__":
    main()
