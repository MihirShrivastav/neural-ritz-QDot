from __future__ import annotations

import argparse
from pathlib import Path

from neural_orbital_maps.analysis.runtime import write_runtime_report
from neural_orbital_maps.plotting.figures import runtime_comparison_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare durations from completed run final summaries.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory to include. Pass multiple times.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = write_runtime_report(args.run_dir, args.output_dir)
    runtime_comparison_plot(report, Path(args.output_dir) / "runtime_comparison.png")
    print(Path(args.output_dir))


if __name__ == "__main__":
    main()
