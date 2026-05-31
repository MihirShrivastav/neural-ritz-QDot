from pathlib import Path

import pytest

from neural_orbital_maps.analysis.baselines import write_pair_baseline_comparison
from neural_orbital_maps.io.artifacts import load_json, save_json
from neural_orbital_maps.plotting.figures import pair_baseline_comparison_plot


def _write_exchange(run_dir: Path, j_mev: float) -> None:
    save_json(
        run_dir / "reports" / "pair_exchange.json",
        {
            "J_dimless": j_mev / 10.0,
            "J_meV": j_mev,
            "J_GHz": j_mev * 241.7989348,
        },
    )


def test_pair_baseline_comparison_report_and_plot(tmp_path):
    neural = tmp_path / "neural"
    fd = tmp_path / "fd"
    out = tmp_path / "comparison"
    _write_exchange(neural, 0.12)
    _write_exchange(fd, 0.10)
    report = write_pair_baseline_comparison(neural, fd, out)
    pair_baseline_comparison_plot(report, out / "pair_baseline_comparison.png")
    saved = load_json(out / "pair_baseline_comparison.json")
    assert saved["difference"]["neural_minus_fd_J_meV"] == pytest.approx(0.02)
    assert (out / "pair_baseline_comparison.png").exists()
