from pathlib import Path

from neural_orbital_maps.analysis.runtime import write_runtime_report
from neural_orbital_maps.io.artifacts import load_json, save_json
from neural_orbital_maps.plotting.figures import runtime_comparison_plot


def _write_final(run_dir: Path, mode: str, duration: float) -> None:
    save_json(run_dir / "reports" / "final_summary.json", {"status": "completed", "mode": mode, "duration_sec": duration})


def test_runtime_report_and_plot(tmp_path):
    neural = tmp_path / "neural"
    fd = tmp_path / "fd"
    out = tmp_path / "runtime"
    _write_final(neural, "pair_ci", 12.0)
    _write_final(fd, "finite_difference_pair_ci", 4.0)
    report = write_runtime_report([neural, fd], out)
    runtime_comparison_plot(report, out / "runtime_comparison.png")
    saved = load_json(out / "runtime_report.json")
    assert saved["fastest"]["mode"] == "finite_difference_pair_ci"
    assert saved["speedup_slowest_over_fastest"] == 3.0
    assert (out / "runtime_comparison.png").exists()
