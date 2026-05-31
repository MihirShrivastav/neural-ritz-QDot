from __future__ import annotations

from pathlib import Path

from neural_orbital_maps.io.artifacts import load_json, save_json


def runtime_report(run_dirs: list[str | Path]) -> dict:
    entries = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        final = load_json(run_path / "reports" / "final_summary.json")
        entries.append(
            {
                "run_dir": str(run_path),
                "mode": final.get("mode", "unknown"),
                "duration_sec": float(final.get("duration_sec", 0.0)),
                "status": final.get("status", "unknown"),
            }
        )
    fastest = min(entries, key=lambda item: item["duration_sec"]) if entries else None
    slowest = max(entries, key=lambda item: item["duration_sec"]) if entries else None
    return {
        "entries": entries,
        "fastest": fastest,
        "slowest": slowest,
        "speedup_slowest_over_fastest": (slowest["duration_sec"] / fastest["duration_sec"]) if fastest and slowest and fastest["duration_sec"] > 0 else None,
        "notes": [
            "Runtime reports compare completed run metadata; they do not normalize for hardware load or config mismatches.",
            "Use common configs and repeated runs before treating timing differences as benchmark claims.",
        ],
    }


def write_runtime_report(run_dirs: list[str | Path], output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = runtime_report(run_dirs)
    save_json(output_dir / "runtime_report.json", report)
    return report
