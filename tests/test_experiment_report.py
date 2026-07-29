import json
from pathlib import Path

from vasu.utils.experiment_report import compare_evaluation_snapshots


def test_snapshot_comparison_is_hash_bound_and_numeric_only(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps({"accuracy": 0.1, "meta": {"name": "a", "count": 5}}),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps({"accuracy": 0.2, "meta": {"name": "b", "count": 7}}),
        encoding="utf-8",
    )
    report = compare_evaluation_snapshots(baseline=baseline, candidate=candidate)
    assert report["read_only"] is True
    assert report["shared_numeric_metrics"] == [
        {"metric": "accuracy", "baseline": 0.1, "candidate": 0.2, "delta": 0.1},
        {"metric": "meta.count", "baseline": 5.0, "candidate": 7.0, "delta": 2.0},
    ]
