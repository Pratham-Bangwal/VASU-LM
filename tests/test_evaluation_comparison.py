import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.comparison import (
    compare_verified_arithmetic_runs,
    paired_binary_bootstrap,
)
from evaluation.frozen_snapshots import (
    compare_frozen_evaluation_snapshots,
    create_frozen_evaluation_snapshot,
    render_frozen_snapshot_comparison,
    write_frozen_evaluation_snapshot,
)


def test_paired_bootstrap_is_reproducible_and_reports_delta() -> None:
    result = paired_binary_bootstrap(
        [False, False, True, True], [True, False, True, True], samples=1000, seed=7
    )
    assert result["delta"] == 0.25
    assert result == paired_binary_bootstrap(
        [False, False, True, True], [True, False, True, True], samples=1000, seed=7
    )


def test_paired_bootstrap_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError, match="paired"):
        paired_binary_bootstrap([True], [])


def test_compare_completed_arithmetic_runs_is_identity_bound(tmp_path: Path) -> None:
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    for directory, outcomes in ((base, [False, True]), (candidate, [True, True])):
        directory.mkdir()
        (directory / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "evaluator_version": "v1",
                    "split": "eval",
                    "split_sha256": "split",
                    "manifest_sha256": "manifest",
                    "tokenizer_sha256": "tokenizer",
                    "generation": {"mode": "greedy"},
                }
            ),
            encoding="utf-8",
        )
        (directory / "per_example.jsonl").write_text(
            "".join(
                json.dumps({"id": str(index), "correct": value}) + "\n"
                for index, value in enumerate(outcomes)
            ),
            encoding="utf-8",
        )
    report = compare_verified_arithmetic_runs(
        baseline_dir=base, candidate_dir=candidate, samples=1000
    )
    assert report["paired_exact_accuracy"]["delta"] == 0.5


def test_frozen_snapshot_comparison_is_identity_bound_and_readable(
    tmp_path: Path,
) -> None:
    baseline_source = tmp_path / "baseline_result.json"
    candidate_source = tmp_path / "candidate_result.json"
    baseline_source.write_text(
        json.dumps({"format_version": "suite_v1", "metrics": {"score": 0.2}}),
        encoding="utf-8",
    )
    candidate_source.write_text(
        json.dumps({"format_version": "suite_v1", "metrics": {"score": 0.5}}),
        encoding="utf-8",
    )
    common = {
        "sources": {"suite": baseline_source},
        "metrics": {"exact_accuracy": ("suite", "metrics.score")},
        "comparison_identity": {"suite": "heldout", "generation": "greedy"},
    }
    baseline_snapshot = create_frozen_evaluation_snapshot(
        label="baseline", **common
    )
    candidate_snapshot = create_frozen_evaluation_snapshot(
        label="candidate",
        sources={"suite": candidate_source},
        metrics=common["metrics"],
        comparison_identity=common["comparison_identity"],
    )
    baseline_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    write_frozen_evaluation_snapshot(output=baseline_path, snapshot=baseline_snapshot)
    write_frozen_evaluation_snapshot(output=candidate_path, snapshot=candidate_snapshot)
    report = compare_frozen_evaluation_snapshots(
        baseline=baseline_path, candidate=candidate_path
    )
    assert report["metrics"] == [
        {
            "metric": "exact_accuracy",
            "path": "metrics.score",
            "baseline": 0.2,
            "candidate": 0.5,
            "delta": 0.3,
        }
    ]
    assert "delta=+0.300000" in render_frozen_snapshot_comparison(report)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_frozen_evaluation_snapshot(output=baseline_path, snapshot=baseline_snapshot)


def test_frozen_snapshot_accepts_utf8_bom_source_json(tmp_path: Path) -> None:
    source = tmp_path / "windows_result.json"
    source.write_bytes(
        b"\xef\xbb\xbf"
        + json.dumps({"format_version": "suite_v1", "metrics": {"score": 0.2}}).encode(
            "utf-8"
        )
    )
    snapshot = create_frozen_evaluation_snapshot(
        label="windows-source",
        sources={"suite": source},
        metrics={"score": ("suite", "metrics.score")},
        comparison_identity={"suite": "fixture"},
    )
    assert snapshot["metrics"]["score"]["value"] == 0.2


def test_frozen_snapshot_comparison_rejects_identity_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps({"format_version": "suite_v1", "metrics": {"score": 0.2}}),
        encoding="utf-8",
    )
    common = {
        "sources": {"suite": source},
        "metrics": {"score": ("suite", "metrics.score")},
    }
    baseline = create_frozen_evaluation_snapshot(
        label="baseline", comparison_identity={"split": "dev"}, **common
    )
    candidate = create_frozen_evaluation_snapshot(
        label="candidate", comparison_identity={"split": "heldout"}, **common
    )
    baseline_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    write_frozen_evaluation_snapshot(output=baseline_path, snapshot=baseline)
    write_frozen_evaluation_snapshot(output=candidate_path, snapshot=candidate)
    with pytest.raises(ValueError, match="identities"):
        compare_frozen_evaluation_snapshots(
            baseline=baseline_path, candidate=candidate_path
        )


def test_frozen_snapshot_comparison_rejects_source_format_mismatch(
    tmp_path: Path,
) -> None:
    baseline_source = tmp_path / "baseline_result.json"
    candidate_source = tmp_path / "candidate_result.json"
    baseline_source.write_text(
        json.dumps({"format_version": "suite_v1", "metrics": {"score": 0.2}}),
        encoding="utf-8",
    )
    candidate_source.write_text(
        json.dumps({"format_version": "suite_v2", "metrics": {"score": 0.2}}),
        encoding="utf-8",
    )
    common = {
        "metrics": {"score": ("suite", "metrics.score")},
        "comparison_identity": {"split": "heldout"},
    }
    baseline = create_frozen_evaluation_snapshot(
        label="baseline", sources={"suite": baseline_source}, **common
    )
    candidate = create_frozen_evaluation_snapshot(
        label="candidate", sources={"suite": candidate_source}, **common
    )
    baseline_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    write_frozen_evaluation_snapshot(output=baseline_path, snapshot=baseline)
    write_frozen_evaluation_snapshot(output=candidate_path, snapshot=candidate)
    with pytest.raises(ValueError, match="source format"):
        compare_frozen_evaluation_snapshots(
            baseline=baseline_path, candidate=candidate_path
        )


def test_frozen_snapshot_commands_support_documented_direct_invocation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps({"format_version": "suite_v1", "metrics": {"score": 0.2}}),
        encoding="utf-8",
    )
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    command_base = [
        sys.executable,
        "evaluation/freeze_evaluation_snapshot.py",
        "--source",
        f"suite={source}",
        "--metric",
        "score=suite:metrics.score",
        "--identity",
        "suite=fixture",
    ]
    subprocess.run(
        [*command_base, "--label", "baseline", "--output", str(baseline)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [*command_base, "--label", "candidate", "--output", str(candidate)],
        check=True,
        capture_output=True,
        text=True,
    )
    comparison = subprocess.run(
        [
            sys.executable,
            "evaluation/compare_frozen_evaluation_snapshots.py",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "VASU frozen evaluation snapshot comparison" in comparison.stdout
