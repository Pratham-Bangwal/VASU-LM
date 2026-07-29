import json
from pathlib import Path

import pytest

from evaluation.comparison import (
    compare_verified_arithmetic_runs,
    paired_binary_bootstrap,
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
