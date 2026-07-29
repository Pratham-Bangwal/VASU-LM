"""Regression coverage for identity-bound runtime benchmark contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from vasu.utils.runtime_benchmark import (
    compare_runtime_benchmarks,
    create_runtime_benchmark,
    render_runtime_comparison,
    write_runtime_benchmark,
    write_runtime_comparison,
)


def _raw_report(path: Path, value: float) -> None:
    path.write_text(
        json.dumps({"format_version": "profile_v1", "timing": {"seconds": value}}),
        encoding="utf-8",
    )


def test_runtime_benchmarks_are_identity_bound_and_non_overwriting(tmp_path: Path) -> None:
    baseline_raw, candidate_raw = tmp_path / "base_raw.json", tmp_path / "candidate_raw.json"
    _raw_report(baseline_raw, 2.0)
    _raw_report(candidate_raw, 1.5)
    common = {
        "benchmark_kind": "data_pipeline",
        "metrics": {"step_seconds": "timing.seconds"},
        "workload_identity": {"batch_size": "2", "sequence_length": "256"},
        "environment_identity": {"device": "cuda:0", "torch": "2.x"},
    }
    baseline = create_runtime_benchmark(
        label="baseline",
        raw_report=baseline_raw,
        variant_identity={"workers": "0"},
        **common,
    )
    candidate = create_runtime_benchmark(
        label="candidate",
        raw_report=candidate_raw,
        variant_identity={"workers": "2"},
        **common,
    )
    baseline_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    write_runtime_benchmark(output=baseline_path, benchmark=baseline)
    write_runtime_benchmark(output=candidate_path, benchmark=candidate)
    report = compare_runtime_benchmarks(baseline=baseline_path, candidate=candidate_path)
    assert report["metrics"][0]["delta"] == -0.5
    assert report["metrics"][0]["relative_delta"] == -0.25
    assert "(-25.00%)" in render_runtime_comparison(report)
    assert report["candidate"]["variant_identity"] == {"workers": "2"}
    comparison_path = tmp_path / "comparison.json"
    write_runtime_comparison(output=comparison_path, comparison=report)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_runtime_comparison(output=comparison_path, comparison=report)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_runtime_benchmark(output=baseline_path, benchmark=baseline)


def test_runtime_benchmark_rejects_environment_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    _raw_report(raw, 1.0)
    common = {
        "benchmark_kind": "synthetic_model",
        "raw_report": raw,
        "metrics": {"step_seconds": "timing.seconds"},
        "workload_identity": {"batch_size": "2"},
        "variant_identity": {"optimizer": "standard"},
    }
    baseline = create_runtime_benchmark(
        label="baseline", environment_identity={"device": "cpu"}, **common
    )
    candidate = create_runtime_benchmark(
        label="candidate", environment_identity={"device": "cuda:0"}, **common
    )
    baseline_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    write_runtime_benchmark(output=baseline_path, benchmark=baseline)
    write_runtime_benchmark(output=candidate_path, benchmark=candidate)
    with pytest.raises(ValueError, match="environment_identity"):
        compare_runtime_benchmarks(baseline=baseline_path, candidate=candidate_path)


def test_runtime_benchmark_commands_support_direct_invocation(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    _raw_report(raw, 1.0)
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    command = [
        sys.executable,
        "scripts/profiling/create_runtime_benchmark.py",
        "--kind",
        "fixture",
        "--raw-report",
        str(raw),
        "--metric",
        "seconds=timing.seconds",
        "--workload",
        "batch=1",
        "--environment",
        "device=cpu",
        "--variant",
        "workers=0",
    ]
    for label, output in (("baseline", baseline), ("candidate", candidate)):
        subprocess.run(
            [*command, "--label", label, "--output", str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profiling/compare_runtime_benchmarks.py",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(tmp_path / "comparison.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "VASU runtime benchmark comparison" in result.stdout
    assert (tmp_path / "comparison.json").is_file()
