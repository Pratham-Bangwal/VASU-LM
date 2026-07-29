"""Deterministic paired evaluation statistics for experiment reports."""

from __future__ import annotations

import random
import hashlib
import json
from pathlib import Path
from typing import Sequence


def paired_binary_bootstrap(
    baseline_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Return a reproducible percentile interval for paired accuracy delta."""

    if len(baseline_correct) != len(candidate_correct) or not baseline_correct:
        raise ValueError("paired non-empty outcome sequences are required")
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    differences = [
        int(candidate) - int(baseline)
        for baseline, candidate in zip(baseline_correct, candidate_correct, strict=True)
    ]
    rng = random.Random(seed)
    size = len(differences)
    draws = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return {
        "examples": size,
        "baseline_accuracy": sum(baseline_correct) / size,
        "candidate_accuracy": sum(candidate_correct) / size,
        "delta": sum(differences) / size,
        "ci_95_lower": draws[int(0.025 * samples)],
        "ci_95_upper": draws[int(0.975 * samples) - 1],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_verified_arithmetic_runs(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Compare two completed arithmetic runs with identity-bound paired CIs."""

    def load(directory: Path) -> tuple[dict, list[dict], Path, Path]:
        manifest_path = directory / "run_manifest.json"
        results_path = directory / "per_example.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("arithmetic run is not complete")
        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        identifiers = [str(row.get("id", "")) for row in rows]
        if (
            not rows
            or len(set(identifiers)) != len(identifiers)
            or not all(identifiers)
        ):
            raise ValueError("arithmetic run has missing or duplicate IDs")
        return manifest, rows, manifest_path, results_path

    baseline_manifest, baseline_rows, baseline_manifest_path, baseline_results_path = (
        load(baseline_dir)
    )
    (
        candidate_manifest,
        candidate_rows,
        candidate_manifest_path,
        candidate_results_path,
    ) = load(candidate_dir)
    identity_fields = (
        "evaluator_version",
        "split",
        "split_sha256",
        "manifest_sha256",
        "tokenizer_sha256",
        "generation",
    )
    if any(
        baseline_manifest.get(field) != candidate_manifest.get(field)
        for field in identity_fields
    ):
        raise ValueError("arithmetic run identities are not comparable")
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    candidate_by_id = {str(row["id"]): row for row in candidate_rows}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("arithmetic runs do not cover the same example IDs")
    ids = sorted(baseline_by_id)
    comparison = paired_binary_bootstrap(
        [bool(baseline_by_id[item].get("correct")) for item in ids],
        [bool(candidate_by_id[item].get("correct")) for item in ids],
        samples=samples,
        seed=seed,
    )
    return {
        "format_version": "vasu_verified_arithmetic_paired_comparison_v1",
        "identity": {field: baseline_manifest.get(field) for field in identity_fields},
        "baseline": {
            "run_manifest_sha256": _sha256_file(baseline_manifest_path),
            "per_example_sha256": _sha256_file(baseline_results_path),
        },
        "candidate": {
            "run_manifest_sha256": _sha256_file(candidate_manifest_path),
            "per_example_sha256": _sha256_file(candidate_results_path),
        },
        "paired_exact_accuracy": comparison,
    }
