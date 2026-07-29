"""Deterministic read-only comparisons of versioned evaluation snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from vasu.utils.lineage import sha256_file


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for key, child in value.items():
            result.update(
                _numeric_leaves(child, f"{prefix}.{key}" if prefix else str(key))
            )
        return result
    return {}


def compare_evaluation_snapshots(*, baseline: Path, candidate: Path) -> dict[str, Any]:
    """Compare shared numeric metrics from two immutable JSON snapshots."""

    base_payload = json.loads(baseline.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
    base_metrics = _numeric_leaves(base_payload)
    candidate_metrics = _numeric_leaves(candidate_payload)
    shared = sorted(set(base_metrics) & set(candidate_metrics))
    return {
        "format_version": "vasu_evaluation_comparison_v1",
        "read_only": True,
        "baseline": {"path": baseline.as_posix(), "sha256": sha256_file(baseline)},
        "candidate": {"path": candidate.as_posix(), "sha256": sha256_file(candidate)},
        "shared_numeric_metrics": [
            {
                "metric": key,
                "baseline": base_metrics[key],
                "candidate": candidate_metrics[key],
                "delta": candidate_metrics[key] - base_metrics[key],
            }
            for key in shared
        ],
    }
