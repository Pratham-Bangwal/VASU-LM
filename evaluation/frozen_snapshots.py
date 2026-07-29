"""Identity-bound, immutable evaluation snapshot contracts.

Snapshots copy only explicitly selected scalar metrics while binding every
source report by SHA-256.  They are intentionally evaluation-only artifacts:
they do not select checkpoints or make promotion decisions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SNAPSHOT_FORMAT_VERSION = "vasu_frozen_evaluation_snapshot_v1"
COMPARISON_FORMAT_VERSION = "vasu_frozen_evaluation_snapshot_comparison_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        # ``utf-8-sig`` accepts standard UTF-8 as well as JSON emitted by
        # Windows tooling that prefixes a UTF-8 byte-order mark.  The source
        # hash remains over the original bytes, preserving provenance.
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"source is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"source JSON must be an object: {path}")
    format_version = payload.get("format_version")
    if not isinstance(format_version, str) or not format_version:
        raise ValueError(f"source JSON has no format_version: {path}")
    return payload


def _metric_value(payload: Mapping[str, Any], path: str) -> float:
    value: Any = payload
    for segment in path.split("."):
        if not segment:
            raise ValueError("metric paths cannot contain empty segments")
        if not isinstance(value, Mapping) or segment not in value:
            raise ValueError(f"metric path is absent: {path}")
        value = value[segment]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric path is not a numeric scalar: {path}")
    return float(value)


def create_frozen_evaluation_snapshot(
    *,
    label: str,
    sources: Mapping[str, Path],
    metrics: Mapping[str, tuple[str, str]],
    comparison_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Create a hash-bound evaluation snapshot from explicit scalar metrics.

    ``comparison_identity`` must describe the invariant suite, split, parser,
    and generation settings that a later candidate must match.  Checkpoint
    identity belongs in the source report and is deliberately not inferred as
    a comparison invariant.
    """

    if not label.strip():
        raise ValueError("snapshot label must be non-empty")
    if not sources:
        raise ValueError("at least one source is required")
    if len(set(sources)) != len(sources) or any(not name for name in sources):
        raise ValueError("source names must be unique and non-empty")
    if not metrics:
        raise ValueError("at least one metric is required")
    if len(set(metrics)) != len(metrics) or any(not name for name in metrics):
        raise ValueError("metric names must be unique and non-empty")
    if not comparison_identity:
        raise ValueError("comparison identity is required")
    if any(not key or not value for key, value in comparison_identity.items()):
        raise ValueError("comparison identity keys and values must be non-empty")

    payloads = {name: _read_json_object(path) for name, path in sources.items()}
    snapshot_sources = {
        name: {
            "path": path.as_posix(),
            "sha256": _sha256_file(path),
            "format_version": payloads[name]["format_version"],
        }
        for name, path in sorted(sources.items())
    }
    snapshot_metrics: dict[str, dict[str, Any]] = {}
    for name, (source_name, metric_path) in sorted(metrics.items()):
        if source_name not in payloads:
            raise ValueError(f"metric {name!r} references unknown source {source_name!r}")
        snapshot_metrics[name] = {
            "source": source_name,
            "path": metric_path,
            "value": _metric_value(payloads[source_name], metric_path),
        }
    return {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "read_only": True,
        "label": label,
        "comparison_identity": dict(sorted(comparison_identity.items())),
        "sources": snapshot_sources,
        "metrics": snapshot_metrics,
    }


def write_frozen_evaluation_snapshot(*, output: Path, snapshot: Mapping[str, Any]) -> None:
    """Atomically write a new snapshot without overwriting an existing artifact."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary_path, output)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing snapshot: {output}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_snapshot(path: Path) -> dict[str, Any]:
    snapshot = _read_json_object(path)
    if snapshot.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise ValueError(f"not a supported frozen evaluation snapshot: {path}")
    if snapshot.get("read_only") is not True:
        raise ValueError(f"snapshot must declare read_only=true: {path}")
    if not isinstance(snapshot.get("comparison_identity"), dict):
        raise ValueError(f"snapshot has no comparison identity: {path}")
    if not isinstance(snapshot.get("metrics"), dict) or not snapshot["metrics"]:
        raise ValueError(f"snapshot has no metrics: {path}")
    return snapshot


def compare_frozen_evaluation_snapshots(
    *, baseline: Path, candidate: Path
) -> dict[str, Any]:
    """Compare only snapshots with an identical evaluation contract."""

    baseline_snapshot = _read_snapshot(baseline)
    candidate_snapshot = _read_snapshot(candidate)
    if baseline_snapshot["comparison_identity"] != candidate_snapshot["comparison_identity"]:
        raise ValueError("frozen snapshot comparison identities are not comparable")
    baseline_sources = baseline_snapshot.get("sources")
    candidate_sources = candidate_snapshot.get("sources")
    if not isinstance(baseline_sources, dict) or not isinstance(candidate_sources, dict):
        raise ValueError("frozen snapshot source definitions are invalid")
    if set(baseline_sources) != set(candidate_sources):
        raise ValueError("frozen snapshots do not define the same source names")
    for name in baseline_sources:
        baseline_source = baseline_sources[name]
        candidate_source = candidate_sources[name]
        if (
            not isinstance(baseline_source, dict)
            or not isinstance(candidate_source, dict)
            or baseline_source.get("format_version")
            != candidate_source.get("format_version")
        ):
            raise ValueError(f"source format is not comparable: {name}")
    baseline_metrics = baseline_snapshot["metrics"]
    candidate_metrics = candidate_snapshot["metrics"]
    if set(baseline_metrics) != set(candidate_metrics):
        raise ValueError("frozen snapshots do not define the same metric names")

    comparisons: list[dict[str, Any]] = []
    for name in sorted(baseline_metrics):
        baseline_metric = baseline_metrics[name]
        candidate_metric = candidate_metrics[name]
        if (
            not isinstance(baseline_metric, dict)
            or not isinstance(candidate_metric, dict)
            or baseline_metric.get("source") != candidate_metric.get("source")
            or baseline_metric.get("path") != candidate_metric.get("path")
        ):
            raise ValueError(f"metric definition is not comparable: {name}")
        baseline_value = baseline_metric.get("value")
        candidate_value = candidate_metric.get("value")
        if (
            isinstance(baseline_value, bool)
            or isinstance(candidate_value, bool)
            or not isinstance(baseline_value, (int, float))
            or not isinstance(candidate_value, (int, float))
        ):
            raise ValueError(f"metric value is not numeric: {name}")
        comparisons.append(
            {
                "metric": name,
                "path": baseline_metric["path"],
                "baseline": float(baseline_value),
                "candidate": float(candidate_value),
                "delta": float(candidate_value) - float(baseline_value),
            }
        )
    return {
        "format_version": COMPARISON_FORMAT_VERSION,
        "read_only": True,
        "comparison_identity": baseline_snapshot["comparison_identity"],
        "baseline": {"label": baseline_snapshot["label"], "sha256": _sha256_file(baseline)},
        "candidate": {"label": candidate_snapshot["label"], "sha256": _sha256_file(candidate)},
        "metrics": comparisons,
    }


def render_frozen_snapshot_comparison(report: Mapping[str, Any]) -> str:
    """Render a deterministic, human-readable evaluation dashboard summary."""

    if report.get("format_version") != COMPARISON_FORMAT_VERSION:
        raise ValueError("unsupported frozen snapshot comparison report")
    baseline = report["baseline"]
    candidate = report["candidate"]
    lines = [
        "VASU frozen evaluation snapshot comparison",
        f"Baseline: {baseline['label']} ({baseline['sha256']})",
        f"Candidate: {candidate['label']} ({candidate['sha256']})",
        "Evaluation identity:",
    ]
    lines.extend(
        f"  {key}: {value}"
        for key, value in sorted(report["comparison_identity"].items())
    )
    lines.append("Metrics:")
    lines.extend(
        "  {metric}: baseline={baseline:.6f}, candidate={candidate:.6f}, delta={delta:+.6f}"
        .format(**metric)
        for metric in report["metrics"]
    )
    lines.append(
        "Interpretation: this report is descriptive only; it makes no checkpoint "
        "selection or promotion decision."
    )
    return "\n".join(lines)
