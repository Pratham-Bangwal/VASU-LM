"""Versioned, identity-bound contracts for VASU runtime measurements.

Raw profiler output is intentionally kept separate from the immutable benchmark
contract.  This permits existing profilers to evolve while comparisons fail
closed unless their workload and runtime environment are explicitly identical.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


RUNTIME_BENCHMARK_FORMAT_VERSION = "vasu_runtime_benchmark_v1"
RUNTIME_COMPARISON_FORMAT_VERSION = "vasu_runtime_benchmark_comparison_v1"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of the exact bytes used for a benchmark input."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"benchmark source is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark source JSON must be an object: {path}")
    return payload


def _require_identity(name: str, value: Mapping[str, str]) -> dict[str, str]:
    if not value or any(not key or not item for key, item in value.items()):
        raise ValueError(f"{name} must contain non-empty keys and values")
    return dict(sorted(value.items()))


def _numeric_path(payload: Mapping[str, Any], path: str) -> float:
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


def create_runtime_benchmark(
    *,
    label: str,
    benchmark_kind: str,
    raw_report: Path,
    workload_identity: Mapping[str, str],
    environment_identity: Mapping[str, str],
    metrics: Mapping[str, str],
) -> dict[str, Any]:
    """Bind selected raw-report metrics to workload and environment identity."""

    if not label.strip() or not benchmark_kind.strip():
        raise ValueError("benchmark label and kind must be non-empty")
    if not metrics or any(not name or not path for name, path in metrics.items()):
        raise ValueError("at least one named metric path is required")
    raw_payload = _read_json_object(raw_report)
    return {
        "format_version": RUNTIME_BENCHMARK_FORMAT_VERSION,
        "read_only": True,
        "label": label,
        "benchmark_kind": benchmark_kind,
        "raw_report": {
            "path": raw_report.as_posix(),
            "sha256": sha256_file(raw_report),
            "format_version": raw_payload.get("format_version"),
        },
        "workload_identity": _require_identity("workload identity", workload_identity),
        "environment_identity": _require_identity(
            "environment identity", environment_identity
        ),
        "metrics": {
            name: {"path": path, "value": _numeric_path(raw_payload, path)}
            for name, path in sorted(metrics.items())
        },
    }


def write_runtime_benchmark(*, output: Path, benchmark: Mapping[str, Any]) -> None:
    """Atomically create a benchmark artifact without replacing existing evidence."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary_path, output)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing benchmark: {output}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_benchmark(path: Path) -> dict[str, Any]:
    benchmark = _read_json_object(path)
    if benchmark.get("format_version") != RUNTIME_BENCHMARK_FORMAT_VERSION:
        raise ValueError(f"not a supported runtime benchmark: {path}")
    if benchmark.get("read_only") is not True:
        raise ValueError(f"runtime benchmark must declare read_only=true: {path}")
    if not isinstance(benchmark.get("metrics"), dict) or not benchmark["metrics"]:
        raise ValueError(f"runtime benchmark has no metrics: {path}")
    return benchmark


def compare_runtime_benchmarks(*, baseline: Path, candidate: Path) -> dict[str, Any]:
    """Compare runtime benchmarks only when their contract is identical."""

    baseline_report = _read_benchmark(baseline)
    candidate_report = _read_benchmark(candidate)
    for field in ("benchmark_kind", "workload_identity", "environment_identity"):
        if baseline_report.get(field) != candidate_report.get(field):
            raise ValueError(f"runtime benchmarks have incompatible {field}")
    baseline_metrics = baseline_report["metrics"]
    candidate_metrics = candidate_report["metrics"]
    if set(baseline_metrics) != set(candidate_metrics):
        raise ValueError("runtime benchmarks do not define the same metric names")
    comparisons: list[dict[str, Any]] = []
    for name in sorted(baseline_metrics):
        baseline_metric = baseline_metrics[name]
        candidate_metric = candidate_metrics[name]
        if (
            not isinstance(baseline_metric, dict)
            or not isinstance(candidate_metric, dict)
            or baseline_metric.get("path") != candidate_metric.get("path")
        ):
            raise ValueError(f"runtime metric definition is incompatible: {name}")
        baseline_value = baseline_metric.get("value")
        candidate_value = candidate_metric.get("value")
        if (
            isinstance(baseline_value, bool)
            or isinstance(candidate_value, bool)
            or not isinstance(baseline_value, (int, float))
            or not isinstance(candidate_value, (int, float))
        ):
            raise ValueError(f"runtime metric value is invalid: {name}")
        delta = float(candidate_value) - float(baseline_value)
        comparisons.append(
            {
                "metric": name,
                "path": baseline_metric["path"],
                "baseline": float(baseline_value),
                "candidate": float(candidate_value),
                "delta": delta,
                "relative_delta": delta / float(baseline_value)
                if baseline_value != 0
                else None,
            }
        )
    return {
        "format_version": RUNTIME_COMPARISON_FORMAT_VERSION,
        "read_only": True,
        "benchmark_kind": baseline_report["benchmark_kind"],
        "workload_identity": baseline_report["workload_identity"],
        "environment_identity": baseline_report["environment_identity"],
        "baseline": {
            "label": baseline_report["label"],
            "sha256": sha256_file(baseline),
        },
        "candidate": {
            "label": candidate_report["label"],
            "sha256": sha256_file(candidate),
        },
        "metrics": comparisons,
    }


def render_runtime_comparison(report: Mapping[str, Any]) -> str:
    """Render a deterministic, descriptive performance comparison summary."""

    if report.get("format_version") != RUNTIME_COMPARISON_FORMAT_VERSION:
        raise ValueError("unsupported runtime benchmark comparison report")
    lines = [
        "VASU runtime benchmark comparison",
        f"Kind: {report['benchmark_kind']}",
        f"Baseline: {report['baseline']['label']} ({report['baseline']['sha256']})",
        f"Candidate: {report['candidate']['label']} ({report['candidate']['sha256']})",
        "Workload identity:",
    ]
    lines.extend(
        f"  {key}: {value}" for key, value in report["workload_identity"].items()
    )
    lines.append("Environment identity:")
    lines.extend(
        f"  {key}: {value}" for key, value in report["environment_identity"].items()
    )
    lines.append("Metrics:")
    for metric in report["metrics"]:
        relative = (
            "n/a"
            if metric["relative_delta"] is None
            else f"{metric['relative_delta']:+.2%}"
        )
        lines.append(
            f"  {metric['metric']}: baseline={metric['baseline']:.6f}, "
            f"candidate={metric['candidate']:.6f}, delta={metric['delta']:+.6f} "
            f"({relative})"
        )
    lines.append(
        "Interpretation: descriptive timing evidence only; this report does not "
        "authorize training or select a checkpoint."
    )
    return "\n".join(lines)
