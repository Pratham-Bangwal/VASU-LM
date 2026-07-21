"""Production recovery and validation for the historical FineWeb extension.

The recovered artifact is a concatenation of independently valid gzip members.
One member is committed per 1,000 accepted records.  The progress file records
the last committed byte boundary, allowing deterministic truncation on resume.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests

from .extension_recovery import (
    HistoricalRecoveryRecord,
    MATCH_STATUSES,
    RecoveryError,
    RecoveryResult,
    RetrievedSourceRecord,
    atomic_write_json,
    atomic_write_text,
    classify_retrieval,
    historical_extension_fingerprint,
    load_historical_records,
    utc_now,
)
from .fineweb_index import (
    FineWebDocumentIndex,
    build_fineweb_document_index,
    sha256_file,
)
from .normalization import NORMALIZATION_VERSION
from .recovery_benchmark import (
    BatchedDatasetServerClient,
    PermanentProviderError,
    ProviderHttpClient,
    RetryExhaustedError,
    RetryPolicy,
)
from .schemas import FineWebIndexConfig, FineWebSource, INDEX_FORMAT_VERSION


PRODUCTION_FORMAT_VERSION = 1
EXPECTED_RECORD_COUNT = 379_247
EXPECTED_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
EXPECTED_REPOSITORY = "HuggingFaceFW/fineweb-edu"
EXPECTED_CONFIG = "CC-MAIN-2025-26"
EXPECTED_SPLIT = "train"
GIB = 1024**3


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_production_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Production recovery config missing: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Production recovery config is invalid JSON: {error}") from error
    required = {
        "format_version", "dataset_repository", "dataset_config", "split",
        "dataset_revision", "source_id_field", "text_field", "metadata_path",
        "historical_binary_path", "historical_binary_sha256",
        "historical_database_path", "historical_database_sha256", "artifact_path",
        "progress_path", "json_report_path", "text_report_path", "batch_size",
        "fallback_batch_sizes", "checkpoint_interval_documents", "concurrency",
        "requests_per_second", "connect_timeout_seconds", "read_timeout_seconds",
        "retry_count", "initial_backoff_seconds", "maximum_backoff_seconds",
        "jitter_seconds", "retry_seed", "maximum_response_bytes",
        "minimum_free_disk_bytes", "historical_fingerprint", "acquisition_method",
        "allow_revision_fallback",
    }
    missing, extra = required - set(payload), set(payload) - required
    if missing or extra:
        raise ValueError(f"Production config fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    expected_identity = (EXPECTED_REPOSITORY, EXPECTED_CONFIG, EXPECTED_SPLIT, EXPECTED_REVISION)
    identity = (
        payload["dataset_repository"], payload["dataset_config"], payload["split"],
        payload["dataset_revision"],
    )
    if identity != expected_identity:
        raise ValueError(f"Production source identity mismatch: {identity!r}")
    fixed = {
        "format_version": PRODUCTION_FORMAT_VERSION,
        "batch_size": 25,
        "fallback_batch_sizes": [10, 5, 1],
        "checkpoint_interval_documents": 1000,
        "concurrency": 1,
        "requests_per_second": 2.0,
        "connect_timeout_seconds": 10.0,
        "read_timeout_seconds": 120.0,
        "retry_count": 4,
        "maximum_backoff_seconds": 60.0,
        "retry_seed": 42,
        "historical_fingerprint": "sha256_utf8_python_str_strip_v1",
        "acquisition_method": "huggingface_dataset_viewer_composite_or_filter",
        "allow_revision_fallback": False,
    }
    for key, value in fixed.items():
        if payload[key] != value:
            raise ValueError(f"Production config requires {key}={value!r}")
    if int(payload["minimum_free_disk_bytes"]) < 10 * GIB:
        raise ValueError("Production config must require at least 10 GiB free")
    return payload


def _repository_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Production path must be repository-relative: {value}")
    return root / path


def _metadata_revision_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    url = (
        f"https://huggingface.co/api/datasets/{config['dataset_repository']}"
        f"/revision/{config['dataset_revision']}"
    )
    response = requests.get(
        url,
        headers={"User-Agent": "VASU-extension-production-preflight/1.0"},
        timeout=(float(config["connect_timeout_seconds"]), float(config["read_timeout_seconds"])),
    )
    response.raise_for_status()
    payload = response.json()
    observed = payload.get("sha") if isinstance(payload, dict) else None
    if observed != config["dataset_revision"]:
        raise RecoveryError(
            f"Pinned metadata revision mismatch: expected {config['dataset_revision']}, got {observed!r}"
        )
    return {"url": url, "status_code": response.status_code, "observed_revision": observed}


def _is_git_ignored(root: Path, path: Path) -> bool:
    import subprocess

    relative = path.resolve().relative_to(root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", relative], cwd=root, check=False
    )
    return result.returncode == 0


def preflight_production(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    provider_probe: Callable[[Mapping[str, Any]], dict[str, Any]] = _metadata_revision_probe,
) -> dict[str, Any]:
    root = repository_root.resolve()
    binary = _repository_path(root, str(config["historical_binary_path"]))
    database = _repository_path(root, str(config["historical_database_path"]))
    metadata_path = _repository_path(root, str(config["metadata_path"]))
    artifact = _repository_path(root, str(config["artifact_path"]))
    progress = _repository_path(root, str(config["progress_path"]))
    free = shutil.disk_usage(root).free
    if free < int(config["minimum_free_disk_bytes"]):
        raise RecoveryError(f"Insufficient free disk: {free} bytes")
    if sha256_file(binary) != config["historical_binary_sha256"]:
        raise RecoveryError("Historical extension binary SHA-256 mismatch")
    if sha256_file(database) != config["historical_database_sha256"]:
        raise RecoveryError("Historical fingerprint database SHA-256 mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in ("dataset_repository", "dataset_config", "split", "dataset_revision"):
        if metadata.get(key) != config[key]:
            raise RecoveryError(f"Historical metadata conflicts on {key}")
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        count, unique_ids, unique_hashes, invalid = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT source_id), COUNT(DISTINCT fingerprint),
              SUM(CASE WHEN source_id IS NULL OR source_id='' OR length(fingerprint)!=64
                       OR fingerprint!=lower(fingerprint)
                       OR fingerprint GLOB '*[^0-9a-f]*' THEN 1 ELSE 0 END)
            FROM fingerprints WHERE origin='extension'
            """
        ).fetchone()
    if integrity != "ok":
        raise RecoveryError(f"Historical fingerprint DB integrity failed: {integrity}")
    if (int(count), int(unique_ids), int(unique_hashes), int(invalid or 0)) != (
        EXPECTED_RECORD_COUNT, EXPECTED_RECORD_COUNT, EXPECTED_RECORD_COUNT, 0
    ):
        raise RecoveryError("Historical IDs/fingerprints do not satisfy the 379,247-record contract")
    if artifact.exists() and not progress.exists():
        raise RecoveryError("Recovery artifact exists without resumable progress")
    if artifact.with_name(artifact.name + ".complete").exists():
        raise RecoveryError("A completed production artifact conflicts with a new acquisition")
    for ignored_path in (artifact.parent, progress.parent):
        ignored_path.mkdir(parents=True, exist_ok=True)
        if not _is_git_ignored(root, ignored_path):
            raise RecoveryError(f"Production output directory is not ignored by Git: {ignored_path}")
    probe = provider_probe(config)
    return {
        "status": "passed",
        "free_disk_bytes": free,
        "historical_binary_sha256": config["historical_binary_sha256"],
        "historical_database_sha256": config["historical_database_sha256"],
        "historical_source_ids": int(unique_ids),
        "historical_fingerprints": int(unique_hashes),
        "database_integrity": integrity,
        "configuration_hash": canonical_hash(config),
        "provider_revision_probe": probe,
        "estimated_requests": math.ceil(EXPECTED_RECORD_COUNT / int(config["batch_size"])),
        "estimated_accepted_text_bytes": 2_031_251_862,
        "estimated_downloaded_response_bytes": 2_242_383_218,
        "artifact_path": str(config["artifact_path"]),
        "progress_path": str(config["progress_path"]),
    }


def _initial_progress(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": PRODUCTION_FORMAT_VERSION,
        "status": "in_progress",
        "configuration_hash": canonical_hash(config),
        "source_revision": config["dataset_revision"],
        "strategy": {
            "batch_size": config["batch_size"],
            "fallback_batch_sizes": config["fallback_batch_sizes"],
            "concurrency": config["concurrency"],
            "requests_per_second": config["requests_per_second"],
        },
        "completed_source_ids": [],
        "failed_source_ids": {},
        "request_counts": {},
        "retry_count": 0,
        "transient_error_count": 0,
        "timeout_count": 0,
        "http_status_counts": {},
        "downloaded_response_bytes": 0,
        "accepted_text_bytes": 0,
        "committed_artifact_bytes": 0,
        "committed_members": 0,
        "fallback_events": [],
        "current_batch": 0,
        "started_at": utc_now(),
        "updated_at": utc_now(),
    }


def _load_or_initialize_progress(
    config: Mapping[str, Any], artifact: Path, progress_path: Path, *, restart: bool
) -> dict[str, Any]:
    if restart:
        for path in (artifact, progress_path, artifact.with_name(artifact.name + ".complete")):
            if path.exists():
                path.unlink()
    if not progress_path.exists():
        if artifact.exists():
            raise RecoveryError("Artifact exists without progress; use explicit --restart after review")
        return _initial_progress(config)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("configuration_hash") != canonical_hash(config):
        raise RecoveryError("Production progress configuration mismatch")
    if progress.get("source_revision") != config["dataset_revision"]:
        raise RecoveryError("Production progress revision mismatch")
    completed = progress.get("completed_source_ids")
    if not isinstance(completed, list) or len(completed) != len(set(completed)):
        raise RecoveryError("Production progress completed IDs are malformed or duplicated")
    committed = int(progress.get("committed_artifact_bytes", 0))
    actual = artifact.stat().st_size if artifact.exists() else 0
    if actual < committed:
        raise RecoveryError("Recovery artifact is shorter than its committed byte boundary")
    if actual > committed:
        with artifact.open("r+b") as handle:
            handle.truncate(committed)
    return progress


def _encode_gzip_member(records: Sequence[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    # mtime=0 makes committed members deterministic for identical records.
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            handle.write(b"\n")
    return buffer.getvalue()


def _commit_member(artifact: Path, records: Sequence[dict[str, Any]]) -> int:
    member = _encode_gzip_member(records)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("ab") as handle:
        handle.write(member)
        handle.flush()
        os.fsync(handle.fileno())
    return len(member)


def _next_fallback_size(size: int) -> int | None:
    if size > 10:
        return 10
    if size > 5:
        return 5
    if size > 1:
        return 1
    return None


def retrieve_with_fallback(
    records: Sequence[HistoricalRecoveryRecord],
    client: BatchedDatasetServerClient,
    *,
    batch_identifier: str,
    fallback_events: list[dict[str, Any]],
) -> list[tuple[HistoricalRecoveryRecord, RecoveryResult, RetrievedSourceRecord | None]]:
    """Retrieve a batch, falling back deterministically 25 -> 10 -> 5 -> 1."""
    ids = [item.source_id for item in records]
    try:
        grouped = client.retrieve_many(ids)
    except Exception as error:
        next_size = _next_fallback_size(len(records))
        if next_size is None:
            result = RecoveryResult(
                source_id=records[0].source_id,
                expected_hash=records[0].expected_hash,
                observed_hash=None,
                match_status="retrieval_error",
                provider_shard=None,
                stable_row_reference=None,
                raw_character_count=None,
                raw_utf8_bytes=None,
                retrieval_timestamp=utc_now(),
                diagnostic_note=f"{type(error).__name__}: {str(error)[:300]}",
            )
            return [(records[0], result, None)]
        fallback_events.append(
            {
                "batch_identifier": batch_identifier,
                "failed_size": len(records),
                "fallback_size": next_size,
                "reason": f"{type(error).__name__}: {str(error)[:300]}",
                "timestamp": utc_now(),
            }
        )
        output: list[tuple[HistoricalRecoveryRecord, RecoveryResult, RetrievedSourceRecord | None]] = []
        for offset in range(0, len(records), next_size):
            output.extend(
                retrieve_with_fallback(
                    records[offset : offset + next_size], client,
                    batch_identifier=f"{batch_identifier}.fallback-{next_size}-{offset // next_size}",
                    fallback_events=fallback_events,
                )
            )
        return output
    output = []
    for historical in records:
        provider_records = grouped[historical.source_id]
        result = classify_retrieval(historical, provider_records)
        provider = provider_records[0] if len(provider_records) == 1 else None
        output.append((historical, result, provider))
    return output


def _artifact_record(
    historical: HistoricalRecoveryRecord,
    result: RecoveryResult,
    provider: RetrievedSourceRecord,
    batch_identifier: str,
) -> dict[str, Any]:
    return {
        "historical_source_id": historical.source_id,
        "text": provider.text,
        "historical_expected_fingerprint": historical.expected_hash,
        "observed_fingerprint": result.observed_hash,
        "pinned_revision": provider.revision,
        "provider_shard": provider.provider_shard,
        "stable_row_reference": provider.stable_row_reference,
        "raw_character_count": result.raw_character_count,
        "retrieval_batch_identifier": batch_identifier,
        "retrieval_timestamp": result.retrieval_timestamp,
    }


def _http_snapshot(client: ProviderHttpClient) -> dict[str, Any]:
    return {
        "network_requests": client.network_request_count,
        "retry_count": client.retry_count,
        "transient_error_count": client.transient_error_count,
        "timeout_count": client.timeout_count,
        "http_status_counts": dict(sorted(client.http_status_counts.items())),
        "downloaded_response_bytes": client.downloaded_bytes,
    }


def _render_progress(completed: int, total: int, client: ProviderHttpClient, failed: int) -> str:
    remaining_requests = math.ceil((total - completed) / 25)
    return (
        f"Recovery {completed:,}/{total:,}; exact={completed / total:.4%}; "
        f"requests={client.network_request_count:,}; retries={client.retry_count:,}; "
        f"failures={failed:,}; downloaded={client.downloaded_bytes:,} bytes; "
        f"estimated requests remaining={remaining_requests:,}"
    )


def run_production_recovery(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    restart: bool = False,
    stop_after_documents: int | None = None,
    client: BatchedDatasetServerClient | None = None,
    http: ProviderHttpClient | None = None,
) -> dict[str, Any]:
    root = repository_root.resolve()
    artifact = _repository_path(root, str(config["artifact_path"]))
    progress_path = _repository_path(root, str(config["progress_path"]))
    historical = load_historical_records(
        _repository_path(root, str(config["historical_database_path"]))
    )
    if len(historical) != EXPECTED_RECORD_COUNT:
        raise RecoveryError("Historical record count changed after preflight")
    progress = _load_or_initialize_progress(config, artifact, progress_path, restart=restart)
    completed_ids = list(progress["completed_source_ids"])
    expected_prefix = [item.source_id for item in historical[: len(completed_ids)]]
    if completed_ids != expected_prefix:
        raise RecoveryError("Completed IDs are not the deterministic historical-ID prefix")
    if http is None:
        policy = RetryPolicy(
            requests_per_second=float(config["requests_per_second"]),
            connect_timeout_seconds=float(config["connect_timeout_seconds"]),
            read_timeout_seconds=float(config["read_timeout_seconds"]),
            retry_count=int(config["retry_count"]),
            initial_backoff_seconds=float(config["initial_backoff_seconds"]),
            maximum_backoff_seconds=float(config["maximum_backoff_seconds"]),
            jitter_seconds=float(config["jitter_seconds"]),
            seed=int(config["retry_seed"]),
        )
        http = ProviderHttpClient(
            policy, maximum_response_bytes=int(config["maximum_response_bytes"])
        )
    if client is None:
        client = BatchedDatasetServerClient(
            repository=str(config["dataset_repository"]),
            dataset_config=str(config["dataset_config"]),
            split=str(config["split"]),
            revision=str(config["dataset_revision"]),
            http=http,
        )
    partition: list[dict[str, Any]] = []
    partition_ids: list[str] = []
    accepted_uncommitted = 0
    pending = historical[len(completed_ids) :]
    started_monotonic = time.monotonic()
    base_accepted = int(progress.get("accepted_text_bytes", 0))
    start_batch = int(progress.get("current_batch", 0))
    prior_http = {
        "network_requests": int(progress.get("network_requests", 0)),
        "retry_count": int(progress.get("retry_count", 0)),
        "transient_error_count": int(progress.get("transient_error_count", 0)),
        "timeout_count": int(progress.get("timeout_count", 0)),
        "downloaded_response_bytes": int(progress.get("downloaded_response_bytes", 0)),
        "http_status_counts": dict(progress.get("http_status_counts", {})),
    }

    def cumulative_http() -> dict[str, Any]:
        current = _http_snapshot(http)
        statuses = dict(prior_http["http_status_counts"])
        for status, count in current["http_status_counts"].items():
            statuses[status] = int(statuses.get(status, 0)) + int(count)
        return {
            key: int(prior_http[key]) + int(current[key])
            for key in (
                "network_requests", "retry_count", "transient_error_count",
                "timeout_count", "downloaded_response_bytes",
            )
        } | {"http_status_counts": dict(sorted(statuses.items()))}

    batches = [pending[offset : offset + 25] for offset in range(0, len(pending), 25)]
    for batch_number, batch in enumerate(batches, start=start_batch + 1):
        batch_id = f"batch-{batch_number:06d}"
        retrieved = retrieve_with_fallback(
            batch, client, batch_identifier=batch_id,
            fallback_events=progress["fallback_events"],
        )
        failures = [(item, result) for item, result, _ in retrieved if result.match_status != "exact_match"]
        if failures:
            for item, result in failures:
                progress["failed_source_ids"][item.source_id] = asdict(result)
            progress.update({**cumulative_http(), "current_batch": batch_number, "updated_at": utc_now()})
            atomic_write_json(progress_path, progress)
            raise RecoveryError(
                f"Unresolved recovery failure in {batch_id}: "
                + ", ".join(f"{item.source_id}={result.match_status}" for item, result in failures)
            )
        for item, result, provider in retrieved:
            if provider is None:
                raise AssertionError("Exact match lacks provider record")
            progress["failed_source_ids"].pop(item.source_id, None)
            partition.append(_artifact_record(item, result, provider, batch_id))
            partition_ids.append(item.source_id)
            accepted_uncommitted += int(result.raw_utf8_bytes or 0)
        should_commit = len(partition) >= int(config["checkpoint_interval_documents"])
        final_batch = batch_number == start_batch + len(batches)
        bounded_stop = stop_after_documents is not None and len(completed_ids) + len(partition) >= stop_after_documents
        if should_commit or final_batch or bounded_stop:
            written = _commit_member(artifact, partition)
            completed_ids.extend(partition_ids)
            progress.update(
                {
                    "completed_source_ids": completed_ids,
                    "accepted_text_bytes": base_accepted + accepted_uncommitted,
                    "committed_artifact_bytes": int(progress["committed_artifact_bytes"]) + written,
                    "committed_members": int(progress["committed_members"]) + 1,
                    "current_batch": batch_number,
                    "updated_at": utc_now(),
                    **cumulative_http(),
                }
            )
            atomic_write_json(progress_path, progress)
            print(_render_progress(len(completed_ids), len(historical), http, len(progress["failed_source_ids"])), flush=True)
            partition = []
            partition_ids = []
            base_accepted = int(progress["accepted_text_bytes"])
            accepted_uncommitted = 0
        if bounded_stop:
            progress["bounded_stop"] = True
            atomic_write_json(progress_path, progress)
            return progress
    progress.update(
        {
            "status": "acquisition_complete_pending_validation",
            "completed_at": utc_now(),
            "runtime_seconds_this_invocation": round(time.monotonic() - started_monotonic, 3),
            **cumulative_http(),
        }
    )
    atomic_write_json(progress_path, progress)
    return progress


def iter_recovered_records(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RecoveryError(f"Malformed recovery artifact line {line_number}") from error
            if not isinstance(value, dict):
                raise RecoveryError(f"Non-object recovery artifact line {line_number}")
            yield value


def validate_recovered_artifact(
    config: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    root = repository_root.resolve()
    artifact = _repository_path(root, str(config["artifact_path"]))
    progress_path = _repository_path(root, str(config["progress_path"]))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    historical = load_historical_records(
        _repository_path(root, str(config["historical_database_path"]))
    )
    expected = {item.source_id: item.expected_hash for item in historical}
    seen: set[str] = set()
    count = total_characters = accepted_bytes = 0
    mismatches = malformed = duplicates = 0
    for record in iter_recovered_records(artifact):
        required = {
            "historical_source_id", "text", "historical_expected_fingerprint",
            "observed_fingerprint", "pinned_revision", "provider_shard",
            "stable_row_reference", "raw_character_count", "retrieval_batch_identifier",
            "retrieval_timestamp",
        }
        if set(record) != required:
            malformed += 1
            continue
        source_id, text = record["historical_source_id"], record["text"]
        if source_id in seen:
            duplicates += 1
            continue
        seen.add(source_id)
        if not isinstance(text, str) or source_id not in expected:
            malformed += 1
            continue
        observed = historical_extension_fingerprint(text)
        if (
            observed != expected[source_id]
            or record["historical_expected_fingerprint"] != expected[source_id]
            or record["observed_fingerprint"] != observed
            or record["pinned_revision"] != config["dataset_revision"]
            or record["raw_character_count"] != len(text)
        ):
            mismatches += 1
        count += 1
        total_characters += len(text)
        accepted_bytes += len(text.encode("utf-8"))
    counts = {status: 0 for status in sorted(MATCH_STATUSES)}
    counts["exact_match"] = count - mismatches
    counts["source_id_found_hash_mismatch"] = mismatches
    for item in progress.get("failed_source_ids", {}).values():
        status = item.get("match_status", "retrieval_error")
        counts[status if status in counts else "retrieval_error"] += 1
    passed = (
        count == EXPECTED_RECORD_COUNT
        and len(seen) == EXPECTED_RECORD_COUNT
        and mismatches == malformed == duplicates == 0
        and all(counts[name] == 0 for name in counts if name != "exact_match")
        and progress.get("source_revision") == config["dataset_revision"]
        and len(progress.get("completed_source_ids", [])) == EXPECTED_RECORD_COUNT
    )
    report = {
        "format_version": PRODUCTION_FORMAT_VERSION,
        "status": "complete" if passed else "blocked",
        "repository": config["dataset_repository"],
        "dataset_config": config["dataset_config"],
        "split": config["split"],
        "revision": config["dataset_revision"],
        "configuration_hash": canonical_hash(config),
        "historical_source_ids": len(historical),
        "accepted_records": count,
        "unique_source_ids": len(seen),
        "counts": counts,
        "malformed_artifact_records": malformed,
        "duplicate_artifact_source_ids": duplicates,
        "accepted_text_bytes": accepted_bytes,
        "total_characters": total_characters,
        "provider_statistics": {
            "network_requests": progress.get("network_requests", 0),
            "retry_count": progress.get("retry_count", 0),
            "transient_error_count": progress.get("transient_error_count", 0),
            "timeout_count": progress.get("timeout_count", 0),
            "http_status_counts": progress.get("http_status_counts", {}),
            "downloaded_response_bytes": progress.get("downloaded_response_bytes", 0),
            "fallback_events": progress.get("fallback_events", []),
        },
        "artifact_path": config["artifact_path"],
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256_file(artifact),
        "started_at": progress.get("started_at"),
        "completed_at": utc_now(),
        "full_text_in_report": False,
        "training_started": False,
    }
    json_path = _repository_path(root, str(config["json_report_path"]))
    text_path = _repository_path(root, str(config["text_report_path"]))
    atomic_write_json(json_path, report)
    atomic_write_text(text_path, render_production_report(report))
    if passed:
        atomic_write_text(artifact.with_name(artifact.name + ".complete"), report["artifact_sha256"] + "\n")
        progress.update({"status": "complete", "validated_at": report["completed_at"]})
        atomic_write_json(progress_path, progress)
    return report


def render_production_report(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    provider = report["provider_statistics"]
    return "\n".join(
        [
            "FineWeb Extension Production Recovery",
            "=" * 42,
            f"Status: {report['status']}",
            f"Pinned revision: {report['revision']}",
            f"Historical IDs: {report['historical_source_ids']:,}",
            f"Exact matches: {counts['exact_match']:,}",
            f"Hash mismatches: {counts['source_id_found_hash_mismatch']:,}",
            f"Missing IDs: {counts['source_id_not_found']:,}",
            f"Duplicate provider IDs: {counts['duplicate_source_id_in_provider']:,}",
            f"Malformed records: {counts['malformed_source_record']:,}",
            f"Retrieval errors: {counts['retrieval_error']:,}",
            f"Provider requests: {provider['network_requests']:,}",
            f"Retries: {provider['retry_count']:,}",
            f"Fallback batches: {len(provider['fallback_events']):,}",
            f"Downloaded response bytes: {provider['downloaded_response_bytes']:,}",
            f"Accepted text bytes: {report['accepted_text_bytes']:,}",
            f"Artifact size: {report['artifact_size_bytes']:,}",
            f"Artifact SHA-256: {report['artifact_sha256']}",
            "",
            "No source text is stored in this report.",
        ]
    ) + "\n"


def build_extension_index(config: Mapping[str, Any], *, repository_root: Path, resume: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    report = json.loads(
        _repository_path(root, str(config["json_report_path"])).read_text(encoding="utf-8")
    )
    if report.get("status") != "complete":
        raise RecoveryError("Recovery validation is not complete; extension index remains blocked")
    artifact = _repository_path(root, str(config["artifact_path"]))
    if sha256_file(artifact) != report.get("artifact_sha256"):
        raise RecoveryError("Recovered artifact hash changed after validation")
    index_config = FineWebIndexConfig(
        format_version=INDEX_FORMAT_VERSION,
        output_path="data/manifests/pretrain/fineweb_extension_document_index.sqlite3",
        metadata_path="data/manifests/pretrain/fineweb_extension_document_index_metadata.json",
        progress_path="data/interim/pretrain/fineweb_extension_document_index_progress.json",
        normalization_version=NORMALIZATION_VERSION,
        shingle_size=5,
        signature_size=64,
        bands=8,
        batch_size=1000,
        maximum_documents=None,
        expected_document_count=EXPECTED_RECORD_COUNT,
        coverage=("fineweb_extension",),
        missing_coverage=("fineweb_original",),
        sources=(
            FineWebSource(
                source_id="fineweb_edu_extension_train",
                source_revision=EXPECTED_REVISION,
                source_shard=EXPECTED_CONFIG,
                path=str(config["artifact_path"]),
                format="jsonl_gzip",
                text_field="text",
                document_id_field="historical_source_id",
                provenance_completeness="complete",
                expected_sha256=report["artifact_sha256"],
            ),
        ),
    )
    metadata = build_fineweb_document_index(
        index_config,
        repository_root=root,
        resume=resume,
        progress_callback=lambda item: print(
            f"Extension index {item['indexed_documents']:,}/{EXPECTED_RECORD_COUNT:,}; "
            f"rejected={item['rejected_records']:,}; duplicates={item['duplicate_hashes']:,}",
            flush=True,
        ),
    )
    output = root / index_config.output_path
    with sqlite3.connect(f"file:{output.as_posix()}?mode=ro", uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        input_documents = connection.execute(
            "SELECT COALESCE(SUM(processed+rejected+duplicate_hashes),0) FROM source_progress"
        ).fetchone()[0]
    if integrity != "ok" or int(input_documents) != EXPECTED_RECORD_COUNT:
        raise RecoveryError("Completed extension index failed integrity or input coverage validation")
    metadata.update(
        {
            "sqlite_integrity": integrity,
            "validated_source_artifact_sha256": report["artifact_sha256"],
            "expected_source_identity": "fineweb_extension",
        }
    )
    atomic_write_json(root / index_config.metadata_path, metadata)
    return metadata


def write_combined_coverage(*, repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    entries = []
    for source_id, db_name in (
        ("fineweb_original", "fineweb_document_index.sqlite3"),
        ("fineweb_extension", "fineweb_extension_document_index.sqlite3"),
    ):
        database = root / "data/manifests/pretrain" / db_name
        metadata_path = database.with_name(f"{database.stem}_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("completion_status") != "complete":
            raise RecoveryError(f"{source_id} index metadata is not complete")
        if metadata.get("normalization_version") != NORMALIZATION_VERSION:
            raise RecoveryError(f"{source_id} normalization version mismatch")
        observed = sha256_file(database)
        if observed != metadata.get("output_sha256"):
            raise RecoveryError(f"{source_id} index SHA-256 mismatch")
        index = FineWebDocumentIndex(database)
        index.close()
        entries.append(
            {
                "source_identity": source_id,
                "index_path": database.relative_to(root).as_posix(),
                "metadata_path": metadata_path.relative_to(root).as_posix(),
                "index_sha256": observed,
                "normalization_version": metadata["normalization_version"],
                "completion_status": metadata["completion_status"],
                "indexed_documents": metadata["indexed_documents"],
                "validated_source_hashes": metadata.get("source_files", []),
            }
        )
    payload = {
        "format_version": 1,
        "status": "complete",
        "normalization_version": NORMALIZATION_VERSION,
        "coverage_sources": ["fineweb_original", "fineweb_extension"],
        "missing_coverage": [],
        "training_ready": True,
        "indexes": entries,
        "created_at": utc_now(),
    }
    atomic_write_json(root / "data/manifests/pretrain/fineweb_combined_coverage.json", payload)
    return payload
