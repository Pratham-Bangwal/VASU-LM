"""Run a bounded, source-ID-based FineWeb extension recovery smoke test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from vasu.data.deduplication.extension_recovery import (
    DatasetServerSourceIdClient,
    RecoveryError,
    RecoveryResult,
    ScalabilityBlockedError,
    atomic_write_json,
    atomic_write_text,
    audit_extension_recovery,
    build_report,
    classify_retrieval,
    configuration_hash,
    deterministic_spread_sample,
    load_historical_records,
    render_report_text,
    utc_now,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/data/deduplication/fineweb_extension_recovery.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Audit local evidence and probe bounded provider lookup.")
    mode.add_argument("--smoke-test", action="store_true", help="Retrieve exactly the configured deterministic sample.")
    parser.add_argument("--restart", action="store_true", help="Discard compatible progress and query the sample again.")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Recovery config is missing: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid recovery config JSON: {error}") from error
    required = {
        "format_version", "dataset_repository", "dataset_config", "split",
        "dataset_revision", "source_id_field", "text_field", "metadata_path",
        "historical_database_path", "progress_path", "json_report_path",
        "text_report_path", "sample_size", "maximum_matches",
        "maximum_accepted_text_bytes", "maximum_response_bytes",
        "request_timeout_seconds", "acquisition_method",
        "allow_parquet_scan_fallback", "historical_fingerprint",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(f"Recovery config fields mismatch; missing={missing}, extra={extra}")
    if payload["format_version"] != 1:
        raise ValueError("Unsupported recovery config format_version")
    if payload["sample_size"] != 100 or payload["maximum_matches"] != 100:
        raise ValueError("Smoke recovery must select and retrieve at most exactly 100 IDs")
    if payload["maximum_accepted_text_bytes"] > 20_000_000:
        raise ValueError("Smoke accepted-text limit may not exceed 20 MB")
    if payload["allow_parquet_scan_fallback"] is not False:
        raise ValueError("Parquet/full-source scan fallback must remain disabled")
    if payload["source_id_field"] != "id" or payload["text_field"] != "text":
        raise ValueError("Historical provider fields must remain id and text")
    if payload["historical_fingerprint"] != "sha256_utf8_python_str_strip_v1":
        raise ValueError("Historical fingerprint contract mismatch")
    return payload


def repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def validate_config_against_audit(config: dict[str, Any], audit: Any) -> None:
    expected = (
        config["dataset_repository"], config["dataset_config"],
        config["dataset_revision"], config["split"],
    )
    observed = (audit.repository, audit.dataset_config, audit.revision, audit.split)
    if expected != observed:
        raise ValueError(f"Recovery configuration conflicts with historical evidence: {expected!r} != {observed!r}")
    if audit.classification != "source_id_reacquisition_possible":
        raise ValueError(f"Historical evidence is not safe for source-ID recovery: {audit.classification}")
    if audit.duplicate_source_ids or audit.invalid_historical_hashes or audit.missing_source_ids:
        raise ValueError("Historical evidence contains missing, duplicate, or invalid IDs/hashes")


def load_progress(path: Path, *, expected_hash: str, revision: str, restart: bool) -> dict[str, Any]:
    if restart or not path.exists():
        return {
            "format_version": 1,
            "configuration_hash": expected_hash,
            "revision": revision,
            "completed": {},
            "accepted_text_bytes": 0,
            "downloaded_response_bytes": 0,
            "accessed_endpoints": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("configuration_hash") != expected_hash:
        raise ValueError("Progress configuration hash mismatch; use --restart only after review")
    if payload.get("revision") != revision:
        raise ValueError("Progress revision mismatch; refusing revision fallback")
    if not isinstance(payload.get("completed"), dict):
        raise ValueError("Progress completed map is malformed")
    return payload


def pending_historical_records(selected: list[Any], completed: dict[str, Any]) -> list[Any]:
    """Return only unfinished IDs so resume never re-queries completed work."""
    return [item for item in selected if item.source_id not in completed]


def print_audit(config: dict[str, Any], audit: Any, selected: list[Any]) -> None:
    print(f"Historical source-ID path: {repository_path(config['historical_database_path'])}::fingerprints.source_id")
    print(f"Historical fingerprint path: {repository_path(config['historical_database_path'])}::fingerprints.fingerprint")
    print(f"Historical source IDs: {audit.retained_source_ids:,}")
    print(f"Historical fingerprints: {audit.extension_fingerprints:,}")
    print(f"Duplicate source IDs: {audit.duplicate_source_ids}")
    print(f"Duplicate historical hashes: {audit.duplicate_historical_hashes}")
    print(f"Expected source-ID field: {audit.source_id_field}")
    print(f"Historical hash contract: {audit.hash_contract}")
    print(f"Smoke IDs selected: {len(selected)}")
    print("Acquisition mechanism: official Dataset Viewer exact-ID /filter; no Parquet scan fallback")


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    metadata_path = repository_path(config["metadata_path"])
    database_path = repository_path(config["historical_database_path"])
    audit = audit_extension_recovery(metadata_path, database_path)
    validate_config_against_audit(config, audit)
    records = load_historical_records(database_path)
    selected = deterministic_spread_sample(records, config["sample_size"])
    config_digest = configuration_hash(config, selected)
    print_audit(config, audit, selected)

    client = DatasetServerSourceIdClient(
        repository=config["dataset_repository"],
        dataset_config=config["dataset_config"],
        split=config["split"],
        revision=config["dataset_revision"],
        timeout_seconds=float(config["request_timeout_seconds"]),
        maximum_response_bytes=int(config["maximum_response_bytes"]),
    )
    client.verify_source()
    print(f"Pinned revision verified: {config['dataset_revision']}")
    if args.dry_run:
        try:
            # Probe one selected ID. A loading/unavailable index blocks the run
            # instead of falling back to a 44+ GB Parquet scan.
            probe = client.retrieve(selected[0].source_id)
        except ScalabilityBlockedError as error:
            print(f"Scalability status: blocked - {error}")
            return 3
        print("Scalability status: bounded filter available")
        print(f"Probe rows returned: {len(probe)}")
        print("Dry run complete; no recovery report or training artifact was created.")
        return 0

    progress_path = repository_path(config["progress_path"])
    progress = load_progress(
        progress_path,
        expected_hash=config_digest,
        revision=config["dataset_revision"],
        restart=args.restart,
    )
    completed: dict[str, dict[str, Any]] = progress["completed"]
    pending = pending_historical_records(selected, completed)
    probe: list[Any] | None = None
    if pending:
        try:
            # Probe only the first unfinished ID so resume never repeats a
            # completed source-ID request.
            probe = client.retrieve(pending[0].source_id)
        except ScalabilityBlockedError as error:
            print(f"Scalability status: blocked - {error}")
            return 3
    print("Scalability status: bounded filter available")
    started_at = progress.get("started_at") or utc_now()
    progress["started_at"] = started_at
    accepted_bytes = int(progress.get("accepted_text_bytes", 0))
    prior_downloaded_bytes = int(progress.get("downloaded_response_bytes", 0))
    prior_endpoints = list(progress.get("accessed_endpoints", []))
    # The capability probe is the actual first unfinished retrieval; reuse it.
    prefetched = {pending[0].source_id: probe} if pending and probe is not None else {}
    for historical in pending:
        try:
            provider_records = prefetched.pop(historical.source_id, None)
            if provider_records is None:
                provider_records = client.retrieve(historical.source_id)
            result = classify_retrieval(historical, provider_records)
        except ScalabilityBlockedError:
            raise
        except Exception as error:  # a per-ID provider failure is reportable and resumable
            result = RecoveryResult(
                source_id=historical.source_id,
                expected_hash=historical.expected_hash,
                observed_hash=None,
                match_status="retrieval_error",
                provider_shard=None,
                stable_row_reference=None,
                raw_character_count=None,
                raw_utf8_bytes=None,
                retrieval_timestamp=utc_now(),
                diagnostic_note=f"{type(error).__name__}: {str(error)[:300]}",
            )
        record_bytes = int(result.raw_utf8_bytes or 0)
        if accepted_bytes + record_bytes > int(config["maximum_accepted_text_bytes"]):
            raise RecoveryError("Accepted text would exceed the configured 20 MB smoke limit")
        accepted_bytes += record_bytes
        completed[historical.source_id] = asdict(result)
        progress.update(
            {
                "accepted_text_bytes": accepted_bytes,
                "downloaded_response_bytes": prior_downloaded_bytes + client.downloaded_bytes,
                "accessed_endpoints": prior_endpoints + client.accessed_endpoints,
                "updated_at": utc_now(),
            }
        )
        atomic_write_json(progress_path, progress)

    ordered_results = [RecoveryResult(**completed[item.source_id]) for item in selected]
    completed_at = utc_now()
    report = build_report(
        audit=audit,
        selected=selected,
        results=ordered_results,
        accessed_endpoints=list(progress["accessed_endpoints"]),
        downloaded_bytes=int(progress["downloaded_response_bytes"]),
        accepted_text_bytes=accepted_bytes,
        config_hash=config_digest,
        started_at=started_at,
        completed_at=completed_at,
        scalability_status="bounded_filter_available",
    )
    json_report = repository_path(config["json_report_path"])
    text_report = repository_path(config["text_report_path"])
    atomic_write_json(json_report, report)
    atomic_write_text(text_report, render_report_text(report))
    print(render_report_text(report), end="")
    print(f"JSON report: {json_report}")
    print(f"Text report: {text_report}")
    return 0 if report["full_reacquisition_authorized"] else 2


def main() -> None:
    try:
        raise SystemExit(run())
    except (RecoveryError, ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
