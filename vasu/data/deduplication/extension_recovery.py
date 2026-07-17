"""Bounded, revision-pinned recovery helpers for the FineWeb extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from typing import Any, Callable, Iterable, Mapping
import urllib.error
import urllib.parse
import urllib.request


RECOVERY_CLASSIFICATIONS = {
    "exact_reconstruction_possible",
    "source_id_reacquisition_possible",
    "approximate_reconstruction_only",
    "unrecoverable",
}
MATCH_STATUSES = {
    "exact_match",
    "source_id_found_hash_mismatch",
    "source_id_not_found",
    "duplicate_source_id_in_provider",
    "malformed_source_record",
    "retrieval_error",
}
DATASET_SERVER_BASE_URL = "https://datasets-server.huggingface.co"
HUB_BASE_URL = "https://huggingface.co"


@dataclass(frozen=True)
class ExtensionRecoveryAudit:
    classification: str
    repository: str | None
    dataset_config: str | None
    revision: str | None
    split: str | None
    extension_fingerprints: int
    retained_source_ids: int
    missing_source_ids: int
    duplicate_source_ids: int = 0
    duplicate_historical_hashes: int = 0
    invalid_historical_hashes: int = 0
    source_id_field: str = "id"
    hash_contract: str = "sha256(text.strip().encode('utf-8')).hexdigest()"


@dataclass(frozen=True)
class HistoricalRecoveryRecord:
    source_id: str
    expected_hash: str


@dataclass(frozen=True)
class RetrievedSourceRecord:
    source_id: str
    text: str
    provider_shard: str | None
    stable_row_reference: str | None
    endpoint: str
    revision: str


@dataclass(frozen=True)
class RecoveryResult:
    source_id: str
    expected_hash: str
    observed_hash: str | None
    match_status: str
    provider_shard: str | None
    stable_row_reference: str | None
    raw_character_count: int | None
    raw_utf8_bytes: int | None
    retrieval_timestamp: str
    diagnostic_note: str


class RecoveryError(RuntimeError):
    """Base error for bounded extension recovery."""


class ScalabilityBlockedError(RecoveryError):
    """Raised when no bounded provider lookup is currently available."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def historical_extension_fingerprint(text: str) -> str:
    """Reproduce the recorded v1 extension exact-deduplication hash."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def historical_hash_matches(text: str, expected_sha256: str) -> bool:
    return historical_extension_fingerprint(text) == expected_sha256.casefold()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required recovery evidence is missing: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def audit_extension_recovery(metadata_path: Path, database_path: Path) -> ExtensionRecoveryAudit:
    metadata_path = Path(metadata_path)
    database_path = Path(database_path)
    metadata = _read_json(metadata_path)
    if not database_path.is_file():
        raise FileNotFoundError(f"Historical fingerprint database is missing: {database_path}")
    uri = f"file:{database_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(fingerprints)")}
        if not {"fingerprint", "origin", "source_id"}.issubset(columns):
            raise ValueError("historical extension database schema is incompatible")
        total, retained, distinct_ids, distinct_hashes, invalid_hashes = connection.execute(
            """
            SELECT COUNT(*), COUNT(source_id), COUNT(DISTINCT source_id),
                   COUNT(DISTINCT fingerprint),
                   SUM(CASE WHEN length(fingerprint) != 64
                                  OR fingerprint != lower(fingerprint)
                                  OR fingerprint GLOB '*[^0-9a-f]*'
                            THEN 1 ELSE 0 END)
              FROM fingerprints WHERE origin='extension'
            """
        ).fetchone()
    total = int(total)
    retained = int(retained)
    distinct_ids = int(distinct_ids)
    distinct_hashes = int(distinct_hashes)
    invalid_hashes = int(invalid_hashes or 0)
    missing = total - retained
    duplicate_ids = retained - distinct_ids
    duplicate_hashes = total - distinct_hashes
    evidence_complete = all(
        metadata.get(field)
        for field in ("dataset_repository", "dataset_config", "dataset_revision", "split")
    )
    if (
        evidence_complete
        and total > 0
        and retained == total
        and duplicate_ids == 0
        and invalid_hashes == 0
    ):
        classification = "source_id_reacquisition_possible"
    elif evidence_complete and total > 0:
        classification = "approximate_reconstruction_only"
    else:
        classification = "unrecoverable"
    return ExtensionRecoveryAudit(
        classification=classification,
        repository=metadata.get("dataset_repository"),
        dataset_config=metadata.get("dataset_config"),
        revision=metadata.get("dataset_revision"),
        split=metadata.get("split"),
        extension_fingerprints=total,
        retained_source_ids=retained,
        missing_source_ids=missing,
        duplicate_source_ids=duplicate_ids,
        duplicate_historical_hashes=duplicate_hashes,
        invalid_historical_hashes=invalid_hashes,
    )


def load_historical_records(database_path: Path) -> list[HistoricalRecoveryRecord]:
    """Load retained extension evidence in a deterministic source-ID order."""
    database_path = Path(database_path)
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    uri = f"file:{database_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT source_id, fingerprint FROM fingerprints "
            "WHERE origin='extension' ORDER BY source_id"
        ).fetchall()
    records = [HistoricalRecoveryRecord(str(source_id), str(fingerprint)) for source_id, fingerprint in rows]
    if any(not item.source_id for item in records):
        raise ValueError("Historical extension evidence contains an empty source ID")
    return records


def deterministic_spread_sample(
    records: Iterable[HistoricalRecoveryRecord], sample_size: int
) -> list[HistoricalRecoveryRecord]:
    """Select stable, evenly spread records without touching global RNG state."""
    ordered = sorted(records, key=lambda item: item.source_id)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if sample_size > len(ordered):
        raise ValueError("sample_size exceeds available historical source IDs")
    if sample_size == 1:
        return [ordered[len(ordered) // 2]]
    # Integer arithmetic keeps selection stable across Python versions.
    indices = [(position * (len(ordered) - 1)) // (sample_size - 1) for position in range(sample_size)]
    selected = [ordered[index] for index in indices]
    if len({item.source_id for item in selected}) != sample_size:
        raise ValueError("deterministic spread sampling produced duplicate source IDs")
    return selected


def configuration_hash(config: Mapping[str, Any], selected: Iterable[HistoricalRecoveryRecord]) -> str:
    payload = {
        "config": dict(config),
        "selected": [asdict(item) for item in selected],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _bounded_json_request(
    url: str,
    *,
    timeout_seconds: float,
    maximum_response_bytes: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, Any], Mapping[str, str], int]:
    request = urllib.request.Request(url, headers={"User-Agent": "VASU-bounded-extension-recovery/1.0"})
    try:
        with opener(request, timeout=timeout_seconds) as response:
            data = response.read(maximum_response_bytes + 1)
            if len(data) > maximum_response_bytes:
                raise RecoveryError(f"Provider response exceeded {maximum_response_bytes} bytes")
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise RecoveryError("Provider returned a non-object JSON response")
            return payload, response.headers, len(data)
    except urllib.error.HTTPError as error:
        detail = error.read(2_000).decode("utf-8", errors="replace")
        if error.code in {429, 500, 502, 503, 504} and "index is loading" in detail.casefold():
            raise ScalabilityBlockedError(
                "The official Dataset Viewer filter index is not available; refusing a full source scan"
            ) from error
        raise RecoveryError(f"Provider request failed with HTTP {error.code}: {detail[:500]}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryError(f"Provider request failed: {error}") from error


class DatasetServerSourceIdClient:
    """Exact-ID client for the official revision-reporting Dataset Viewer API."""

    def __init__(
        self,
        *,
        repository: str,
        dataset_config: str,
        split: str,
        revision: str,
        timeout_seconds: float = 120.0,
        maximum_response_bytes: int = 2_000_000,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.repository = repository
        self.dataset_config = dataset_config
        self.split = split
        self.revision = revision
        self.timeout_seconds = timeout_seconds
        self.maximum_response_bytes = maximum_response_bytes
        self.opener = opener
        self.accessed_endpoints: list[str] = []
        self.downloaded_bytes = 0

    def _request(self, endpoint: str, parameters: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, str]]:
        url = f"{DATASET_SERVER_BASE_URL}/{endpoint}?{urllib.parse.urlencode(parameters)}"
        self.accessed_endpoints.append(url)
        payload, headers, size = _bounded_json_request(
            url,
            timeout_seconds=self.timeout_seconds,
            maximum_response_bytes=self.maximum_response_bytes,
            opener=self.opener,
        )
        self.downloaded_bytes += size
        served_revision = headers.get("x-revision") or headers.get("X-Revision")
        if served_revision != self.revision:
            raise RecoveryError(
                f"Dataset Viewer revision mismatch: expected {self.revision}, got {served_revision!r}"
            )
        return payload, headers

    def verify_source(self) -> None:
        payload, _ = self._request(
            "rows",
            {
                "dataset": self.repository,
                "config": self.dataset_config,
                "split": self.split,
                "offset": 0,
                "length": 1,
            },
        )
        if not isinstance(payload.get("rows"), list):
            raise RecoveryError("Dataset Viewer rows response is malformed")

    def retrieve(self, source_id: str) -> list[RetrievedSourceRecord]:
        escaped = source_id.replace("'", "''")
        payload, _ = self._request(
            "filter",
            {
                "dataset": self.repository,
                "config": self.dataset_config,
                "split": self.split,
                "where": f'"id"=\'{escaped}\'',
                "offset": 0,
                # One returned document per selected ID keeps the entire smoke
                # under its 100-document cap. num_rows_total still exposes a
                # duplicate provider ID without downloading the second text.
                "length": 1,
            },
        )
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise RecoveryError("Dataset Viewer filter response is malformed")
        total_matches = payload.get("num_rows_total")
        if not isinstance(total_matches, int) or total_matches < len(rows):
            raise RecoveryError("Dataset Viewer filter response lacks a valid num_rows_total")
        records: list[RetrievedSourceRecord] = []
        for item in rows:
            if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
                raise RecoveryError("Dataset Viewer returned a malformed source record")
            row = item["row"]
            records.append(
                RetrievedSourceRecord(
                    source_id=row.get("id") if isinstance(row.get("id"), str) else "",
                    text=row.get("text") if isinstance(row.get("text"), str) else "",
                    provider_shard=row.get("file_path") if isinstance(row.get("file_path"), str) else None,
                    stable_row_reference=(
                        f"dataset-viewer-row:{item['row_idx']}"
                        if isinstance(item.get("row_idx"), int)
                        else None
                    ),
                    endpoint=self.accessed_endpoints[-1],
                    revision=self.revision,
                )
            )
        if total_matches > 1:
            # A metadata-only sentinel triggers duplicate classification while
            # retaining at most one provider document body.
            records.append(
                RetrievedSourceRecord(
                    source_id=source_id,
                    text="",
                    provider_shard=None,
                    stable_row_reference=f"dataset-viewer-match-count:{total_matches}",
                    endpoint=self.accessed_endpoints[-1],
                    revision=self.revision,
                )
            )
        return records


def classify_retrieval(
    historical: HistoricalRecoveryRecord,
    provider_records: list[RetrievedSourceRecord],
    *,
    timestamp: str | None = None,
) -> RecoveryResult:
    timestamp = timestamp or utc_now()
    if not provider_records:
        status = "source_id_not_found"
        observed = None
        note = "No provider row matched the exact historical source ID."
        record = None
    elif len(provider_records) > 1:
        status = "duplicate_source_id_in_provider"
        observed = None
        note = "The provider returned more than one row for the exact source ID."
        record = provider_records[0]
    else:
        record = provider_records[0]
        if record.source_id != historical.source_id or not record.text:
            status = "malformed_source_record"
            observed = None
            note = "Provider row lacks the exact ID or a non-empty text field."
        else:
            observed = historical_extension_fingerprint(record.text)
            status = "exact_match" if observed == historical.expected_hash.casefold() else "source_id_found_hash_mismatch"
            note = "Historical text.strip() SHA-256 matched." if status == "exact_match" else "Historical text.strip() SHA-256 did not match."
    text = record.text if record is not None and record.text else None
    return RecoveryResult(
        source_id=historical.source_id,
        expected_hash=historical.expected_hash,
        observed_hash=observed,
        match_status=status,
        provider_shard=record.provider_shard if record else None,
        stable_row_reference=record.stable_row_reference if record else None,
        raw_character_count=len(text) if text is not None else None,
        raw_utf8_bytes=len(text.encode("utf-8")) if text is not None else None,
        retrieval_timestamp=timestamp,
        diagnostic_note=note,
    )


def validate_report(report: Mapping[str, Any]) -> None:
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("Recovery report results must be a list")
    for item in results:
        if not isinstance(item, dict) or item.get("match_status") not in MATCH_STATUSES:
            raise ValueError("Recovery report contains an invalid match status")
        forbidden = {"text", "raw_text", "content", "document"}.intersection(item)
        if forbidden:
            raise ValueError(f"Recovery report leaks full text fields: {sorted(forbidden)}")
    counts = report.get("counts")
    if not isinstance(counts, dict) or sum(int(counts.get(status, 0)) for status in MATCH_STATUSES) != len(results):
        raise ValueError("Recovery report status counts do not match results")


def build_report(
    *,
    audit: ExtensionRecoveryAudit,
    selected: list[HistoricalRecoveryRecord],
    results: list[RecoveryResult],
    accessed_endpoints: list[str],
    downloaded_bytes: int,
    accepted_text_bytes: int,
    config_hash: str,
    started_at: str,
    completed_at: str,
    scalability_status: str,
) -> dict[str, Any]:
    counts = {status: 0 for status in sorted(MATCH_STATUSES)}
    for result in results:
        counts[result.match_status] += 1
    passed = (
        len(results) == len(selected) == 100
        and counts["exact_match"] == 100
        and scalability_status == "bounded_filter_available"
    )
    average_bytes = accepted_text_bytes / counts["exact_match"] if counts["exact_match"] else None
    estimate = None
    if average_bytes is not None:
        estimated_raw = round(average_bytes * audit.retained_source_ids)
        estimate = {
            "records": audit.retained_source_ids,
            "estimated_raw_text_bytes": estimated_raw,
            "estimated_temporary_storage_bytes": estimated_raw * 2,
            "estimated_extension_index_bytes": None,
            "recommended_resumable_batch_size": 1_000,
            "recommended_free_disk_bytes": max(10 * 1024**3, estimated_raw * 3),
            "provider_requests": audit.retained_source_ids,
            "runtime_note": "Measure request throughput during an authorized larger pilot; no runtime is invented from a blocked or tiny sample.",
        }
    report = {
        "format_version": 1,
        "classification": audit.classification,
        "repository": audit.repository,
        "dataset_config": audit.dataset_config,
        "split": audit.split,
        "revision": audit.revision,
        "configuration_hash": config_hash,
        "started_at": started_at,
        "completed_at": completed_at,
        "historical_evidence": asdict(audit),
        "sampled_source_ids": [item.source_id for item in selected],
        "results": [asdict(item) for item in results],
        "counts": counts,
        "downloaded_response_bytes": downloaded_bytes,
        "accepted_text_bytes": accepted_text_bytes,
        "accessed_endpoints": accessed_endpoints,
        "scalability_status": scalability_status,
        "full_reacquisition_authorized": passed,
        "production_estimate": estimate,
        "cross_source_index_compatibility": "vasu_cross_source_nfc_casefold_ws_v1 (separate from historical hash verification)",
    }
    validate_report(report)
    return report


def render_report_text(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "FineWeb Extension Recovery Smoke",
        "=" * 40,
        f"Source: {report['repository']} / {report['dataset_config']} / {report['split']}",
        f"Pinned revision: {report['revision']}",
        f"Smoke IDs selected: {len(report['sampled_source_ids'])}",
        f"Exact matches: {counts['exact_match']}",
        f"Hash mismatches: {counts['source_id_found_hash_mismatch']}",
        f"Missing IDs: {counts['source_id_not_found']}",
        f"Duplicate provider IDs: {counts['duplicate_source_id_in_provider']}",
        f"Malformed records: {counts['malformed_source_record']}",
        f"Retrieval errors: {counts['retrieval_error']}",
        f"Downloaded response bytes: {report['downloaded_response_bytes']}",
        f"Accepted text bytes: {report['accepted_text_bytes']}",
        f"Scalability: {report['scalability_status']}",
        f"Full reacquisition authorized: {report['full_reacquisition_authorized']}",
        "",
        "No source text is stored in this report.",
    ]
    return "\n".join(lines) + "\n"


def random_state_unchanged_by_sampling(records: list[HistoricalRecoveryRecord], sample_size: int) -> bool:
    """Test helper documenting that sampling is arithmetic, not RNG based."""
    before = random.getstate()
    deterministic_spread_sample(records, sample_size)
    return before == random.getstate()
