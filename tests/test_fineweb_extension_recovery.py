from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import random
import sqlite3
from typing import Any
import urllib.error
import urllib.parse

import pytest

from scripts.reacquire_fineweb_extension import (
    load_config,
    load_progress,
    pending_historical_records,
    repository_path,
    validate_config_against_audit,
)
from vasu.data.deduplication.extension_recovery import (
    DatasetServerSourceIdClient,
    ExtensionRecoveryAudit,
    HistoricalRecoveryRecord,
    RecoveryError,
    RetrievedSourceRecord,
    ScalabilityBlockedError,
    atomic_write_json,
    audit_extension_recovery,
    build_report,
    classify_retrieval,
    configuration_hash,
    deterministic_spread_sample,
    historical_extension_fingerprint,
    load_historical_records,
    random_state_unchanged_by_sampling,
    validate_report,
)


REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def make_evidence(
    tmp_path: Path,
    rows: list[tuple[str, str]] | None = None,
    *,
    primary_key: bool = True,
) -> tuple[Path, Path]:
    rows = rows or [("id-1", historical_extension_fingerprint(" alpha "))]
    metadata = tmp_path / "metadata.json"
    database = tmp_path / "history.sqlite3"
    metadata.write_text(
        json.dumps(
            {
                "dataset_repository": "HuggingFaceFW/fineweb-edu",
                "dataset_config": "CC-MAIN-2025-26",
                "dataset_revision": REVISION,
                "split": "train",
            }
        ),
        encoding="utf-8",
    )
    key = " PRIMARY KEY" if primary_key else ""
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"CREATE TABLE fingerprints(fingerprint TEXT{key}, origin TEXT NOT NULL, source_id TEXT)"
        )
        connection.executemany(
            "INSERT INTO fingerprints VALUES (?, 'extension', ?)",
            [(fingerprint, source_id) for source_id, fingerprint in rows],
        )
    return metadata, database


def retrieved(source_id: str, text: str) -> RetrievedSourceRecord:
    return RetrievedSourceRecord(source_id, text, "shard", "row:1", "endpoint", REVISION)


class FakeResponse:
    def __init__(self, payload: dict[str, Any], revision: str = REVISION) -> None:
        self.data = json.dumps(payload).encode()
        self.headers = {"x-revision": revision}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, _: int) -> bytes:
        return self.data


def test_historical_evidence_count_validation(tmp_path: Path) -> None:
    metadata, database = make_evidence(tmp_path, [("a", "a" * 64), ("b", "b" * 64)])
    audit = audit_extension_recovery(metadata, database)
    assert audit.extension_fingerprints == audit.retained_source_ids == 2
    assert audit.classification == "source_id_reacquisition_possible"


def test_duplicate_source_id_rejection(tmp_path: Path) -> None:
    metadata, database = make_evidence(
        tmp_path, [("same", "a" * 64), ("same", "b" * 64)], primary_key=False
    )
    audit = audit_extension_recovery(metadata, database)
    assert audit.duplicate_source_ids == 1
    assert audit.classification == "approximate_reconstruction_only"


def test_duplicate_historical_fingerprint_accounting(tmp_path: Path) -> None:
    metadata, database = make_evidence(
        tmp_path, [("a", "c" * 64), ("b", "c" * 64)], primary_key=False
    )
    assert audit_extension_recovery(metadata, database).duplicate_historical_hashes == 1


def test_deterministic_spread_sampling() -> None:
    records = [HistoricalRecoveryRecord(f"id-{i:03}", f"{i:064x}") for i in range(201)]
    first = deterministic_spread_sample(records, 100)
    assert first == deterministic_spread_sample(reversed(records), 100)
    assert first[0] == records[0] and first[-1] == records[-1]
    assert len({item.source_id for item in first}) == 100


def test_sampling_does_not_mutate_global_random_state() -> None:
    records = [HistoricalRecoveryRecord(str(i), "a" * 64) for i in range(10)]
    random.seed(9182)
    assert random_state_unchanged_by_sampling(records, 5)


def test_exact_strip_hash_reproduction() -> None:
    assert historical_extension_fingerprint(" \talpha\r\n") == historical_extension_fingerprint("alpha")


def test_hash_preserves_raw_unicode() -> None:
    assert historical_extension_fingerprint("é") != historical_extension_fingerprint("e\u0301")


def test_exact_match_classification() -> None:
    expected = HistoricalRecoveryRecord("id", historical_extension_fingerprint(" text "))
    result = classify_retrieval(expected, [retrieved("id", " text ")], timestamp="now")
    assert result.match_status == "exact_match"
    assert result.raw_character_count == 6


def test_hash_mismatch_classification() -> None:
    expected = HistoricalRecoveryRecord("id", "0" * 64)
    assert classify_retrieval(expected, [retrieved("id", "text")]).match_status == "source_id_found_hash_mismatch"


def test_missing_id_classification() -> None:
    expected = HistoricalRecoveryRecord("id", "0" * 64)
    assert classify_retrieval(expected, []).match_status == "source_id_not_found"


def test_duplicate_provider_id_classification() -> None:
    expected = HistoricalRecoveryRecord("id", "0" * 64)
    assert classify_retrieval(expected, [retrieved("id", "a"), retrieved("id", "b")]).match_status == "duplicate_source_id_in_provider"


def test_malformed_record_classification() -> None:
    expected = HistoricalRecoveryRecord("id", "0" * 64)
    assert classify_retrieval(expected, [retrieved("", "")]).match_status == "malformed_source_record"


def test_pinned_revision_enforcement() -> None:
    def opener(*_: object, **__: object) -> FakeResponse:
        return FakeResponse({"rows": []}, revision="wrong")

    client = DatasetServerSourceIdClient(
        repository="r", dataset_config="c", split="train", revision=REVISION, opener=opener
    )
    with pytest.raises(RecoveryError, match="revision mismatch"):
        client.verify_source()


def test_configuration_mismatch_rejection() -> None:
    audit = ExtensionRecoveryAudit(
        "source_id_reacquisition_possible", "other", "c", REVISION, "train", 1, 1, 0
    )
    config = {
        "dataset_repository": "expected", "dataset_config": "c",
        "dataset_revision": REVISION, "split": "train",
    }
    with pytest.raises(ValueError, match="conflicts"):
        validate_config_against_audit(config, audit)


def test_resume_skips_completed_ids(tmp_path: Path) -> None:
    selected = [HistoricalRecoveryRecord("a", "a" * 64), HistoricalRecoveryRecord("b", "b" * 64)]
    progress_path = tmp_path / "progress.json"
    atomic_write_json(
        progress_path,
        {"configuration_hash": "hash", "revision": REVISION, "completed": {"a": {}}},
    )
    progress = load_progress(progress_path, expected_hash="hash", revision=REVISION, restart=False)
    assert [item.source_id for item in pending_historical_records(selected, progress["completed"])] == ["b"]


def test_atomic_progress_write(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "progress.json"
    atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}
    assert not target.with_name("progress.json.tmp").exists()


def test_atomic_progress_retries_transient_windows_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "progress.json"
    original = Path.replace
    attempts = [0]

    def replace(path: Path, destination: Path) -> Path:
        attempts[0] += 1
        if attempts[0] < 3:
            raise PermissionError("scanner lock")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", replace)
    atomic_write_json(target, {"safe": True})
    assert attempts[0] == 3
    assert json.loads(target.read_text()) == {"safe": True}


def test_report_validation_and_no_full_text_leak(tmp_path: Path) -> None:
    audit = ExtensionRecoveryAudit(
        "source_id_reacquisition_possible", "r", "c", REVISION, "train", 1, 1, 0
    )
    selected = [HistoricalRecoveryRecord("id", historical_extension_fingerprint("hello"))]
    result = classify_retrieval(selected[0], [retrieved("id", "hello")])
    report = build_report(
        audit=audit, selected=selected, results=[result], accessed_endpoints=[],
        downloaded_bytes=20, accepted_text_bytes=5, config_hash="x",
        started_at="a", completed_at="b", scalability_status="bounded_filter_available",
    )
    validate_report(report)
    serialized = json.dumps(report)
    assert '"text"' not in serialized and "hello" not in serialized


def test_report_rejects_full_text_field() -> None:
    report = {
        "results": [{"match_status": "exact_match", "text": "secret"}],
        "counts": {status: int(status == "exact_match") for status in (
            "exact_match", "source_id_found_hash_mismatch", "source_id_not_found",
            "duplicate_source_id_in_provider", "malformed_source_record", "retrieval_error",
        )},
    }
    with pytest.raises(ValueError, match="leaks"):
        validate_report(report)


def test_scalability_blocker_handling() -> None:
    body = BytesIO(b'{"error":"the dataset index is loading, this can take a minute"}')
    error = urllib.error.HTTPError("url", 500, "error", {}, body)

    def opener(*_: object, **__: object) -> Any:
        raise error

    client = DatasetServerSourceIdClient(
        repository="r", dataset_config="c", split="train", revision=REVISION, opener=opener
    )
    with pytest.raises(ScalabilityBlockedError):
        client.retrieve("id")


def test_dataset_server_exact_id_filter_and_row_reference() -> None:
    seen_urls: list[str] = []

    def opener(request: Any, **_: object) -> FakeResponse:
        seen_urls.append(request.full_url)
        return FakeResponse(
            {"num_rows_total": 1, "rows": [{"row_idx": 42, "row": {"id": "source'id", "text": "raw", "file_path": "warc"}}]}
        )

    client = DatasetServerSourceIdClient(
        repository="r", dataset_config="c", split="train", revision=REVISION, opener=opener
    )
    rows = client.retrieve("source'id")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(seen_urls[0]).query)
    assert query["where"] == ['"id"=\'source\'\'id\'']
    assert query["length"] == ["1"]
    assert rows[0].stable_row_reference == "dataset-viewer-row:42"


def test_duplicate_provider_count_does_not_download_second_document() -> None:
    def opener(*_: object, **__: object) -> FakeResponse:
        return FakeResponse(
            {"num_rows_total": 2, "rows": [{"row_idx": 1, "row": {"id": "id", "text": "one"}}]}
        )

    client = DatasetServerSourceIdClient(
        repository="r", dataset_config="c", split="train", revision=REVISION, opener=opener
    )
    rows = client.retrieve("id")
    assert len(rows) == 2
    assert rows[1].text == ""
    expected = HistoricalRecoveryRecord("id", historical_extension_fingerprint("one"))
    assert classify_retrieval(expected, rows).match_status == "duplicate_source_id_in_provider"


def test_no_current_working_directory_assumption(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert repository_path("configs/data/deduplication/fineweb_extension_recovery.json").is_file()


def test_config_hash_protects_selected_ids() -> None:
    one = [HistoricalRecoveryRecord("a", "a" * 64)]
    two = [HistoricalRecoveryRecord("b", "b" * 64)]
    assert configuration_hash({"revision": REVISION}, one) != configuration_hash({"revision": REVISION}, two)


def test_repository_config_has_hard_smoke_limits() -> None:
    config = load_config(repository_path("configs/data/deduplication/fineweb_extension_recovery.json"))
    assert config["sample_size"] == config["maximum_matches"] == 100
    assert config["maximum_accepted_text_bytes"] == 20_000_000
    assert config["allow_parquet_scan_fallback"] is False


def test_load_historical_records_is_sorted(tmp_path: Path) -> None:
    _, database = make_evidence(tmp_path, [("z", "a" * 64), ("a", "b" * 64)])
    assert [item.source_id for item in load_historical_records(database)] == ["a", "z"]
