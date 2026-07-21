"""Focused production recovery tests without provider access or full data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from vasu.data.deduplication import extension_production as production
from vasu.data.deduplication.extension_recovery import (
    HistoricalRecoveryRecord,
    RetrievedSourceRecord,
    historical_extension_fingerprint,
)
from vasu.data.deduplication.fineweb_index import build_fineweb_document_index
from vasu.data.deduplication.schemas import FineWebIndexConfig, FineWebSource, INDEX_FORMAT_VERSION


REVISION = production.EXPECTED_REVISION


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_evidence(root: Path, count: int = 50) -> tuple[Path, Path, Path]:
    binary = root / "extension.bin"
    binary.write_bytes(b"tokens")
    database = root / "extension.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE fingerprints(fingerprint TEXT PRIMARY KEY, origin TEXT, source_id TEXT)"
        )
        connection.executemany(
            "INSERT INTO fingerprints VALUES(?,?,?)",
            [
                (historical_extension_fingerprint(f"text-{index}"), "extension", f"id-{index:04d}")
                for index in range(count)
            ],
        )
    metadata = root / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "dataset_repository": production.EXPECTED_REPOSITORY,
                "dataset_config": production.EXPECTED_CONFIG,
                "split": production.EXPECTED_SPLIT,
                "dataset_revision": REVISION,
            }
        ),
        encoding="utf-8",
    )
    return binary, database, metadata


def config_for(root: Path, count: int = 50) -> dict[str, object]:
    binary, database, metadata = make_evidence(root, count)
    return {
        "format_version": 1,
        "dataset_repository": production.EXPECTED_REPOSITORY,
        "dataset_config": production.EXPECTED_CONFIG,
        "split": production.EXPECTED_SPLIT,
        "dataset_revision": REVISION,
        "source_id_field": "id",
        "text_field": "text",
        "metadata_path": metadata.name,
        "historical_binary_path": binary.name,
        "historical_binary_sha256": sha(binary),
        "historical_database_path": database.name,
        "historical_database_sha256": sha(database),
        "artifact_path": "ignored/recovered.jsonl.gz",
        "progress_path": "ignored/progress.json",
        "json_report_path": "report.json",
        "text_report_path": "report.txt",
        "batch_size": 25,
        "fallback_batch_sizes": [10, 5, 1],
        "checkpoint_interval_documents": 1000,
        "concurrency": 1,
        "requests_per_second": 2.0,
        "connect_timeout_seconds": 10.0,
        "read_timeout_seconds": 120.0,
        "retry_count": 4,
        "initial_backoff_seconds": 1.0,
        "maximum_backoff_seconds": 60.0,
        "jitter_seconds": 0.25,
        "retry_seed": 42,
        "maximum_response_bytes": 20_000_000,
        "minimum_free_disk_bytes": 10 * production.GIB,
        "historical_fingerprint": "sha256_utf8_python_str_strip_v1",
        "acquisition_method": "huggingface_dataset_viewer_composite_or_filter",
        "allow_revision_fallback": False,
    }


class FakeHttp:
    network_request_count = 0
    retry_count = 0
    transient_error_count = 0
    timeout_count = 0
    downloaded_bytes = 0
    http_status_counts: dict[str, int] = {}


class FakeClient:
    def __init__(self, fail_sizes: set[int] | None = None) -> None:
        self.fail_sizes = fail_sizes or set()
        self.calls: list[tuple[str, ...]] = []

    def retrieve_many(self, source_ids: list[str]) -> dict[str, list[RetrievedSourceRecord]]:
        self.calls.append(tuple(source_ids))
        if len(source_ids) in self.fail_sizes:
            raise RuntimeError(f"unsupported batch size {len(source_ids)}")
        return {
            source_id: [
                RetrievedSourceRecord(
                    source_id=source_id,
                    text=f"text-{int(source_id.split('-')[-1])}",
                    provider_shard="shard.parquet",
                    stable_row_reference=f"row:{source_id}",
                    endpoint="test",
                    revision=REVISION,
                )
            ]
            for source_id in source_ids
        }


def test_production_config_contract_loads() -> None:
    path = Path("configs/data/deduplication/fineweb_extension_recovery_production.json")
    assert production.load_production_config(path)["batch_size"] == 25


def test_preflight_checks_hash_counts_and_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = config_for(tmp_path)
    monkeypatch.setattr(production, "EXPECTED_RECORD_COUNT", 50)
    monkeypatch.setattr(production, "_is_git_ignored", lambda root, path: True)
    monkeypatch.setattr(
        production.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100, used=0, free=20 * production.GIB),
    )
    report = production.preflight_production(
        config,
        repository_root=tmp_path,
        provider_probe=lambda _: {"observed_revision": REVISION},
    )
    assert report["status"] == "passed"
    assert report["historical_source_ids"] == 50


def test_preflight_rejects_hash_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = config_for(tmp_path)
    config["historical_binary_sha256"] = "0" * 64
    monkeypatch.setattr(
        production.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100, used=0, free=20 * production.GIB),
    )
    with pytest.raises(production.RecoveryError, match="binary SHA-256"):
        production.preflight_production(config, repository_root=tmp_path)


def test_deterministic_fallback_25_to_10() -> None:
    historical = [
        HistoricalRecoveryRecord(f"id-{index:04d}", historical_extension_fingerprint(f"text-{index}"))
        for index in range(25)
    ]
    client = FakeClient({25})
    events: list[dict[str, object]] = []
    results = production.retrieve_with_fallback(
        historical, client, batch_identifier="batch-1", fallback_events=events  # type: ignore[arg-type]
    )
    assert len(results) == 25
    assert [len(call) for call in client.calls] == [25, 10, 10, 5]
    assert events[0]["fallback_size"] == 10


def test_atomic_gzip_members_and_resume_do_not_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(tmp_path)
    monkeypatch.setattr(production, "EXPECTED_RECORD_COUNT", 50)
    client = FakeClient()
    http = FakeHttp()
    first = production.run_production_recovery(
        config,
        repository_root=tmp_path,
        client=client,  # type: ignore[arg-type]
        http=http,  # type: ignore[arg-type]
        stop_after_documents=25,
    )
    assert len(first["completed_source_ids"]) == 25
    artifact = tmp_path / str(config["artifact_path"])
    with artifact.open("ab") as handle:
        handle.write(b"uncommitted")
    second_client = FakeClient()
    second = production.run_production_recovery(
        config,
        repository_root=tmp_path,
        client=second_client,  # type: ignore[arg-type]
        http=FakeHttp(),  # type: ignore[arg-type]
    )
    assert len(second["completed_source_ids"]) == 50
    assert all(not any(source_id in call for call in second_client.calls) for source_id in first["completed_source_ids"])
    records = list(production.iter_recovered_records(artifact))
    assert len(records) == len({item["historical_source_id"] for item in records}) == 50


def test_full_validation_recomputes_every_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(tmp_path)
    monkeypatch.setattr(production, "EXPECTED_RECORD_COUNT", 50)
    production.run_production_recovery(
        config,
        repository_root=tmp_path,
        client=FakeClient(),  # type: ignore[arg-type]
        http=FakeHttp(),  # type: ignore[arg-type]
    )
    report = production.validate_recovered_artifact(config, repository_root=tmp_path)
    assert report["status"] == "complete"
    assert report["accepted_records"] == 50
    assert "text-0" not in json.dumps(report)


def test_gzip_source_uses_existing_index_builder(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl.gz"
    source.write_bytes(
        production._encode_gzip_member(
            [
                {"id": "a", "text": "one two three four five six"},
                {"id": "b", "text": "seven eight nine ten eleven twelve"},
            ]
        )
    )
    config = FineWebIndexConfig(
        format_version=INDEX_FORMAT_VERSION,
        output_path="index.sqlite3",
        metadata_path="metadata.json",
        progress_path="progress.json",
        normalization_version=production.NORMALIZATION_VERSION,
        shingle_size=5,
        signature_size=64,
        bands=8,
        batch_size=1,
        maximum_documents=None,
        expected_document_count=2,
        coverage=("fineweb_extension",),
        missing_coverage=("fineweb_original",),
        sources=(
            FineWebSource(
                source_id="extension",
                source_revision=REVISION,
                source_shard="test",
                path=source.name,
                format="jsonl_gzip",
                document_id_field="id",
                provenance_completeness="complete",
                expected_sha256=sha(source),
            ),
        ),
    )
    metadata = build_fineweb_document_index(config, repository_root=tmp_path)
    assert metadata["indexed_documents"] == 2
    assert metadata["records_with_complete_ids"] == 2


def test_report_renderer_never_includes_source_text() -> None:
    report = {
        "status": "complete",
        "revision": REVISION,
        "historical_source_ids": 1,
        "counts": {status: int(status == "exact_match") for status in production.MATCH_STATUSES},
        "provider_statistics": {
            "network_requests": 1,
            "retry_count": 0,
            "fallback_events": [],
            "downloaded_response_bytes": 100,
        },
        "accepted_text_bytes": 50,
        "artifact_size_bytes": 20,
        "artifact_sha256": "a" * 64,
    }
    assert "source text" in production.render_production_report(report)
    assert "text-0" not in production.render_production_report(report)
