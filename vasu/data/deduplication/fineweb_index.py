"""Resumable SQLite FineWeb document index builder and reader."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import gzip
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Iterator

from .normalization import (
    NORMALIZATION_VERSION,
    bucket_keys,
    minhash_signature,
    normalize_for_matching,
)
from .schemas import FineWebIndexConfig, FineWebSource, INDEX_FORMAT_VERSION, IndexRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_index_config(path: Path) -> FineWebIndexConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"FineWeb index config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"FineWeb index config is invalid JSON: {path}") from error
    sources = tuple(FineWebSource(**item) for item in payload.pop("sources"))
    payload["coverage"] = tuple(payload.get("coverage", ("fineweb_original", "fineweb_extension")))
    payload["missing_coverage"] = tuple(payload.get("missing_coverage", ()))
    config = FineWebIndexConfig(sources=sources, **payload)
    config.validate()
    return config


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents (
          document_id TEXT PRIMARY KEY,
          source_id TEXT NOT NULL,
          source_revision TEXT NOT NULL,
          source_shard TEXT NOT NULL,
          source_url TEXT,
          normalized_sha256 TEXT NOT NULL UNIQUE,
          signature TEXT NOT NULL,
          normalized_character_count INTEGER NOT NULL,
          normalization_version TEXT NOT NULL,
          provenance_completeness TEXT NOT NULL,
          source_content_reference TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS buckets (
          bucket_key TEXT NOT NULL,
          document_id TEXT NOT NULL,
          PRIMARY KEY (bucket_key, document_id),
          FOREIGN KEY (document_id) REFERENCES documents(document_id)
        );
        CREATE INDEX IF NOT EXISTS idx_buckets_key ON buckets(bucket_key);
        CREATE TABLE IF NOT EXISTS source_progress (
          source_id TEXT PRIMARY KEY,
          source_sha256 TEXT NOT NULL,
          next_line INTEGER NOT NULL,
          processed INTEGER NOT NULL,
          rejected INTEGER NOT NULL,
          duplicate_hashes INTEGER NOT NULL,
          complete INTEGER NOT NULL
        );
        """
    )


def _signature_text(signature: tuple[int, ...]) -> str:
    return ",".join(f"{value:016x}" for value in signature)


def _parse_signature(value: str) -> tuple[int, ...]:
    return tuple(int(item, 16) for item in value.split(","))


def _iter_jsonl(
    source_path: Path,
    start_line: int,
    *,
    compressed: bool = False,
) -> Iterator[tuple[int, dict[str, Any] | None]]:
    opener = gzip.open if compressed else Path.open
    with opener(source_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line_number < start_line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None
                continue
            yield line_number, value if isinstance(value, dict) else None


def build_fineweb_document_index(
    config: FineWebIndexConfig,
    *,
    repository_root: Path,
    resume: bool = False,
    stop_after_documents: int | None = None,
    progress_callback: Callable[[dict[str, int | str]], None] | None = None,
) -> dict[str, Any]:
    """Build a bounded index into a temporary DB and atomically promote it."""
    started = time.monotonic()
    config.validate()
    root = repository_root.resolve()
    output = root / config.output_path
    partial = output.with_name(output.name + ".partial")
    metadata_path = root / config.metadata_path
    progress_path = root / config.progress_path
    config_hash = _canonical_hash(asdict(config))
    if output.exists() and not resume:
        raise FileExistsError(f"completed FineWeb index already exists: {output}")
    if resume and not partial.exists():
        raise FileNotFoundError(f"no partial FineWeb index to resume: {partial}")
    if not resume:
        partial.parent.mkdir(parents=True, exist_ok=True)
        if partial.exists():
            raise FileExistsError(f"partial FineWeb index exists; use --resume: {partial}")

    source_info: list[tuple[FineWebSource, Path, str]] = []
    for source in config.sources:
        path = root / source.path
        if not path.is_file():
            raise FileNotFoundError(f"FineWeb document source missing: {path}")
        source_hash = sha256_file(path)
        if source.expected_sha256 and source_hash != source.expected_sha256:
            raise ValueError(f"source-file hash mismatch for {source.source_id}")
        source_info.append((source, path, source_hash))

    connection = sqlite3.connect(partial)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    _create_schema(connection)
    stored_hash = connection.execute(
        "SELECT value FROM metadata WHERE key='configuration_hash'"
    ).fetchone()
    if stored_hash and stored_hash[0] != config_hash:
        connection.close()
        raise ValueError("FineWeb partial index configuration mismatch")
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key,value) VALUES('configuration_hash',?)",
        (config_hash,),
    )
    connection.execute(
        "INSERT OR IGNORE INTO metadata(key,value) VALUES('normalization_version',?)",
        (config.normalization_version,),
    )
    connection.commit()

    total_processed = 0
    total_rejected = 0
    total_duplicates = 0
    stopped_by_bound = False
    indexed_now = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    for source, path, source_hash in source_info:
        row = connection.execute(
            "SELECT source_sha256,next_line,processed,rejected,duplicate_hashes,complete "
            "FROM source_progress WHERE source_id=?",
            (source.source_id,),
        ).fetchone()
        if row and row[0] != source_hash:
            connection.close()
            raise ValueError(f"source-file hash mismatch on resume for {source.source_id}")
        next_line, processed, rejected, duplicates, complete = (
            (int(row[1]), int(row[2]), int(row[3]), int(row[4]), bool(row[5]))
            if row
            else (0, 0, 0, 0, False)
        )
        if complete:
            total_processed += processed
            total_rejected += rejected
            total_duplicates += duplicates
            continue
        bucket_buffer: list[tuple[str, str]] = []
        for line_number, payload in _iter_jsonl(
            path,
            next_line,
            compressed=source.format == "jsonl_gzip",
        ):
            if config.maximum_documents is not None and indexed_now >= config.maximum_documents:
                stopped_by_bound = True
                break
            text = payload.get(source.text_field) if payload is not None else None
            if not isinstance(text, str) or not text.strip():
                rejected += 1
            else:
                normalized = normalize_for_matching(text)
                fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                raw_id = payload.get(source.document_id_field) if source.document_id_field else None
                document_id = str(raw_id) if raw_id not in (None, "") else f"{source.source_id}:line:{line_number}"
                source_url = payload.get(source.source_url_field) if source.source_url_field else None
                signature = minhash_signature(
                    normalized,
                    shingle_size=config.shingle_size,
                    signature_size=config.signature_size,
                )
                record = IndexRecord(
                    document_id=document_id,
                    source_id=source.source_id,
                    source_revision=source.source_revision,
                    source_shard=source.source_shard,
                    source_url=str(source_url) if source_url else None,
                    normalized_sha256=fingerprint,
                    signature=signature,
                    normalized_character_count=len(normalized),
                    normalization_version=NORMALIZATION_VERSION,
                    provenance_completeness=source.provenance_completeness,
                    source_content_reference=f"{source.path}#line={line_number + 1}",
                )
                record.validate()
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record.document_id,
                        record.source_id,
                        record.source_revision,
                        record.source_shard,
                        record.source_url,
                        record.normalized_sha256,
                        _signature_text(record.signature),
                        record.normalized_character_count,
                        record.normalization_version,
                        record.provenance_completeness,
                        record.source_content_reference,
                    ),
                ).rowcount
                if inserted:
                    bucket_buffer.extend(
                        (key, record.document_id)
                        for key in bucket_keys(signature, config.bands)
                    )
                    processed += 1
                    indexed_now += 1
                else:
                    duplicates += 1
            next_line = line_number + 1
            if (processed + rejected + duplicates) % config.batch_size == 0:
                if bucket_buffer:
                    connection.executemany(
                        "INSERT OR IGNORE INTO buckets(bucket_key,document_id) VALUES(?,?)",
                        bucket_buffer,
                    )
                    bucket_buffer.clear()
                connection.execute(
                    "INSERT OR REPLACE INTO source_progress VALUES(?,?,?,?,?,?,0)",
                    (source.source_id, source_hash, next_line, processed, rejected, duplicates),
                )
                connection.commit()
                atomic_write_json(
                    progress_path,
                    {
                        "status": "in_progress",
                        "configuration_hash": config_hash,
                        "source_id": source.source_id,
                        "next_line": next_line,
                        "indexed_documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                    },
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "source_id": source.source_id,
                            "next_line": next_line,
                            "indexed_documents": int(
                                connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                            ),
                            "rejected_records": rejected,
                            "duplicate_hashes": duplicates,
                        }
                    )
            if stop_after_documents is not None and connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0] >= stop_after_documents:
                if bucket_buffer:
                    connection.executemany(
                        "INSERT OR IGNORE INTO buckets(bucket_key,document_id) VALUES(?,?)",
                        bucket_buffer,
                    )
                connection.execute(
                    "INSERT OR REPLACE INTO source_progress VALUES(?,?,?,?,?,?,0)",
                    (source.source_id, source_hash, next_line, processed, rejected, duplicates),
                )
                connection.commit()
                connection.close()
                atomic_write_json(
                    progress_path,
                    {
                        "status": "in_progress",
                        "configuration_hash": config_hash,
                        "source_id": source.source_id,
                        "next_line": next_line,
                        "indexed_documents": stop_after_documents,
                    },
                )
                raise InterruptedError("synthetic bounded interruption")
        if bucket_buffer:
            connection.executemany(
                "INSERT OR IGNORE INTO buckets(bucket_key,document_id) VALUES(?,?)",
                bucket_buffer,
            )
        connection.execute(
            "INSERT OR REPLACE INTO source_progress VALUES(?,?,?,?,?,?,0)",
            (source.source_id, source_hash, next_line, processed, rejected, duplicates),
        )
        connection.commit()
        total_processed += processed
        total_rejected += rejected
        total_duplicates += duplicates
        if stopped_by_bound:
            break
        connection.execute(
            "INSERT OR REPLACE INTO source_progress VALUES(?,?,?,?,?,?,1)",
            (source.source_id, source_hash, next_line, processed, rejected, duplicates),
        )
        connection.commit()

    indexed_documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    input_documents = int(
        connection.execute(
            "SELECT COALESCE(SUM(processed + rejected + duplicate_hashes),0) FROM source_progress"
        ).fetchone()[0]
    )
    complete = not stopped_by_bound
    if complete and config.expected_document_count is not None and input_documents != config.expected_document_count:
        connection.close()
        raise ValueError(
            f"input document count mismatch: expected={config.expected_document_count}, "
            f"actual={input_documents}"
        )
    complete_ids = int(
        connection.execute(
            "SELECT COUNT(*) FROM documents WHERE document_id NOT LIKE '%:line:%'"
        ).fetchone()[0]
    )
    url_records = int(
        connection.execute("SELECT COUNT(*) FROM documents WHERE source_url IS NOT NULL").fetchone()[0]
    )
    incomplete_provenance = int(
        connection.execute(
            "SELECT COUNT(*) FROM documents WHERE provenance_completeness='incomplete'"
        ).fetchone()[0]
    )
    bucket_count = int(connection.execute("SELECT COUNT(*) FROM buckets").fetchone()[0])
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES('completion_status',?)",
        ("complete" if complete else "bounded_complete",),
    )
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    os.replace(partial, output)
    output_hash = sha256_file(output)
    metadata = {
        "format_version": INDEX_FORMAT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "configuration_hash": config_hash,
        "completion_status": "complete" if complete else "bounded_complete",
        "coverage": "configured_sources_only",
        "coverage_sources": list(config.coverage),
        "missing_coverage": list(config.missing_coverage),
        "training_ready": complete and not config.missing_coverage,
        "input_documents": input_documents,
        "indexed_documents": indexed_documents,
        "rejected_records": total_rejected,
        "duplicate_hashes": total_duplicates,
        "records_with_complete_ids": complete_ids,
        "records_with_urls": url_records,
        "records_with_incomplete_provenance": incomplete_provenance,
        "lsh_bucket_count": bucket_count,
        "source_files": [
            {"source_id": source.source_id, "path": source.path, "sha256": source_hash}
            for source, _, source_hash in source_info
        ],
        "output_path": config.output_path,
        "output_sha256": output_hash,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "created_at": _utc_now(),
    }
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(progress_path, {**metadata, "status": metadata["completion_status"]})
    return metadata


class FineWebDocumentIndex:
    """Read-only exact and LSH candidate lookup."""

    def __init__(
        self,
        path: Path,
        *,
        expected_normalization_version: str = NORMALIZATION_VERSION,
        allow_bounded: bool = False,
    ):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"FineWeb document index not found: {self.path}")
        self.connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        rows = dict(self.connection.execute("SELECT key,value FROM metadata"))
        version = rows.get("normalization_version")
        if version != expected_normalization_version:
            self.connection.close()
            raise ValueError(
                f"FineWeb normalization version mismatch: index={version!r}, "
                f"required={expected_normalization_version!r}"
            )
        allowed_statuses = {"complete", "bounded_complete"} if allow_bounded else {"complete"}
        if rows.get("completion_status") not in allowed_statuses:
            self.connection.close()
            raise ValueError("FineWeb document index is incomplete or only a bounded pilot")

    def close(self) -> None:
        self.connection.close()

    def exact(self, fingerprint: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT document_id,source_id,source_revision,source_shard,source_url,"
            "normalized_sha256,signature,normalized_character_count,normalization_version,"
            "provenance_completeness,source_content_reference FROM documents "
            "WHERE normalized_sha256=?",
            (fingerprint,),
        ).fetchone()
        return self._row(row) if row else None

    def candidates(self, signature: tuple[int, ...], bands: int = 8) -> list[dict[str, Any]]:
        keys = bucket_keys(signature, bands)
        placeholders = ",".join("?" for _ in keys)
        rows = self.connection.execute(
            "SELECT DISTINCT d.document_id,d.source_id,d.source_revision,d.source_shard,"
            "d.source_url,d.normalized_sha256,d.signature,d.normalized_character_count,"
            "d.normalization_version,d.provenance_completeness,d.source_content_reference "
            "FROM documents d JOIN buckets b ON b.document_id=d.document_id "
            f"WHERE b.bucket_key IN ({placeholders}) ORDER BY d.document_id",
            keys,
        ).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: tuple[Any, ...]) -> dict[str, Any]:
        keys = (
            "document_id", "source_id", "source_revision", "source_shard", "source_url",
            "normalized_sha256", "signature", "normalized_character_count",
            "normalization_version", "provenance_completeness", "source_content_reference",
        )
        value = dict(zip(keys, row))
        value["signature"] = _parse_signature(value["signature"])
        return value


class FederatedFineWebDocumentIndex:
    """Read-only lookup across independently validated compatible indexes."""

    def __init__(self, paths: list[Path] | tuple[Path, ...]):
        if not paths:
            raise ValueError("A federated FineWeb index needs at least one database")
        self.indexes = [FineWebDocumentIndex(path) for path in paths]

    def close(self) -> None:
        for index in self.indexes:
            index.close()

    def exact(self, fingerprint: str) -> dict[str, Any] | None:
        for index in self.indexes:
            result = index.exact(fingerprint)
            if result is not None:
                return result
        return None

    def candidates(self, signature: tuple[int, ...], bands: int = 8) -> list[dict[str, Any]]:
        combined: dict[tuple[str, str], dict[str, Any]] = {}
        for index in self.indexes:
            for result in index.candidates(signature, bands):
                combined[(result["source_id"], result["document_id"])] = result
        return [combined[key] for key in sorted(combined)]
