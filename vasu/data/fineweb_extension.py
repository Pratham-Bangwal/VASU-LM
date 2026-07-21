"""Reusable preparation and manifest helpers for FineWeb extension shards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from vasu.tokenizer.tokenizer import VASUTokenizer


FORMAT_VERSION = 1
PREPARATION_SCRIPT_VERSION = "fineweb_extension_v1"
PREPROCESSING_VERSION = "strip_text_append_literal_eos_v1"
SEPARATOR = "\n[EOS]\n"
DTYPE = np.dtype(np.uint16)
VOCAB_SIZE = 32_000
MIN_FREE_DISK_GIB = 10

ORIGINAL_DATA_PATH = Path("data/processed/pretrain/fineweb_1m.bin")
ORIGINAL_RAW_PATH = Path("data/raw/pretrain/fineweb_1m.jsonl")
ORIGINAL_DATA_SHA256 = (
    "cd42417dd6456f438fb3fcde2f47d6a604d6dcf9cb76402e16ff3ae10f15aee6"
)
ORIGINAL_TOTAL_TOKENS = 1_271_317_605
ORIGINAL_TRAIN_END = 1_245_891_252
ORIGINAL_VALIDATION_END = 1_271_317_605

REQUIRED_METADATA_FIELDS = {
    "format_version",
    "dataset_repository",
    "dataset_config",
    "dataset_revision",
    "split",
    "source_selection_reason",
    "source_overlap_policy",
    "requested_target_tokens",
    "actual_written_tokens",
    "dtype",
    "bytes_per_token",
    "file_size_bytes",
    "tokenizer_path",
    "tokenizer_sha256",
    "vocab_size",
    "separator_policy",
    "normalization_policy",
    "filtering_policy",
    "documents_seen",
    "documents_written",
    "documents_dropped",
    "malformed_documents",
    "empty_documents",
    "duplicates_within_extension",
    "source_ids_preserved",
    "minimum_token_id",
    "maximum_token_id",
    "buffer_token_limit",
    "creation_started_at",
    "creation_completed_at",
    "output_sha256",
    "preparation_script_version",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_document(text: str) -> str:
    """Match the original FineWeb preparation's normalization exactly."""
    return text.strip()


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_token_ids(token_ids: list[int]) -> None:
    if not token_ids:
        raise ValueError("Tokenized document contains no tokens.")
    minimum = min(token_ids)
    maximum = max(token_ids)
    if minimum < 0 or maximum >= VOCAB_SIZE:
        raise ValueError(
            f"Token ID range [{minimum}, {maximum}] is outside "
            f"[0, {VOCAB_SIZE})."
        )


def write_uint16(handle: Any, token_ids: list[int]) -> int:
    validate_token_ids(token_ids)
    array = np.asarray(token_ids, dtype=DTYPE)
    array.tofile(handle)
    return int(array.size)


@dataclass
class Progress:
    dataset_repository: str
    dataset_config: str
    dataset_revision: str
    split: str
    seed: int
    tokenizer_sha256: str
    tokenizer_path: str
    output_temporary_path: str
    deduplication_database_path: str
    preprocessing_version: str
    dtype: str = "uint16"
    documents_seen: int = 0
    documents_written: int = 0
    documents_dropped: int = 0
    malformed_documents: int = 0
    empty_documents: int = 0
    duplicates_within_extension: int = 0
    cross_shard_exact_duplicates: int = 0
    missing_source_ids: int = 0
    tokens_written: int = 0
    minimum_token_id: int | None = None
    maximum_token_id: int | None = None
    creation_started_at: str = ""


def _connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS fingerprints (
            fingerprint TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            source_id TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def _get_state(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM state WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else str(row[0])


def _set_state(
    connection: sqlite3.Connection,
    key: str,
    value: str,
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)",
        (key, value),
    )


def seed_original_fingerprints(
    connection: sqlite3.Connection,
    original_raw_path: Path,
    commit_interval: int = 10_000,
) -> tuple[int, int]:
    """Seed normalized original-document hashes for exact overlap rejection."""
    if _get_state(connection, "original_seed_complete") == "1":
        unique = int(_get_state(connection, "original_unique_documents") or 0)
        duplicates = int(_get_state(connection, "original_duplicate_documents") or 0)
        return unique, duplicates

    if not original_raw_path.exists():
        raise FileNotFoundError(
            "Original raw FineWeb JSONL is required for exact cross-shard "
            f"deduplication: {original_raw_path}"
        )

    # A prior interruption may have committed a partial seed. Rebuild that
    # portion deterministically so the reported unique/duplicate counts remain
    # exact; extension rows cannot exist before seeding completes.
    connection.execute("DELETE FROM fingerprints WHERE origin = 'original'")
    connection.commit()

    unique = 0
    duplicates = 0
    with original_raw_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                text = normalize_document(row["text"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if not text:
                continue
            fingerprint = fingerprint_text(text)
            cursor = connection.execute(
                "INSERT OR IGNORE INTO fingerprints "
                "(fingerprint, origin, source_id) VALUES (?, 'original', NULL)",
                (fingerprint,),
            )
            if cursor.rowcount:
                unique += 1
            else:
                duplicates += 1
            if index % commit_interval == 0:
                connection.commit()

    _set_state(connection, "original_unique_documents", str(unique))
    _set_state(connection, "original_duplicate_documents", str(duplicates))
    _set_state(connection, "original_seed_complete", "1")
    connection.commit()
    return unique, duplicates


def _progress_from_database(
    connection: sqlite3.Connection,
) -> Progress | None:
    value = _get_state(connection, "progress")
    if value is None:
        return None
    return Progress(**json.loads(value))


def _persist_progress(
    connection: sqlite3.Connection,
    progress: Progress,
    state_path: Path,
    pending_fingerprints: list[tuple[str, str | None]],
) -> None:
    with connection:
        for fingerprint, source_id in pending_fingerprints:
            connection.execute(
                "INSERT INTO fingerprints "
                "(fingerprint, origin, source_id) "
                "VALUES (?, 'extension', ?)",
                (fingerprint, source_id),
            )
        _set_state(connection, "progress", json.dumps(asdict(progress)))
    atomic_write_json(state_path, asdict(progress))


def _validate_resume_identity(
    progress: Progress,
    expected: Progress,
) -> None:
    identity_fields = (
        "dataset_repository",
        "dataset_config",
        "dataset_revision",
        "split",
        "seed",
        "tokenizer_sha256",
        "tokenizer_path",
        "output_temporary_path",
        "deduplication_database_path",
        "preprocessing_version",
        "dtype",
    )
    mismatches = [
        field
        for field in identity_fields
        if getattr(progress, field) != getattr(expected, field)
    ]
    if mismatches:
        raise ValueError(
            "Resume state does not match the requested preparation: "
            + ", ".join(mismatches)
        )


def _free_disk_gib(path: Path) -> float:
    anchor = path.parent if path.parent.exists() else Path.cwd()
    return shutil.disk_usage(anchor).free / (1024**3)


def _source_rows_after(
    rows: Iterable[Mapping[str, Any]],
    documents_seen: int,
) -> Iterator[Mapping[str, Any]]:
    for index, row in enumerate(rows):
        if index < documents_seen:
            continue
        yield row


def prepare_extension_from_rows(
    *,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: VASUTokenizer,
    tokenizer_path: Path,
    output_path: Path,
    metadata_path: Path,
    dataset_repository: str,
    dataset_config: str,
    dataset_revision: str,
    split: str,
    target_tokens: int,
    buffer_token_limit: int,
    seed: int,
    resume: bool,
    max_documents: int | None,
    original_raw_path: Path = ORIGINAL_RAW_PATH,
    original_data_path: Path = ORIGINAL_DATA_PATH,
    expected_original_sha256: str = ORIGINAL_DATA_SHA256,
    minimum_free_disk_gib: float = MIN_FREE_DISK_GIB,
    stop_after_documents: int | None = None,
) -> dict[str, Any]:
    """Prepare one shard; ``stop_after_documents`` exists for resume tests."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if buffer_token_limit <= 0:
        raise ValueError("buffer_token_limit must be positive")
    if output_path.resolve() == original_data_path.resolve():
        raise ValueError("Refusing to overwrite the original FineWeb binary.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if minimum_free_disk_gib and _free_disk_gib(output_path) < minimum_free_disk_gib:
        raise RuntimeError(
            f"At least {minimum_free_disk_gib:.2f} GiB free disk is required."
        )
    original_hash = sha256_file(original_data_path)
    if original_hash.lower() != expected_original_sha256.lower():
        raise ValueError("The original FineWeb binary hash changed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    state_path = output_path.with_suffix(output_path.suffix + ".state.json")
    database_path = output_path.with_suffix(output_path.suffix + ".dedup.sqlite3")
    tokenizer_hash = sha256_file(tokenizer_path)

    expected = Progress(
        dataset_repository=dataset_repository,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        split=split,
        seed=seed,
        tokenizer_sha256=tokenizer_hash,
        tokenizer_path=tokenizer_path.as_posix(),
        output_temporary_path=temporary_path.as_posix(),
        deduplication_database_path=database_path.as_posix(),
        preprocessing_version=PREPROCESSING_VERSION,
        creation_started_at=utc_now(),
    )

    if not resume and any(
        path.exists() for path in (temporary_path, state_path, database_path)
    ):
        raise FileExistsError(
            "Temporary preparation state exists; pass --resume or choose "
            "a different output path."
        )

    connection = _connect_database(database_path)
    try:
        original_unique, original_duplicates = seed_original_fingerprints(
            connection, original_raw_path
        )
        progress = _progress_from_database(connection)
        if progress is None:
            progress = expected
            temporary_path.touch(exist_ok=False)
            _persist_progress(connection, progress, state_path, [])
        else:
            if not resume:
                raise FileExistsError("Existing progress requires --resume.")
            _validate_resume_identity(progress, expected)

        expected_size = progress.tokens_written * DTYPE.itemsize
        actual_size = temporary_path.stat().st_size
        if actual_size < expected_size:
            raise ValueError(
                "Temporary output is shorter than committed SQLite progress."
            )
        if actual_size > expected_size:
            # A crash can occur after fsync but before the SQLite commit.
            with temporary_path.open("r+b") as handle:
                handle.truncate(expected_size)
        atomic_write_json(state_path, asdict(progress))

        buffer: list[int] = []
        pending_fingerprints: list[tuple[str, str | None]] = []
        pending_hashes: set[str] = set()
        documents_this_run = 0

        def flush() -> None:
            nonlocal buffer, pending_fingerprints, pending_hashes
            if buffer:
                if minimum_free_disk_gib and _free_disk_gib(output_path) < minimum_free_disk_gib:
                    raise RuntimeError("Free disk fell below the safe threshold.")
                with temporary_path.open("ab") as output_handle:
                    write_uint16(output_handle, buffer)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
            _persist_progress(
                connection,
                progress,
                state_path,
                pending_fingerprints,
            )
            buffer = []
            pending_fingerprints = []
            pending_hashes = set()

        for row in _source_rows_after(rows, progress.documents_seen):
            if max_documents is not None and progress.documents_seen >= max_documents:
                break
            progress.documents_seen += 1
            documents_this_run += 1

            if not isinstance(row, Mapping):
                progress.malformed_documents += 1
                progress.documents_dropped += 1
            else:
                raw_text = row.get("text")
                if not isinstance(raw_text, str):
                    progress.malformed_documents += 1
                    progress.documents_dropped += 1
                else:
                    text = normalize_document(raw_text)
                    if not text:
                        progress.empty_documents += 1
                        progress.documents_dropped += 1
                    else:
                        fingerprint = fingerprint_text(text)
                        existing = connection.execute(
                            "SELECT origin FROM fingerprints "
                            "WHERE fingerprint = ?",
                            (fingerprint,),
                        ).fetchone()
                        if fingerprint in pending_hashes:
                            existing = ("extension",)
                        if existing is not None:
                            if existing[0] == "original":
                                progress.cross_shard_exact_duplicates += 1
                            else:
                                progress.duplicates_within_extension += 1
                            progress.documents_dropped += 1
                        else:
                            token_ids = tokenizer.encode(text + SEPARATOR)
                            validate_token_ids(token_ids)
                            source_id_value = row.get("id")
                            source_id = (
                                str(source_id_value)
                                if source_id_value not in (None, "")
                                else None
                            )
                            if source_id is None:
                                progress.missing_source_ids += 1
                            pending_hashes.add(fingerprint)
                            pending_fingerprints.append((fingerprint, source_id))
                            buffer.extend(token_ids)
                            progress.documents_written += 1
                            progress.tokens_written += len(token_ids)
                            progress.minimum_token_id = min(
                                progress.minimum_token_id
                                if progress.minimum_token_id is not None
                                else min(token_ids),
                                min(token_ids),
                            )
                            progress.maximum_token_id = max(
                                progress.maximum_token_id
                                if progress.maximum_token_id is not None
                                else max(token_ids),
                                max(token_ids),
                            )

            if len(buffer) >= buffer_token_limit:
                flush()
                print(
                    f"documents_seen={progress.documents_seen:,} "
                    f"documents_written={progress.documents_written:,} "
                    f"tokens_written={progress.tokens_written:,} "
                    f"duplicates_removed="
                    f"{progress.duplicates_within_extension + progress.cross_shard_exact_duplicates:,} "
                    f"free_disk={_free_disk_gib(output_path):.2f} GiB"
                )
            elif progress.documents_seen % 1_000 == 0:
                flush()

            if progress.tokens_written >= target_tokens:
                flush()
                break
            if (
                stop_after_documents is not None
                and documents_this_run >= stop_after_documents
            ):
                flush()
                raise InterruptedError("Controlled preparation interruption.")

        flush()
        if progress.tokens_written < target_tokens:
            raise RuntimeError(
                "Source stopped before the requested target was reached; "
                "temporary state was preserved for resume."
            )

        if temporary_path.stat().st_size != progress.tokens_written * 2:
            raise ValueError("Temporary file size does not match token count.")
        if progress.minimum_token_id is None or progress.maximum_token_id is None:
            raise ValueError("No valid tokens were written.")

        os.replace(temporary_path, output_path)
        output_hash = sha256_file(output_path)
        metadata: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "dataset_repository": dataset_repository,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "split": split,
            "source_selection_reason": (
                "A distinct FineWeb-Edu Common Crawl dump from the verified "
                "original CC-MAIN-2013-20 source region."
            ),
            "source_overlap_policy": (
                "Exact normalized-text SHA-256 matches are rejected against "
                "the retained original raw JSONL and within the extension; "
                "semantic and near-duplicate overlap is not detectable."
            ),
            "requested_target_tokens": target_tokens,
            "actual_written_tokens": progress.tokens_written,
            "dtype": "uint16",
            "bytes_per_token": 2,
            "file_size_bytes": output_path.stat().st_size,
            "tokenizer_path": tokenizer_path.as_posix(),
            "tokenizer_sha256": tokenizer_hash,
            "vocab_size": VOCAB_SIZE,
            "separator_policy": "append exact text \\n[EOS]\\n to each complete document",
            "separator_token_ids": tokenizer.encode(SEPARATOR),
            "normalization_policy": "Python str.strip()",
            "filtering_policy": (
                "drop empty/malformed rows and exact normalized-text duplicates"
            ),
            "documents_seen": progress.documents_seen,
            "documents_written": progress.documents_written,
            "documents_dropped": progress.documents_dropped,
            "malformed_documents": progress.malformed_documents,
            "empty_documents": progress.empty_documents,
            "duplicates_within_extension": progress.duplicates_within_extension,
            "cross_shard_exact_duplicates": progress.cross_shard_exact_duplicates,
            "original_unique_fingerprints": original_unique,
            "original_duplicate_fingerprints": original_duplicates,
            "source_ids_preserved": progress.missing_source_ids == 0,
            "source_ids_storage": database_path.as_posix(),
            "minimum_token_id": progress.minimum_token_id,
            "maximum_token_id": progress.maximum_token_id,
            "buffer_token_limit": buffer_token_limit,
            "creation_started_at": progress.creation_started_at,
            "creation_completed_at": utc_now(),
            "output_sha256": output_hash,
            "preparation_script_version": PREPARATION_SCRIPT_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "packing_strategy": "complete documents; final document may exceed target",
            "shuffle": False,
            "seed": seed,
            "max_documents": max_documents,
            "deduplication_database_path": database_path.as_posix(),
            "resume_state_finalized": True,
            "original_data_path": original_data_path.as_posix(),
            "original_data_sha256": original_hash,
            "original_total_tokens": ORIGINAL_TOTAL_TOKENS,
            "original_train_end": ORIGINAL_TRAIN_END,
            "original_validation_start": ORIGINAL_TRAIN_END,
            "original_validation_end": ORIGINAL_VALIDATION_END,
            "output_path": output_path.as_posix(),
        }
        atomic_write_json(metadata_path, metadata)
        state_path.unlink(missing_ok=True)
        _set_state(connection, "finalized", "1")
        connection.commit()
        return metadata
    finally:
        connection.close()


def validate_source_selection(
    repository: str,
    config: str,
    revision: str,
    split: str,
) -> None:
    if repository != "HuggingFaceFW/fineweb-edu":
        raise ValueError("Only the verified FineWeb-Edu repository is compatible.")
    if not config.startswith("CC-MAIN-"):
        raise ValueError("Use a specific CC-MAIN FineWeb-Edu config, not default/sample.")
    if config == "CC-MAIN-2013-20":
        raise ValueError("CC-MAIN-2013-20 is the verified original source region.")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError("dataset revision must be a full 40-character commit hash")
    if split != "train":
        raise ValueError("Only the verified train split is supported.")


def build_manifest(
    *,
    extension_metadata: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    tokenizer_hash = str(extension_metadata["tokenizer_sha256"])
    manifest = {
        "format_version": 1,
        "dtype": "uint16",
        "tokenizer_path": "assets/tokenizer.json",
        "tokenizer_sha256": tokenizer_hash,
        "sequence_length": 256,
        "training_shards": [
            {
                "path": ORIGINAL_DATA_PATH.as_posix(),
                "start_token": 0,
                "end_token": ORIGINAL_TRAIN_END,
                "role": "original_train",
                "sha256": ORIGINAL_DATA_SHA256,
            },
            {
                "path": str(extension_metadata["output_path"]),
                "start_token": 0,
                "end_token": int(extension_metadata["actual_written_tokens"]),
                "role": "extension_train",
                "sha256": str(extension_metadata["output_sha256"]),
            },
        ],
        "validation": {
            "path": ORIGINAL_DATA_PATH.as_posix(),
            "start_token": ORIGINAL_TRAIN_END,
            "end_token": ORIGINAL_VALIDATION_END,
            "role": "fixed_original_validation",
        },
        "logical_training_tokens": (
            ORIGINAL_TRAIN_END + int(extension_metadata["actual_written_tokens"])
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def map_logical_training_offset(
    manifest: Mapping[str, Any],
    logical_offset: int,
) -> tuple[str, int]:
    if logical_offset < 0:
        raise IndexError("logical offset must be non-negative")
    remaining = logical_offset
    for shard in manifest["training_shards"]:
        length = int(shard["end_token"]) - int(shard["start_token"])
        if remaining < length:
            return str(shard["path"]), int(shard["start_token"]) + remaining
        remaining -= length
    raise IndexError("logical training offset is outside all training shards")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("dtype") != "uint16":
        raise ValueError("Manifest dtype must be uint16.")
    shards = manifest.get("training_shards")
    if not isinstance(shards, list) or len(shards) != 2:
        raise ValueError("Manifest must contain original and extension shards.")
    original = shards[0]
    if original.get("role") != "original_train":
        raise ValueError("First shard must be original_train.")
    if int(original["start_token"]) != 0 or int(original["end_token"]) != ORIGINAL_TRAIN_END:
        raise ValueError("Original training boundary changed.")
    validation = manifest.get("validation", {})
    if (
        validation.get("path") != ORIGINAL_DATA_PATH.as_posix()
        or int(validation.get("start_token", -1)) != ORIGINAL_TRAIN_END
        or int(validation.get("end_token", -1)) != ORIGINAL_VALIDATION_END
    ):
        raise ValueError("Original validation region was not preserved.")
    # Prove the boundary maps to the extension rather than validation.
    mapped_path, mapped_offset = map_logical_training_offset(
        manifest, ORIGINAL_TRAIN_END
    )
    if mapped_path != shards[1]["path"] or mapped_offset != int(shards[1]["start_token"]):
        raise ValueError("Original-to-extension logical boundary is invalid.")
