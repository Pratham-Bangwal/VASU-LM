"""Bounded exact and near-duplicate detection for pilot preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3

from vasu.data.deduplication.fineweb_index import FineWebDocumentIndex
from vasu.data.deduplication.fineweb_index import sha256_file as index_sha256_file
from vasu.data.deduplication.normalization import NORMALIZATION_VERSION

from .filters import comparison_normalize


WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(comparison_normalize(text).encode("utf-8")).hexdigest()


def word_shingles(text: str, size: int = 5) -> frozenset[tuple[str, ...]]:
    words = WORD_PATTERN.findall(comparison_normalize(text))
    if len(words) < size:
        return frozenset({tuple(words)}) if words else frozenset()
    return frozenset(tuple(words[index : index + size]) for index in range(len(words) - size + 1))


def jaccard_similarity(
    left: frozenset[tuple[str, ...]],
    right: frozenset[tuple[str, ...]],
) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class PilotDeduplicator:
    """Exact Jaccard comparisons are intentional and bounded to the pilot."""

    def __init__(self, *, exact_enabled: bool, near_enabled: bool, threshold: float):
        self.exact_enabled = exact_enabled
        self.near_enabled = near_enabled
        self.threshold = threshold
        self.exact_hashes: set[str] = set()
        self.signatures: list[frozenset[tuple[str, ...]]] = []
        self.candidate_comparisons = 0

    def add_existing(self, text: str, fingerprint: str | None = None) -> None:
        self.exact_hashes.add(fingerprint or normalized_sha256(text))
        if self.near_enabled:
            self.signatures.append(word_shingles(text))

    def classify(self, text: str) -> tuple[str | None, str]:
        fingerprint = normalized_sha256(text)
        if self.exact_enabled and fingerprint in self.exact_hashes:
            return "exact_duplicate", fingerprint
        signature = word_shingles(text)
        if self.near_enabled:
            for existing in self.signatures:
                self.candidate_comparisons += 1
                if jaccard_similarity(signature, existing) >= self.threshold:
                    return "near_duplicate", fingerprint
        return None, fingerprint

    def accept(self, text: str, fingerprint: str) -> None:
        """Commit a retained document after all other checks have passed."""
        self.exact_hashes.add(fingerprint)
        if self.near_enabled:
            self.signatures.append(word_shingles(text))


def fineweb_document_index_status(
    repository_root: Path,
    configured_path: str = "data/manifests/pretrain/fineweb_document_index.sqlite3",
) -> dict[str, object]:
    preferred = repository_root / configured_path
    if preferred.is_file():
        try:
            index = FineWebDocumentIndex(preferred)
            index.close()
            metadata_path = preferred.with_name(f"{preferred.stem}_metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("completion_status") != "complete":
                raise ValueError("FineWeb document index metadata is not complete")
            if metadata.get("normalization_version") != NORMALIZATION_VERSION:
                raise ValueError("FineWeb document index metadata normalization mismatch")
            if metadata.get("output_sha256") != index_sha256_file(preferred):
                raise ValueError("FineWeb document index output hash mismatch")
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            ValueError,
            sqlite3.DatabaseError,
        ) as error:
            return {
                "status": "blocked",
                "path": preferred.relative_to(repository_root).as_posix(),
                "reason": str(error),
                "normalization_version": NORMALIZATION_VERSION,
            }
        return {
            "status": "available",
            "path": preferred.relative_to(repository_root).as_posix(),
            "normalization_version": NORMALIZATION_VERSION,
        }
    candidates = (
        repository_root / "data/manifests/pretrain/fineweb_document_hashes.jsonl",
        repository_root / "data/manifests/pretrain/fineweb_document_hashes.sqlite",
        repository_root / "data/processed/pretrain/fineweb_document_hashes.txt",
    )
    available = next((path for path in candidates if path.is_file()), None)
    if available is not None:
        return {"status": "available", "path": available.relative_to(repository_root).as_posix()}
    return {
        "status": "blocked",
        "path": None,
        "required_artifact": (
            "A versioned document-level index containing normalized-text SHA-256 "
            "hashes and compatible word-5-gram MinHash signatures for all FineWeb "
            "training sources, with normalization version and source provenance."
        ),
    }


def load_fineweb_exact_hashes(repository_root: Path) -> tuple[set[str], dict[str, object]]:
    """Load a versioned document-level hash index when one is available.

    Token binaries are intentionally never treated as document indexes.  Text,
    JSONL, and SQLite indexes must expose normalized SHA-256 values explicitly.
    """
    status = fineweb_document_index_status(repository_root)
    if status["status"] != "available":
        return set(), status
    relative = status["path"]
    assert isinstance(relative, str)
    path = repository_root / relative
    hashes: set[str] = set()
    if path.suffix == ".txt":
        candidates = path.read_text(encoding="utf-8").splitlines()
    elif path.suffix == ".jsonl":
        candidates = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid FineWeb hash JSONL at line {line_number}"
                    ) from error
                value = record.get("normalized_sha256", record.get("sha256"))
                candidates.append(value)
    elif path.suffix == ".sqlite":
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            selected: tuple[str, str] | None = None
            for (table,) in tables:
                columns = {
                    row[1]
                    for row in connection.execute(f'PRAGMA table_info("{table}")')
                }
                for column in ("normalized_sha256", "sha256"):
                    if column in columns:
                        selected = (table, column)
                        break
                if selected:
                    break
            if selected is None:
                raise ValueError("FineWeb SQLite index lacks a SHA-256 column")
            table, column = selected
            candidates = [
                row[0]
                for row in connection.execute(
                    f'SELECT "{column}" FROM "{table}"'
                )
            ]
    else:  # pragma: no cover - candidates are fixed above
        raise ValueError(f"Unsupported FineWeb document index: {path}")
    for value in candidates:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError(f"FineWeb document index contains an invalid SHA-256: {value!r}")
        hashes.add(value.casefold())
    status = {**status, "loaded_exact_hashes": len(hashes)}
    return hashes, status
