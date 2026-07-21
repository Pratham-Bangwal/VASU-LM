"""Typed schemas for the FineWeb document index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .normalization import NORMALIZATION_VERSION


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
INDEX_FORMAT_VERSION = "fineweb_document_index_v1"


@dataclass(frozen=True)
class FineWebSource:
    source_id: str
    source_revision: str
    source_shard: str
    path: str
    format: str = "jsonl_text"
    text_field: str = "text"
    source_url_field: str | None = None
    document_id_field: str | None = None
    provenance_completeness: str = "incomplete"
    expected_sha256: str | None = None

    def validate(self) -> None:
        for name in ("source_id", "source_revision", "source_shard", "path"):
            if not getattr(self, name):
                raise ValueError(f"FineWeb source {name} must be non-empty")
        if self.format not in {"jsonl_text", "jsonl_gzip"}:
            raise ValueError(f"unsupported FineWeb source format: {self.format}")
        if self.provenance_completeness not in {"complete", "incomplete"}:
            raise ValueError("provenance_completeness must be complete or incomplete")
        if self.expected_sha256 is not None and not SHA256_PATTERN.fullmatch(
            self.expected_sha256.casefold()
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class FineWebIndexConfig:
    format_version: str
    output_path: str
    metadata_path: str
    progress_path: str
    normalization_version: str
    shingle_size: int
    signature_size: int
    bands: int
    batch_size: int
    maximum_documents: int | None
    sources: tuple[FineWebSource, ...]
    expected_document_count: int | None = None
    replacement_policy: str = "refuse_existing_use_explicit_restart"
    coverage: tuple[str, ...] = ("fineweb_original", "fineweb_extension")
    missing_coverage: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.format_version != INDEX_FORMAT_VERSION:
            raise ValueError(f"unsupported index format version: {self.format_version}")
        if self.normalization_version != NORMALIZATION_VERSION:
            raise ValueError("unsupported normalization version")
        if not self.sources:
            raise ValueError("FineWeb index needs at least one source")
        if self.shingle_size < 1 or self.signature_size < 1 or self.bands < 1:
            raise ValueError("shingle/signature/band values must be positive")
        if self.signature_size % self.bands:
            raise ValueError("signature_size must be divisible by bands")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.maximum_documents is not None and self.maximum_documents < 1:
            raise ValueError("maximum_documents must be positive or null")
        if self.expected_document_count is not None and self.expected_document_count < 1:
            raise ValueError("expected_document_count must be positive or null")
        if self.replacement_policy not in {
            "refuse_existing_use_explicit_restart",
            "refuse_existing",
        }:
            raise ValueError("unsupported index replacement policy")
        if not self.coverage or any(not value for value in self.coverage):
            raise ValueError("index coverage must be non-empty")
        if set(self.coverage) & set(self.missing_coverage):
            raise ValueError("coverage and missing_coverage must not overlap")
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("duplicate FineWeb source IDs")
        for source in self.sources:
            source.validate()
        for raw in (self.output_path, self.metadata_path, self.progress_path):
            path = Path(raw)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("index paths must be repository-relative")


@dataclass(frozen=True)
class IndexRecord:
    document_id: str
    source_id: str
    source_revision: str
    source_shard: str
    source_url: str | None
    normalized_sha256: str
    signature: tuple[int, ...]
    normalized_character_count: int
    normalization_version: str
    provenance_completeness: str
    source_content_reference: str

    def validate(self) -> None:
        if not self.document_id or not self.source_id or not self.source_content_reference:
            raise ValueError("index record identity fields must be non-empty")
        if not SHA256_PATTERN.fullmatch(self.normalized_sha256):
            raise ValueError("index record contains invalid normalized SHA-256")
        if not self.signature:
            raise ValueError("index record signature cannot be empty")
        if self.normalized_character_count < 0:
            raise ValueError("normalized character count cannot be negative")
        if self.normalization_version != NORMALIZATION_VERSION:
            raise ValueError("index record normalization version mismatch")
        if self.provenance_completeness not in {"complete", "incomplete"}:
            raise ValueError("invalid provenance completeness")
