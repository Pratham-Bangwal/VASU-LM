"""Typed schemas for dataset provenance and approval metadata."""

from __future__ import annotations

from dataclasses import dataclass


APPROVAL_STATUSES = frozenset({"pending", "approved", "rejected", "blocked"})


@dataclass(frozen=True)
class DataSourceRecord:
    source_id: str
    display_name: str
    provider: str
    dataset_name: str
    dataset_card_url: str
    homepage_url: str
    access_method: str
    subset: str
    split: str
    pinned_revision: str
    license_name: str
    license_url: str
    commercial_use_allowed: bool | None
    attribution_required: bool | None
    redistribution_allowed: bool | None
    gated_access: bool | None
    requires_authentication: bool | None
    expected_download_size_bytes: int | None
    expected_raw_examples: int | None
    expected_raw_tokens: int | None
    supported_domains: tuple[str, ...]
    intended_purpose: str
    quality_risks: tuple[str, ...]
    required_quality_filters: tuple[str, ...]
    deduplication_requirements: tuple[str, ...]
    contamination_risks: tuple[str, ...]
    local_raw_path: str
    local_processed_path: str
    approval_status: str
    approval_notes: str
    reviewed_by: str
    reviewed_at: str
    content_hash: str | None
    notes: str


@dataclass(frozen=True)
class SourceRegistry:
    records: tuple[DataSourceRecord, ...]

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(record.source_id for record in self.records)

