"""Typed, trainer-independent schemas for VASU data-mixture planning."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_DOMAINS = frozenset(
    {
        "general",
        "educational",
        "factual",
        "code",
        "mathematics",
        "reasoning",
        "conversation",
    }
)
SUPPORTED_FORMATS = frozenset({"token_bin", "jsonl", "parquet", "text"})
SUPPORTED_MANIFEST_VERSIONS = frozenset({"1.0"})


@dataclass(frozen=True)
class MixtureSource:
    id: str
    domain: str
    path: str
    format: str
    tokenizer_path: str
    token_count: int
    weight: float
    license: str
    source_url: str
    dataset_revision: str
    attribution_required: bool
    commercial_use_allowed: bool
    split: str
    content_hash: str | None
    deduplication_status: str
    quality_filters: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class MixtureManifest:
    manifest_version: str
    experiment_id: str
    description: str
    tokenizer_path: str
    target_tokens: int
    random_seed: int
    sources: tuple[MixtureSource, ...]
    created_at: str
    notes: str
    sampling_with_replacement: bool = False
    allow_duplicate_paths: bool = False


@dataclass(frozen=True)
class SourceAllocation:
    source_id: str
    domain: str
    weight: float
    allocated_tokens: int
    available_tokens: int
    replacement_required: bool


@dataclass(frozen=True)
class SamplingPlan:
    manifest_version: str
    experiment_id: str
    random_seed: int
    target_tokens: int
    sampling_with_replacement: bool
    allocations: tuple[SourceAllocation, ...]
    deterministic_source_order: tuple[str, ...]

    @property
    def allocated_tokens(self) -> int:
        return sum(item.allocated_tokens for item in self.allocations)

    def allocation_for(self, source_id: str) -> SourceAllocation:
        for allocation in self.allocations:
            if allocation.source_id == source_id:
                return allocation
        raise KeyError(f"Unknown source ID in sampling plan: {source_id}")
