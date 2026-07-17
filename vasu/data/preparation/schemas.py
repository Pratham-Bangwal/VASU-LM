"""Typed configuration and result schemas for bounded data preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


MAX_PILOT_DOWNLOAD_BYTES = 1_000_000_000
MAX_PILOT_RAW_EXAMPLES = 10_000
MAX_PILOT_ACCEPTED_DOCUMENTS = 2_000
MAX_PILOT_OUTPUT_TOKENS = 2_000_000


@dataclass(frozen=True)
class PreparationOutputPaths:
    raw_directory: str
    interim_directory: str
    processed_directory: str
    output_jsonl: str
    manifest_json: str
    progress_json: str
    summary_json: str
    summary_text: str


@dataclass(frozen=True)
class WikimediaPreparationConfig:
    source_id: str
    dataset_name: str
    subset: str
    split: str
    pinned_revision: str
    shard_identifier: str
    tokenizer_path: str
    random_seed: int
    max_download_bytes: int
    max_raw_examples: int
    max_accepted_documents: int
    max_output_tokens: int
    minimum_document_characters: int
    maximum_document_characters: int
    exact_deduplication_enabled: bool
    near_deduplication_enabled: bool
    near_duplicate_similarity_threshold: float
    contamination_check_enabled: bool
    contamination_ngram_words: int
    chunking_enabled: bool
    target_chunk_tokens: int
    maximum_chunk_tokens: int
    minimum_chunk_tokens: int
    chunk_overlap_tokens: int
    review_sampling_enabled: bool
    reference_section_behavior: str
    quality_warning_threshold: int
    review_max_chunks_per_article: int
    output_paths: PreparationOutputPaths
    resume_enabled: bool

    def for_smoke_test(self) -> "WikimediaPreparationConfig":
        base = self.output_paths
        smoke = PreparationOutputPaths(
            raw_directory=base.raw_directory,
            interim_directory=base.interim_directory,
            processed_directory=f"{base.processed_directory}_smoke",
            output_jsonl=str(Path(base.output_jsonl).with_name("documents_smoke.jsonl")),
            manifest_json=str(Path(base.manifest_json).with_name("wikimedia_pilot_smoke.json")),
            progress_json=str(Path(base.progress_json).with_name("progress_smoke.json")),
            summary_json=str(Path(base.summary_json).with_name("summary_smoke.json")),
            summary_text=str(Path(base.summary_text).with_name("summary_smoke.txt")),
        )
        return replace(
            self,
            max_raw_examples=min(self.max_raw_examples, 100),
            max_accepted_documents=min(self.max_accepted_documents, 20),
            max_output_tokens=min(self.max_output_tokens, 20_000),
            output_paths=smoke,
        )

    def for_review_sample(self) -> "WikimediaPreparationConfig":
        base = self.output_paths
        review = PreparationOutputPaths(
            raw_directory=base.raw_directory,
            interim_directory=base.interim_directory,
            processed_directory=f"{base.processed_directory}_review",
            output_jsonl=str(Path(base.output_jsonl).with_name("documents_review.jsonl")),
            manifest_json=str(Path(base.manifest_json).with_name("wikimedia_pilot_review.json")),
            progress_json=str(Path(base.progress_json).with_name("progress_review.json")),
            summary_json=str(Path(base.summary_json).with_name("summary_review.json")),
            summary_text=str(Path(base.summary_text).with_name("summary_review.txt")),
        )
        return replace(
            self,
            max_raw_examples=min(self.max_raw_examples, 500),
            max_accepted_documents=min(self.max_accepted_documents, 50),
            max_output_tokens=min(self.max_output_tokens, 50_000),
            review_sampling_enabled=True,
            reference_section_behavior="flag",
            review_max_chunks_per_article=5,
            output_paths=review,
        )


@dataclass
class PreparationProgress:
    configuration_hash: str
    source_id: str
    pinned_revision: str
    shard_identifier: str
    last_processed_row: int = -1
    raw_examples: int = 0
    accepted_documents: int = 0
    output_tokens: int = 0
    total_characters: int = 0
    output_size_bytes: int = 0
    rejection_counts: dict[str, int] | None = None
    seen_exact_hashes: list[str] | None = None
    contamination_matches: list[dict[str, str]] | None = None
    near_duplicate_candidate_comparisons: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    encoding_repairs: int = 0
    quality_rejections: int = 0
    active_row_index: int | None = None
    next_chunk_index: int = 0
    inspected_row_indices: list[int] | None = None
    reference_section_detections: int = 0
    status: str = "in_progress"

    def __post_init__(self) -> None:
        if self.rejection_counts is None:
            self.rejection_counts = {}
        if self.seen_exact_hashes is None:
            self.seen_exact_hashes = []
        if self.contamination_matches is None:
            self.contamination_matches = []
        if self.inspected_row_indices is None:
            self.inspected_row_indices = []
