"""Bounded, resumable Wikimedia Parquet pilot preparation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics
import warnings
from collections import Counter
from typing import Any, Iterator, Mapping

import pyarrow.parquet as pq

from vasu.data.sources import get_source, load_source_registry
from vasu.data.sources.validation import validate_manifest_sources
from vasu.data.mixtures import load_manifest
from vasu.data.deduplication import (
    FederatedFineWebDocumentIndex,
    FineWebDocumentIndex,
    MatchDecision,
    match_text,
)

from .contamination import check_contamination, load_prompt_evidence
from .chunking import (
    BOUNDARY_TYPES,
    TextChunk,
    chunk_document,
    chunk_document_detailed,
    contains_reference_section_heading,
    is_reference_section_heading,
)
from .deduplication import (
    PilotDeduplicator,
    fineweb_document_index_status,
    load_fineweb_exact_hashes,
    normalized_sha256,
)
from .filters import filter_wikimedia_record
from .progress import (
    load_progress,
    reconcile_output,
    save_progress,
    validate_resume_identity,
)
from .quality import assess_final_chunk, assess_text_quality
from .reporting import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_hash,
    format_wikimedia_summary,
    sha256_file,
)
from .schemas import (
    MAX_PILOT_ACCEPTED_CHUNKS,
    MAX_PILOT_ACCEPTED_PARENT_DOCUMENTS,
    MAX_PILOT_DOWNLOAD_BYTES,
    MAX_PILOT_OUTPUT_TOKENS,
    MAX_PILOT_RAW_EXAMPLES,
    MAX_LIKELIHOOD_ACCEPTED_CHUNKS,
    MAX_LIKELIHOOD_ACCEPTED_PARENT_DOCUMENTS,
    MAX_LIKELIHOOD_OUTPUT_TOKENS,
    PreparationOutputPaths,
    PreparationProgress,
    WikimediaPreparationConfig,
)
from .token_count import count_tokens, load_vasu_tokenizer


REQUIRED_PARQUET_FIELDS = frozenset({"id", "url", "title", "text"})
CONFIG_FIELDS = frozenset(WikimediaPreparationConfig.__dataclass_fields__)
LEGACY_CHUNK_LIMIT_FIELD = "max_accepted_documents"
OUTPUT_PATH_FIELDS = frozenset(PreparationOutputPaths.__dataclass_fields__)
_DEFAULT_CHUNK_DOCUMENT = chunk_document


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_to_dict(config: WikimediaPreparationConfig) -> dict[str, Any]:
    return asdict(config)


def load_preparation_config(path: Path) -> WikimediaPreparationConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Preparation config does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Preparation config is invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Preparation config must be a JSON object")
    legacy_present = LEGACY_CHUNK_LIMIT_FIELD in payload
    explicit_chunk_present = "max_accepted_chunks" in payload
    if legacy_present and explicit_chunk_present:
        raise ValueError(
            "Use either 'max_accepted_documents' or 'max_accepted_chunks', not both"
        )
    if legacy_present:
        warnings.warn(
            "'max_accepted_documents' is deprecated and is interpreted as "
            "'max_accepted_chunks' because that matches historical behavior",
            DeprecationWarning,
            stacklevel=2,
        )
        payload["max_accepted_chunks"] = payload.pop(LEGACY_CHUNK_LIMIT_FIELD)
        # Historical configurations had no independent parent ceiling. The raw
        # and chunk caps together provide a safe non-tightening compatibility
        # bound because accepted parents can never exceed accepted chunks.
        payload.setdefault(
            "max_accepted_parent_documents",
            min(
                payload.get("max_raw_examples", 0),
                payload["max_accepted_chunks"],
            ),
        )
    payload.setdefault("maximum_list_like_line_ratio", 0.8)
    payload.setdefault("minimum_prose_sentences_for_list_chunk", 2)
    payload.setdefault("preparation_profile", "pilot_v1")
    missing = sorted(CONFIG_FIELDS - set(payload))
    unknown = sorted(set(payload) - CONFIG_FIELDS)
    if missing:
        raise ValueError(f"Preparation config missing fields: {missing}")
    if unknown:
        raise ValueError(f"Preparation config has unknown fields: {unknown}")
    raw_paths = payload["output_paths"]
    if not isinstance(raw_paths, dict):
        raise ValueError("output_paths must be a JSON object")
    path_missing = sorted(OUTPUT_PATH_FIELDS - set(raw_paths))
    path_unknown = sorted(set(raw_paths) - OUTPUT_PATH_FIELDS)
    if path_missing or path_unknown:
        raise ValueError(
            f"output_paths fields invalid; missing={path_missing}, unknown={path_unknown}"
        )
    payload["output_paths"] = PreparationOutputPaths(**raw_paths)
    config = WikimediaPreparationConfig(**payload)
    validate_preparation_config(config)
    return config


def _positive_int(value: object, name: str, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be greater than zero")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds hard pilot limit {maximum:,}: {value:,}")


def validate_preparation_config(config: WikimediaPreparationConfig) -> None:
    for field in (
        "source_id",
        "dataset_name",
        "subset",
        "split",
        "pinned_revision",
        "shard_identifier",
        "tokenizer_path",
    ):
        value = getattr(config, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    if config.dataset_name != "wikimedia/wikipedia":
        raise ValueError("dataset_name must remain exactly 'wikimedia/wikipedia'")
    if config.subset != "20231101.en" or config.split != "train":
        raise ValueError("pilot is restricted to subset 20231101.en and split train")
    if config.pinned_revision != "e6057dc557255a03c9c3c47ceab0eb44353b1bc5":
        raise ValueError("pinned_revision does not match the approved immutable revision")
    expected_prefix = f"{config.subset}/"
    if (
        not config.shard_identifier.startswith(expected_prefix)
        or not config.shard_identifier.endswith(".parquet")
        or any(character in config.shard_identifier for character in "*?[]")
    ):
        raise ValueError("shard_identifier must select one explicit Parquet shard")
    if isinstance(config.random_seed, bool) or not isinstance(config.random_seed, int) or config.random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if config.preparation_profile not in {
        "pilot_v1",
        "vasu_140m_likelihood_source_v1",
    }:
        raise ValueError("preparation_profile is unsupported")
    likelihood_profile = (
        config.preparation_profile == "vasu_140m_likelihood_source_v1"
    )
    parent_limit = (
        MAX_LIKELIHOOD_ACCEPTED_PARENT_DOCUMENTS
        if likelihood_profile
        else MAX_PILOT_ACCEPTED_PARENT_DOCUMENTS
    )
    chunk_limit = (
        MAX_LIKELIHOOD_ACCEPTED_CHUNKS
        if likelihood_profile
        else MAX_PILOT_ACCEPTED_CHUNKS
    )
    token_limit = (
        MAX_LIKELIHOOD_OUTPUT_TOKENS
        if likelihood_profile
        else MAX_PILOT_OUTPUT_TOKENS
    )
    _positive_int(config.max_download_bytes, "max_download_bytes", MAX_PILOT_DOWNLOAD_BYTES)
    _positive_int(config.max_raw_examples, "max_raw_examples", MAX_PILOT_RAW_EXAMPLES)
    _positive_int(
        config.max_accepted_parent_documents,
        "max_accepted_parent_documents",
        parent_limit,
    )
    _positive_int(
        config.max_accepted_chunks,
        "max_accepted_chunks",
        chunk_limit,
    )
    _positive_int(config.max_output_tokens, "max_output_tokens", token_limit)
    _positive_int(config.minimum_document_characters, "minimum_document_characters")
    _positive_int(config.maximum_document_characters, "maximum_document_characters")
    if config.minimum_document_characters >= config.maximum_document_characters:
        raise ValueError("minimum_document_characters must be less than maximum_document_characters")
    for field in (
        "exact_deduplication_enabled",
        "near_deduplication_enabled",
        "contamination_check_enabled",
        "chunking_enabled",
        "review_sampling_enabled",
        "resume_enabled",
        "fineweb_index_required",
    ):
        if not isinstance(getattr(config, field), bool):
            raise ValueError(f"{field} must be boolean")
    if isinstance(config.near_duplicate_similarity_threshold, bool) or not isinstance(
        config.near_duplicate_similarity_threshold, (int, float)
    ) or not 0 < float(config.near_duplicate_similarity_threshold) <= 1:
        raise ValueError("near_duplicate_similarity_threshold must be in (0, 1]")
    _positive_int(config.contamination_ngram_words, "contamination_ngram_words")
    _positive_int(config.target_chunk_tokens, "target_chunk_tokens")
    _positive_int(config.maximum_chunk_tokens, "maximum_chunk_tokens")
    _positive_int(config.minimum_chunk_tokens, "minimum_chunk_tokens")
    if isinstance(config.chunk_overlap_tokens, bool) or not isinstance(
        config.chunk_overlap_tokens, int
    ) or config.chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must be a non-negative integer")
    if not (
        config.chunk_overlap_tokens
        < config.minimum_chunk_tokens
        <= config.target_chunk_tokens
        <= config.maximum_chunk_tokens
    ):
        raise ValueError(
            "Chunk limits must satisfy overlap < minimum <= target <= maximum"
        )
    if config.reference_section_behavior not in {"keep", "flag", "exclude"}:
        raise ValueError("reference_section_behavior must be keep, flag, or exclude")
    if isinstance(config.maximum_list_like_line_ratio, bool) or not isinstance(
        config.maximum_list_like_line_ratio, (int, float)
    ) or not 0 < float(config.maximum_list_like_line_ratio) <= 1:
        raise ValueError("maximum_list_like_line_ratio must be in (0, 1]")
    _positive_int(
        config.minimum_prose_sentences_for_list_chunk,
        "minimum_prose_sentences_for_list_chunk",
    )
    fineweb_path = Path(config.fineweb_index_path)
    if (
        not config.fineweb_index_path
        or fineweb_path.is_absolute()
        or ".." in fineweb_path.parts
    ):
        raise ValueError("fineweb_index_path must be repository-relative and traversal-free")
    if isinstance(config.quality_warning_threshold, bool) or not isinstance(
        config.quality_warning_threshold, int
    ) or config.quality_warning_threshold < 0:
        raise ValueError("quality_warning_threshold must be a non-negative integer")
    if isinstance(config.review_max_chunks_per_article, bool) or not isinstance(
        config.review_max_chunks_per_article, int
    ) or config.review_max_chunks_per_article < 0:
        raise ValueError("review_max_chunks_per_article must be a non-negative integer")
    if config.review_sampling_enabled and config.review_max_chunks_per_article < 1:
        raise ValueError("Review sampling requires a positive per-article chunk cap")
    values = []
    for field in OUTPUT_PATH_FIELDS:
        value = getattr(config.output_paths, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"output_paths.{field} must be a non-empty string")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"output_paths.{field} must be repository-relative and traversal-free")
        values.append(path.as_posix().casefold())
    if len(values) != len(set(values)):
        raise ValueError("output_paths entries must be distinct")


def resolve_paths(config: WikimediaPreparationConfig, repository_root: Path) -> dict[str, Path]:
    root = Path(repository_root).resolve()
    paths = {field: root / getattr(config.output_paths, field) for field in OUTPUT_PATH_FIELDS}
    paths["tokenizer"] = root / config.tokenizer_path
    paths["prompts"] = root / "evaluation/prompts.json"
    paths["registry"] = root / "configs/data/sources"
    paths["factual_manifest"] = root / "configs/data/vasu_60m_factual_pilot.json"
    paths["fineweb_index"] = root / config.fineweb_index_path
    return paths


def validate_registry_approval(config: WikimediaPreparationConfig, repository_root: Path) -> None:
    root = Path(repository_root)
    registry = load_source_registry(root / "configs/data/sources")
    manifest = load_manifest(root / "configs/data/vasu_60m_factual_pilot.json")
    validate_manifest_sources(manifest, registry, require_approved=True)
    source = get_source(registry, config.source_id)
    if source.dataset_name != config.dataset_name:
        raise ValueError("Preparation dataset_name conflicts with approved registry record")
    expected_revision = source.pinned_revision.split(":", 1)[0]
    if config.pinned_revision != expected_revision:
        raise ValueError("Preparation revision conflicts with approved registry record")
    if source.subset != config.subset or source.split != config.split:
        raise ValueError("Preparation subset/split conflicts with approved registry record")
    if source.approval_status != "approved":
        raise ValueError(f"Source {config.source_id!r} is not approved")


def inspect_parquet_schema(path: Path) -> tuple[str, ...]:
    names = tuple(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(REQUIRED_PARQUET_FIELDS - set(names))
    if missing:
        raise ValueError(f"Selected Wikimedia Parquet shard lacks required fields: {missing}")
    return names


def iter_parquet_rows(path: Path, start_row: int = 0) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    row_index = 0
    columns = [name for name in ("id", "url", "title", "text", "language") if name in parquet.schema_arrow.names]
    for batch in parquet.iter_batches(batch_size=256, columns=columns):
        for row in batch.to_pylist():
            if row_index >= start_row:
                yield row
            row_index += 1


def select_review_row_indices(
    total_rows: int,
    maximum_rows: int,
    seed: int,
) -> tuple[int, ...]:
    """Select unique, shard-spanning rows without mutating global RNG state."""
    if total_rows < 1 or maximum_rows < 1:
        raise ValueError("total_rows and maximum_rows must be positive")
    count = min(total_rows, maximum_rows)
    offset = random.Random(seed).random()
    indices = tuple(
        min(total_rows - 1, int((position + offset) * total_rows / count))
        for position in range(count)
    )
    if len(set(indices)) != len(indices):  # Defensive; spacing should be unique.
        raise AssertionError("Deterministic review row selection produced duplicates")
    return indices


def iter_selected_parquet_rows(
    path: Path,
    selected_indices: tuple[int, ...],
    *,
    minimum_row: int = 0,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read only row groups containing selected rows."""
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    selected = tuple(
        index for index in selected_indices if index >= minimum_row
    )
    if any(index < 0 or index >= total_rows for index in selected):
        raise IndexError("Selected review row is outside the Parquet shard")
    columns = [
        name
        for name in ("id", "url", "title", "text", "language")
        if name in parquet.schema_arrow.names
    ]
    position = 0
    row_group_start = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group_rows = parquet.metadata.row_group(row_group_index).num_rows
        row_group_end = row_group_start + row_group_rows
        group_indices: list[int] = []
        while position < len(selected) and selected[position] < row_group_end:
            group_indices.append(selected[position])
            position += 1
        if group_indices:
            table = parquet.read_row_group(row_group_index, columns=columns)
            rows = table.to_pylist()
            for absolute_index in group_indices:
                yield absolute_index, rows[absolute_index - row_group_start]
        row_group_start = row_group_end
        if position >= len(selected):
            break


def is_reference_section(section_title: str | None) -> bool:
    """Backward-compatible wrapper around canonical heading matching."""
    return is_reference_section_heading(section_title)


def acquire_pinned_shard(
    config: WikimediaPreparationConfig,
    repository_root: Path,
) -> tuple[Path, dict[str, Any]]:
    paths = resolve_paths(config, repository_root)
    local_candidate = paths["raw_directory"] / config.shard_identifier
    acquisition_path = paths["interim_directory"] / "acquisition.json"
    if local_candidate.is_file() and acquisition_path.is_file():
        metadata = json.loads(acquisition_path.read_text(encoding="utf-8"))
        expected = {
            "dataset_name": config.dataset_name,
            "subset": config.subset,
            "split": config.split,
            "pinned_revision": config.pinned_revision,
            "shard_identifier": config.shard_identifier,
        }
        mismatches = [
            field
            for field, value in expected.items()
            if metadata.get(field) != value
        ]
        if mismatches:
            raise ValueError(
                f"Cached shard acquisition metadata mismatch: {mismatches}"
            )
        actual_size = local_candidate.stat().st_size
        if actual_size > config.max_download_bytes:
            raise ValueError("Cached shard exceeds configured download limit")
        if metadata.get("downloaded_size_bytes") != actual_size:
            raise ValueError("Cached shard size differs from acquisition metadata")
        if metadata.get("input_sha256") != sha256_file(local_candidate):
            raise ValueError("Cached shard hash differs from acquisition metadata")
        return local_candidate, metadata

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(
        config.dataset_name,
        revision=config.pinned_revision,
        files_metadata=True,
    )
    sibling = next(
        (item for item in info.siblings if item.rfilename == config.shard_identifier),
        None,
    )
    if sibling is None:
        raise FileNotFoundError(
            f"Pinned revision does not contain shard {config.shard_identifier!r}"
        )
    remote_size = getattr(sibling, "size", None)
    if not isinstance(remote_size, int):
        lfs = getattr(sibling, "lfs", None)
        remote_size = getattr(lfs, "size", None) if lfs is not None else None
    if not isinstance(remote_size, int) or remote_size < 1:
        raise ValueError("Provider metadata did not expose a valid shard size")
    if remote_size > config.max_download_bytes:
        raise ValueError(
            f"Selected shard is {remote_size:,} bytes, above configured "
            f"max_download_bytes={config.max_download_bytes:,}"
        )
    local_path = Path(
        hf_hub_download(
            repo_id=config.dataset_name,
            repo_type="dataset",
            filename=config.shard_identifier,
            revision=config.pinned_revision,
            local_dir=paths["raw_directory"],
        )
    )
    actual_size = local_path.stat().st_size
    if actual_size != remote_size or actual_size > config.max_download_bytes:
        raise ValueError(
            f"Downloaded shard size mismatch: expected={remote_size}, actual={actual_size}"
        )
    metadata = {
        "dataset_name": config.dataset_name,
        "subset": config.subset,
        "split": config.split,
        "pinned_revision": config.pinned_revision,
        "shard_identifier": config.shard_identifier,
        "provider_reported_size_bytes": remote_size,
        "downloaded_size_bytes": actual_size,
        "downloaded_at": utc_now(),
        "input_sha256": sha256_file(local_path),
        "local_path": local_path.relative_to(repository_root).as_posix(),
    }
    atomic_write_json(paths["interim_directory"] / "acquisition.json", metadata)
    return local_path, metadata


def _load_existing_output(
    output_path: Path,
    deduplicator: PilotDeduplicator,
) -> tuple[Counter[str], set[str], int, int]:
    parent_counts: Counter[str] = Counter()
    accepted_parent_ids: set[str] = set()
    accepted_chunks = 0
    total_tokens = 0
    if not output_path.exists():
        return parent_counts, accepted_parent_ids, accepted_chunks, total_tokens
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                deduplicator.add_existing(record["cleaned_text"], record["normalized_sha256"])
                parent_id = str(
                    record.get("parent_document_id", record["document_id"])
                )
                parent_counts[parent_id] += 1
                accepted_parent_ids.add(parent_id)
                accepted_chunks += 1
                total_tokens += int(record["token_count"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid existing output JSONL at line {line_number}") from error
    return parent_counts, accepted_parent_ids, accepted_chunks, total_tokens


def _increment(progress: PreparationProgress, reason: str) -> None:
    assert progress.rejection_counts is not None
    progress.rejection_counts[reason] = progress.rejection_counts.get(reason, 0) + 1


def summarize_review_diversity(
    output_path: Path,
    *,
    maximum_chunk_tokens: int,
    quality_warning_threshold: int,
) -> dict[str, Any]:
    parent_counts: Counter[str] = Counter()
    tokens: list[int] = []
    titles: set[str] = set()
    source_rows: set[int] = set()
    warning_count = 0
    reference_flags = 0
    replacement_characters = 0
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            parent_counts[str(record["parent_document_id"])] += 1
            tokens.append(int(record["token_count"]))
            titles.add(str(record["title"]))
            source_rows.add(int(record["provenance_metadata"]["source_row_index"]))
            warnings = list(record.get("quality_warnings", []))
            warning_count += len(warnings)
            reference_flags += int("reference_section" in warnings)
            replacement_characters += str(record["cleaned_text"]).count("\ufffd")
    chunk_count = len(tokens)
    largest_count = max(parent_counts.values(), default=0)
    largest_proportion = largest_count / chunk_count if chunk_count else 0.0
    warnings: list[str] = []
    if len(parent_counts) < 10:
        warnings.append("fewer_than_10_parent_articles")
    if largest_proportion > 0.25:
        warnings.append("largest_article_exceeds_25_percent")
    if warning_count > quality_warning_threshold:
        warnings.append("quality_warning_threshold_exceeded")
    if replacement_characters:
        warnings.append("replacement_characters_present")
    if tokens and max(tokens) > maximum_chunk_tokens:
        warnings.append("chunk_maximum_exceeded")
    return {
        "distinct_parent_articles": len(parent_counts),
        "chunks_per_article": dict(sorted(parent_counts.items())),
        "chunk_tokens": {
            "minimum": min(tokens, default=0),
            "median": statistics.median(tokens) if tokens else 0,
            "mean": statistics.fmean(tokens) if tokens else 0.0,
            "maximum": max(tokens, default=0),
        },
        "titles_represented": sorted(titles),
        "source_row_indices": sorted(source_rows),
        "quality_warning_count": warning_count,
        "reference_section_flags": reference_flags,
        "replacement_character_count": replacement_characters,
        "largest_article_chunk_count": largest_count,
        "largest_article_proportion": largest_proportion,
        "warnings": warnings,
    }


def _restart(paths: Mapping[str, Path]) -> None:
    for key in ("output_jsonl", "manifest_json", "progress_json", "summary_json", "summary_text"):
        paths[key].unlink(missing_ok=True)


def prepare_wikimedia_pilot(
    config: WikimediaPreparationConfig,
    *,
    repository_root: Path,
    input_parquet: Path,
    acquisition_metadata: Mapping[str, Any] | None = None,
    resume: bool = False,
    restart: bool = False,
    tokenizer: Any | None = None,
    stop_after_rows: int | None = None,
) -> dict[str, Any]:
    validate_preparation_config(config)
    validate_registry_approval(config, repository_root)
    paths = resolve_paths(config, repository_root)
    if restart:
        _restart(paths)
    config_payload = config_to_dict(config)
    configuration_hash = canonical_json_hash(config_payload)
    output_path = paths["output_jsonl"]
    progress_path = paths["progress_json"]

    if progress_path.exists():
        if not resume or not config.resume_enabled:
            raise FileExistsError("Existing progress requires --resume or --restart")
        progress = load_progress(progress_path)
        validate_resume_identity(
            progress,
            configuration_hash=configuration_hash,
            source_id=config.source_id,
            pinned_revision=config.pinned_revision,
            shard_identifier=config.shard_identifier,
        )
        reconcile_output(output_path, progress.output_size_bytes)
    else:
        if resume:
            raise FileNotFoundError("--resume requested but no progress file exists")
        progress = PreparationProgress(
            configuration_hash=configuration_hash,
            source_id=config.source_id,
            pinned_revision=config.pinned_revision,
            shard_identifier=config.shard_identifier,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=False)
        save_progress(progress_path, progress)

    inspect_parquet_schema(input_parquet)
    tokenizer = tokenizer or load_vasu_tokenizer(paths["tokenizer"])
    prompt_evidence = (
        load_prompt_evidence(paths["prompts"], config.contamination_ngram_words)
        if config.contamination_check_enabled
        else tuple()
    )
    deduplicator = PilotDeduplicator(
        exact_enabled=config.exact_deduplication_enabled,
        near_enabled=config.near_deduplication_enabled,
        threshold=config.near_duplicate_similarity_threshold,
    )
    fineweb_status = fineweb_document_index_status(
        Path(repository_root), config.fineweb_index_path
    )
    if config.fineweb_index_required and not fineweb_status.get("training_ready", False):
        raise RuntimeError(
            "Default factual preparation is blocked until completed compatible "
            "FineWeb original and extension document-index coverage is available"
        )
    fineweb_index: FineWebDocumentIndex | FederatedFineWebDocumentIndex | None = None
    fineweb_hashes: set[str] = set()
    if fineweb_status["status"] == "available" and fineweb_status.get("index_paths"):
        fineweb_index = FederatedFineWebDocumentIndex(
            [Path(repository_root) / path for path in fineweb_status["index_paths"]]
        )
    elif fineweb_status["status"] == "available" and paths["fineweb_index"].is_file():
        fineweb_index = FineWebDocumentIndex(paths["fineweb_index"])
    elif fineweb_status["status"] == "available":
        # Backward compatibility for a legacy exact-hash-only artifact.
        fineweb_hashes, fineweb_status = load_fineweb_exact_hashes(Path(repository_root))
        deduplicator.exact_hashes.update(fineweb_hashes)
    (
        retained_parent_counts,
        output_parent_ids,
        output_chunk_count,
        output_token_count,
    ) = _load_existing_output(output_path, deduplicator)
    assert progress.accepted_parent_document_ids is not None
    progress_parent_ids = set(progress.accepted_parent_document_ids)
    if output_parent_ids != progress_parent_ids:
        raise ValueError(
            "Existing output parent IDs do not match committed preparation progress"
        )
    if output_chunk_count != progress.accepted_chunks:
        raise ValueError(
            "Existing output chunk count does not match committed preparation progress"
        )
    if output_token_count != progress.total_vasu_tokens:
        raise ValueError(
            "Existing output token count does not match committed preparation progress"
        )
    if len(progress_parent_ids) != progress.accepted_parent_documents:
        raise ValueError(
            "Committed parent-document count does not match its deterministic ID list"
        )
    started_at = utc_now()
    completion_reason = "source_exhausted"
    parquet_row_count = pq.ParquetFile(input_parquet).metadata.num_rows
    selected_row_indices = (
        select_review_row_indices(
            parquet_row_count,
            config.max_raw_examples,
            config.random_seed,
        )
        if config.review_sampling_enabled
        else tuple()
    )

    start_row = (
        progress.active_row_index
        if progress.active_row_index is not None
        else progress.last_processed_row + 1
    )
    stop_processing = False
    if config.review_sampling_enabled:
        row_iterator: Iterator[tuple[int, dict[str, Any]]] = iter_selected_parquet_rows(
            input_parquet,
            selected_row_indices,
            minimum_row=start_row,
        )
    else:
        row_iterator = enumerate(iter_parquet_rows(input_parquet, start_row), start=start_row)
    with output_path.open("ab") as output:
        for row_index, row in row_iterator:
            resuming_row = progress.active_row_index == row_index
            if not resuming_row:
                if progress.raw_examples >= config.max_raw_examples:
                    completion_reason = "raw_example_limit"
                    break
                progress.raw_examples += 1
                assert progress.inspected_row_indices is not None
                progress.inspected_row_indices.append(row_index)
                progress.active_row_index = row_index
                progress.next_chunk_index = 0
                save_progress(progress_path, progress)
            filtered = filter_wikimedia_record(
                row,
                minimum_characters=config.minimum_document_characters,
                maximum_characters=config.maximum_document_characters,
            )
            if not filtered.accepted:
                _increment(progress, filtered.reason or "filter_rejected")
                if filtered.reason in {
                    "replacement_character",
                    "control_character",
                    "low_confidence_encoding_corruption",
                    "implausible_non_ascii_corruption",
                }:
                    progress.quality_rejections += 1
                progress.last_processed_row = row_index
                progress.active_row_index = None
                progress.next_chunk_index = 0
                save_progress(progress_path, progress)
                continue
            if not resuming_row and bool(filtered.metadata.get("encoding_repaired")):
                progress.encoding_repairs += 1
            if config.chunking_enabled:
                chunk_arguments = {
                    "target_tokens": config.target_chunk_tokens,
                    "maximum_tokens": config.maximum_chunk_tokens,
                    "minimum_tokens": config.minimum_chunk_tokens,
                    "overlap_tokens": config.chunk_overlap_tokens,
                    "reference_section_behavior": config.reference_section_behavior,
                }
                if chunk_document is _DEFAULT_CHUNK_DOCUMENT:
                    chunking_result = chunk_document_detailed(
                        filtered.cleaned_text,
                        tokenizer,
                        **chunk_arguments,
                    )
                    chunks = list(chunking_result.chunks)
                    chunking_rejections = chunking_result.rejection_counts
                    excluded_reference_sections = (
                        chunking_result.excluded_reference_sections
                    )
                else:
                    # Preserve the existing focused-test/custom integration
                    # seam without weakening production's detailed result.
                    chunks = chunk_document(
                        filtered.cleaned_text,
                        tokenizer,
                        **chunk_arguments,
                    )
                    chunking_rejections = {}
                    excluded_reference_sections = 0
                if not resuming_row:
                    for reason, count in chunking_rejections.items():
                        for _ in range(count):
                            _increment(progress, reason)
                    progress.quality_rejections += chunking_rejections.get(
                        "below_minimum_chunk_tokens",
                        0,
                    )
                    progress.reference_section_detections += excluded_reference_sections
            else:
                chunks = [
                    TextChunk(
                        filtered.cleaned_text,
                        count_tokens(tokenizer, filtered.cleaned_text),
                        None,
                    )
                ]
            parent_document_id = str(row["id"])
            for chunk_index in range(progress.next_chunk_index, len(chunks)):
                chunk = chunks[chunk_index]
                chunk_id = f"{parent_document_id}:{chunk_index:04d}"
                if (
                    config.review_sampling_enabled
                    and retained_parent_counts[parent_document_id]
                    >= config.review_max_chunks_per_article
                ):
                    _increment(progress, "review_article_chunk_cap")
                    progress.next_chunk_index = chunk_index + 1
                    save_progress(progress_path, progress)
                    continue
                reference_section = is_reference_section(chunk.section_title)
                if reference_section:
                    progress.reference_section_detections += 1
                    if config.reference_section_behavior == "exclude":
                        _increment(progress, "reference_section_excluded")
                        progress.next_chunk_index = chunk_index + 1
                        save_progress(progress_path, progress)
                        continue
                chunk_warnings = list(filtered.metadata.get("quality_warnings", []))
                if reference_section and config.reference_section_behavior == "flag":
                    chunk_warnings.append("reference_section")
                # Chunking is a text transformation boundary of its own. Check
                # its final output before deduplication, accounting, or writing
                # so a corrupt chunk cannot contaminate downstream state while
                # clean siblings from the same article remain eligible.
                final_quality = assess_final_chunk(
                    chunk.text,
                    token_count=chunk.token_count,
                    minimum_tokens=config.minimum_chunk_tokens,
                    maximum_list_like_line_ratio=config.maximum_list_like_line_ratio,
                    minimum_prose_sentences_for_list_chunk=(
                        config.minimum_prose_sentences_for_list_chunk
                    ),
                    boundary_start_type=chunk.boundary_start_type,
                    boundary_end_type=chunk.boundary_end_type,
                    training_mode=config.reference_section_behavior == "exclude",
                )
                if final_quality.rejection_reason:
                    _increment(progress, final_quality.rejection_reason)
                    progress.quality_rejections += 1
                    progress.next_chunk_index = chunk_index + 1
                    save_progress(progress_path, progress)
                    continue
                chunk_warnings.extend(final_quality.warnings)
                exact_contamination, contamination_matches = check_contamination(
                    chunk.text, chunk_id, prompt_evidence
                )
                assert progress.contamination_matches is not None
                progress.contamination_matches.extend(contamination_matches)
                if exact_contamination:
                    _increment(progress, "contamination_exact_prompt")
                    progress.next_chunk_index = chunk_index + 1
                    save_progress(progress_path, progress)
                    continue
                duplicate_reason, fingerprint = deduplicator.classify(chunk.text)
                if duplicate_reason:
                    if duplicate_reason == "exact_duplicate" and fingerprint in fineweb_hashes:
                        duplicate_reason = "fineweb_exact_duplicate"
                    _increment(progress, duplicate_reason)
                    if duplicate_reason in {"exact_duplicate", "fineweb_exact_duplicate"}:
                        progress.exact_duplicates += 1
                    else:
                        progress.near_duplicates += 1
                    progress.next_chunk_index = chunk_index + 1
                    save_progress(progress_path, progress)
                    continue
                fineweb_overlap: dict[str, Any] | None = None
                if fineweb_index is not None:
                    overlap = match_text(fineweb_index, chunk.text, chunk_id=chunk_id)
                    if overlap.match_type:
                        fineweb_overlap = asdict(overlap)
                        fineweb_overlap["decision"] = overlap.decision.value
                        assert progress.fineweb_matches is not None
                        progress.fineweb_matches.append(fineweb_overlap)
                    if overlap.decision == MatchDecision.REJECT:
                        reason = (
                            "fineweb_exact_duplicate"
                            if overlap.match_type == "exact"
                            else "fineweb_near_duplicate"
                        )
                        _increment(progress, reason)
                        if overlap.match_type == "exact":
                            progress.exact_duplicates += 1
                        else:
                            progress.near_duplicates += 1
                        progress.next_chunk_index = chunk_index + 1
                        save_progress(progress_path, progress)
                        continue
                    if overlap.decision == MatchDecision.REVIEW:
                        chunk_warnings.append("fineweb_ambiguous_overlap")
                if (
                    parent_document_id not in progress_parent_ids
                    and progress.accepted_parent_documents
                    >= config.max_accepted_parent_documents
                ):
                    completion_reason = "parent_document_limit"
                    stop_processing = True
                    save_progress(progress_path, progress)
                    break
                if progress.accepted_chunks >= config.max_accepted_chunks:
                    completion_reason = "accepted_chunk_limit"
                    stop_processing = True
                    save_progress(progress_path, progress)
                    break
                if (
                    progress.total_vasu_tokens + chunk.token_count
                    > config.max_output_tokens
                ):
                    _increment(progress, "token_limit")
                    completion_reason = "token_limit"
                    stop_processing = True
                    save_progress(progress_path, progress)
                    break
                deduplicator.accept(chunk.text, fingerprint)
                output_record = {
                    "format_version": "wikimedia_pilot_document_v3",
                    "document_id": chunk_id,
                    "parent_document_id": parent_document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "section_title": chunk.section_title,
                    "boundary_start_type": chunk.boundary_start_type,
                    "boundary_end_type": chunk.boundary_end_type,
                    "overlap_characters": chunk.overlap_characters,
                    "source_id": config.source_id,
                    "source_revision": config.pinned_revision,
                    "source_shard": config.shard_identifier,
                    "source_url": str(row["url"]),
                    "title": str(row["title"]),
                    "cleaned_text": chunk.text,
                    "token_count": chunk.token_count,
                    "normalized_sha256": fingerprint,
                    "encoding_repaired": bool(filtered.metadata.get("encoding_repaired")),
                    "quality_warnings": list(dict.fromkeys(chunk_warnings)),
                    "filtering_metadata": filtered.metadata,
                    "quality_metrics": final_quality.metrics,
                    "fineweb_overlap": fineweb_overlap,
                    "provenance_metadata": {
                        "dataset_name": config.dataset_name,
                        "subset": config.subset,
                        "split": config.split,
                        "source_row_index": row_index,
                        "parent_document_id": parent_document_id,
                    },
                }
                encoded = (json.dumps(output_record, ensure_ascii=False) + "\n").encode("utf-8")
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
                if parent_document_id not in progress_parent_ids:
                    progress_parent_ids.add(parent_document_id)
                    progress.accepted_parent_document_ids.append(parent_document_id)
                    progress.accepted_parent_documents += 1
                progress.accepted_chunks += 1
                retained_parent_counts[parent_document_id] += 1
                progress.total_vasu_tokens += chunk.token_count
                progress.total_characters += len(chunk.text)
                progress.output_size_bytes = output.tell()
                progress.next_chunk_index = chunk_index + 1
                assert progress.seen_exact_hashes is not None
                progress.seen_exact_hashes.append(fingerprint)
                progress.near_duplicate_candidate_comparisons = deduplicator.candidate_comparisons
                save_progress(progress_path, progress)
                if stop_after_rows is not None and progress.raw_examples >= stop_after_rows:
                    raise InterruptedError("Synthetic interruption after committed progress")
            if stop_processing:
                break
            progress.last_processed_row = row_index
            progress.active_row_index = None
            progress.next_chunk_index = 0
            save_progress(progress_path, progress)
        else:
            completion_reason = "source_exhausted"

    progress.status = "complete"
    progress.completion_status = completion_reason
    save_progress(progress_path, progress)
    input_hash = sha256_file(input_parquet)
    tokenizer_hash = sha256_file(paths["tokenizer"])
    prompt_hash = sha256_file(paths["prompts"])
    output_hash = sha256_file(output_path)
    diversity = summarize_review_diversity(
        output_path,
        maximum_chunk_tokens=config.maximum_chunk_tokens,
        quality_warning_threshold=config.quality_warning_threshold,
    )
    rejected = sum((progress.rejection_counts or {}).values())
    manifest = {
        "format_version": "wikimedia_pilot_v4",
        "source_registry_id": config.source_id,
        "dataset_name": config.dataset_name,
        "source_revision": config.pinned_revision,
        "subset": config.subset,
        "split": config.split,
        "shard_identifier": config.shard_identifier,
        "acquisition_timestamp": (acquisition_metadata or {}).get("downloaded_at"),
        "input_file_size_bytes": input_parquet.stat().st_size,
        "input_sha256": input_hash,
        "configuration": config_payload,
        "configuration_hash": configuration_hash,
        "tokenizer_path": config.tokenizer_path,
        "tokenizer_sha256": tokenizer_hash,
        "prompt_suite_path": "evaluation/prompts.json",
        "prompt_suite_sha256": prompt_hash,
        "processed_row_range": [
            0 if progress.raw_examples else None,
            max(
                progress.last_processed_row,
                progress.active_row_index if progress.active_row_index is not None else -1,
            ),
        ],
        "row_selection": {
            "mode": "deterministic_broad_review"
            if config.review_sampling_enabled
            else "sequential",
            "strategy": "evenly_spaced_with_seeded_offset"
            if config.review_sampling_enabled
            else "sequential_from_start",
            "shard_row_count": parquet_row_count,
            "selected_row_indices": list(selected_row_indices),
            "inspected_row_indices": progress.inspected_row_indices,
        },
        "raw_examples": progress.raw_examples,
        "accepted_parent_documents": progress.accepted_parent_documents,
        "accepted_chunks": progress.accepted_chunks,
        "rejected_items": rejected,
        "rejection_reasons": progress.rejection_counts,
        "exact_duplicates": progress.exact_duplicates,
        "near_duplicates": progress.near_duplicates,
        "near_duplicate_candidate_comparisons": progress.near_duplicate_candidate_comparisons,
        "near_duplicate_similarity_threshold": config.near_duplicate_similarity_threshold,
        "encoding_repairs": progress.encoding_repairs,
        "quality_rejections": progress.quality_rejections,
        "reference_section_detections": progress.reference_section_detections,
        "review_diversity": diversity,
        "contamination_matches": progress.contamination_matches,
        "fineweb_cross_deduplication": fineweb_status,
        "fineweb_overlap_matches": progress.fineweb_matches,
        "total_characters": progress.total_characters,
        "total_vasu_tokens": progress.total_vasu_tokens,
        "tokens_per_parent_document": (
            progress.total_vasu_tokens / progress.accepted_parent_documents
            if progress.accepted_parent_documents else 0.0
        ),
        "tokens_per_chunk": (
            progress.total_vasu_tokens / progress.accepted_chunks
            if progress.accepted_chunks else 0.0
        ),
        "characters_per_token": (
            progress.total_characters / progress.total_vasu_tokens
            if progress.total_vasu_tokens else 0.0
        ),
        "output_artifact_paths": {"documents_jsonl": config.output_paths.output_jsonl},
        "output_artifact_hashes": {"documents_jsonl_sha256": output_hash},
        "completion_status": completion_reason,
        "resumability_status": "complete; progress retained for validation",
        "started_at": started_at,
        "completed_at": utc_now(),
    }
    summary = {
        "source_id": config.source_id,
        "shard_identifier": config.shard_identifier,
        "completion_status": completion_reason,
        "raw_examples": progress.raw_examples,
        "accepted_parent_documents": progress.accepted_parent_documents,
        "accepted_chunks": progress.accepted_chunks,
        "rejected_items": rejected,
        "total_vasu_tokens": progress.total_vasu_tokens,
        "total_characters": progress.total_characters,
        "rejection_reasons": progress.rejection_counts,
        "exact_duplicates": progress.exact_duplicates,
        "near_duplicates": progress.near_duplicates,
        "encoding_repairs": progress.encoding_repairs,
        "quality_rejections": progress.quality_rejections,
        "reference_section_detections": progress.reference_section_detections,
        "review_diversity": diversity,
        "contamination_match_count": len(progress.contamination_matches or []),
        "fineweb_cross_deduplication": fineweb_status,
        "fineweb_overlap_match_count": len(progress.fineweb_matches or []),
    }
    atomic_write_json(paths["manifest_json"], manifest)
    atomic_write_json(paths["summary_json"], summary)
    review_lines = ["", "Manual review sample (maximum 20 chunks)"]
    with output_path.open("r", encoding="utf-8") as review_input:
        for review_index, line in enumerate(review_input):
            if review_index >= 20:
                break
            record = json.loads(line)
            excerpt = " ".join(record["cleaned_text"][:500].split())
            review_lines.extend(
                [
                    "",
                    f"Title: {record['title']}",
                    f"Chunk: {record['chunk_id']}",
                    f"Tokens: {record['token_count']}",
                    f"Encoding repaired: {record['encoding_repaired']}",
                    f"Quality warnings: {record['quality_warnings']}",
                    f"Excerpt: {excerpt}",
                ]
            )
    atomic_write_text(
        paths["summary_text"],
        format_wikimedia_summary(summary, review_lines),
    )
    if fineweb_index is not None:
        fineweb_index.close()
    return manifest


def validate_preparation_output(
    config: WikimediaPreparationConfig,
    *,
    repository_root: Path,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    paths = resolve_paths(config, repository_root)
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    format_version = manifest.get("format_version")
    if format_version not in {
        "wikimedia_pilot_v2",
        "wikimedia_pilot_v3",
        "wikimedia_pilot_v4",
    }:
        raise ValueError(
            "Unsupported preparation output format; expected wikimedia_pilot_v2 "
            "through wikimedia_pilot_v4"
        )
    resolved_config = config_to_dict(config)
    if format_version in {"wikimedia_pilot_v2", "wikimedia_pilot_v3"}:
        stored_config = manifest.get("configuration")
        if not isinstance(stored_config, dict):
            raise ValueError("Legacy output manifest lacks its resolved configuration")
        stored_config = dict(stored_config)
        if format_version == "wikimedia_pilot_v2":
            legacy_limit = stored_config.pop(LEGACY_CHUNK_LIMIT_FIELD, None)
            if legacy_limit is not None:
                stored_config.setdefault("max_accepted_chunks", legacy_limit)
                stored_config.setdefault(
                    "max_accepted_parent_documents",
                    stored_config.get("max_raw_examples"),
                )
        stored_config.setdefault("maximum_list_like_line_ratio", 0.8)
        stored_config.setdefault("minimum_prose_sentences_for_list_chunk", 2)
        if stored_config != resolved_config:
            raise ValueError("Output manifest configuration mismatch")
    elif manifest.get("configuration_hash") != canonical_json_hash(resolved_config):
        raise ValueError("Output manifest configuration hash mismatch")
    output = paths["output_jsonl"]
    expected_hash = manifest["output_artifact_hashes"]["documents_jsonl_sha256"]
    if sha256_file(output) != expected_hash:
        raise ValueError("Output JSONL hash mismatch")
    documents = 0
    tokens = 0
    parent_ids: set[str] = set()
    tokenizer = tokenizer or load_vasu_tokenizer(paths["tokenizer"])
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    with output.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                document_id = str(record["document_id"])
                fingerprint = str(record["normalized_sha256"])
                token_count = int(record["token_count"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid output JSONL line {line_number}") from error
            if document_id in seen_ids or fingerprint in seen_hashes:
                raise ValueError(f"Duplicate document/hash in output at line {line_number}")
            seen_ids.add(document_id)
            seen_hashes.add(fingerprint)
            parent_ids.add(str(record.get("parent_document_id", document_id)))
            record_version = record.get("format_version")
            if record_version not in {
                "wikimedia_pilot_document_v2",
                "wikimedia_pilot_document_v3",
            }:
                raise ValueError(f"Unsupported document format at line {line_number}")
            required = (
                "parent_document_id",
                "chunk_id",
                "chunk_index",
                "chunk_count",
                "section_title",
                "encoding_repaired",
                "quality_warnings",
                "provenance_metadata",
            )
            missing = [field for field in required if field not in record]
            if missing:
                raise ValueError(f"Output JSONL line {line_number} missing fields: {missing}")
            if record_version == "wikimedia_pilot_document_v3":
                boundary_fields = (
                    "boundary_start_type",
                    "boundary_end_type",
                    "overlap_characters",
                    "quality_metrics",
                )
                boundary_missing = [
                    field for field in boundary_fields if field not in record
                ]
                if boundary_missing:
                    raise ValueError(
                        f"Output JSONL line {line_number} missing boundary fields: "
                        f"{boundary_missing}"
                    )
            text = record.get("cleaned_text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Output JSONL line {line_number} has invalid text")
            if normalized_sha256(text) != fingerprint:
                raise ValueError(f"Output JSONL line {line_number} hash mismatch")
            exact_count = count_tokens(tokenizer, text)
            if token_count != exact_count:
                raise ValueError(f"Output JSONL line {line_number} token count mismatch")
            if token_count > config.maximum_chunk_tokens:
                raise ValueError(f"Output JSONL line {line_number} exceeds chunk maximum")
            if (
                record_version == "wikimedia_pilot_document_v3"
                and token_count < config.minimum_chunk_tokens
            ):
                raise ValueError(f"Output JSONL line {line_number} is below chunk minimum")
            if "\ufffd" in text:
                raise ValueError(f"Output JSONL line {line_number} contains replacement text")
            _, quality_rejection = assess_text_quality(text)
            if quality_rejection:
                raise ValueError(
                    f"Output JSONL line {line_number} fails quality validation: "
                    f"{quality_rejection}"
                )
            if record_version == "wikimedia_pilot_document_v3":
                start_type = record["boundary_start_type"]
                end_type = record["boundary_end_type"]
                if start_type not in BOUNDARY_TYPES or end_type not in BOUNDARY_TYPES:
                    raise ValueError(
                        f"Output JSONL line {line_number} has invalid boundary metadata"
                    )
                overlap_characters = record["overlap_characters"]
                if (
                    isinstance(overlap_characters, bool)
                    or not isinstance(overlap_characters, int)
                    or overlap_characters < 0
                ):
                    raise ValueError(
                        f"Output JSONL line {line_number} has invalid overlap metadata"
                    )
                if config.reference_section_behavior == "exclude" and (
                    is_reference_section(record.get("section_title"))
                    or contains_reference_section_heading(text)
                ):
                    raise ValueError(
                        f"Output JSONL line {line_number} retained a reference section"
                    )
                final_quality = assess_final_chunk(
                    text,
                    token_count=token_count,
                    minimum_tokens=config.minimum_chunk_tokens,
                    maximum_list_like_line_ratio=(
                        config.maximum_list_like_line_ratio
                    ),
                    minimum_prose_sentences_for_list_chunk=(
                        config.minimum_prose_sentences_for_list_chunk
                    ),
                    boundary_start_type=start_type,
                    boundary_end_type=end_type,
                    training_mode=config.reference_section_behavior == "exclude",
                )
                if final_quality.rejection_reason:
                    raise ValueError(
                        f"Output JSONL line {line_number} fails final quality validation: "
                        f"{final_quality.rejection_reason}"
                    )
            documents += 1
            tokens += token_count
    manifest_chunks = (
        manifest.get("accepted_chunks")
        if format_version in {"wikimedia_pilot_v3", "wikimedia_pilot_v4"}
        else manifest.get("accepted_documents", manifest.get("accepted_chunks"))
    )
    if documents != manifest_chunks or tokens != manifest["total_vasu_tokens"]:
        raise ValueError("Output JSONL counts do not match manifest")
    if (
        format_version in {"wikimedia_pilot_v3", "wikimedia_pilot_v4"}
        and len(parent_ids) != manifest["accepted_parent_documents"]
    ):
        raise ValueError("Output parent-document count does not match manifest")
    return {
        "valid": True,
        "parent_documents": len(parent_ids),
        "chunks": documents,
        "tokens": tokens,
        "sha256": expected_hash,
    }
