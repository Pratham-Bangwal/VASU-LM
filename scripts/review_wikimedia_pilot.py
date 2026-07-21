"""Deterministic, read-only manual review tooling for Wikimedia pilot chunks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.preparation.reporting import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    canonical_json_hash,
    sha256_file,
)
from vasu.data.preparation.chunking import (  # noqa: E402
    is_reference_section_heading,
)


CONFIG_FORMAT = "wikimedia_manual_review_config_v1"
REPORT_FORMAT = "wikimedia_manual_review_v1"
VALID_STATUSES = ("pending", "pass", "minor_issue", "reject")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "â€™", "â€œ", "â€", "ðŸ")
SELECTION_GROUPS = (
    "random",
    "shortest",
    "longest",
    "near_maximum",
    "evenly_spaced",
    "distinct_parent",
    "quality_warning",
    "reference_section",
    "suspicious_metadata",
)
CHECKLIST = (
    "No mojibake or replacement characters",
    "No joined-word corruption",
    "Chunk begins naturally",
    "Chunk ends naturally",
    "No broken sentence caused by chunking",
    "No citation/reference debris",
    "No table or navigation garbage",
    "No excessive headings or lists",
    "No repeated prose",
    "No malformed formulas, dates, or units",
    "Useful factual or explanatory content",
    "Provenance fields are present",
)
BLOCKING_PRECHECKS = {
    "control_character",
    "empty_text",
    "missing_parent_or_chunk_provenance",
    "missing_title",
    "missing_url",
    "mojibake_marker",
    "preview_generation_failure",
    "replacement_character",
    "token_count_above_maximum",
}


class ReviewError(ValueError):
    """Raised when review configuration, data, or state is invalid."""


class ReviewValidationError(ReviewError):
    """Raised when a manual review does not satisfy acceptance policy."""


@dataclass(frozen=True)
class ReviewConfig:
    format_version: str
    input_jsonl: Path
    input_sha256: str
    manifest_path: Path
    output_json: Path
    output_text: Path
    random_seed: int
    random_sample_count: int
    shortest_sample_count: int
    longest_sample_count: int
    near_max_sample_count: int
    evenly_spaced_sample_count: int
    distinct_parent_sample_count: int
    preview_start_characters: int
    preview_end_characters: int
    minimum_token_threshold: int
    maximum_token_count: int
    configuration_hash: str


@dataclass(frozen=True)
class DatasetChunk:
    jsonl_line_number: int
    record: Mapping[str, Any]

    @property
    def chunk_id(self) -> str:
        return str(self.record.get("chunk_id", ""))

    @property
    def parent_document_id(self) -> str:
        return str(self.record.get("parent_document_id", ""))

    @property
    def token_count(self) -> int:
        value = self.record.get("token_count")
        return value if isinstance(value, int) and not isinstance(value, bool) else -1


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[DatasetChunk, ...]
    reasons: Mapping[int, tuple[str, ...]]
    selection_counts: Mapping[str, int]
    duplicate_selections_removed: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(value: object, repository_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReviewError("Review paths must be non-empty strings")
    path = Path(value)
    return path if path.is_absolute() else (repository_root / path).resolve()


def _positive_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReviewError(f"{field} must be an integer")
    if value <= 0:
        raise ReviewError(f"{field} must be greater than zero")
    return value


def load_review_config(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> ReviewConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReviewError(f"Review configuration does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReviewError(f"Review configuration is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ReviewError("Review configuration must be a JSON object")
    if payload.get("format_version") != CONFIG_FORMAT:
        raise ReviewError(
            f"Unsupported review configuration format: {payload.get('format_version')!r}"
        )
    input_sha256 = payload.get("input_sha256")
    if not isinstance(input_sha256, str) or not SHA256_PATTERN.fullmatch(input_sha256):
        raise ReviewError("input_sha256 must be a lowercase SHA-256 hash")
    seed = payload.get("random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReviewError("random_seed must be an integer")
    config_hash = canonical_json_hash(payload)
    return ReviewConfig(
        format_version=CONFIG_FORMAT,
        input_jsonl=_resolve_path(payload.get("input_jsonl"), repository_root),
        input_sha256=input_sha256,
        manifest_path=_resolve_path(payload.get("manifest_path"), repository_root),
        output_json=_resolve_path(payload.get("output_json"), repository_root),
        output_text=_resolve_path(payload.get("output_text"), repository_root),
        random_seed=seed,
        random_sample_count=_positive_int(payload, "random_sample_count"),
        shortest_sample_count=_positive_int(payload, "shortest_sample_count"),
        longest_sample_count=_positive_int(payload, "longest_sample_count"),
        near_max_sample_count=_positive_int(payload, "near_max_sample_count"),
        evenly_spaced_sample_count=_positive_int(
            payload, "evenly_spaced_sample_count"
        ),
        distinct_parent_sample_count=_positive_int(
            payload, "distinct_parent_sample_count"
        ),
        preview_start_characters=_positive_int(
            payload, "preview_start_characters"
        ),
        preview_end_characters=_positive_int(payload, "preview_end_characters"),
        minimum_token_threshold=_positive_int(payload, "minimum_token_threshold"),
        maximum_token_count=_positive_int(payload, "maximum_token_count"),
        configuration_hash=config_hash,
    )


def verify_input_hash(config: ReviewConfig) -> str:
    if not config.input_jsonl.is_file():
        raise ReviewError(f"Input JSONL does not exist: {config.input_jsonl}")
    actual = sha256_file(config.input_jsonl)
    if actual != config.input_sha256:
        raise ReviewError(
            "Input JSONL SHA-256 mismatch: "
            f"expected {config.input_sha256}, found {actual}"
        )
    return actual


def load_dataset(config: ReviewConfig) -> list[DatasetChunk]:
    verify_input_hash(config)
    chunks: list[DatasetChunk] = []
    seen_ids: set[str] = set()
    with config.input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewError(f"Invalid JSON on input line {line_number}") from exc
            if not isinstance(record, dict):
                raise ReviewError(f"Input line {line_number} is not a JSON object")
            chunk = DatasetChunk(line_number, record)
            if not chunk.chunk_id:
                raise ReviewError(f"Input line {line_number} has no chunk_id")
            if chunk.chunk_id in seen_ids:
                raise ReviewError(f"Duplicate chunk_id in input: {chunk.chunk_id}")
            if not isinstance(record.get("cleaned_text"), str):
                raise ReviewError(f"Input chunk {chunk.chunk_id} has no cleaned_text")
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
    if not chunks:
        raise ReviewError("Input JSONL contains no chunks")
    _validate_manifest(config, chunks)
    return chunks


def _validate_manifest(config: ReviewConfig, chunks: Sequence[DatasetChunk]) -> None:
    try:
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReviewError(f"Pilot manifest does not exist: {config.manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ReviewError(f"Pilot manifest is invalid JSON: {config.manifest_path}") from exc
    if manifest.get("accepted_chunks") != len(chunks):
        raise ReviewError("Pilot manifest accepted_chunks does not match input JSONL")
    parent_count = len({chunk.parent_document_id for chunk in chunks})
    if manifest.get("accepted_parent_documents") != parent_count:
        raise ReviewError(
            "Pilot manifest accepted_parent_documents does not match input JSONL"
        )
    artifact_hash = manifest.get("output_artifact_hashes", {}).get(
        "documents_jsonl_sha256"
    )
    if artifact_hash != config.input_sha256:
        raise ReviewError("Pilot manifest output hash does not match review input hash")


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    count = min(count, length)
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _quality_warnings(chunk: DatasetChunk) -> list[str]:
    warnings = chunk.record.get("quality_warnings", [])
    return [str(item) for item in warnings] if isinstance(warnings, list) else []


def _is_reference_section(chunk: DatasetChunk) -> bool:
    section = chunk.record.get("section_title")
    text = str(chunk.record.get("cleaned_text", ""))
    return (
        is_reference_section_heading(str(section or ""))
        or any(is_reference_section_heading(line) for line in text.splitlines())
        or "reference_section"
        in {item.casefold() for item in _quality_warnings(chunk)}
    )


def _has_suspicious_metadata(chunk: DatasetChunk) -> bool:
    record = chunk.record
    if record.get("encoding_repaired") is True:
        return True
    metadata = record.get("filtering_metadata")
    if not isinstance(metadata, dict):
        return True
    repeated_ratio = metadata.get("repeated_line_ratio", 0.0)
    if isinstance(repeated_ratio, (int, float)) and repeated_ratio > 0.1:
        return True
    metadata_warnings = metadata.get("quality_warnings", [])
    return bool(metadata_warnings)


def select_chunks(
    chunks: Sequence[DatasetChunk],
    config: ReviewConfig,
) -> SelectionResult:
    reasons: dict[int, set[str]] = defaultdict(set)
    selection_counts: dict[str, int] = {}
    selection_events = 0

    def add(group: str, selected: Iterable[DatasetChunk]) -> None:
        nonlocal selection_events
        materialized = list(selected)
        selection_counts[group] = len(materialized)
        selection_events += len(materialized)
        for chunk in materialized:
            reasons[chunk.jsonl_line_number].add(group)

    rng = random.Random(config.random_seed)
    random_count = min(config.random_sample_count, len(chunks))
    random_indices = rng.sample(range(len(chunks)), random_count)
    add("random", (chunks[index] for index in random_indices))

    add(
        "shortest",
        sorted(chunks, key=lambda item: (item.token_count, item.jsonl_line_number))[
            : config.shortest_sample_count
        ],
    )
    add(
        "longest",
        sorted(chunks, key=lambda item: (-item.token_count, item.jsonl_line_number))[
            : config.longest_sample_count
        ],
    )
    add(
        "near_maximum",
        sorted(
            chunks,
            key=lambda item: (
                abs(config.maximum_token_count - item.token_count),
                item.jsonl_line_number,
            ),
        )[: config.near_max_sample_count],
    )
    add(
        "evenly_spaced",
        (chunks[index] for index in _evenly_spaced_indices(
            len(chunks), config.evenly_spaced_sample_count
        )),
    )

    first_by_parent: dict[str, DatasetChunk] = {}
    for chunk in chunks:
        first_by_parent.setdefault(chunk.parent_document_id, chunk)
    parents = sorted(first_by_parent)
    parent_indices = _evenly_spaced_indices(
        len(parents), config.distinct_parent_sample_count
    )
    add("distinct_parent", (first_by_parent[parents[index]] for index in parent_indices))
    add("quality_warning", (chunk for chunk in chunks if _quality_warnings(chunk)))
    add("reference_section", (chunk for chunk in chunks if _is_reference_section(chunk)))
    add(
        "suspicious_metadata",
        (chunk for chunk in chunks if _has_suspicious_metadata(chunk)),
    )

    selected = tuple(
        chunk for chunk in chunks if chunk.jsonl_line_number in reasons
    )
    ordered_reasons = {
        line_number: tuple(
            group for group in SELECTION_GROUPS if group in selected_reasons
        )
        for line_number, selected_reasons in reasons.items()
    }
    return SelectionResult(
        selected=selected,
        reasons=ordered_reasons,
        selection_counts={group: selection_counts.get(group, 0) for group in SELECTION_GROUPS},
        duplicate_selections_removed=selection_events - len(selected),
    )


def build_previews(text: str, start_limit: int, end_limit: int) -> tuple[str, str]:
    start = text[:start_limit]
    if len(text) <= start_limit:
        return start, ""
    end_start = max(start_limit, len(text) - end_limit)
    return start, text[end_start:]


def _has_control_characters(text: str) -> bool:
    return any(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in text
    )


def _source_row_index(chunk: DatasetChunk) -> int | None:
    metadata = chunk.record.get("provenance_metadata")
    value = metadata.get("source_row_index") if isinstance(metadata, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _automatic_prechecks(
    chunk: DatasetChunk,
    config: ReviewConfig,
    duplicate_hashes: set[str],
) -> list[str]:
    record = chunk.record
    text = str(record.get("cleaned_text", ""))
    failures: list[str] = []
    if "\ufffd" in text:
        failures.append("replacement_character")
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        failures.append("mojibake_marker")
    if _has_control_characters(text):
        failures.append("control_character")
    if not text.strip():
        failures.append("empty_text")
    normalized_hash = record.get("normalized_sha256")
    if normalized_hash in duplicate_hashes:
        failures.append("duplicate_chunk_hash")
    if chunk.token_count > config.maximum_token_count:
        failures.append("token_count_above_maximum")
    if chunk.token_count < config.minimum_token_threshold:
        failures.append("below_minimum_token_threshold")
    if not str(record.get("title", "")).strip():
        failures.append("missing_title")
    if not str(record.get("source_url", "")).strip():
        failures.append("missing_url")
    if (
        not chunk.parent_document_id
        or not chunk.chunk_id
        or _source_row_index(chunk) is None
        or not isinstance(record.get("chunk_index"), int)
        or not isinstance(record.get("chunk_count"), int)
    ):
        failures.append("missing_parent_or_chunk_provenance")
    if _is_reference_section(chunk):
        failures.append("reference_section")
    return failures


def _review_record(
    chunk: DatasetChunk,
    reasons: Sequence[str],
    config: ReviewConfig,
    duplicate_hashes: set[str],
) -> dict[str, Any]:
    record = chunk.record
    text = str(record["cleaned_text"])
    prechecks = _automatic_prechecks(chunk, config, duplicate_hashes)
    try:
        preview_start, preview_end = build_previews(
            text,
            config.preview_start_characters,
            config.preview_end_characters,
        )
    except Exception as exc:  # defensive report boundary
        preview_start, preview_end = "", ""
        prechecks.append(f"preview_generation_failure:{type(exc).__name__}")
    return {
        "jsonl_line_number": chunk.jsonl_line_number,
        "chunk_id": chunk.chunk_id,
        "parent_document_id": chunk.parent_document_id,
        "chunk_index": record.get("chunk_index"),
        "chunk_count": record.get("chunk_count"),
        "title": record.get("title"),
        "source_url": record.get("source_url"),
        "source_row_index": _source_row_index(chunk),
        "section_title": record.get("section_title"),
        "token_count": chunk.token_count,
        "quality_warnings": _quality_warnings(chunk),
        "encoding_repaired": record.get("encoding_repaired"),
        "normalized_sha256": record.get("normalized_sha256"),
        "selection_reasons": list(reasons),
        "automatic_prechecks": prechecks,
        "preview_start": preview_start,
        "preview_end": preview_end,
        "review_status": "pending",
        "review_notes": "",
    }


def _add_duplicate_preview_prechecks(samples: list[dict[str, Any]]) -> None:
    previews: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (sample["preview_start"], sample["preview_end"])
        if any(key):
            previews[key].append(sample)
    for matching in previews.values():
        if len(matching) > 1:
            for sample in matching:
                sample["automatic_prechecks"].append("exact_duplicate_preview")


def _status_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(sample.get("review_status")) for sample in samples)
    return {status: counts.get(status, 0) for status in VALID_STATUSES}


def _build_summary(
    config: ReviewConfig,
    chunks: Sequence[DatasetChunk],
    result: SelectionResult,
    samples: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    token_counts = [int(sample["token_count"]) for sample in samples]
    status_counts = _status_counts(samples)
    return {
        "input_dataset_path": str(config.input_jsonl),
        "input_sha256": config.input_sha256,
        "manifest_path": str(config.manifest_path),
        "configuration_hash": config.configuration_hash,
        "random_seed": config.random_seed,
        "total_chunks_in_dataset": len(chunks),
        "total_parent_articles": len({chunk.parent_document_id for chunk in chunks}),
        "total_sampled_chunks": len(samples),
        "selection_counts_by_category": dict(result.selection_counts),
        "duplicate_selections_removed": result.duplicate_selections_removed,
        "minimum_sampled_token_count": min(token_counts),
        "median_sampled_token_count": statistics.median(token_counts),
        "mean_sampled_token_count": statistics.fmean(token_counts),
        "maximum_sampled_token_count": max(token_counts),
        "titles_represented": sorted({str(sample["title"]) for sample in samples}),
        "parent_articles_represented": len(
            {str(sample["parent_document_id"]) for sample in samples}
        ),
        "quality_warning_count": sum(bool(sample["quality_warnings"]) for sample in samples),
        "reference_section_count": sum(
            "reference_section" in sample["automatic_prechecks"] for sample in samples
        ),
        "automatic_precheck_failure_count": sum(
            bool(sample["automatic_prechecks"]) for sample in samples
        ),
        "pending_review_count": status_counts["pending"],
        "pass_count": status_counts["pass"],
        "minor_issue_count": status_counts["minor_issue"],
        "reject_count": status_counts["reject"],
        "generation_timestamp": generated_at,
    }


def _config_for_report(config: ReviewConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("configuration_hash")
    for key in ("input_jsonl", "manifest_path", "output_json", "output_text"):
        payload[key] = str(payload[key])
    return payload


def render_text_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "VASU Wikimedia pilot manual review",
        "=" * 80,
        f"Input dataset path: {summary['input_dataset_path']}",
        f"Input SHA-256: {summary['input_sha256']}",
        f"Manifest path: {summary['manifest_path']}",
        f"Configuration hash: {summary['configuration_hash']}",
        f"Random seed: {summary['random_seed']}",
        f"Total chunks in dataset: {summary['total_chunks_in_dataset']}",
        f"Total parent articles: {summary['total_parent_articles']}",
        f"Total sampled chunks: {summary['total_sampled_chunks']}",
        f"Selection counts by category: {summary['selection_counts_by_category']}",
        f"Duplicate selections removed: {summary['duplicate_selections_removed']}",
        "Sample token min/median/mean/max: "
        f"{summary['minimum_sampled_token_count']}/"
        f"{summary['median_sampled_token_count']}/"
        f"{summary['mean_sampled_token_count']:.3f}/"
        f"{summary['maximum_sampled_token_count']}",
        f"Titles represented: {summary['titles_represented']}",
        f"Parent articles represented: {summary['parent_articles_represented']}",
        f"Quality-warning count: {summary['quality_warning_count']}",
        f"Reference-section count: {summary['reference_section_count']}",
        f"Automatic precheck failures: {summary['automatic_precheck_failure_count']}",
        f"Pending review count: {summary['pending_review_count']}",
        f"Pass count: {summary['pass_count']}",
        f"Minor-issue count: {summary['minor_issue_count']}",
        f"Reject count: {summary['reject_count']}",
        f"Generation timestamp: {summary['generation_timestamp']}",
    ]
    if report.get("updated_at"):
        lines.append(f"Last status update: {report['updated_at']}")
    for sample in report["samples"]:
        lines.extend(
            [
                "",
                "-" * 80,
                f"Chunk ID: {sample['chunk_id']} (JSONL line {sample['jsonl_line_number']})",
                f"Parent: {sample['parent_document_id']} | chunk "
                f"{sample['chunk_index']}/{sample['chunk_count']}",
                f"Title: {sample['title']}",
                f"URL: {sample['source_url']}",
                f"Source row: {sample['source_row_index']}",
                f"Section: {sample['section_title']}",
                f"Tokens: {sample['token_count']}",
                f"Selection reasons: {sample['selection_reasons']}",
                f"Quality warnings: {sample['quality_warnings']}",
                f"Automatic prechecks: {sample['automatic_prechecks']}",
                f"Review status: {sample['review_status']}",
                f"Review notes: {sample['review_notes']}",
                "Preview start:",
                sample["preview_start"],
                "Preview end:",
                sample["preview_end"] or "[not needed; start preview contains the short chunk]",
                "Checklist:",
                *(f"[ ] {item}" for item in CHECKLIST),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _write_report(config: ReviewConfig, report: Mapping[str, Any]) -> None:
    atomic_write_json(config.output_json, report)
    atomic_write_text(config.output_text, render_text_report(report))


def generate_review(
    config_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    force: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    config = load_review_config(config_path, repository_root=repository_root)
    if (config.output_json.exists() or config.output_text.exists()) and not force:
        raise ReviewError(
            "Review report already exists; refusing to regenerate a possibly reviewed sample. "
            "Use --force only when an intentional fresh selection is required."
        )
    chunks = load_dataset(config)
    result = select_chunks(chunks, config)
    hash_counts = Counter(
        str(chunk.record.get("normalized_sha256", "")) for chunk in chunks
    )
    duplicate_hashes = {value for value, count in hash_counts.items() if value and count > 1}
    samples = [
        _review_record(
            chunk,
            result.reasons[chunk.jsonl_line_number],
            config,
            duplicate_hashes,
        )
        for chunk in result.selected
    ]
    _add_duplicate_preview_prechecks(samples)
    timestamp = generated_at or utc_now()
    report = {
        "format_version": REPORT_FORMAT,
        "configuration": _config_for_report(config),
        "summary": _build_summary(config, chunks, result, samples, timestamp),
        "samples": samples,
    }
    _write_report(config, report)
    return report


def _load_report(config: ReviewConfig) -> dict[str, Any]:
    try:
        report = json.loads(config.output_json.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReviewError(f"Review report does not exist: {config.output_json}") from exc
    except json.JSONDecodeError as exc:
        raise ReviewError(f"Review report is invalid JSON: {config.output_json}") from exc
    if report.get("format_version") != REPORT_FORMAT:
        raise ReviewError("Unsupported manual-review report format")
    summary = report.get("summary", {})
    if summary.get("configuration_hash") != config.configuration_hash:
        raise ReviewError("Review report configuration hash does not match config")
    if summary.get("input_sha256") != config.input_sha256:
        raise ReviewError("Review report input hash does not match config")
    return report


def _refresh_status_summary(report: dict[str, Any]) -> None:
    counts = _status_counts(report["samples"])
    report["summary"]["pending_review_count"] = counts["pending"]
    report["summary"]["pass_count"] = counts["pass"]
    report["summary"]["minor_issue_count"] = counts["minor_issue"]
    report["summary"]["reject_count"] = counts["reject"]


def set_review_status(
    config_path: Path,
    chunk_id: str,
    status: str,
    notes: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    updated_at: str | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ReviewError(
            f"Invalid review status {status!r}; expected one of {', '.join(VALID_STATUSES)}"
        )
    if status in {"minor_issue", "reject"} and not notes.strip():
        raise ReviewError(f"{status} requires non-empty review notes")
    config = load_review_config(config_path, repository_root=repository_root)
    report = _load_report(config)
    matching = [sample for sample in report["samples"] if sample["chunk_id"] == chunk_id]
    if not matching:
        raise ReviewError(f"Unknown sampled chunk ID: {chunk_id}")
    if len(matching) != 1:
        raise ReviewError(f"Review report contains duplicate chunk ID: {chunk_id}")
    matching[0]["review_status"] = status
    matching[0]["review_notes"] = notes
    report["updated_at"] = updated_at or utc_now()
    _refresh_status_summary(report)
    _write_report(config, report)
    return report


def review_summary(
    config_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Mapping[str, Any]:
    config = load_review_config(config_path, repository_root=repository_root)
    return _load_report(config)["summary"]


def validate_review(
    config_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    config = load_review_config(config_path, repository_root=repository_root)
    verify_input_hash(config)
    report = _load_report(config)
    errors: list[str] = []
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("review report contains no samples")
        samples = []
    for sample in samples:
        if "cleaned_text" in sample:
            errors.append(f"{sample.get('chunk_id')}: complete text leaked into report")
        if len(str(sample.get("preview_start", ""))) > config.preview_start_characters:
            errors.append(f"{sample.get('chunk_id')}: start preview exceeds limit")
        if len(str(sample.get("preview_end", ""))) > config.preview_end_characters:
            errors.append(f"{sample.get('chunk_id')}: end preview exceeds limit")
        status = sample.get("review_status")
        if status not in VALID_STATUSES:
            errors.append(f"{sample.get('chunk_id')}: invalid review status")
        elif status == "pending":
            errors.append(f"{sample.get('chunk_id')}: review is pending")
        elif status == "reject":
            errors.append(f"{sample.get('chunk_id')}: review was rejected")
        elif status == "minor_issue" and not str(sample.get("review_notes", "")).strip():
            errors.append(f"{sample.get('chunk_id')}: minor issue is undocumented")
        prechecks = {
            str(value).split(":", 1)[0]
            for value in sample.get("automatic_prechecks", [])
        }
        blocking = sorted(prechecks & BLOCKING_PRECHECKS)
        if blocking:
            errors.append(
                f"{sample.get('chunk_id')}: unresolved blocking prechecks {blocking}"
            )
    _refresh_status_summary(report)
    if report["summary"]["reject_count"]:
        errors.append("reject count must be zero")
    if errors:
        raise ReviewValidationError("Manual review validation failed:\n- " + "\n- ".join(errors))
    return report


def _print_summary(summary: Mapping[str, Any]) -> None:
    for key, value in summary.items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", action="store_true")
    action.add_argument("--set-status", action="store_true")
    action.add_argument("--summary", action="store_true")
    action.add_argument("--validate", action="store_true")
    parser.add_argument("--chunk-id")
    parser.add_argument("--status", choices=VALID_STATUSES)
    parser.add_argument("--notes", default="")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.generate:
            report = generate_review(args.config, force=args.force)
            _print_summary(report["summary"])
        elif args.set_status:
            if not args.chunk_id or not args.status:
                raise ReviewError("--set-status requires --chunk-id and --status")
            report = set_review_status(
                args.config,
                args.chunk_id,
                args.status,
                args.notes,
            )
            _print_summary(report["summary"])
        elif args.summary:
            _print_summary(review_summary(args.config))
        else:
            report = validate_review(args.config)
            print(
                "Manual review valid: "
                f"samples={report['summary']['total_sampled_chunks']}"
            )
    except ReviewError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
