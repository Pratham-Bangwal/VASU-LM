"""Small, interruption-safe reporting helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Retain atomic replace semantics across transient Windows file locks."""
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def format_wikimedia_summary(
    summary: Mapping[str, Any],
    review_lines: list[str],
) -> str:
    """Render the unambiguous human-readable Wikimedia preparation summary."""
    diversity = summary["review_diversity"]
    lines = [
        "VASU Wikimedia pilot preparation",
        f"Completion status: {summary['completion_status']}",
        f"Raw examples: {summary['raw_examples']}",
        f"Accepted parent documents: {summary['accepted_parent_documents']}",
        f"Accepted chunks: {summary['accepted_chunks']}",
        f"Rejected items: {summary['rejected_items']}",
        f"VASU tokens: {summary['total_vasu_tokens']}",
        f"FineWeb cross-deduplication: {summary['fineweb_cross_deduplication']['status']}",
        f"Distinct parent count: {diversity['distinct_parent_articles']}",
        f"Chunks per parent article: {diversity['chunks_per_article']}",
        f"Largest parent-article contribution: {diversity['largest_article_chunk_count']} "
        f"chunks ({diversity['largest_article_proportion']:.6f})",
        f"Chunk token min/median/mean/max: {diversity['chunk_tokens']}",
        f"Titles represented: {diversity['titles_represented']}",
        f"Source row indices: {diversity['source_row_indices']}",
        f"Encoding repairs: {summary['encoding_repairs']}",
        f"Quality warnings: {diversity['quality_warning_count']}",
        f"Quality rejections: {summary['quality_rejections']}",
        f"Reference-section flags: {diversity['reference_section_flags']}",
        f"Contamination matches: {summary['contamination_match_count']}",
        f"Review warnings: {diversity['warnings']}",
        *review_lines,
        "",
    ]
    return "\n".join(lines)
