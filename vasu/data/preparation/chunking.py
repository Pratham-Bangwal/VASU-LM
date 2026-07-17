"""Deterministic exact-token document chunking."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    text: str
    token_count: int
    section_title: str | None


def _decode(tokenizer: Any, ids: list[int]) -> str:
    if not hasattr(tokenizer, "decode"):
        raise TypeError("Tokenizer must expose decode() for token-boundary chunking")
    return str(tokenizer.decode(ids)).strip()


def _is_heading(paragraph: str) -> bool:
    words = paragraph.split()
    return (
        "\n" not in paragraph
        and bool(paragraph)
        and paragraph[0].isupper()
        and 1 <= len(words) <= 12
        and len(paragraph) <= 120
        and not paragraph.rstrip().endswith((".", "!", "?", ";", ":", ","))
    )


def _atomic_units(text: str, tokenizer: Any, maximum_tokens: int) -> list[tuple[str, str | None]]:
    units: list[tuple[str, str | None]] = []
    section: str | None = None
    for paragraph in (part.strip() for part in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        if _is_heading(paragraph):
            section = paragraph
            continue
        if len(tokenizer.encode(paragraph)) <= maximum_tokens:
            units.append((paragraph, section))
            continue
        sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(paragraph) if part.strip()]
        for sentence in sentences:
            ids = list(tokenizer.encode(sentence))
            if len(ids) <= maximum_tokens:
                units.append((sentence, section))
                continue
            for start in range(0, len(ids), maximum_tokens):
                piece = _decode(tokenizer, ids[start : start + maximum_tokens])
                if piece:
                    units.append((piece, section))
    return units


def chunk_document(
    text: str,
    tokenizer: Any,
    *,
    target_tokens: int,
    maximum_tokens: int,
    minimum_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    if not 0 <= overlap_tokens < minimum_tokens <= target_tokens <= maximum_tokens:
        raise ValueError(
            "Chunk limits must satisfy 0 <= overlap < minimum <= target <= maximum"
        )
    units = _atomic_units(text, tokenizer, maximum_tokens)
    chunks: list[TextChunk] = []
    current = ""
    current_section: str | None = None

    def emit() -> None:
        nonlocal current, current_section
        value = current.strip()
        if value:
            count = len(tokenizer.encode(value))
            if count > maximum_tokens:
                raise AssertionError("Chunker emitted an oversized chunk")
            chunks.append(TextChunk(value, count, current_section))
        current = ""
        current_section = None

    for unit, section in units:
        if current and section is not None and section != current_section:
            emit()
        candidate = unit if not current else f"{current}\n\n{unit}"
        candidate_count = len(tokenizer.encode(candidate))
        if current and (candidate_count > maximum_tokens or len(tokenizer.encode(current)) >= target_tokens):
            previous_ids = list(tokenizer.encode(current))
            previous_section = current_section
            emit()
            overlap = _decode(tokenizer, previous_ids[-overlap_tokens:]) if overlap_tokens else ""
            current = overlap
            current_section = previous_section
            candidate = unit if not current else f"{current}\n\n{unit}"
            if len(tokenizer.encode(candidate)) > maximum_tokens:
                current = ""
                current_section = None
                candidate = unit
        current = candidate
        current_section = current_section or section
    emit()

    if len(chunks) >= 2 and chunks[-1].token_count < minimum_tokens:
        merged = f"{chunks[-2].text}\n\n{chunks[-1].text}"
        merged_count = len(tokenizer.encode(merged))
        if merged_count <= maximum_tokens:
            chunks[-2:] = [
                TextChunk(merged, merged_count, chunks[-2].section_title)
            ]
    return chunks
