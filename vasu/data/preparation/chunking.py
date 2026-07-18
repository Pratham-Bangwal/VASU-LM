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


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def _largest_prefix_within_limit(
    text: str,
    tokenizer: Any,
    maximum_tokens: int,
) -> str:
    """Return a source-text prefix without decoding a partial byte-token slice.

    The tokenizer is byte-level, so decoding an arbitrary token slice can begin
    or end within a multi-byte Unicode scalar and introduce U+FFFD. Searching
    source character boundaries keeps every emitted character traceable to the
    clean input text.
    """
    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if _token_count(tokenizer, text[:middle]) <= maximum_tokens:
            low = middle
        else:
            high = middle - 1
    if low == 0:
        raise ValueError("A single Unicode character exceeds the chunk token limit")
    return text[:low]


def _split_at_character_boundaries(
    text: str,
    tokenizer: Any,
    maximum_tokens: int,
) -> list[str]:
    pieces: list[str] = []
    remaining = text
    while remaining:
        if _token_count(tokenizer, remaining) <= maximum_tokens:
            piece = remaining
            remaining = ""
        else:
            piece = _largest_prefix_within_limit(
                remaining,
                tokenizer,
                maximum_tokens,
            )
            remaining = remaining[len(piece) :]
        piece = piece.strip()
        remaining = remaining.lstrip()
        if piece:
            pieces.append(piece)
    return pieces


def _source_suffix_within_limit(
    text: str,
    tokenizer: Any,
    maximum_tokens: int,
) -> str:
    """Return a character-aligned overlap suffix bounded by token count."""
    if maximum_tokens == 0:
        return ""
    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[len(text) - middle :]
        if _token_count(tokenizer, candidate) <= maximum_tokens:
            low = middle
        else:
            high = middle - 1
    return text[len(text) - low :].strip() if low else ""


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
            for piece in _split_at_character_boundaries(
                sentence,
                tokenizer,
                maximum_tokens,
            ):
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
            previous_text = current
            previous_section = current_section
            emit()
            overlap = _source_suffix_within_limit(
                previous_text,
                tokenizer,
                overlap_tokens,
            )
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
