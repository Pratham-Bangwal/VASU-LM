"""Deterministic, Unicode-safe, structure-aware document chunking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
HEADING_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
REFERENCE_SECTION_TITLES = frozenset(
    {
        "bibliography",
        "citations",
        "editions",
        "external links",
        "further reading",
        "general and cited references",
        "notes",
        "publications",
        "references",
        "see also",
        "sources",
        "works cited",
    }
)
BOUNDARY_TYPES = frozenset(
    {"section", "paragraph", "sentence", "word_fallback", "token_fallback"}
)


@dataclass(frozen=True)
class TextChunk:
    text: str
    token_count: int
    section_title: str | None
    boundary_start_type: str = "paragraph"
    boundary_end_type: str = "paragraph"
    overlap_characters: int = 0


@dataclass(frozen=True)
class ChunkingResult:
    chunks: tuple[TextChunk, ...]
    rejection_counts: dict[str, int]
    excluded_reference_sections: int


@dataclass(frozen=True)
class _AtomicUnit:
    text: str
    section_title: str | None
    boundary_start_type: str
    boundary_end_type: str


def normalize_section_heading(value: str) -> str:
    """Canonicalize a standalone section heading without matching prose."""
    normalized = HEADING_PUNCTUATION.sub(" ", value.casefold())
    return " ".join(normalized.split())


def is_reference_section_heading(value: str | None) -> bool:
    return bool(value) and normalize_section_heading(value) in REFERENCE_SECTION_TITLES


def contains_reference_section_heading(text: str) -> bool:
    return any(is_reference_section_heading(line) for line in text.splitlines())


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def _largest_prefix_length(text: str, tokenizer: Any, maximum_tokens: int) -> int:
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
    return low


def _safe_source_split(text: str, maximum_length: int) -> tuple[str, str, str]:
    """Split at a source character boundary, preferring a full word."""
    for position in range(maximum_length, 0, -1):
        before = text[position - 1]
        after = text[position] if position < len(text) else ""
        if not after or not (before.isalnum() and after.isalnum()):
            return text[:position].rstrip(), text[position:].lstrip(), "word_fallback"
    # A single overlong alphanumeric token has no safe lexical boundary. It is
    # retained only as an explicitly marked fallback so training policy can
    # reject it rather than silently emitting a mid-word boundary.
    return text[:maximum_length], text[maximum_length:], "token_fallback"


def _split_overlong_sentence(
    text: str,
    tokenizer: Any,
    maximum_tokens: int,
    start_type: str,
    end_type: str,
) -> list[tuple[str, str, str]]:
    pieces: list[tuple[str, str, str]] = []
    remaining = text
    first = True
    while remaining:
        if _token_count(tokenizer, remaining) <= maximum_tokens:
            piece = remaining.strip()
            if piece:
                pieces.append(
                    (piece, start_type if first else "word_fallback", end_type)
                )
            break
        maximum_length = _largest_prefix_length(remaining, tokenizer, maximum_tokens)
        piece, remaining, fallback = _safe_source_split(remaining, maximum_length)
        if not piece:
            raise ValueError("Unable to construct a non-empty bounded text chunk")
        pieces.append((piece, start_type if first else fallback, fallback))
        first = False
    return pieces


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


def _paragraph_units(
    paragraph: str,
    section: str | None,
    tokenizer: Any,
    maximum_tokens: int,
) -> list[_AtomicUnit]:
    if _token_count(tokenizer, paragraph) <= maximum_tokens:
        return [_AtomicUnit(paragraph, section, "paragraph", "paragraph")]
    sentences = [
        part.strip() for part in SENTENCE_BOUNDARY.split(paragraph) if part.strip()
    ]
    units: list[_AtomicUnit] = []
    for index, sentence in enumerate(sentences):
        start_type = "paragraph" if index == 0 else "sentence"
        end_type = "paragraph" if index == len(sentences) - 1 else "sentence"
        if _token_count(tokenizer, sentence) <= maximum_tokens:
            units.append(_AtomicUnit(sentence, section, start_type, end_type))
            continue
        units.extend(
            _AtomicUnit(piece, section, piece_start, piece_end)
            for piece, piece_start, piece_end in _split_overlong_sentence(
                sentence,
                tokenizer,
                maximum_tokens,
                start_type,
                end_type,
            )
        )
    return units


def _atomic_units(
    text: str,
    tokenizer: Any,
    maximum_tokens: int,
    reference_section_behavior: str,
) -> tuple[list[_AtomicUnit], int]:
    units: list[_AtomicUnit] = []
    section: str | None = None
    excluded_reference_sections = 0
    for raw_paragraph in (part.strip() for part in re.split(r"\n\s*\n", text)):
        if not raw_paragraph:
            continue
        lines = [line.strip() for line in raw_paragraph.splitlines() if line.strip()]
        reference_index = next(
            (
                index
                for index, line in enumerate(lines)
                if is_reference_section_heading(line)
            ),
            None,
        )
        if reference_index is not None:
            prefix = "\n".join(lines[:reference_index]).strip()
            if prefix:
                units.extend(_paragraph_units(prefix, section, tokenizer, maximum_tokens))
            reference_title = lines[reference_index]
            if reference_section_behavior == "exclude":
                # The source does not expose reliable heading hierarchy. Once a
                # reference appendix begins, conservatively exclude to article
                # end rather than leaking later bibliography/navigation text.
                excluded_reference_sections += 1
                break
            section = reference_title
            suffix = "\n".join(lines[reference_index + 1 :]).strip()
            if suffix:
                units.extend(_paragraph_units(suffix, section, tokenizer, maximum_tokens))
            continue
        if _is_heading(raw_paragraph):
            section = raw_paragraph
            continue
        units.extend(_paragraph_units(raw_paragraph, section, tokenizer, maximum_tokens))
    return units, excluded_reference_sections


def _sentence_aligned_overlap(
    text: str,
    tokenizer: Any,
    maximum_tokens: int,
) -> tuple[str, str]:
    """Return only an overlap that begins at a complete sentence boundary."""
    if maximum_tokens == 0:
        return "", "sentence"
    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[len(text) - middle :]
        if _token_count(tokenizer, candidate) <= maximum_tokens:
            low = middle
        else:
            high = middle - 1
    suffix = text[len(text) - low :] if low else ""
    for match in SENTENCE_BOUNDARY.finditer(suffix):
        candidate = suffix[match.end() :].strip()
        if candidate:
            return candidate, "sentence"
    # A mid-sentence overlap was the source of visibly malformed chunk starts.
    # Omitting it is safer than displaying an arbitrary fragment.
    return "", "sentence"


def _without_overlap(chunk: TextChunk) -> str:
    if chunk.overlap_characters <= 0:
        return chunk.text
    return chunk.text[chunk.overlap_characters :].lstrip()


def _merge_or_reject_small_chunks(
    chunks: list[TextChunk],
    tokenizer: Any,
    minimum_tokens: int,
    maximum_tokens: int,
) -> tuple[list[TextChunk], int]:
    retained: list[TextChunk] = []
    rejected = 0
    pending = list(chunks)
    index = 0
    while index < len(pending):
        chunk = pending[index]
        if chunk.token_count >= minimum_tokens:
            retained.append(chunk)
            index += 1
            continue
        unique_text = _without_overlap(chunk)
        if retained and retained[-1].section_title == chunk.section_title:
            merged_text = f"{retained[-1].text}\n\n{unique_text}".strip()
            merged_count = _token_count(tokenizer, merged_text)
            if merged_count <= maximum_tokens:
                previous = retained[-1]
                retained[-1] = TextChunk(
                    merged_text,
                    merged_count,
                    previous.section_title,
                    previous.boundary_start_type,
                    chunk.boundary_end_type,
                    previous.overlap_characters,
                )
                index += 1
                continue
        if index + 1 < len(pending) and pending[index + 1].section_title == chunk.section_title:
            following = pending[index + 1]
            following_unique = _without_overlap(following)
            merged_text = f"{unique_text}\n\n{following_unique}".strip()
            merged_count = _token_count(tokenizer, merged_text)
            if merged_count <= maximum_tokens:
                pending[index + 1] = TextChunk(
                    merged_text,
                    merged_count,
                    chunk.section_title,
                    chunk.boundary_start_type,
                    following.boundary_end_type,
                    chunk.overlap_characters,
                )
                index += 1
                continue
        rejected += 1
        index += 1
    return retained, rejected


def chunk_document_detailed(
    text: str,
    tokenizer: Any,
    *,
    target_tokens: int,
    maximum_tokens: int,
    minimum_tokens: int,
    overlap_tokens: int,
    reference_section_behavior: str = "keep",
) -> ChunkingResult:
    if not 0 <= overlap_tokens < minimum_tokens <= target_tokens <= maximum_tokens:
        raise ValueError(
            "Chunk limits must satisfy 0 <= overlap < minimum <= target <= maximum"
        )
    if reference_section_behavior not in {"keep", "flag", "exclude"}:
        raise ValueError("reference_section_behavior must be keep, flag, or exclude")
    units, excluded_sections = _atomic_units(
        text,
        tokenizer,
        maximum_tokens,
        reference_section_behavior,
    )
    chunks: list[TextChunk] = []
    current = ""
    current_section: str | None = None
    current_start_type = "paragraph"
    current_end_type = "paragraph"
    current_overlap_characters = 0

    def emit() -> None:
        nonlocal current, current_section, current_overlap_characters
        value = current.strip()
        if value:
            count = _token_count(tokenizer, value)
            if count > maximum_tokens:
                raise AssertionError("Chunker emitted an oversized chunk")
            chunks.append(
                TextChunk(
                    value,
                    count,
                    current_section,
                    current_start_type,
                    current_end_type,
                    current_overlap_characters,
                )
            )
        current = ""
        current_section = None
        current_overlap_characters = 0

    for unit in units:
        if current and unit.section_title is not None and unit.section_title != current_section:
            emit()
        candidate = unit.text if not current else f"{current}\n\n{unit.text}"
        candidate_count = _token_count(tokenizer, candidate)
        if current and (
            candidate_count > maximum_tokens
            or _token_count(tokenizer, current) >= target_tokens
        ):
            previous_text = current
            previous_section = current_section
            emit()
            overlap, overlap_start_type = _sentence_aligned_overlap(
                previous_text,
                tokenizer,
                overlap_tokens,
            )
            current = overlap
            current_section = previous_section
            current_start_type = overlap_start_type if overlap else unit.boundary_start_type
            current_end_type = unit.boundary_end_type
            current_overlap_characters = len(overlap)
            candidate = unit.text if not current else f"{current}\n\n{unit.text}"
            if _token_count(tokenizer, candidate) > maximum_tokens:
                current = ""
                current_section = None
                current_start_type = unit.boundary_start_type
                current_overlap_characters = 0
                candidate = unit.text
        elif not current:
            current_start_type = unit.boundary_start_type
            current_overlap_characters = 0
        current = candidate
        current_section = current_section or unit.section_title
        current_end_type = unit.boundary_end_type
    emit()

    retained, below_minimum = _merge_or_reject_small_chunks(
        chunks,
        tokenizer,
        minimum_tokens,
        maximum_tokens,
    )
    rejections = Counter()
    if below_minimum:
        rejections["below_minimum_chunk_tokens"] = below_minimum
    if excluded_sections:
        rejections["reference_section_excluded"] = excluded_sections
    return ChunkingResult(tuple(retained), dict(rejections), excluded_sections)


def chunk_document(
    text: str,
    tokenizer: Any,
    *,
    target_tokens: int,
    maximum_tokens: int,
    minimum_tokens: int,
    overlap_tokens: int,
    reference_section_behavior: str = "keep",
) -> list[TextChunk]:
    """Backward-compatible list-returning chunker entry point."""
    return list(
        chunk_document_detailed(
            text,
            tokenizer,
            target_tokens=target_tokens,
            maximum_tokens=maximum_tokens,
            minimum_tokens=minimum_tokens,
            overlap_tokens=overlap_tokens,
            reference_section_behavior=reference_section_behavior,
        ).chunks
    )
