"""Deterministic, conservative Wikimedia pilot filtering."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Mapping, Any


CITATION_PATTERN = re.compile(r"\[(?:\d{1,4}|citation needed)\]", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class FilterResult:
    accepted: bool
    reason: str | None
    cleaned_text: str
    metadata: dict[str, Any]


def comparison_normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    lines = [SPACE_PATTERN.sub(" ", line).strip() for line in normalized.splitlines()]
    return "\n".join(line for line in lines if line).casefold()


def clean_training_text(text: str) -> tuple[str, int]:
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    without_citations, count = CITATION_PATTERN.subn("", normalized)
    lines = [SPACE_PATTERN.sub(" ", line).strip() for line in without_citations.splitlines()]
    return "\n".join(line for line in lines if line).strip(), count


def _repeated_line_ratio(text: str) -> float:
    lines = [line.casefold() for line in text.splitlines() if len(line.split()) >= 2]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def filter_wikimedia_record(
    record: Mapping[str, Any],
    *,
    minimum_characters: int,
    maximum_characters: int,
) -> FilterResult:
    raw_text = record.get("text")
    title = record.get("title")
    document_id = record.get("id")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return FilterResult(False, "empty_text", "", {})
    if not isinstance(title, str) or not title.strip() or document_id in (None, ""):
        return FilterResult(False, "missing_identity", "", {})
    language = record.get("language")
    if language is not None and str(language).casefold() not in {"en", "english"}:
        return FilterResult(False, "non_english", "", {})
    try:
        raw_text.encode("utf-8", errors="strict")
        title.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return FilterResult(False, "malformed_unicode", "", {})

    lowered_title = title.strip().casefold()
    lowered_start = raw_text.lstrip()[:200].casefold()
    if lowered_start.startswith("#redirect") or lowered_start.startswith("redirect:"):
        return FilterResult(False, "redirect", "", {})
    if lowered_title.endswith("(disambiguation)") or "may refer to:" in lowered_start or "can refer to:" in lowered_start:
        return FilterResult(False, "disambiguation", "", {})

    cleaned, citations_removed = clean_training_text(raw_text)
    if len(cleaned) < minimum_characters:
        return FilterResult(False, "too_short", "", {})
    if len(cleaned) > maximum_characters:
        return FilterResult(False, "too_long", "", {})
    if not any(character.isalpha() for character in cleaned):
        return FilterResult(False, "no_alphabetic_text", "", {})
    repeated_ratio = _repeated_line_ratio(cleaned)
    if len(cleaned.splitlines()) >= 6 and repeated_ratio > 0.5:
        return FilterResult(False, "repeated_low_information_lines", "", {})
    boilerplate_markers = sum(
        cleaned.casefold().count(marker)
        for marker in ("cookie policy", "privacy policy", "all rights reserved", "sign up to continue")
    )
    if boilerplate_markers >= 3:
        return FilterResult(False, "excessive_boilerplate", "", {})
    return FilterResult(
        True,
        None,
        cleaned,
        {
            "citations_removed": citations_removed,
            "repeated_line_ratio": round(repeated_ratio, 6),
            "unicode_normalization": "NFC",
        },
    )

