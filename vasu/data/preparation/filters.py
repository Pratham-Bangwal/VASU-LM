"""Deterministic, conservative Wikimedia pilot filtering."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Mapping, Any

from .quality import assess_text_quality, repair_mojibake


CITATION_PATTERN = re.compile(r"\[(?:\d{1,4}|citation needed)\]", re.IGNORECASE)
REFERENCE_PATTERN = re.compile(
    r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>",
    re.IGNORECASE | re.DOTALL,
)
TEMPLATE_PATTERN = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
SPACE_PATTERN = re.compile(r"[ \t]+")
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")


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


def clean_training_text(text: str) -> tuple[str, dict[str, int | bool | list[str]], str | None]:
    encoding = repair_mojibake(text)
    if encoding.rejection_reason:
        return "", {
            "encoding_repaired": encoding.repaired,
            "encoding_repair_count": encoding.repair_count,
            "quality_warnings": list(encoding.warnings),
        }, encoding.rejection_reason
    normalized = unicodedata.normalize("NFC", encoding.text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned, comments = COMMENT_PATTERN.subn(" ", normalized)
    cleaned, references = REFERENCE_PATTERN.subn(" ", cleaned)
    templates = 0
    while True:
        cleaned, count = TEMPLATE_PATTERN.subn(" ", cleaned)
        templates += count
        if count == 0:
            break
    cleaned, citations = CITATION_PATTERN.subn(" ", cleaned)
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in cleaned.splitlines():
        line = SPACE_PATTERN.sub(" ", raw_line).strip()
        line = SPACE_BEFORE_PUNCTUATION.sub(r"\1", line)
        if line:
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    result = "\n\n".join(paragraphs).strip()
    quality_warnings, rejection = assess_text_quality(result)
    warnings = list(dict.fromkeys((*encoding.warnings, *quality_warnings)))
    return result, {
        "encoding_repaired": encoding.repaired,
        "encoding_repair_count": encoding.repair_count,
        "quality_warnings": warnings,
        "citations_removed": citations,
        "references_removed": references,
        "templates_removed": templates,
        "comments_removed": comments,
    }, rejection


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

    cleaned, cleanup_metadata, quality_rejection = clean_training_text(raw_text)
    if quality_rejection:
        return FilterResult(False, quality_rejection, "", cleanup_metadata)
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
            **cleanup_metadata,
            "repeated_line_ratio": round(repeated_ratio, 6),
            "unicode_normalization": "NFC",
        },
    )
