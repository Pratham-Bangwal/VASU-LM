"""Deterministic, conservative Wikimedia pilot filtering."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Mapping, Any

from vasu.data.deduplication.normalization import normalize_for_matching

from .quality import assess_text_quality, repair_mojibake


CITATION_PATTERN = re.compile(r"\[(?:\d{1,4}|citation needed)\]", re.IGNORECASE)
REFERENCE_PATTERN = re.compile(
    r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>",
    re.IGNORECASE | re.DOTALL,
)
TEMPLATE_PATTERN = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
ITALIC_BOLD_PATTERN = re.compile(r"'{2,5}([^'\n]+?)'{2,5}")
SPACE_PATTERN = re.compile(r"[ \t]+")
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")


@dataclass(frozen=True)
class FilterResult:
    accepted: bool
    reason: str | None
    cleaned_text: str
    metadata: dict[str, Any]


def _template_parts(template: str) -> tuple[str, list[str], dict[str, str]]:
    fields = [field.strip() for field in template[2:-2].split("|")]
    name = fields[0].casefold().replace("_", " ").strip()
    positional: list[str] = []
    named: dict[str, str] = {}
    for field in fields[1:]:
        key = field.split("=", 1)[0].casefold().strip() if "=" in field else ""
        if "=" in field and (
            key.isdigit()
            or key
            in {"text", "quote", "title", "lang", "abbr", "disp", "output"}
        ):
            key, value = field.split("=", 1)
            named[key.casefold().strip()] = value.strip()
        elif field:
            positional.append(field)
    return name, positional, named


def _render_readable_template(template: str) -> str | None:
    """Render only template families with an unambiguous readable value.

    Unknown/maintenance templates still disappear with surrounding whitespace.
    This deliberately avoids pretending to be a complete MediaWiki expander.
    """
    name, positional, named = _template_parts(template)
    if name in {"chem", "chem2", "chemical formula"}:
        return "".join(positional) or named.get("1")
    if name in {"math", "mvar", "var"}:
        return positional[0] if positional else named.get("1")
    if name in {"sfrac", "frac"} and len(positional) >= 2:
        return f"{positional[-2]}/{positional[-1]}"
    if name in {"convert", "cvt"} and len(positional) >= 2:
        return f"{positional[0]} {positional[1]}"
    if name == "lang" or name.startswith("lang-"):
        if name == "lang" and len(positional) >= 2:
            return positional[1]
        return positional[0] if positional else named.get("text")
    if name in {"transl", "transliteration"}:
        return positional[-1] if positional else named.get("text")
    if name in {"quote", "quotation", "blockquote"}:
        return named.get("text") or named.get("quote") or (
            positional[0] if positional else None
        )
    if name in {"ubl", "unbulleted list", "plainlist", "flatlist"}:
        return "\n".join(positional) if positional else named.get("1")
    if name in {"nowrap", "nobr", "italic title", "title"}:
        return positional[0] if positional else named.get("1")
    if name.startswith("cite "):
        return named.get("title")
    return None


def _replace_templates(text: str) -> tuple[str, int, int]:
    removed = 0
    preserved = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal removed, preserved
        rendered = _render_readable_template(match.group(0))
        if rendered is None or not rendered.strip():
            removed += 1
            return " "
        preserved += 1
        return f" {rendered.strip()} "

    current = text
    while True:
        current, count = TEMPLATE_PATTERN.subn(replace, current)
        if count == 0:
            break
    return current, removed, preserved


def comparison_normalize(text: str) -> str:
    """Backward-compatible entry point for the shared cross-source contract."""
    return normalize_for_matching(text)


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
    cleaned, templates, templates_preserved = _replace_templates(cleaned)
    cleaned = ITALIC_BOLD_PATTERN.sub(r"\1", cleaned)
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
        "templates_preserved": templates_preserved,
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
