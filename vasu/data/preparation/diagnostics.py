"""Named, dataset-wide diagnostics for known Wikimedia extraction defects."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .quality import CATALOGUE_ENTRY, PUBLICATION_ENTRY, _list_metrics


@dataclass(frozen=True)
class DefectFinding:
    """One bounded diagnostic finding and its required disposition."""

    family: str
    disposition: str
    rejection_reason: str | None
    start: int
    end: int
    evidence: str


EMPTY_PARENTHETICAL = re.compile(r"\(\s*[,;:]+\s*\)")
CONNECTOR_ONLY_FIELD = re.compile(
    r"\(\s*[,;:]*\s*(?:(?:or|and)\s*[,;:]*\s*)+\)",
    re.IGNORECASE,
)
MISSING_ETYMOLOGY = re.compile(
    r"\b(?:derived|borrowed|taken)(?:\s+either)?\s+from\s*,\s*"
    r"(?:meaning|signifying|or\b)|\bfrom\s+and\s+related\s+words\b",
    re.IGNORECASE,
)
MISSING_MEASUREMENT = re.compile(
    r"\b(?:distance|mass|length|height|width|area|volume|weight|speed)\s+"
    r"of\s+(?:about\s+|approximately\s+)?(?:earth\s+radii|kilograms?|"
    r"kilomet(?:er|re)s?|meters?|metres?|miles?|feet|seconds?|degrees?|"
    r"lit(?:er|re)s?|watts?|volts?|kelvins?)\b|"
    r"\b(?:distance|mass|length|height|width|area|volume|weight|speed|"
    r"parallax)\s+of\s*[,.;]",
    re.IGNORECASE,
)
MISSING_ISOTOPE_IDENTIFIER = re.compile(
    r"\b(?:isotope|nuclide)\s+(?:of\s*)?(?:[,.;]|(?:is|was)\s*[,.;])",
    re.IGNORECASE,
)
JOINED_LIFESPAN_DATE = re.compile(
    r"\b(?:1\d{3}|20\d{2})(?=(?:0?[1-9]|[12]\d|3[01])\s+"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b)",
    re.IGNORECASE,
)
MISSING_MATH_OPERAND = re.compile(
    r"\bis\s+equivalent\s+to\s*[,.;]|"
    r"\bmeans\s*,\s*which\s+is\s+not\s+equivalent\b|"
    r"(?:=|≈|≃|~)\s*(?:[,.;]|$)",
    re.IGNORECASE,
)
PROMISED_COLON_AT_END = re.compile(r":\s*$")
REPEATED_COMMAS = re.compile(r",{2,}")
SEMICOLON_COMMA = re.compile(r";\s*,")
WORD_PERIOD_COMMA = re.compile(r"\b(?P<word>[A-Za-z]{4,})\.\s*,")
BROKEN_PARENTHETICAL_PUNCTUATION = re.compile(r"\)\s*\.\s*;")
MIXED_ALPHANUMERIC_JOIN = re.compile(r"\b[A-Za-z]*[a-z]{2,}\d+\b")
UNIT_EXPONENT = re.compile(r"^(?:cm|mm|km|ft|yd|in|mi|sp|mc|dw|sdh)\d+$", re.I)
JOINED_DATE_OR_FUNCTION_WORD = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|age|or|the|figure|hours?|minutes?|seconds?)"
    r"\d+\b",
    re.IGNORECASE,
)

PROMISED_MATERIAL = re.compile(
    r"\b(?:following|quotation|quote|verses?|words|statement|passage|"
    r"anecdote|theme|examples?|types?|classified|designated)\b[^\n]{0,180}:$",
    re.IGNORECASE,
)
TRANSITION_WITHOUT_PROMISED_MATERIAL = re.compile(
    r"^(?:Recognition\b|Upon\s+the\s+conclusion\b|"
    r"In\s+(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{4}\b|"
    r"This\s+is\s+(?:a|the)\s+(?:celebrated|famous|well-known)\s+passage\b)",
    re.IGNORECASE,
)
def _finding(
    family: str,
    match: re.Match[str],
    *,
    disposition: str = "reject",
    rejection_reason: str | None = "missing_source_value",
) -> DefectFinding:
    return DefectFinding(
        family=family,
        disposition=disposition,
        rejection_reason=rejection_reason,
        start=match.start(),
        end=match.end(),
        evidence=match.group(0)[:240],
    )


def _find_missing_promised_material(text: str) -> DefectFinding | None:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    offset = 0
    for index, paragraph in enumerate(paragraphs):
        match = PROMISED_MATERIAL.search(paragraph)
        paragraph_start = text.find(paragraph, offset)
        offset = paragraph_start + len(paragraph)
        if match is None or index + 1 >= len(paragraphs):
            continue
        following = paragraphs[index + 1]
        if TRANSITION_WITHOUT_PROMISED_MATERIAL.search(following):
            start = paragraph_start + match.start()
            end = paragraph_start + match.end()
            return DefectFinding(
                family="heading_or_commentary_after_omitted_material",
                disposition="reject",
                rejection_reason="missing_source_value",
                start=start,
                end=end,
                evidence=text[start:end][:240],
            )
    return None


def _find_malformed_punctuation(text: str) -> DefectFinding | None:
    for family, pattern in (
        ("repeated_punctuation_from_removed_markup", REPEATED_COMMAS),
        ("repeated_punctuation_from_removed_markup", SEMICOLON_COMMA),
        ("repeated_punctuation_from_removed_markup", BROKEN_PARENTHETICAL_PUNCTUATION),
    ):
        if match := pattern.search(text):
            return _finding(
                family,
                match,
                rejection_reason="malformed_source_text",
            )
    for match in WORD_PERIOD_COMMA.finditer(text):
        if match.group("word").casefold() != "ibid":
            return _finding(
                "repeated_punctuation_from_removed_markup",
                match,
                rejection_reason="malformed_source_text",
            )
    return None


def _find_catalogue_dominance(text: str) -> DefectFinding | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    publication_lines = sum(bool(PUBLICATION_ENTRY.search(line)) for line in lines)
    catalogue_lines = sum(bool(CATALOGUE_ENTRY.search(line)) for line in lines)
    if max(publication_lines, catalogue_lines) < 4:
        return None
    if max(publication_lines, catalogue_lines) / len(lines) < 0.5:
        return None
    metrics = _list_metrics(text)
    review_only = metrics["prose_token_ratio"] >= 0.65
    return DefectFinding(
        family="publication_adaptation_or_catalogue_dominated",
        disposition="manual_review" if review_only else "reject",
        rejection_reason=None if review_only else "list_dominated",
        start=0,
        end=min(len(text), 240),
        evidence=text[:240],
    )


def _find_fragmented_list(text: str) -> DefectFinding | None:
    metrics = _list_metrics(text)
    if (
        metrics["maximum_consecutive_list_like_lines"] < 6
        and metrics["list_block_count"] < 2
    ):
        return None
    if metrics["list_like_line_count"] < 6 or metrics["prose_token_ratio"] >= 0.65:
        return None
    return DefectFinding(
        family="fragmented_or_structured_list",
        disposition="reject",
        rejection_reason="list_dominated",
        start=0,
        end=min(len(text), 240),
        evidence=text[:240],
    )


def find_known_defects(text: str) -> tuple[DefectFinding, ...]:
    """Return one named finding per known defect family in ``text``."""
    findings: list[DefectFinding] = []
    named_patterns = (
        ("empty_pronunciation_or_alternate_name", EMPTY_PARENTHETICAL, "missing_source_value"),
        ("connector_only_language_field", CONNECTOR_ONLY_FIELD, "missing_source_value"),
        ("missing_etymological_source_term", MISSING_ETYMOLOGY, "missing_source_value"),
        ("missing_measurement_unit_or_quantity", MISSING_MEASUREMENT, "missing_source_value"),
        ("missing_isotope_identifier", MISSING_ISOTOPE_IDENTIFIER, "missing_source_value"),
        ("malformed_or_joined_lifespan_date", JOINED_LIFESPAN_DATE, "malformed_source_text"),
        ("missing_formula_or_mathematical_operand", MISSING_MATH_OPERAND, "missing_source_value"),
        ("missing_quoted_passage_or_enumeration", PROMISED_COLON_AT_END, "missing_source_value"),
    )
    for family, pattern, reason in named_patterns:
        if match := pattern.search(text):
            findings.append(_finding(family, match, rejection_reason=reason))
    if finding := _find_missing_promised_material(text):
        findings.append(finding)
    if finding := _find_malformed_punctuation(text):
        findings.append(finding)
    if finding := _find_catalogue_dominance(text):
        findings.append(finding)
    if finding := _find_fragmented_list(text):
        findings.append(finding)
    if joined := JOINED_DATE_OR_FUNCTION_WORD.search(text):
        findings.append(
            _finding(
                "joined_date_or_function_word",
                joined,
                rejection_reason="malformed_source_text",
            )
        )
    mixed = next(
        (
            match
            for match in MIXED_ALPHANUMERIC_JOIN.finditer(text)
            if not UNIT_EXPONENT.fullmatch(match.group(0))
            and not JOINED_DATE_OR_FUNCTION_WORD.fullmatch(match.group(0))
        ),
        None,
    )
    if mixed is not None:
        findings.append(
            _finding(
                "possible_joined_alphanumeric_formatting",
                mixed,
                disposition="manual_review",
                rejection_reason=None,
            )
        )
    return tuple(findings)
