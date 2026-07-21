"""Conservative Unicode repair and text-quality checks for factual sources."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata


SUSPICIOUS_SEQUENCES = (
    "Ã¢",
    "Ãƒ",
    "ÃŽ",
    "Â°",
    "Â ",
    "Î¼",
    "Î¸",
    "â€",
    "âˆ",
    "ï¿½",
)
JOINED_WORD_PATTERNS = (
    re.compile(r"\bforauthority\b", re.IGNORECASE),
    re.compile(r"\bendof\b", re.IGNORECASE),
    re.compile(r"\bwithinanarchist\b", re.IGNORECASE),
    re.compile(r"\bfromthe\b", re.IGNORECASE),
    re.compile(r"\basdistinct\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class EncodingResult:
    text: str
    repaired: bool
    repair_count: int
    warnings: tuple[str, ...]
    rejection_reason: str | None = None


@dataclass(frozen=True)
class ChunkQualityResult:
    warnings: tuple[str, ...]
    rejection_reason: str | None
    metrics: dict[str, int | float]


LIST_PREFIX = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+")
YEAR_NAME_ENTRY = re.compile(r"^\s*(?:c\.\s*)?\d{1,4}\s*[–—-]\s*\S+")
DATE_ENTRY = re.compile(
    r"^\s*(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2}\b",
    re.IGNORECASE,
)
BIBLIOGRAPHY_ENTRY = re.compile(
    r"^\s*(?:[A-Z][\w'’.-]+,\s*(?:[A-Z]\.|[A-Z][\w'’.-]+)|"
    r".+\(\d{4}[a-z]?\)\.|ISBN\b|doi\s*:)",
)
BIBLIOGRAPHY_YEAR_ENTRY = re.compile(
    r"^\s*[^.!?\n]{2,220}(?::|,)\s*[^.!?\n]{1,220}"
    r"\b(?:1[5-9]\d{2}|20\d{2})\b[.!]?\s*$"
)
PUBLICATION_ENTRY = re.compile(
    r"(?:\b(?:translated|edited)\s+by\b|\bvols?\.?\s+\d+\b|"
    r"\([^():\n]{1,60}:\s*[^,\n]{2,100},\s*(?:1[5-9]\d{2}|20\d{2})\)|"
    r"\[(?:contains?|translations?)\b)",
    re.IGNORECASE,
)
CATALOGUE_ENTRY = re.compile(
    r"^(?:\d{4}\s*[–—-]|(?:Film|Television|Radio|Stage|Novel|Series|Album|"
    r"Edition|Volume)\s*:|[^.!?\n]{2,100}\(\d{4}\)\s*$)",
    re.IGNORECASE,
)
NAMES_ONLY_ENTRY = re.compile(
    r"^\s*(?:[A-Z][\w'’.-]+(?:\s+|$)){2,6}$"
)
CATEGORY_ENTRY = re.compile(r"^\s*(?:category|class|family|genus|order)\s*:", re.IGNORECASE)
TAXONOMIC_ENTRY = re.compile(
    r"^\s*(?:order|family|subfamily|tribe|genus|species)\s+\S+",
    re.IGNORECASE,
)
INSTITUTION_ENTRY = re.compile(
    r"\b(?:academy|battalion|chapel|church|college|detachment|faculty|hospital|"
    r"institute|institution|ministry|regiment|school|service|subdivision|"
    r"university|unit)\b",
    re.IGNORECASE,
)
CATEGORY_LABEL = re.compile(r"^\s*[A-Z][\w &'/-]{0,40}:\s*$")
GLOSSARY_ENTRY = re.compile(r"^\s*\S.{1,220}?\s+[–—-]\s+\S")
LEXICAL_TOKEN = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", re.UNICODE)
SENTENCE_END = re.compile(r"[.!?](?:[\"'’”)]*)\s*(?:$|\s)")
LIST_INTRODUCTION_WITHOUT_CONTENT = re.compile(
    r"\b(?:as follows|the following|include(?:s|d)?|consist(?:s|ed)? of)\s*:\s*$",
    re.IGNORECASE,
)
MISSING_VALUE_PATTERNS = (
    re.compile(r"\(\s*[,;:]+\s*\)"),
    # A chunk must not retain an enumeration promise without its values.
    re.compile(
        r"\b(?:examples?\s+are|the\s+following\s+are|"
        r"the\s+(?:principal\s+)?types\s+are|"
        r"(?:is|are|was|were)\s+classified\s+as|"
        r"designated\s+the\s+types(?:\s+of\s+[^:\n]{1,100})?)\s*:\s*$",
        re.IGNORECASE,
    ),
    # A retained chunk ending in a colon omits the promised value, list,
    # quotation, formula, or explanation at its training boundary.
    re.compile(r":\s*$"),
    re.compile(r"\b(?:currently|approximately|about|nearly)\s+[.,;]", re.IGNORECASE),
    re.compile(r"\b(?:total\s+)?(?:land\s+)?area\s+of\s*[,.;]", re.IGNORECASE),
    re.compile(r"\bfrom\s+(?:almost|approximately|about)\s+(?:for|to|and)\b", re.IGNORECASE),
    re.compile(r"\b(?:more|less|fewer)\s+than\s+(?:of\b|[,.;])", re.IGNORECASE),
    re.compile(r"\b(?:over|under|approximately|about|nearly)\s+of\b", re.IGNORECASE),
    re.compile(
        r"\b(?:delivering|spilling|containing|producing|covering)\s+of\b",
        re.IGNORECASE,
    ),
    re.compile(r",\s*(?:of|as|is|from)\s*,", re.IGNORECASE),
    re.compile(
        r"\b(?:summarized|expressed|written|represented)\s+"
        r"(?:formally\s+)?as\s+(?:or|and)?\s*[,.;]",
        re.IGNORECASE,
    ),
    re.compile(r"\bfrom\s+(?:the|an?)\s*,", re.IGNORECASE),
    re.compile(
        r"\b(?:definition|formula|quotation|quote|example)\b[^.\n]{0,120}:"
        r"[ \t]*\n[ \t]*\n",
        re.IGNORECASE,
    ),
    # Removed linguistic forms can leave a connector as the first apparent
    # item inside parentheses, or leave grammatical-number forms empty.
    re.compile(r"\b(?:is|are|was|were)\s*\(\s*(?:or|and)\b", re.IGNORECASE),
    re.compile(r"\bvariously\s+(?:or|and)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:plural|singular|comparative|superlative)\s+form\s+"
        r"(?:is|are|was|were)\s*[)\],.;]",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:or|and)\s+(?:or|and)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:used|called|named|written|spelled|rendered)\s*,\s*"
        r"(?:while|whereas|with|and)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:called|named)\s+the\s*,", re.IGNORECASE),
    re.compile(r"\b(?:river|term|word|name)\s+as\s*[.,;]", re.IGNORECASE),
    # Missing etymological source forms leave only definitions/connectors.
    re.compile(
        r"\b(?:derived|borrowed|taken)(?:\s+either)?\s+from\s*,\s*"
        r"(?:meaning|signifying|or\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\bfrom\s+and\s+related\s+words\b", re.IGNORECASE),
    # Mathematical examples/results with removed operands or sample values.
    re.compile(
        r"\b(?:mean|median|average|value|result|sum|difference|product|"
        r"quotient|variance|probability)\s+(?:is|are|was|were)\s*[,.;]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:for example|for instance),?\s+(?:consider|take|suppose)\s+"
        r"(?:the|a|this)\s+(?:data\s+)?(?:sample|set|sequence|values?|numbers?)"
        r"\s*[.;]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsuch as\s*,\s*(?:the|a|an|this|that|these|those|it|they|we)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\(\s*(?:[A-Z][\w -]*|[a-z]{2,3})\s*:\s*\)"),
    re.compile(r"[\"“]\s*[\"”]"),
    re.compile(r"\bwith\s+(?:the\s+)?formula\s*[.;]", re.IGNORECASE),
    # A physical quantity cannot consist solely of its unit.
    re.compile(
        r"\b(?:distance|mass|length|height|width|area|volume|weight|speed)\s+"
        r"of\s+(?:about\s+|approximately\s+)?(?:earth\s+radii|kilograms?|"
        r"kilomet(?:er|re)s?|meters?|metres?|miles?|feet|seconds?|degrees?|"
        r"lit(?:er|re)s?|watts?|volts?|kelvins?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:distance|mass|length|height|width|area|volume|weight|speed|"
        r"parallax)\s+of\s*[,.;]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:isotope|nuclide)\s+(?:of\s*)?(?:[,.;]|(?:is|was)\s*[,.;])",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*\((?:triangle|parallelogram|rectangle|square|circle|polygon|"
        r"formula|equation)\)\s*[.;]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"\b(?:at|nearly)\s+Mach\s+\d+(?:\.\d+)?\s+or\s*"
        r"(?:on\b|[,.;]|$|\n)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhaving\s+reached\s+on\s+\d{1,2}\s+[A-Za-z]+\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bfor\s+and\s+respectively\b", re.IGNORECASE),
    re.compile(
        r"\b(?:extends?|stretches?)\s+more\s+than\s+"
        r"(?:through|across|along|over)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+following\s+(?:principles|items|examples)|"
        r"with\s+these\s+words|the\s+following\s+are|"
        r"statistics[^.\n]{0,80}(?:released|reported)[^.\n]{0,40})\s*:"
        r"[ \t]*\n[ \t]*\n[ \t]*(?![-*•▪◦]|\d+[.)]\s|[\"“])",
        re.IGNORECASE,
    ),
    # A displayed example/reaction was removed before the next section.
    re.compile(
        r"\b(?:an?\s+example\s+is|examples?\s+are)\s+"
        r"[^\n.!?:]{2,240}\s*\n\s*\n\s*[A-Z][^\n]{0,100}\n",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reacts?|forms?|converts?|alkylates?|hydrogenates?)\b"
        r"[^\n.]{0,240}:\s*\n\s*\n\s*[A-Z][^\n]{0,100}\n",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:following|quotation|quote|verses?|words|statement|passage|"
        r"anecdote|theme|examples?|types?|classified|designated)\b"
        r"[^\n]{0,180}:\s*\n\s*\n\s*(?:Recognition\b|"
        r"Upon\s+the\s+conclusion\b|"
        r"In\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b|"
        r"This\s+is\s+(?:a|the)\s+(?:celebrated|famous|well-known)\s+"
        r"passage\b)",
        re.IGNORECASE,
    ),
    # Promised mathematical expressions and operands must remain readable.
    re.compile(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:possible\s+)?(?:ways|forms|expressions|equations|identities|cases)"
        r"\s*:\s*\n\s*\n\s*(?=[A-Z])",
        re.IGNORECASE,
    ),
    re.compile(r"\bis\s+equivalent\s+to\s*[,.;]", re.IGNORECASE),
    re.compile(r"\bmeans\s*,\s*which\s+is\s+not\s+equivalent\b", re.IGNORECASE),
    re.compile(
        r"\bwritten\s+(?:unambiguously\s+)?as\s*\n\s*\n\s*(?=[A-Z])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\(\s*[,;]*\s*(?:(?:or|and)\s*[,;]*\s*)+\)",
        re.IGNORECASE,
    ),
    re.compile(r"\(\s*;"),
    re.compile(
        r"\b(?:about|approximately|around|nearly)\s+"
        r"(?:north|south|east|west)\s+of\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bmeasures?\s+long\b", re.IGNORECASE),
    re.compile(r"\band\s+across\s+at\s+(?:its|the)\b", re.IGNORECASE),
    re.compile(r"\b(?:its|the)\s+area\s+is\s+(?:and\b|[,.;])", re.IGNORECASE),
    re.compile(
        r"(?:[.!?]\s*,\s*[A-Za-z]+\b|"
        r"(?:^|\n)\s*,\s*(?:the|it|there)\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"\bthe\s+\w+(?:ical|ive|ary|ous)\s+"
        r"\(\s*\d{3,4}\s*[–—-]\s*\d{2,4}\s*\)",
        re.IGNORECASE,
    ),
    re.compile(r"\(\s*\)"),
    re.compile(r"(?:=|≈|≃|~)\s*(?:[,.;]|$)"),
)
MALFORMED_SOURCE_PATTERNS = (
    re.compile(r"\bof\s+allowed\b", re.IGNORECASE),
    re.compile(r"\bAfter\s+s\s+", re.IGNORECASE),
    re.compile(r"\bproject\s*,\s*,", re.IGNORECASE),
    re.compile(r"\b\w+\s*,\s*,\s*\w+"),
    re.compile(r"(?:^|\n\s*\n)\s*,\s*[A-Z]", re.MULTILINE),
    re.compile(r"\)\s*\.\s*;"),
    re.compile(r"(?:^|\n)\s*[^\n()]{1,100}\)\s*(?:\n|$)"),
    re.compile(r"\b\d+,\s*(?:$|\n)", re.MULTILINE),
    re.compile(r",{2,}"),
    re.compile(r";\s*,"),
    re.compile(r"\b(?!Ibid\b)[A-Za-z]{4,}\.\s*,", re.IGNORECASE),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\d+\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:age|or|the|figure|hours?|minutes?|seconds?)\d+\b", re.IGNORECASE),
    # Joined lifespan dates such as "149020 March 1568" have lost a dash.
    re.compile(
        r"\b(?:1\d{3}|20\d{2})(?=(?:0?[1-9]|[12]\d|3[01])\s+"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\b)",
        re.IGNORECASE,
    ),
)


def _suspicion_score(text: str) -> int:
    score = sum(text.count(marker) for marker in SUSPICIOUS_SEQUENCES)
    score += len(re.findall(r"Ã[\u0080-\u00bf]", text))
    score += len(re.findall(r"Â[\u0080-\u00bf]", text))
    score += len(re.findall(r"Î[\u0080-\u00bf]", text))
    score += text.count("\ufffd") * 4
    score += sum(1 for character in text if 0x80 <= ord(character) <= 0x9F)
    return score


def _decode_candidate(text: str, encoding: str) -> str | None:
    try:
        return text.encode(encoding).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def repair_mojibake(text: str, max_passes: int = 2) -> EncodingResult:
    """Repair only reversible UTF-8-as-single-byte decoding corruption.

    A candidate is accepted only when a strict round trip succeeds and reduces
    known corruption markers without introducing replacement characters.
    Already-correct Unicode therefore remains byte-for-byte unchanged.
    """
    if "\ufffd" in text:
        return EncodingResult(text, False, 0, (), "replacement_character")
    original_score = _suspicion_score(text)
    if original_score == 0:
        return EncodingResult(text, False, 0, ())
    current = text
    repairs = 0
    for _ in range(max_passes):
        current_score = _suspicion_score(current)
        candidates = [
            candidate
            for encoding in ("cp1252", "latin-1")
            if (candidate := _decode_candidate(current, encoding)) is not None
            and "\ufffd" not in candidate
            and _suspicion_score(candidate) < current_score
        ]
        if not candidates:
            break
        current = min(candidates, key=lambda value: (_suspicion_score(value), value))
        repairs += 1
        if _suspicion_score(current) == 0:
            break
    remaining = _suspicion_score(current)
    if repairs and remaining < original_score:
        warnings = ("encoding_repaired",) if remaining == 0 else (
            "encoding_repaired",
            "residual_mojibake_markers",
        )
        rejection = None if remaining == 0 else "low_confidence_encoding_corruption"
        return EncodingResult(current, True, repairs, warnings, rejection)
    return EncodingResult(
        text,
        False,
        0,
        ("suspicious_mojibake_markers",),
        "low_confidence_encoding_corruption",
    )


def assess_text_quality(text: str) -> tuple[tuple[str, ...], str | None]:
    warnings: list[str] = []
    if "\ufffd" in text:
        return (), "replacement_character"
    controls = [
        character
        for character in text
        if unicodedata.category(character) == "Cc" and character not in "\n\t"
    ]
    if controls:
        return (), "control_character"
    suspicious = _suspicion_score(text)
    if suspicious:
        rate = suspicious / max(1, len(text))
        if rate > 0.001 or suspicious >= 2:
            return (), "implausible_non_ascii_corruption"
        warnings.append("suspicious_mojibake_marker")
    if any(pattern.search(text) for pattern in JOINED_WORD_PATTERNS):
        warnings.append("suspected_joined_words")
    return tuple(warnings), None


def _list_line_kind(line: str) -> str | None:
    """Classify structural list lines without treating ordinary prose as a list."""
    if LIST_PREFIX.search(line):
        return "explicit"
    if YEAR_NAME_ENTRY.search(line) or DATE_ENTRY.search(line):
        return "dated"
    if (
        BIBLIOGRAPHY_ENTRY.search(line)
        or BIBLIOGRAPHY_YEAR_ENTRY.search(line)
        or PUBLICATION_ENTRY.search(line)
        or CATALOGUE_ENTRY.search(line)
    ):
        return "bibliography"
    if TAXONOMIC_ENTRY.search(line) or CATEGORY_ENTRY.search(line):
        return "taxonomy"
    if CATEGORY_LABEL.fullmatch(line):
        return "category"
    if GLOSSARY_ENTRY.search(line):
        return "glossary"
    if (
        len(line) <= 120
        and INSTITUTION_ENTRY.search(line)
        and not SENTENCE_END.search(line)
    ):
        return "institution"
    if NAMES_ONLY_ENTRY.fullmatch(line):
        return "name"
    if (
        len(line) <= 100
        and any(character.isalpha() for character in line)
        and not SENTENCE_END.search(line)
    ):
        return "short"
    return None


def _list_metrics(text: str) -> dict[str, int | float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    list_like = 0
    short_lines = 0
    strong_list_lines = 0
    list_tokens = 0
    prose_tokens = 0
    prose_sentences = 0
    consecutive_list_lines = 0
    maximum_consecutive_list_lines = 0
    list_blocks = 0
    previous_was_list = False
    kind_counts: dict[str, int] = {}
    for line in lines:
        kind = _list_line_kind(line)
        token_count = len(LEXICAL_TOKEN.findall(line))
        if len(line) <= 100:
            short_lines += 1
        if kind is not None:
            if not previous_was_list:
                list_blocks += 1
            previous_was_list = True
            list_like += 1
            list_tokens += token_count
            consecutive_list_lines += 1
            maximum_consecutive_list_lines = max(
                maximum_consecutive_list_lines,
                consecutive_list_lines,
            )
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            if kind != "short":
                strong_list_lines += 1
        else:
            prose_tokens += token_count
            consecutive_list_lines = 0
            previous_was_list = False
            prose_sentences += len(SENTENCE_END.findall(line))
    count = len(lines)
    total_tokens = list_tokens + prose_tokens
    repeated_structure_lines = max(kind_counts.values(), default=0)
    return {
        "nonempty_line_count": count,
        "list_like_line_count": list_like,
        "list_like_line_ratio": list_like / count if count else 0.0,
        "strong_list_like_line_count": strong_list_lines,
        "short_line_count": short_lines,
        "short_line_ratio": short_lines / count if count else 0.0,
        "maximum_consecutive_list_like_lines": maximum_consecutive_list_lines,
        "list_block_count": list_blocks,
        "repeated_structure_line_ratio": (
            repeated_structure_lines / count if count else 0.0
        ),
        "prose_token_ratio": prose_tokens / total_tokens if total_tokens else 0.0,
        "average_line_length": (
            sum(len(line) for line in lines) / count if count else 0.0
        ),
        "prose_sentence_count": prose_sentences,
    }


def _is_low_information(text: str) -> bool:
    stripped = text.strip()
    if not stripped or not any(character.isalpha() for character in stripped):
        return True
    words = stripped.split()
    if (
        "\n" not in stripped
        and len(words) <= 12
        and not stripped.endswith((".", "!", "?"))
        and stripped[0].isupper()
        and all(
            not any(character.isalpha() for character in word)
            or word[0].isupper()
            for word in words
        )
    ):
        return True
    return bool(LIST_INTRODUCTION_WITHOUT_CONTENT.search(stripped))


def assess_final_chunk(
    text: str,
    *,
    token_count: int,
    minimum_tokens: int,
    maximum_list_like_line_ratio: float,
    minimum_prose_sentences_for_list_chunk: int,
    boundary_start_type: str,
    boundary_end_type: str,
    training_mode: bool,
) -> ChunkQualityResult:
    """Apply structural checks after chunking and before any accounting."""
    warnings, rejection = assess_text_quality(text)
    metrics = _list_metrics(text)
    if rejection:
        return ChunkQualityResult(warnings, rejection, metrics)
    if token_count < minimum_tokens:
        return ChunkQualityResult(warnings, "below_minimum_chunk_tokens", metrics)
    if _is_low_information(text):
        return ChunkQualityResult(warnings, "low_information", metrics)
    if any(pattern.search(text) for pattern in MISSING_VALUE_PATTERNS):
        return ChunkQualityResult(warnings, "missing_source_value", metrics)
    if any(pattern.search(text) for pattern in MALFORMED_SOURCE_PATTERNS):
        return ChunkQualityResult(warnings, "malformed_source_text", metrics)
    line_count = int(metrics["nonempty_line_count"])
    minimum_prose_sentences = max(
        minimum_prose_sentences_for_list_chunk,
        math.ceil(line_count * (1.0 - maximum_list_like_line_ratio)),
    )
    ratio_dominated = (
        line_count >= 4
        and metrics["list_like_line_ratio"] >= maximum_list_like_line_ratio
        and metrics["prose_sentence_count"] < minimum_prose_sentences
    )
    consecutive_dominated = (
        metrics["maximum_consecutive_list_like_lines"] >= 6
        and metrics["prose_token_ratio"] < 0.85
    )
    repeated_dominated = (
        metrics["strong_list_like_line_count"] >= 8
        and metrics["list_like_line_ratio"] >= 0.35
        and metrics["prose_token_ratio"] < 0.75
    )
    multiple_blocks_dominated = (
        metrics["list_block_count"] >= 2
        and metrics["list_like_line_count"] >= 6
        and metrics["prose_token_ratio"] < 0.65
    )
    structured_statistics_dominated = (
        line_count >= 8
        and metrics["list_like_line_ratio"] >= 0.6
        and metrics["maximum_consecutive_list_like_lines"] >= 4
        and metrics["prose_token_ratio"] < 0.6
    )
    if (
        ratio_dominated
        or consecutive_dominated
        or repeated_dominated
        or multiple_blocks_dominated
        or structured_statistics_dominated
    ):
        return ChunkQualityResult(warnings, "list_dominated", metrics)
    if boundary_start_type == "token_fallback" or boundary_end_type == "token_fallback":
        if training_mode:
            return ChunkQualityResult(warnings, "token_fallback_boundary", metrics)
        warnings = (*warnings, "token_fallback_boundary")
    elif boundary_start_type == "word_fallback" or boundary_end_type == "word_fallback":
        warnings = (*warnings, "word_fallback_boundary")
    return ChunkQualityResult(tuple(dict.fromkeys(warnings)), None, metrics)
