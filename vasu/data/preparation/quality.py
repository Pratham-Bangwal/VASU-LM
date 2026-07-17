"""Conservative Unicode repair and text-quality checks for factual sources."""

from __future__ import annotations

from dataclasses import dataclass
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
