"""Dataset-wide regression tests for named Wikimedia defect families."""

from __future__ import annotations

import pytest

from vasu.data.preparation.diagnostics import find_known_defects
from vasu.data.preparation.quality import assess_final_chunk


def _families(text: str) -> set[str]:
    return {finding.family for finding in find_known_defects(text)}


def _quality_rejection(text: str) -> str | None:
    return assess_final_chunk(
        text,
        token_count=max(3, len(text.split())),
        minimum_tokens=3,
        maximum_list_like_line_ratio=0.8,
        minimum_prose_sentences_for_list_chunk=2,
        boundary_start_type="paragraph",
        boundary_end_type="paragraph",
        training_mode=True,
    ).rejection_reason


@pytest.mark.parametrize(
    ("text", "family"),
    [
        (
            "The city (,; ) is the capital of the region.",
            "empty_pronunciation_or_alternate_name",
        ),
        (
            "The language (or, ) is widely spoken.",
            "connector_only_language_field",
        ),
        (
            'The word is derived from, meaning "stone".',
            "missing_etymological_source_term",
        ),
        (
            "The estimate gave a distance of Earth radii.",
            "missing_measurement_unit_or_quantity",
        ),
        (
            "The isotope of, has a short half-life.",
            "missing_isotope_identifier",
        ),
        (
            "The ruler lived from 17 May 149020 March 1568.",
            "malformed_or_joined_lifespan_date",
        ),
        (
            "The first expression is equivalent to, but the second differs.",
            "missing_formula_or_mathematical_operand",
        ),
        (
            "The four examples are:",
            "missing_quoted_passage_or_enumeration",
        ),
        (
            "The quoted passage follows:\n\nRecognition\nLater events are described.",
            "heading_or_commentary_after_omitted_material",
        ),
        (
            "The result was recorded,,, and later verified.",
            "repeated_punctuation_from_removed_markup",
        ),
        (
            "The meeting ran from July5 to July 8.",
            "joined_date_or_function_word",
        ),
    ],
)
def test_named_defect_families_are_detected(text: str, family: str) -> None:
    assert family in _families(text)


@pytest.mark.parametrize(
    "text",
    [
        "The city (,; ) is the capital of the region.",
        "The estimate gave a distance of Earth radii.",
        "The isotope of, has a short half-life.",
        "The ruler lived from 17 May 149020 March 1568.",
        "The first expression is equivalent to, but the second differs.",
        "The four examples are:",
        "The quoted passage follows:\n\nRecognition\nLater events are described.",
        "The result was recorded,,, and later verified.",
        "The meeting ran from July5 to July 8.",
    ],
)
def test_high_precision_diagnostics_are_automatic_rejections(text: str) -> None:
    assert _quality_rejection(text) is not None


def test_bibliography_dominated_chunk_is_an_automatic_reject() -> None:
    text = "\n".join(
        [
            "First work, translated by A. Editor, vol. 1, (Oxford: Press, 1998)",
            "Second work, translated by B. Editor, vol. 2, (London: Press, 1999)",
            "Third work, edited by C. Scholar, vol. 3, (Paris: Press, 2000)",
            "Fourth work, translated by D. Scholar, vol. 4, (Dublin: Press, 2001)",
        ]
    )
    findings = find_known_defects(text)
    finding = next(
        item
        for item in findings
        if item.family == "publication_adaptation_or_catalogue_dominated"
    )
    assert finding.disposition == "reject"
    assert finding.rejection_reason == "list_dominated"


def test_mixed_film_catalogue_with_substantial_prose_requires_review() -> None:
    text = (
        "The director made several influential films and described his methods "
        "in detailed interviews. His features are:\n"
        "First Film (1962)\nSecond Film (1966)\nThird Film (1972)\n"
        "Fourth Film (1975)\nFifth Film (1979)\nSixth Film (1983)\n\n"
        "The later paragraph explains the director's technique in several "
        "complete sentences. It discusses production, editing, and reception. "
        "The catalogue supports rather than replaces that explanation. The "
        "director also described how actors developed each role and how the "
        "camera shaped the audience's perspective. Contemporary critics then "
        "compared those methods with earlier films, providing detailed context "
        "for the titles. Later scholarship examined the same body of work in "
        "relation to theatre, literature, and visual art."
    )
    findings = find_known_defects(text)
    finding = next(
        item
        for item in findings
        if item.family == "publication_adaptation_or_catalogue_dominated"
    )
    assert finding.disposition == "manual_review"


def test_fragmented_structured_list_is_detected() -> None:
    text = (
        "Churches:\nNorth Church\nSouth Church\nIsland Church\n"
        "A short note.\nSchools:\nHarbor School\nVillage School\nCommunity School"
    )
    assert "fragmented_or_structured_list" in _families(text)


@pytest.mark.parametrize(
    "text",
    [
        "The city (Spanish: Ciudad Ejemplo) is the regional capital.",
        'The term derives from Greek λίθος, meaning "stone".',
        "The estimate gave a distance of 1,210 Earth radii.",
        "The isotope astatine-211 has a half-life of 7.2 hours.",
        "The ruler lived from 17 May 1490 – 20 March 1568.",
        "The associative law is (a × b) × c = a × (b × c).",
        "Examples are:\n- first complete example\n- second complete example",
        "The author made a statement:\n\n\"The complete quotation is here.\"",
        "The article cites Smith (1998) and Jones (2001) in explanatory prose.",
        "The Apollo 11 mission landed on the Moon.",
    ],
)
def test_valid_negative_controls_have_no_automatic_reject_findings(text: str) -> None:
    assert all(
        finding.disposition != "reject" for finding in find_known_defects(text)
    )
    assert _quality_rejection(text) is None


def test_ambiguous_compact_proper_name_is_review_only() -> None:
    findings = find_known_defects("The Hayabusa2 mission returned a sample.")
    assert len(findings) == 1
    assert findings[0].family == "possible_joined_alphanumeric_formatting"
    assert findings[0].disposition == "manual_review"
