"""Build a frozen, assistant-authored VASU-140M internal diagnostic fixture.

This module deliberately creates development-only fixtures. It never creates
held-out material, claims independent curation, or authorizes evaluation or
training.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from evaluation.framework.vasu_140m_base_v2 import canonical_json
from evaluation.framework.vasu_140m_base_v2_inventory_builder import (
    AUTHORING_SCHEMA_ID,
    build_fixture_inventory,
)


SUITE_ID = "vasu_140m_assistant_authored_internal_v1"
REPORT_SCHEMA_ID = "vasu_140m_assistant_authored_internal_suite_report_v1"
COUNTS = {
    "factuality": 200,
    "arithmetic": 1_000,
    "repetition": 120,
    "robustness": 120,
    "manual_review": 60,
}
SCORER_PATH = "evaluation/framework/vasu_140m_base_v2_tasks.py"
RETRIEVED_AT = "2026-08-02T00:00:00+00:00"

# Deliberately small, stable fact seeds. The suite is a diagnostic, not an
# externally sourced factual benchmark; provenance records say so explicitly.
FACTS = (
    ("the Red Planet", "Mars"),
    ("the largest planet in the Solar System", "Jupiter"),
    ("the planet closest to the Sun", "Mercury"),
    ("Earth's natural satellite", "the Moon"),
    ("the chemical symbol H", "hydrogen"),
    ("the chemical symbol O", "oxygen"),
    ("the chemical formula H2O", "water"),
    ("the process plants use to convert light into chemical energy", "photosynthesis"),
    ("the number of metres in one kilometre", "1000"),
    ("the number of continents customarily counted on Earth", "7"),
    ("the largest ocean on Earth", "the Pacific Ocean"),
    ("the capital of France", "Paris"),
    ("the capital of Japan", "Tokyo"),
    ("the capital of Canada", "Ottawa"),
    ("the capital of Australia", "Canberra"),
    ("the organ that pumps blood through the human body", "the heart"),
    ("the number of chambers in a typical human heart", "4"),
    ("the gas humans require for ordinary aerobic respiration", "oxygen"),
    ("the first element in the periodic table", "hydrogen"),
    ("the nearest star to Earth", "the Sun"),
)
FACT_TEMPLATES = (
    "What is {subject}?",
    "Name {subject}.",
    "Give the standard answer for {subject}.",
    "In basic general knowledge, identify {subject}.",
    "Which answer correctly names {subject}?",
    "Provide only the answer for {subject}.",
    "Complete this factual prompt: {subject} is",
    "State {subject}.",
    "What term or value corresponds to {subject}?",
    "Answer this concise question: what is {subject}?",
)


def _provenance() -> dict[str, object]:
    return {
        "source_name": "VASU assistant-authored internal diagnostic suite",
        "source_url": "https://github.com/Pratham-Bangwal/VASU-LM",
        "license_name": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_revision": "assistant-authored-internal-v1",
        "citation": (
            "Assistant-authored internal diagnostic under explicit project-owner "
            "authorization; not independent curation or source-admission evidence."
        ),
        "retrieved_at": RETRIEVED_AT,
        "human_authored": False,
    }


def _item(
    *,
    dimension: str,
    ordinal: int,
    strata: Mapping[str, str],
    content: Mapping[str, object],
    scoring: Mapping[str, object],
) -> dict[str, object]:
    item_id = f"assistant-{dimension}-{ordinal:04d}"
    family = f"assistant-{dimension}-family-{ordinal:04d}"
    return {
        "schema_id": AUTHORING_SCHEMA_ID,
        "item_id": item_id,
        "dimension": dimension,
        "semantic_family_id": family,
        "parent_document_id": f"assistant-authored/{dimension}/{ordinal:04d}",
        "strata": {"semantic_family": family, **dict(strata)},
        "content": dict(content),
        "scoring": dict(scoring),
        "provenance": _provenance(),
        # This records the project owner's explicit authorization to create the
        # fixture, not a claim that the prompt text was human-authored.
        "human_approved": True,
    }


def factuality_items() -> list[dict[str, object]]:
    answers = [answer for _, answer in FACTS]
    records = []
    for ordinal in range(200):
        fact_index = ordinal % len(FACTS)
        subject, answer = FACTS[fact_index]
        distractors = [
            answers[(fact_index + offset) % len(answers)]
            for offset in (3, 7, 11)
        ]
        choices = [answer]
        for distractor in distractors:
            if distractor not in choices:
                choices.append(distractor)
        while len(choices) < 4:
            candidate = answers[(fact_index + len(choices) + 13) % len(answers)]
            if candidate not in choices:
                choices.append(candidate)
        records.append(
            _item(
                dimension="factuality",
                ordinal=ordinal + 1,
                strata={"task_family": "assistant_authored_general_knowledge"},
                content={
                    "prompt": FACT_TEMPLATES[ordinal // len(FACTS)].format(subject=subject),
                    "choices": [
                        {"choice_id": chr(97 + index), "text": value}
                        for index, value in enumerate(choices)
                    ],
                },
                scoring={"correct_choice_id": "a"},
            )
        )
    return records


def arithmetic_items() -> list[dict[str, object]]:
    records = []
    operation_names = {"+": "addition", "-": "subtraction", "*": "multiplication", "/": "division"}
    for ordinal in range(1, 1_001):
        block, offset = divmod(ordinal - 1, 250)
        if block == 0:
            left, right, operation = 1_003 + 17 * offset, 2_009 + 13 * offset, "+"
            answer = left + right
        elif block == 1:
            left, right, operation = 10_007 + 23 * offset, 131 + 7 * offset, "-"
            answer = left - right
        elif block == 2:
            left, right, operation = 31 + offset, 17 + (offset % 19), "*"
            answer = left * right
        else:
            right, answer, operation = 2 + (offset % 23), 13 + offset, "/"
            left = right * answer
        records.append(
            _item(
                dimension="arithmetic",
                ordinal=ordinal,
                strata={
                    "operation": operation_names[operation],
                    "difficulty": "internal_diagnostic",
                    "template_family": f"assistant_{operation_names[operation]}_v1",
                },
                content={"prompt": f"Calculate exactly: {left} {operation} {right} = ?"},
                scoring={"answer_type": "integer", "expected_answer": str(answer)},
            )
        )
    return records


def repetition_items() -> list[dict[str, object]]:
    settings = ("harbor", "library", "orchard", "workshop", "station", "garden", "museum", "valley", "theatre", "observatory")
    objects = ("map", "ledger", "lantern", "compass", "notebook", "clock", "bridge", "telescope", "cabinet", "fountain", "gate", "path")
    return [
        _item(
            dimension="repetition",
            ordinal=ordinal,
            strata={"prompt_family": "assistant_scene_continuation"},
            content={
                "prompt": (
                    f"Continue this short scene coherently: at the {settings[(ordinal - 1) % len(settings)]}, "
                    f"the caretaker checked the {objects[(ordinal - 1) % len(objects)]} before recording observation {ordinal}."
                )
            },
            scoring={"loop_ngram_size": 3},
        )
        for ordinal in range(1, 121)
    ]


def robustness_items() -> list[dict[str, object]]:
    return [
        _item(
            dimension="robustness",
            ordinal=ordinal,
            strata={"variant_kind": "case_punctuation_instruction_paraphrase"},
            content={
                "baseline_prompt": f"Calculate exactly: {41 + 7 * (ordinal - 1)} + {9 + ((ordinal - 1) % 11)} = ?",
                "variant_prompt": f"calculate EXACTLY — {41 + 7 * (ordinal - 1)} plus {9 + ((ordinal - 1) % 11)}; answer with only the integer.",
            },
            scoring={"accepted_answers": [str(50 + 7 * (ordinal - 1) + ((ordinal - 1) % 11))]},
        )
        for ordinal in range(1, 121)
    ]


def manual_review_items() -> list[dict[str, object]]:
    topics = ("rainfall", "photosynthesis", "a library catalog", "a bicycle repair", "a volcanic eruption", "a community garden", "a solar eclipse", "a data backup", "a recycling program", "a river ecosystem", "a telescope", "a local history archive", "a cooking recipe", "a map legend", "a school science fair", "a public transit route", "a forest trail", "a weather report", "a museum exhibit", "a computer program")
    styles = ("Explain", "Give a concise introduction to", "Describe the main idea behind")
    return [
        _item(
            dimension="manual_review",
            ordinal=ordinal,
            strata={"category": "coherence_and_grounding"},
            content={"prompt": f"{styles[(ordinal - 1) // len(topics)]} {topics[(ordinal - 1) % len(topics)]} for a curious beginner."},
            scoring={"rubric_dimensions": ["coherence", "factual_support", "degeneration"]},
        )
        for ordinal in range(1, 61)
    ]


def generated_authoring_items() -> dict[str, list[dict[str, object]]]:
    records = {
        "factuality": factuality_items(),
        "arithmetic": arithmetic_items(),
        "repetition": repetition_items(),
        "robustness": robustness_items(),
        "manual_review": manual_review_items(),
    }
    if {name: len(values) for name, values in records.items()} != COUNTS:
        raise AssertionError("unexpected internal-suite record count")
    return records


def build_suite(*, repository_root: Path, output_directory: str, repository_commit: str) -> dict[str, object]:
    """Create immutable development fixtures and return a non-authorizing report."""

    if len(repository_commit) != 40 or any(char not in "0123456789abcdef" for char in repository_commit):
        raise ValueError("repository_commit must be a lowercase Git commit")
    root = repository_root.resolve()
    relative = Path(output_directory.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output directory must be repository-relative")
    output = root / relative
    if output.exists():
        raise FileExistsError("internal suite output already exists")
    if not output.parent.is_dir():
        raise ValueError("internal suite output parent must already exist")
    output.mkdir()
    source = generated_authoring_items()
    manifests = {}
    for dimension, items in source.items():
        manifests[dimension] = build_fixture_inventory(
            repository_root=root,
            output_directory=(relative / dimension).as_posix(),
            inventory_id=f"{SUITE_ID}-{dimension}-development",
            suite_id=SUITE_ID,
            repository_commit=repository_commit,
            dimension=dimension,
            scorer={"path": SCORER_PATH, "sha256": hashlib.sha256((root / SCORER_PATH).read_bytes()).hexdigest()},
            authoring_items=items,
        )
    report: dict[str, object] = {
        "schema_id": REPORT_SCHEMA_ID,
        "suite_id": SUITE_ID,
        "repository_commit": repository_commit,
        "authoring": {
            "provenance": "assistant_authored_internal_nonindependent",
            "independently_curated": False,
            "held_out_content_present": False,
            "training_data_eligible": False,
        },
        "inventory_sha256s": {name: manifest["inventory_sha256"] for name, manifest in manifests.items()},
        "record_counts": {name: int(manifest["payload"]["record_count"]) for name, manifest in manifests.items()},
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    with (output / "suite_report.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report
