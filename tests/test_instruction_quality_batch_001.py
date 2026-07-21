from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from vasu.data.instruction_quality import (
    exact_duplicate_groups,
    load_jsonl,
    near_duplicate_candidates,
    validate_format_constraint,
    validate_record,
    write_jsonl,
)
from vasu.data.instruction_quality_batch_001 import (
    EXPECTED_COUNTS,
    EXPECTED_DIFFICULTY,
    author_batch,
    compare_with_demo,
    token_length_audit,
)


SOURCE = Path("data/raw/instruct/vasu_instruction_quality_v1_batch_001.jsonl")
CONFIG = Path("configs/data/vasu_instruction_quality_v1_batch_001.json")
REVIEW = Path("data/reviews/instruct/vasu_instruction_quality_v1_batch_001_review.jsonl")
MANIFEST = Path("data/manifests/instruct/vasu_instruction_quality_v1_batch_001_source.json")
DEMO = Path("data/examples/instruct/vasu_instruction_quality_v1_demo.jsonl")
DEMO_SHA256 = "87e64502e0aefeb91dec8a38cbaec9500024b8041ae9f7c0d1962ff07522cd45"
GATE_V2 = Path("evaluation/results/instruction_quality_batch_001_gate_v2")


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return load_jsonl(SOURCE)


def test_exactly_500_records_are_produced(records: list[dict]) -> None:
    assert len(records) == len(author_batch()) == 500


def test_category_and_difficulty_counts_match(records: list[dict]) -> None:
    assert dict(Counter(row["capability"] for row in records)) == EXPECTED_COUNTS
    assert dict(Counter(row["difficulty"] for row in records)) == EXPECTED_DIFFICULTY


def test_ids_are_unique_sequential_and_ordered(records: list[dict]) -> None:
    assert [row["example_id"] for row in records] == [
        f"viq1_b001_{index:06d}" for index in range(1, 501)
    ]


def test_schema_review_and_language_are_fixed(records: list[dict]) -> None:
    assert {row["schema_version"] for row in records} == {"vasu_instruction_quality_v1"}
    assert {row["language"] for row in records} == {"en"}
    expected_quality = {
        "author_status": "draft",
        "review_status": "unreviewed",
        "reviewer_id": None,
        "review_notes": None,
    }
    assert all(row["quality"] == expected_quality for row in records)


def test_review_decision_file_contains_all_approved_records() -> None:
    reviews = load_jsonl(REVIEW)

    assert len(reviews) == 500
    assert {row["status"] for row in reviews} == {"approved"}
    assert len({row["example_id"] for row in reviews}) == 500


def test_demo_fixture_is_unchanged() -> None:
    assert hashlib.sha256(DEMO.read_bytes()).hexdigest() == DEMO_SHA256


def test_all_records_pass_existing_validator(records: list[dict]) -> None:
    assert all(validate_record(row) == [] for row in records)


def test_no_exact_duplicates_within_or_against_demo(records: list[dict]) -> None:
    assert exact_duplicate_groups(records) == []
    comparison = compare_with_demo(records, load_jsonl(DEMO), 0.85)
    assert comparison["exact_duplicate_groups"] == []


def test_near_duplicate_detection_is_deterministic_and_empty(records: list[dict]) -> None:
    first = near_duplicate_candidates(records, 0.85)
    assert first == near_duplicate_candidates(records, 0.85) == []
    assert compare_with_demo(records, load_jsonl(DEMO), 0.85)["near_duplicate_candidates"] == []


def test_factual_records_have_verified_http_sources(records: list[dict]) -> None:
    factual = [row for row in records if row["capability"] == "short_factual_qa"]
    assert len(factual) == 125
    for row in factual:
        assert row["source_reference"].startswith("https://")
        assert row["facts"]
        assert all(fact["verification_status"] == "verified" for fact in row["facts"])
        assert all(fact["verification_source"].startswith("https://") for fact in row["facts"])
        assert all(fact["time_sensitive"] is False for fact in row["facts"])


def test_broad_unrelated_factual_sources_are_gone(records: list[dict]) -> None:
    banned = {
        "https://science.nasa.gov/solar-system/",
        "https://medlineplus.gov/anatomy.html",
        "https://www.britannica.com/topic/history",
        "https://www.ibm.com/think/topics/computer-science",
        "https://www.merriam-webster.com/dictionary/definition",
        "https://www.bipm.org/en/measurement-units",
    }
    factual = [row for row in records if row["capability"] == "short_factual_qa"]
    assert all(row["source_reference"] not in banned for row in factual)
    assert all(
        fact["verification_source"] == row["source_reference"]
        for row in factual
        for fact in row["facts"]
    )


def test_time_sensitive_facts_require_reference_date(records: list[dict]) -> None:
    row = deepcopy(records[0])
    row["facts"][0]["time_sensitive"] = True
    assert "factual_reference_date_missing" in {finding.code for finding in validate_record(row)}


def test_all_declared_format_constraints_match(records: list[dict]) -> None:
    assert all(validate_format_constraint(row) == [] for row in records)


def test_json_records_parse_and_have_exact_keys(records: list[dict]) -> None:
    rows = [row for row in records if row["capability"] == "json_schema_output"]
    assert len(rows) == 25
    for row in rows:
        parsed = json.loads(row["response"])
        constraint = row["format_constraints"]
        assert set(parsed) == set(constraint["required_keys"])
        assert constraint["strict"] is True


def test_rewriting_rows_have_inputs_and_expected_transformations(records: list[dict]) -> None:
    rows = [row for row in records if row["capability"] == "rewriting_transformation"]
    assert len(rows) == 75
    assert all(row["input"].strip() for row in rows)
    polite = next(row for row in rows if row["input"] == "Move your bag.")
    active = next(row for row in rows if row["input"] == "The room was cleaned by the volunteers.")
    concise = next(row for row in rows if row["input"].startswith("Due to the fact"))
    assert polite["response"] == "Could you please move your bag?"
    assert active["response"] == "The volunteers cleaned the room."
    assert concise["response"] == "The late bus made us arrive after the start."


def test_known_gate_failures_are_corrected(records: list[dict]) -> None:
    by_id = {row["example_id"]: row for row in records}
    assert by_id["viq1_b001_000072"]["instruction"] == (
        "What unit marks angles on a standard school protractor?"
    )
    assert by_id["viq1_b001_000177"]["instruction"] == (
        "Explain what the lungs do in simple terms."
    )
    assert "attachment" in by_id["viq1_b001_000228"]["response"]
    assert "needed materials" in by_id["viq1_b001_000232"]["response"]
    assert "Test yourself" in by_id["viq1_b001_000237"]["response"]
    assert "Restore a test file" in by_id["viq1_b001_000254"]["response"]
    assert by_id["viq1_b001_000379"]["response"] == (
        "I neglected to attach the document."
    )
    assert by_id["viq1_b001_000387"]["response"] == (
        "The event has moved to the main hall!"
    )


def test_known_filler_phrases_do_not_remain(records: list[dict]) -> None:
    banned = (
        "Store it in a safe place.",
        "Review the result.",
        "Check the final result.",
        "Use a clear name.",
        "Keep it organized.",
    )
    responses = "\n".join(row["response"] for row in records)
    assert all(phrase not in responses for phrase in banned)


def test_urgent_and_context_sensitive_rewrites_preserve_meaning(records: list[dict]) -> None:
    by_input = {row["input"]: row for row in records if row["input"]}
    assert by_input["Fix this error now."]["response"].endswith("now?")
    assert "previous" not in by_input["I forgot to attach the document."]["response"]
    assert "See you" not in by_input["The event has moved to the main hall."]["response"]


def test_uncertainty_rows_state_limits_without_fabricating(records: list[dict]) -> None:
    rows = [row for row in records if row["capability"] == "uncertainty_honest_fallback"]
    assert len(rows) == 25
    limit_terms = (
        "cannot",
        "unknown",
        "not know",
        "not enough",
        "unverified",
        "may be",
        "do not have",
        "not credible",
        "not supported",
        "no reliable evidence",
        "unidentified",
        "are missing",
    )
    assert all(any(term in row["response"].casefold() for term in limit_terms) for row in rows)


def test_no_response_contains_prompt_template_markers(records: list[dict]) -> None:
    markers = ("User:", "Assistant:", "### Instruction:", "### Input:", "### Response:")
    assert all(not any(marker in row["response"] for marker in markers) for row in records)


def test_token_length_and_vocabulary_audit(records: list[dict]) -> None:
    audit = token_length_audit(records, Path("assets/tokenizer.json"), 256)
    assert audit["truncated_example_ids"] == []
    assert audit["full_example_tokens"]["maximum"] <= 257
    assert audit["maximum_token_id"] < 32000


def test_authoring_is_byte_deterministic(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_jsonl(left, author_batch())
    write_jsonl(right, author_batch())
    assert left.read_bytes() == right.read_bytes() == SOURCE.read_bytes()


def test_source_manifest_hash_matches_source() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert manifest["record_count"] == 500
    assert manifest["training_authorized"] is False


def test_config_preserves_training_gate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["training_authorized"] is False


def test_production_token_mask_and_release_artifacts_exist() -> None:
    required = (
        Path("data/processed/instruct/vasu_instruction_quality_v1_batch_001.bin"),
        Path("data/processed/instruct/vasu_instruction_quality_v1_batch_001_mask.bin"),
        Path("data/processed/instruct/vasu_instruction_quality_v1_batch_001_release.jsonl"),
        Path("data/manifests/instruct/vasu_instruction_quality_v1_batch_001_release.json"),
    )

    assert all(path.exists() for path in required)
    assert all(path.stat().st_size > 0 for path in required)


def test_authoring_has_no_optimizer_or_training_invocation() -> None:
    paths = (
        Path("vasu/data/instruction_quality_batch_001.py"),
        Path("scripts/author_instruction_quality_batch_001.py"),
        Path("scripts/repair_instruction_quality_batch_001.py"),
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "optimizer.step" not in source
    assert ".backward(" not in source
    assert "Trainer(" not in source


def test_gate_v2_sample_contract_and_hash_binding(records: list[dict]) -> None:
    sample_path = GATE_V2 / "vasu_instruction_quality_v1_batch_001_gate_v2_sample.jsonl"
    manifest_path = GATE_V2 / "vasu_instruction_quality_v1_batch_001_gate_v2_manifest.json"
    sample = load_jsonl(sample_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "short_factual_qa": 25,
        "beginner_explanation": 20,
        "exact_format_following": 20,
        "rewriting_transformation": 15,
        "lists_structured_output": 10,
        "json_schema_output": 5,
        "uncertainty_honest_fallback": 5,
    }
    assert len(sample) == 100
    assert dict(Counter(row["capability"] for row in sample)) == expected
    assert [row["example_id"] for row in sample] == manifest["selected_ids"]
    assert manifest["actual_counts"] == expected
    assert manifest["review_decisions_transferred"] == 0
    assert manifest["training_authorized"] is False
    assert manifest["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert set(manifest["selected_ids"]) <= {row["example_id"] for row in records}


def test_gate_v2_selection_is_deterministic(records: list[dict]) -> None:
    manifest = json.loads(
        (GATE_V2 / "vasu_instruction_quality_v1_batch_001_gate_v2_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected_ids = []
    for capability, count in manifest["target_counts"].items():
        rows = [row for row in records if row["capability"] == capability]
        rows.sort(
            key=lambda row: (
                hashlib.sha256(f"42:{row['example_id']}".encode()).hexdigest(),
                row["example_id"],
            )
        )
        expected_ids.extend(row["example_id"] for row in rows[:count])
    assert sorted(expected_ids) == manifest["selected_ids"]


def test_gate_v2_has_no_decision_artifact() -> None:
    assert not any("decision" in path.name for path in GATE_V2.iterdir())
