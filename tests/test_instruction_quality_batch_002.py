from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from vasu.data.instruction_quality import (
    exact_duplicate_groups,
    near_duplicate_candidates,
    validate_format_constraint,
    validate_record,
    write_jsonl,
)
from vasu.data.instruction_quality_batch_002 import (
    EXPECTED_COUNTS,
    EXPECTED_DIFFICULTY,
    author_batch,
    token_length_audit,
)


CONFIG = Path("config/configs/data/vasu_instruction_quality_v1_batch_002.json")


def records() -> list[dict]:
    return author_batch()


def test_exactly_500_records_are_produced() -> None:
    assert len(records()) == 500


def test_category_and_difficulty_counts_match() -> None:
    rows = records()
    assert dict(Counter(row["capability"] for row in rows)) == EXPECTED_COUNTS
    assert dict(Counter(row["difficulty"] for row in rows)) == EXPECTED_DIFFICULTY


def test_ids_are_unique_sequential_and_ordered() -> None:
    assert [row["example_id"] for row in records()] == [
        f"viq1_b002_{index:06d}" for index in range(1, 501)
    ]


def test_schema_review_language_and_provenance_are_fixed() -> None:
    rows = records()
    assert {row["schema_version"] for row in rows} == {
        "vasu_instruction_quality_v1"
    }
    assert {row["language"] for row in rows} == {"en"}
    assert {
        row["metadata"]["provenance"] for row in rows
    } == {"purpose-written VASU instruction-quality batch 002"}
    expected_quality = {
        "author_status": "draft",
        "review_status": "unreviewed",
        "reviewer_id": None,
        "review_notes": None,
    }
    assert all(row["quality"] == expected_quality for row in rows)


def test_all_records_pass_existing_validator() -> None:
    assert all(validate_record(row) == [] for row in records())


def test_no_exact_or_near_duplicates_within_batch() -> None:
    rows = records()
    assert exact_duplicate_groups(rows) == []
    assert near_duplicate_candidates(rows, 0.85) == []


def test_factual_records_have_verified_http_sources() -> None:
    factual = [
        row for row in records()
        if row["capability"] == "short_factual_qa"
    ]
    assert len(factual) == 125
    for row in factual:
        assert row["source_reference"].startswith("https://")
        assert row["facts"]
        for fact in row["facts"]:
            assert fact["verification_status"] == "verified"
            assert fact["verification_source"] == row["source_reference"]
            assert fact["time_sensitive"] is False


def test_time_sensitive_facts_require_reference_date() -> None:
    row = deepcopy(records()[0])
    row["facts"][0]["time_sensitive"] = True
    codes = {finding.code for finding in validate_record(row)}
    assert "factual_reference_date_missing" in codes


def test_all_declared_format_constraints_match() -> None:
    assert all(validate_format_constraint(row) == [] for row in records())


def test_json_records_parse_and_have_exact_keys() -> None:
    rows = [
        row for row in records()
        if row["capability"] == "json_schema_output"
    ]
    assert len(rows) == 25
    for row in rows:
        parsed = json.loads(row["response"])
        constraint = row["format_constraints"]
        assert set(parsed) == set(constraint["required_keys"])
        assert constraint["strict"] is True


def test_rewriting_rows_have_unique_inputs() -> None:
    rows = [
        row for row in records()
        if row["capability"] == "rewriting_transformation"
    ]
    assert len(rows) == 75
    assert all(row["input"].strip() for row in rows)
    assert len({row["input"] for row in rows}) == 75


def test_uncertainty_rows_state_limits_without_fabricating() -> None:
    rows = [
        row for row in records()
        if row["capability"] == "uncertainty_honest_fallback"
    ]
    assert len(rows) == 25
    limit_terms = (
        "cannot",
        "unknown",
        "not know",
        "not enough",
        "unverified",
        "no reliable",
        "not credible",
        "do not have",
        "not authorized",
        "not necessarily",
        "no.",
    )
    assert all(
        any(term in row["response"].casefold() for term in limit_terms)
        for row in rows
    )


def test_no_response_contains_prompt_template_markers() -> None:
    markers = (
        "User:",
        "Assistant:",
        "### Instruction:",
        "### Input:",
        "### Response:",
    )
    assert all(
        not any(marker in row["response"] for marker in markers)
        for row in records()
    )


def test_token_length_and_vocabulary_audit() -> None:
    audit = token_length_audit(records(), Path("assets/tokenizer.json"), 256)
    assert audit["truncated_example_ids"] == []
    assert audit["full_example_tokens"]["maximum"] <= 257
    assert audit["maximum_token_id"] < 32000


def test_authoring_is_byte_deterministic(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    write_jsonl(left, author_batch())
    write_jsonl(right, author_batch())
    assert left.read_bytes() == right.read_bytes()


def test_config_matches_batch_contract_and_preserves_training_gate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["training_authorized"] is False
    assert config["expected_category_counts"] == EXPECTED_COUNTS
    assert config["expected_difficulty_counts"] == EXPECTED_DIFFICULTY
    assert config["authoring_version"] == (
        "vasu_instruction_quality_batch_002_authoring_v1"
    )


def test_authoring_has_no_training_invocation() -> None:
    source = Path(
        "vasu/data/instruction_quality_batch_002.py"
    ).read_text(encoding="utf-8")
    assert "optimizer.step" not in source
    assert ".backward(" not in source
    assert "Trainer(" not in source
