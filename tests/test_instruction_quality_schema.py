from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from vasu.data.instruction_quality import (
    CAPABILITIES,
    load_jsonl,
    validate_record,
    validate_records,
)


DEMO = Path("data/examples/instruct/vasu_instruction_quality_v1_demo.jsonl")


def records() -> list[dict]:
    return load_jsonl(DEMO)


def codes(record: dict) -> set[str]:
    return {finding.code for finding in validate_record(record)}


def test_valid_demo_records_pass() -> None:
    assert len(records()) == 21
    assert validate_records(records()) == []
    assert {record["capability"] for record in records()} == set(CAPABILITIES)


def test_invalid_schema_version_fails() -> None:
    record = deepcopy(records()[0])
    record["schema_version"] = "future"
    assert "unsupported_schema_version" in codes(record)


def test_unknown_capability_fails() -> None:
    record = deepcopy(records()[0])
    record["capability"] = "unknown"
    assert "unsupported_capability" in codes(record)


def test_empty_instruction_fails() -> None:
    record = deepcopy(records()[0])
    record["instruction"] = ""
    assert "empty_instruction" in codes(record)


def test_empty_response_fails() -> None:
    record = deepcopy(records()[0])
    record["response"] = " "
    assert "empty_response" in codes(record)


def test_duplicate_ids_fail() -> None:
    duplicate = deepcopy(records()[0])
    assert "duplicate_example_id" in {
        finding.code for finding in validate_records([records()[0], duplicate])
    }


def test_wrong_bullet_count_fails() -> None:
    record = deepcopy(records()[12])
    record["response"] = "- One\n- Two"
    assert "wrong_bullet_count" in codes(record)


def test_wrong_sentence_count_fails() -> None:
    record = deepcopy(records()[3])
    record["response"] = "Only one sentence."
    assert "wrong_sentence_count" in codes(record)


def test_one_word_constraint_is_enforced() -> None:
    record = deepcopy(records()[1])
    record["response"] = "The planet Jupiter."
    assert "one_word_violation" in codes(record)


def test_valid_strict_json_passes() -> None:
    assert codes(records()[15]) == set()


def test_invalid_json_fails() -> None:
    record = deepcopy(records()[15])
    record["response"] = "name: VASU"
    assert "invalid_json" in codes(record)


def test_required_json_keys_are_enforced() -> None:
    record = deepcopy(records()[15])
    record["response"] = json.dumps({"name": "VASU"})
    assert "missing_json_keys" in codes(record)


def test_strict_json_rejects_extra_keys() -> None:
    record = deepcopy(records()[15])
    record["response"] = json.dumps(
        {"name": "VASU", "purpose": "research", "extra": True}
    )
    assert "extra_json_keys" in codes(record)


def test_rewriting_record_preserves_required_fields() -> None:
    record = records()[9]
    assert record["input"] == "Send me the file now."
    assert record["response"] == "Could you please send me the file now?"
    assert codes(record) == set()


def test_factual_record_requires_verification() -> None:
    record = deepcopy(records()[0])
    record["facts"] = []
    assert "factual_verification_missing" in codes(record)


def test_template_marker_is_rejected() -> None:
    record = deepcopy(records()[3])
    record["response"] = "Assistant: Gravity pulls objects."
    assert "template_marker" in codes(record)


def test_placeholder_and_truncation_are_rejected() -> None:
    record = deepcopy(records()[3])
    record["response"] = "This remains [TODO]:"
    assert {"placeholder_text", "possibly_truncated"} <= codes(record)


def test_unsupported_fields_fail() -> None:
    record = deepcopy(records()[0])
    record["surprise"] = 1
    assert "unsupported_fields" in codes(record)


def test_non_normalized_line_endings_fail() -> None:
    record = deepcopy(records()[3])
    record["response"] = "First line.\r\nSecond line."
    assert "non_normalized_line_endings" in codes(record)


def test_malformed_jsonl_fails_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text('{"valid": true}\n{broken}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(path)
