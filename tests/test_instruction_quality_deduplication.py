from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from vasu.data.instruction_quality import (
    exact_duplicate_groups,
    load_jsonl,
    load_review_decisions,
    make_review_decision,
    near_duplicate_candidates,
    normalize_for_comparison,
    quality_score,
    source_record_hash,
    transition_allowed,
    validate_review_decisions,
)


DEMO = Path("data/examples/instruct/vasu_instruction_quality_v1_demo.jsonl")


def records() -> list[dict]:
    return load_jsonl(DEMO)


def test_comparison_normalization_does_not_change_source() -> None:
    text = "  Tokyo\r\nIS   a City. "
    assert normalize_for_comparison(text) == "tokyo is a city."
    assert text.startswith("  ")


def test_exact_duplicates_are_reported_not_deleted() -> None:
    left = records()[0]
    right = deepcopy(left)
    right["example_id"] = "viq1_999999"
    groups = exact_duplicate_groups([left, right])
    assert groups
    assert all(len(group["example_ids"]) == 2 for group in groups)


def test_near_duplicate_candidates_are_deterministic() -> None:
    left = records()[3]
    right = deepcopy(left)
    right["example_id"] = "viq1_999999"
    right["response"] += " Today."
    first = near_duplicate_candidates([left, right], 0.5)
    assert first == near_duplicate_candidates([left, right], 0.5)
    assert first[0]["left"] == left["example_id"]


def test_near_duplicate_threshold_is_enforced() -> None:
    assert near_duplicate_candidates(records()[:2], 1.0) == []


def test_stale_review_hash_fails() -> None:
    record = records()[0]
    decision = make_review_decision(record, "approved", "human", "checked")
    decision["source_sha256"] = "0" * 64
    findings = validate_review_decisions([record], {record["example_id"]: decision})
    assert findings[0].code == "stale_review"


def test_current_review_hash_passes() -> None:
    record = records()[0]
    decision = make_review_decision(record, "approved", "human", "checked")
    assert decision["source_sha256"] == source_record_hash(record)
    assert validate_review_decisions([record], {record["example_id"]: decision}) == []


def test_status_transitions_are_explicit() -> None:
    assert transition_allowed("unreviewed", "approved")
    assert not transition_allowed("approved", "unreviewed")
    assert transition_allowed("approved", "needs_fact_check")


def test_quality_score_never_auto_approves() -> None:
    result = quality_score(records()[0], [], False)
    assert result == {
        "score": 90,
        "reasons": [{"code": "very_short_response", "points": -10}],
        "automatic_approval": False,
    }


def test_duplicate_risk_reduces_priority_score() -> None:
    result = quality_score(records()[0], [], True)
    assert result["score"] == 75
    assert result["automatic_approval"] is False


def test_duplicate_review_decisions_fail(tmp_path: Path) -> None:
    decision = make_review_decision(records()[0], "approved", "human", "checked")
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        f"{json.dumps(decision)}\n{json.dumps(decision)}\n",
        encoding="utf-8",
    )
    try:
        load_review_decisions(review_path)
    except ValueError as error:
        assert "duplicate review decision" in str(error)
    else:
        raise AssertionError("duplicate decisions must fail")
