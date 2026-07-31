from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.smoke_vasu_140m_release_plan import run
from vasu.data.vasu_140m_records import sha256_json
from vasu.data.vasu_140m_release_plan import (
    EXPECTED_CAPABILITY_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    FROZEN_PLAN_REPORT_SHA256,
    FROZEN_PLAN_SHA256,
    load_release_plan,
    plan_identity,
    validate_frozen_plan_report,
    validate_plan_report,
    validate_release_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "configs"
    / "data"
    / "releases"
    / "vasu_140m_instruction_seed_v1.plan.json"
)
REPORT_PATH = (
    ROOT
    / "evaluation"
    / "fixtures"
    / "vasu_140m_instruction_seed_v1_plan_report.json"
)


def _plan() -> dict[str, object]:
    return load_release_plan(PLAN_PATH)


def _rehash_plan(plan: dict[str, object]) -> None:
    plan["plan_sha256"] = plan_identity(plan)


def test_repository_plan_matches_frozen_qualification() -> None:
    report = run(PLAN_PATH, ROOT, allow_existing_planned_outputs=True)
    frozen = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report == frozen
    assert report["plan_sha256"] == FROZEN_PLAN_SHA256
    assert report["report_sha256"] == FROZEN_PLAN_REPORT_SHA256
    validate_frozen_plan_report(report)


def test_source_review_deduplication_and_contamination_evidence() -> None:
    report = validate_release_plan(_plan(), ROOT, require_outputs_absent=False)
    assert report["source_example_count"] == 1000
    assert report["eligible_example_count"] == 996
    assert report["capability_counts"] == EXPECTED_CAPABILITY_COUNTS
    assert report["combined_exact_duplicate_groups"] == 0
    assert report["combined_near_duplicate_candidates"] == 0
    assert report["prompt_count"] == 2618
    assert len(report["exact_contamination_findings"]) == 4
    assert report["ngram_only_contamination_findings"] == []
    assert report["eligible_contamination_findings"] == 0


def test_split_policy_preserves_development_and_freezes_assignment() -> None:
    report = validate_release_plan(_plan(), ROOT, require_outputs_absent=False)
    assert report["split_counts"] == EXPECTED_SPLIT_COUNTS
    assert report["assignment_sha256"] == (
        "59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394"
    )


def test_published_plan_remains_non_authorizing_and_default_replay_fails_closed() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="planned production output already exists"):
        validate_release_plan(plan, ROOT)

    report = validate_release_plan(plan, ROOT, require_outputs_absent=False)
    assert plan["production_release_created"] is False
    assert plan["training_authorized"] is False
    assert plan["release_scope"]["base_checkpoint_selected"] is False
    # This is immutable historical evidence, not a claim about current paths.
    assert report["planned_outputs_absent"] is True
    assert report["release_build_permitted"] is False
    assert report["training_permitted"] is False
    for value in report["planned_output_paths"].values():
        assert (ROOT / value).exists() or value.endswith("authorization_receipts")


def test_rehashed_plan_mutation_is_rejected_by_frozen_identity() -> None:
    plan = deepcopy(_plan())
    plan["sources"][0]["source_sha256"] = "0" * 64
    _rehash_plan(plan)
    with pytest.raises(ValueError, match="frozen plan identity"):
        validate_release_plan(plan, ROOT)


def test_authorization_and_quarantine_mutations_fail_closed() -> None:
    plan = deepcopy(_plan())
    plan["training_authorized"] = True
    _rehash_plan(plan)
    with pytest.raises(ValueError, match="training_authorized"):
        validate_release_plan(plan, ROOT)

    plan = deepcopy(_plan())
    plan["release_scope"]["quarantined_example_ids"].pop()
    _rehash_plan(plan)
    with pytest.raises(ValueError, match="four unique IDs"):
        validate_release_plan(plan, ROOT)


def test_frozen_report_rejects_self_consistent_rehashed_mutation() -> None:
    report = deepcopy(validate_release_plan(_plan(), ROOT, require_outputs_absent=False))
    report["eligible_example_count"] = 995
    body = dict(report)
    del body["report_sha256"]
    report["report_sha256"] = sha256_json(body)
    with pytest.raises(ValueError, match="eligible_example_count"):
        validate_plan_report(report)

    report = deepcopy(validate_release_plan(_plan(), ROOT, require_outputs_absent=False))
    report["source_evidence"][0]["license"] = "MIT"
    body = dict(report)
    del body["report_sha256"]
    report["report_sha256"] = sha256_json(body)
    validate_plan_report(report)
    with pytest.raises(ValueError, match="frozen identity"):
        validate_frozen_plan_report(report)
