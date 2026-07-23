"""Unit coverage for VASU's internal capability-suite infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.framework.registry import load_checkpoint_registry, load_suite
from evaluation.framework.reporting import append_jsonl_atomic, promotion_report
from evaluation.framework.runner import (
    load_human_review_results,
    parse_args,
    validate_args,
    validate_resume_manifest,
)
from evaluation.framework.scoring import parse_single_number, repetition_metrics, score_task
from evaluation.framework.schemas import CapabilityTask


def _task(rule: str, scoring: dict) -> CapabilityTask:
    return CapabilityTask("id", "formatting", "prompt", "objective", {"rule": rule, **scoring}, {"source": "test"}, "v1")


def test_numeric_and_structural_scoring() -> None:
    assert parse_single_number("42") == 42.0
    assert parse_single_number("42 and 43") is None
    assert parse_single_number("7 + 8 is 7 + 8.") is None
    assert score_task(_task("number", {"answer": 7}), "7")["passed"]
    assert score_task(_task("json_keys", {"keys": ["a"]}), '{"a": 1}')["passed"]
    assert score_task(_task("count_lines", {"count": 2}), "a\nb")["passed"]


def test_repetition_metrics_are_length_aware() -> None:
    metrics = repetition_metrics("one two one two one two.")
    assert metrics["repeated_bigram_ratio"] > 0
    assert not metrics["empty"]


def test_suite_and_legacy_registry_validation(tmp_path: Path) -> None:
    suite = {"suite_version": "v1", "tasks": [{"id": "a", "category": "x", "prompt": "p", "metric_kind": "objective", "scoring": {"rule": "exact", "answers": ["a"]}, "provenance": {"source": "test"}}]}
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    assert load_suite(suite_path)[1][0].identifier == "a"
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"checkpoints": {"legacy": "model.pt"}}), encoding="utf-8"
    )
    assert load_checkpoint_registry(registry_path)["legacy"].model_config == "vasu_31m"


def test_atomic_resume_does_not_duplicate_records(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    record = {"checkpoint_id": "a", "task_id": "b", "seed": None}
    append_jsonl_atomic(path, record)
    append_jsonl_atomic(path, record)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def _record(checkpoint: str, category: str, passed: bool | None, *, human: dict | None = None) -> dict:
    record = {
        "checkpoint_id": checkpoint,
        "task_id": f"{checkpoint}-{category}-{passed}",
        "category": category,
        "score": {
            "metric_kind": "human" if passed is None else "objective",
            "passed": passed,
            "repetition": {"repeated_bigram_ratio": 0.0},
        },
    }
    if human is not None:
        record["human_review"] = human
    return record


def _gate(kind: str) -> dict:
    gates = {
        "continued_pretraining": {"targeted_categories": ["target"], "general_categories": ["general"], "targeted_objective_delta_minimum": 0.0, "general_objective_regression_maximum": 0.1, "human_review_required": False},
        "instruction": {"formatting_categories": ["formatting"], "factual_categories": ["factual"], "formatting_regression_maximum": 0.1, "factual_regression_maximum": 0.1, "human_review_required": True},
        "conversation": {"objective_categories": ["general"], "objective_regression_maximum": 0.1, "human_review_required": True},
    }
    return gates[kind]


def _report(records: list[dict], kind: str) -> dict:
    return promotion_report(records, parent="parent", candidate="candidate", promotion_type=kind, gate=_gate(kind))


def test_continued_pretraining_passes_and_threshold_equality_is_inclusive() -> None:
    records = [_record("parent", "target", True), _record("candidate", "target", True), _record("parent", "general", True), _record("candidate", "general", True)]
    assert _report(records, "continued_pretraining")["status"] == "passed"

    equality_records = [
        *[_record("parent", "target", True) for _ in range(10)],
        *[_record("candidate", "target", True) for _ in range(10)],
        *[_record("parent", "general", True) for _ in range(10)],
        *[_record("candidate", "general", True) for _ in range(9)],
        _record("candidate", "general", False),
    ]
    assert _report(equality_records, "continued_pretraining")["status"] == "passed"


def test_targeted_improvement_below_threshold_blocks() -> None:
    gate = _gate("continued_pretraining")
    gate["targeted_objective_delta_minimum"] = 0.5
    records = [_record("parent", "target", True), _record("candidate", "target", True), _record("parent", "general", True), _record("candidate", "general", True)]
    assert promotion_report(records, parent="parent", candidate="candidate", promotion_type="continued_pretraining", gate=gate)["status"] == "blocked"


def test_general_and_instruction_regressions_block() -> None:
    continued = [_record("parent", "target", True), _record("candidate", "target", True), _record("parent", "general", True), _record("candidate", "general", False)]
    assert _report(continued, "continued_pretraining")["status"] == "blocked"
    instruction = [_record("parent", "formatting", True), _record("candidate", "formatting", False), _record("parent", "factual", True), _record("candidate", "factual", True), _record("candidate", "human", None, human={"status": "approved", "reviewer": "tester"})]
    assert _report(instruction, "instruction")["status"] == "blocked"


def test_instruction_factual_and_conversation_regressions_block() -> None:
    instruction = [_record("parent", "formatting", True), _record("candidate", "formatting", True), _record("parent", "factual", True), _record("candidate", "factual", False), _record("candidate", "human", None, human={"status": "approved", "reviewer": "tester"})]
    assert _report(instruction, "instruction")["status"] == "blocked"
    conversation = [_record("parent", "general", True), _record("candidate", "general", False), _record("candidate", "human", None, human={"status": "approved", "reviewer": "tester"})]
    assert _report(conversation, "conversation")["status"] == "blocked"


def test_human_review_semantics_and_missing_categories() -> None:
    records = [_record("parent", "formatting", True), _record("candidate", "formatting", True), _record("parent", "factual", True), _record("candidate", "factual", True), _record("candidate", "human", None)]
    assert _report(records, "instruction")["status"] == "blocked"
    records[-1]["human_review"] = {"status": "approved", "reviewer": "tester"}
    report = _report(records, "instruction")
    assert report["status"] == "passed"
    assert "parent_heuristic" in report
    assert "candidate_heuristic" in report
    assert all("passed" in gate for gate in report["gate_results"])
    missing = [_record("parent", "target", True), _record("candidate", "target", True)]
    assert _report(missing, "continued_pretraining")["status"] == "blocked"


def test_runner_requires_promotion_type_with_parent() -> None:
    args = parse_args([
        "--suite", "suite.json", "--checkpoints", "candidate", "--parent",
        "parent", "--output-dir", "out",
    ])
    try:
        validate_args(args)
    except ValueError as error:
        assert "--parent requires --promotion-type" in str(error)
    else:
        raise AssertionError("parent comparisons must require a promotion type")

    inverse = parse_args([
        "--suite", "suite.json", "--checkpoints", "candidate",
        "--promotion-type", "instruction", "--output-dir", "out",
    ])
    try:
        validate_args(inverse)
    except ValueError as error:
        assert "--promotion-type requires --parent" in str(error)
    else:
        raise AssertionError("promotion type must require a parent checkpoint")


def test_human_review_file_requires_structured_human_task_results(
    tmp_path: Path,
) -> None:
    human_task = CapabilityTask(
        "review", "general", "prompt", "human", {"rule": "human"}, {}, "v1"
    )
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps({"review": {"status": "approved", "reviewer": "tester"}}),
        encoding="utf-8",
    )
    assert load_human_review_results(path, [human_task])["review"]["status"] == "approved"
    path.write_text(json.dumps({"review": {"status": "pending"}}), encoding="utf-8")
    try:
        load_human_review_results(path, [human_task])
    except ValueError as error:
        assert "Invalid human review" in str(error)
    else:
        raise AssertionError("invalid human review must fail clearly")


def test_runner_rejects_changed_promotion_configuration_on_resume() -> None:
    existing = {
        "suite_sha256": "suite",
        "checkpoint_ids": ["parent", "candidate"],
        "generation_settings": {"mode": "greedy"},
        "seeds": [None],
        "promotion": {"type": "instruction", "gate_sha256": "old"},
    }
    current = {**existing, "promotion": {"type": "instruction", "gate_sha256": "new"}}
    try:
        validate_resume_manifest(existing, current)
    except ValueError as error:
        assert "promotion differs" in str(error)
    else:
        raise AssertionError("changed gate configuration must reject resume")


def test_unknown_promotion_type_fails_clearly() -> None:
    records = [_record("parent", "general", True), _record("candidate", "general", True)]
    try:
        promotion_report(records, parent="parent", candidate="candidate", promotion_type="unknown", gate={})
    except ValueError as error:
        assert "Unknown promotion type" in str(error)
    else:
        raise AssertionError("unknown promotion type must fail")
