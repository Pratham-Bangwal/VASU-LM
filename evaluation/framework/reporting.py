"""Atomic, resumable capability-run reporting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def atomic_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl_atomic(path: Path, record: dict[str, Any]) -> None:
    """Append one deduplicated result via atomic whole-file replacement."""

    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    identity = (record["checkpoint_id"], record["task_id"], record["seed"])
    if any((item["checkpoint_id"], item["task_id"], item["seed"]) == identity for item in records):
        return
    records.append(record)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    os.replace(temporary, path)


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    categories: dict[str, list[bool]] = {}
    heuristics: dict[str, list[dict[str, Any]]] = {}
    for record in values:
        score = record["score"]
        if score["metric_kind"] == "objective" and score["passed"] is not None:
            categories.setdefault(record["category"], []).append(bool(score["passed"]))
        else:
            heuristics.setdefault(record["category"], []).append(score["repetition"])
    return {
        "objective_categories": {name: {"count": len(items), "accuracy": sum(items) / len(items)} for name, items in categories.items()},
        "heuristic_categories": {
            name: {
                "count": len(items),
                "mean_repeated_bigram_ratio": sum(
                    item.get("repeated_bigram_ratio", 0.0) for item in items
                ) / len(items),
            }
            for name, items in heuristics.items()
        },
    }


def _human_review_state(records: list[dict[str, Any]], required: bool) -> dict[str, Any]:
    human_records = [item for item in records if item["score"]["metric_kind"] == "human"]
    if not required:
        return {"required": False, "complete": True, "approved": True, "reasons": []}
    reasons: list[str] = []
    completed = 0
    approved = True
    if not human_records:
        reasons.append("required human-review tasks are absent")
    for item in human_records:
        review = item.get("human_review")
        try:
            validate_human_review(review)
        except ValueError:
            reasons.append(f"human review missing or malformed for {item['task_id']}")
            continue
        assert isinstance(review, dict)
        completed += 1
        if review["status"] == "rejected":
            approved = False
            reasons.append(f"human review rejected {item['task_id']}")
    return {
        "required": True,
        "complete": completed == len(human_records) and bool(human_records),
        "approved": approved,
        "reasons": reasons,
    }


def validate_human_review(value: object) -> None:
    """Validate the stable, user-supplied human-review record shape."""

    if not isinstance(value, dict) or set(value) - {"status", "reviewer", "notes"}:
        raise ValueError("human review must be an object with status, reviewer, and optional notes")
    if value.get("status") not in {"approved", "rejected"}:
        raise ValueError("human review status must be approved or rejected")
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("human review requires a nonblank reviewer")
    if "notes" in value and not isinstance(value["notes"], str):
        raise ValueError("human review notes must be a string when present")


def _category_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    return summarize(records)["objective_categories"]


def _compare_categories(
    parent: dict[str, dict[str, float | int]],
    candidate: dict[str, dict[str, float | int]],
    categories: list[str],
    *,
    threshold: float,
    kind: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    calculations: list[dict[str, Any]] = []
    blockers: list[str] = []
    for category in categories:
        if category not in parent or category not in candidate:
            blockers.append(f"missing parent or candidate objective category: {category}")
            continue
        parent_accuracy = float(parent[category]["accuracy"])
        candidate_accuracy = float(candidate[category]["accuracy"])
        improvement = candidate_accuracy - parent_accuracy
        regression = parent_accuracy - candidate_accuracy
        passed = improvement >= threshold if kind == "improvement" else regression <= threshold
        calculations.append({"category": category, "parent_accuracy": parent_accuracy, "candidate_accuracy": candidate_accuracy, "improvement_delta": improvement, "regression": regression, "threshold": threshold, "comparison": ">=" if kind == "improvement" else "<=", "passed": passed})
        if not passed:
            blockers.append(f"{category} {kind} gate failed")
    return calculations, blockers


def _gate_result(
    name: str,
    calculations: list[dict[str, Any]],
    blockers: list[str],
) -> dict[str, Any]:
    """Keep every configured gate independently inspectable in the report."""

    return {
        "name": name,
        "passed": not blockers,
        "calculations": calculations,
        "blocking_reasons": blockers,
    }


def promotion_report(
    records: Iterable[dict[str, Any]], *, parent: str, candidate: str,
    promotion_type: str, gate: dict[str, Any],
) -> dict[str, Any]:
    """Apply explicit inclusive promotion gates without a combined score.

    Regression is ``parent_accuracy - candidate_accuracy`` and improvement is
    ``candidate_accuracy - parent_accuracy``. Equality passes every boundary.
    """

    records_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_checkpoint.setdefault(record["checkpoint_id"], []).append(record)
    if parent not in records_by_checkpoint or candidate not in records_by_checkpoint:
        raise ValueError("Parent and candidate both require completed results.")
    if promotion_type not in {"continued_pretraining", "instruction", "conversation"}:
        raise ValueError(f"Unknown promotion type: {promotion_type}")
    parent_summary = summarize(records_by_checkpoint[parent])
    candidate_summary = summarize(records_by_checkpoint[candidate])
    parent_objective = _category_metrics(records_by_checkpoint[parent])
    candidate_objective = _category_metrics(records_by_checkpoint[candidate])
    gate_results: list[dict[str, Any]] = []
    blockers: list[str] = []
    if promotion_type == "continued_pretraining":
        items, failures = _compare_categories(parent_objective, candidate_objective, gate["targeted_categories"], threshold=float(gate["targeted_objective_delta_minimum"]), kind="improvement")
        gate_results.append(_gate_result("targeted_objective_delta", items, failures))
        blockers.extend(failures)
        items, failures = _compare_categories(parent_objective, candidate_objective, gate["general_categories"], threshold=float(gate["general_objective_regression_maximum"]), kind="regression")
        gate_results.append(_gate_result("general_objective_regression", items, failures))
        blockers.extend(failures)
    elif promotion_type == "instruction":
        for label in ("formatting", "factual"):
            items, failures = _compare_categories(parent_objective, candidate_objective, gate[f"{label}_categories"], threshold=float(gate[f"{label}_regression_maximum"]), kind="regression")
            gate_results.append(_gate_result(f"{label}_regression", items, failures))
            blockers.extend(failures)
        if "other_objective_regression_maximum" in gate:
            items, failures = _compare_categories(parent_objective, candidate_objective, gate["other_objective_categories"], threshold=float(gate["other_objective_regression_maximum"]), kind="regression")
            gate_results.append(_gate_result("other_objective_regression", items, failures))
            blockers.extend(failures)
    else:
        items, failures = _compare_categories(parent_objective, candidate_objective, gate["objective_categories"], threshold=float(gate["objective_regression_maximum"]), kind="regression")
        gate_results.append(_gate_result("objective_regression", items, failures))
        blockers.extend(failures)
    human = _human_review_state(records_by_checkpoint[candidate], bool(gate["human_review_required"]))
    human_blockers = []
    if human["required"] and (not human["complete"] or not human["approved"]):
        human_blockers = human["reasons"] or ["required human review is incomplete"]
        blockers.extend(human_blockers)
    gate_results.append(_gate_result("human_review", [], human_blockers))
    return {
        "promotion_type": promotion_type, "resolved_gate": gate,
        "parent": parent, "candidate": candidate,
        "parent_objective": parent_summary["objective_categories"],
        "candidate_objective": candidate_summary["objective_categories"],
        "parent_heuristic": parent_summary["heuristic_categories"],
        "candidate_heuristic": candidate_summary["heuristic_categories"],
        "gate_results": gate_results, "human_review": human,
        "blocking_reasons": blockers, "status": "passed" if not blockers else "blocked",
    }
