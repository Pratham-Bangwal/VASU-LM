"""Deterministic statistics and fixture qualification for VASU-140M eval v2."""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from evaluation.framework.vasu_140m_base_v2 import DIMENSIONS, RESULT_MODES
from evaluation.framework.vasu_140m_base_v2_tasks import (
    canonical_json,
    score_task,
    validate_inventory,
)


QUALIFICATION_SCHEMA_ID = "vasu_140m_base_evaluation_scoring_qualification_v1"
EVIDENCE_SCHEMA_ID = (
    "vasu_140m_base_evaluation_scoring_qualification_evidence_v1"
)
SHA256_HEX_LENGTH = 64


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _git_commit(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character Git commit")
    return value


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    index = int(probability * len(sorted_values))
    return sorted_values[min(max(index, 0), len(sorted_values) - 1)]


def _perplexity(loss: float) -> float:
    if loss > math.log(sys.float_info.max):
        raise ValueError("loss is too large for finite perplexity")
    return math.exp(loss)


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int = 10_000, seed: int = 140
) -> dict[str, float | int]:
    """Return a deterministic percentile interval for a scalar mean."""

    if not values:
        raise ValueError("bootstrap values must not be empty")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    clean = [_finite(value, "bootstrap value") for value in values]
    rng = random.Random(seed)
    count = len(clean)
    means = sorted(
        math.fsum(clean[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return {
        "count": count,
        "mean": math.fsum(clean) / count,
        "ci_95_lower": _percentile(means, 0.025),
        "ci_95_upper": _percentile(means, 0.975),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _strata_summary(
    scores: Sequence[Mapping[str, object]], metric: str
) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for score in scores:
        strata = score["strata"]
        if not isinstance(strata, Mapping):
            raise ValueError("score strata must be an object")
        raw_value = score[metric]
        value = float(raw_value) if isinstance(raw_value, bool) else _finite(
            raw_value, f"score.{metric}"
        )
        for field, name in strata.items():
            grouped[str(field)][str(name)].append(value)
    return {
        field: {
            name: {
                "count": len(values),
                "mean": math.fsum(values) / len(values),
            }
            for name, values in sorted(groups.items())
        }
        for field, groups in sorted(grouped.items())
    }


def summarize_dimension_mode(
    scores: Sequence[Mapping[str, object]],
    *,
    samples: int = 10_000,
    seed: int = 140,
) -> dict[str, object]:
    """Summarize one exact dimension/mode without forming an aggregate score."""

    if not scores:
        raise ValueError("dimension scores must not be empty")
    dimensions = {str(score["dimension"]) for score in scores}
    modes = {str(score["mode"]) for score in scores}
    if len(dimensions) != 1 or len(modes) != 1:
        raise ValueError("summary requires one exact dimension and mode")
    dimension = next(iter(dimensions))
    mode = next(iter(modes))
    if mode not in RESULT_MODES[dimension]:
        raise ValueError("summary dimension/mode mismatch")
    report: dict[str, object] = {
        "dimension": dimension,
        "mode": mode,
        "count": len(scores),
    }

    if dimension == "likelihood":
        tokens = sum(int(score["target_token_count"]) for score in scores)
        total = math.fsum(float(score["total_log_likelihood"]) for score in scores)
        loss = -total / tokens
        report.update(
            {
                "target_token_count": tokens,
                "total_log_likelihood": total,
                "loss": loss,
                "perplexity": _perplexity(loss),
                "document_loss_ci": bootstrap_mean_ci(
                    [float(score["loss"]) for score in scores],
                    samples=samples,
                    seed=seed,
                ),
                "strata": _strata_summary(scores, "loss"),
            }
        )
    elif dimension == "factuality":
        normalized = [float(bool(score["normalized_correct"])) for score in scores]
        report.update(
            {
                "raw_accuracy": sum(bool(score["raw_correct"]) for score in scores)
                / len(scores),
                "normalized_accuracy": math.fsum(normalized) / len(normalized),
                "raw_tie_count": sum(bool(score["raw_tie"]) for score in scores),
                "normalized_tie_count": sum(
                    bool(score["normalized_tie"]) for score in scores
                ),
                "normalized_accuracy_ci": bootstrap_mean_ci(
                    normalized, samples=samples, seed=seed
                ),
                "strata": _strata_summary(scores, "normalized_correct"),
            }
        )
    elif dimension == "arithmetic":
        correctness = [float(bool(score["correct"])) for score in scores]
        outcomes = Counter(str(score["outcome"]) for score in scores)
        report.update(
            {
                "exact_accuracy": math.fsum(correctness) / len(correctness),
                "accuracy_ci": bootstrap_mean_ci(
                    correctness, samples=samples, seed=seed
                ),
                "outcomes": dict(sorted(outcomes.items())),
                "truncated_count": sum(
                    bool(score["truncated"]) for score in scores
                ),
                "prompt_leakage_count": sum(
                    bool(score["prompt_leakage"]) for score in scores
                ),
                "strata": _strata_summary(scores, "correct"),
            }
        )
    elif dimension == "repetition":
        repeated = [float(score["repeated_trigram_rate"]) for score in scores]
        report.update(
            {
                "mean_repeated_trigram_rate": math.fsum(repeated) / len(repeated),
                "mean_unique_token_ratio": math.fsum(
                    float(score["unique_token_ratio"]) for score in scores
                )
                / len(scores),
                "empty_rate": sum(bool(score["empty"]) for score in scores)
                / len(scores),
                "terminal_loop_rate": sum(
                    bool(score["terminal_loop"]) for score in scores
                )
                / len(scores),
                "eos_rate": sum(bool(score["eos_emitted"]) for score in scores)
                / len(scores),
                "truncation_rate": sum(
                    bool(score["truncated"]) for score in scores
                )
                / len(scores),
                "repeated_trigram_ci": bootstrap_mean_ci(
                    repeated, samples=samples, seed=seed
                ),
                "strata": _strata_summary(scores, "repeated_trigram_rate"),
            }
        )
    elif dimension == "robustness":
        consistency = [float(bool(score["consistent"])) for score in scores]
        deltas = [float(score["paired_correctness_delta"]) for score in scores]
        report.update(
            {
                "baseline_accuracy": sum(
                    bool(score["baseline_correct"]) for score in scores
                )
                / len(scores),
                "variant_accuracy": sum(
                    bool(score["variant_correct"]) for score in scores
                )
                / len(scores),
                "baseline_empty_rate": sum(
                    bool(score["baseline_empty"]) for score in scores
                )
                / len(scores),
                "variant_empty_rate": sum(
                    bool(score["variant_empty"]) for score in scores
                )
                / len(scores),
                "consistency_rate": math.fsum(consistency) / len(consistency),
                "paired_accuracy_delta": math.fsum(deltas) / len(deltas),
                "consistency_ci": bootstrap_mean_ci(
                    consistency, samples=samples, seed=seed
                ),
                "paired_delta_ci": bootstrap_mean_ci(
                    deltas, samples=samples, seed=seed + 1
                ),
                "strata": _strata_summary(scores, "consistent"),
            }
        )
    else:
        report.update(
            {
                "human_review_required": True,
                "human_judgments_present": False,
                "objective_score": None,
                "categories": dict(
                    sorted(Counter(str(score["category"]) for score in scores).items())
                ),
            }
        )
    return report


def select_manual_review(
    scores: Sequence[Mapping[str, object]], *, count: int, seed: int
) -> list[dict[str, object]]:
    """Select a deterministic category-stratified manual-review packet."""

    if not scores or count < 1 or count > len(scores):
        raise ValueError("manual sample count must be within the score count")
    if any(score["dimension"] != "manual_review" for score in scores):
        raise ValueError("manual sample may contain only manual-review scores")
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for score in scores:
        groups[str(score["category"])].append(score)
    if count < len(groups):
        raise ValueError("manual sample count must cover every category")

    def rank(score: Mapping[str, object]) -> str:
        material = f"{seed}:{score['item_id']}".encode()
        return hashlib.sha256(material).hexdigest()

    selected: list[Mapping[str, object]] = []
    for category in sorted(groups):
        selected.append(min(groups[category], key=rank))
    remaining = [score for score in scores if score not in selected]
    selected.extend(sorted(remaining, key=rank)[: count - len(selected)])
    return [
        {
            "item_id": score["item_id"],
            "task_sha256": score["task_sha256"],
            "category": score["category"],
            "response_sha256": score["response_sha256"],
            "rubric_dimensions": score["rubric_dimensions"],
            "judgment": None,
        }
        for score in sorted(selected, key=lambda value: str(value["item_id"]))
    ]


def qualification_identity(report: Mapping[str, object]) -> str:
    """Return the report identity excluding its self-hash."""

    body = dict(report)
    body.pop("qualification_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def evidence_identity(evidence: Mapping[str, object]) -> str:
    """Return compact evidence identity excluding its self-hash."""

    body = dict(evidence)
    body.pop("evidence_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def build_fixture_qualification(
    *,
    tasks: Sequence[Mapping[str, object]],
    observations: Sequence[Mapping[str, object]],
    repository_commit: str,
    schema_implementation_sha256: str,
    task_implementation_sha256: str,
    statistics_implementation_sha256: str,
    tests_sha256: str,
    smoke_sha256: str,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 140,
) -> dict[str, object]:
    """Build prompt-free scorer evidence without invoking a model or checkpoint."""

    validate_inventory(tasks)
    if any(task["split"] != "development" for task in tasks):
        raise PermissionError("fixture qualification cannot open held-out tasks")
    if len(observations) != len(tasks):
        raise ValueError("qualification requires one observation per task")
    by_id: dict[str, Mapping[str, object]] = {}
    for observation in observations:
        item_id = observation.get("item_id")
        if not isinstance(item_id, str) or item_id in by_id:
            raise ValueError("observations contain missing or duplicate item IDs")
        by_id[item_id] = observation
    if set(by_id) != {str(task["item_id"]) for task in tasks}:
        raise ValueError("observations do not cover the task inventory")

    scores = [score_task(task, by_id[str(task["item_id"])]) for task in tasks]
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for score in scores:
        grouped[(str(score["dimension"]), str(score["mode"]))].append(score)
    expected = {
        (dimension, mode)
        for dimension in DIMENSIONS
        for mode in RESULT_MODES[dimension]
    }
    if set(grouped) != expected:
        raise ValueError("qualification lacks complete dimension/mode coverage")
    summaries = [
        summarize_dimension_mode(
            grouped[key], samples=bootstrap_samples, seed=bootstrap_seed
        )
        for key in sorted(grouped)
    ]
    manual_scores = [score for score in scores if score["dimension"] == "manual_review"]
    manual_packet = select_manual_review(
        manual_scores,
        count=len(manual_scores),
        seed=bootstrap_seed,
    )
    for value, label in (
        (schema_implementation_sha256, "schema implementation"),
        (task_implementation_sha256, "task implementation"),
        (statistics_implementation_sha256, "statistics implementation"),
        (tests_sha256, "tests"),
        (smoke_sha256, "smoke"),
    ):
        _sha256(value, label)
    _git_commit(repository_commit, "repository_commit")
    report: dict[str, object] = {
        "schema_id": QUALIFICATION_SCHEMA_ID,
        "repository_commit": repository_commit,
        "schema_implementation_sha256": schema_implementation_sha256,
        "task_implementation_sha256": task_implementation_sha256,
        "statistics_implementation_sha256": statistics_implementation_sha256,
        "tests_sha256": tests_sha256,
        "smoke_sha256": smoke_sha256,
        "task_inventory_sha256": hashlib.sha256(canonical_json(tasks)).hexdigest(),
        "observations_sha256": hashlib.sha256(
            canonical_json(observations)
        ).hexdigest(),
        "score_rows_sha256": hashlib.sha256(canonical_json(scores)).hexdigest(),
        "task_count": len(tasks),
        "dimension_mode_count": len(summaries),
        "bootstrap": {
            "confidence_level": 0.95,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "summaries": summaries,
        "manual_review_packet": manual_packet,
        "fixture_only": True,
        "prompt_content_included": False,
        "held_out_opened": False,
        "model_invoked": False,
        "checkpoint_opened": False,
        "production_suite_frozen": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
        "qualification_sha256": "0" * 64,
    }
    report["qualification_sha256"] = qualification_identity(report)
    validate_fixture_qualification(report)
    return report


def validate_fixture_qualification(report: Mapping[str, object]) -> None:
    """Validate the immutable, non-authorizing qualification report."""

    expected = {
        "schema_id",
        "repository_commit",
        "schema_implementation_sha256",
        "task_implementation_sha256",
        "statistics_implementation_sha256",
        "tests_sha256",
        "smoke_sha256",
        "task_inventory_sha256",
        "observations_sha256",
        "score_rows_sha256",
        "task_count",
        "dimension_mode_count",
        "bootstrap",
        "summaries",
        "manual_review_packet",
        "fixture_only",
        "prompt_content_included",
        "held_out_opened",
        "model_invoked",
        "checkpoint_opened",
        "production_suite_frozen",
        "evaluation_run_authorized",
        "training_authorized",
        "qualification_sha256",
    }
    missing = sorted(expected - set(report))
    unknown = sorted(set(report) - expected)
    if missing or unknown:
        raise ValueError(
            f"qualification fields mismatch: missing={missing}, unknown={unknown}"
        )
    if report["schema_id"] != QUALIFICATION_SCHEMA_ID:
        raise ValueError("qualification schema identity mismatch")
    _git_commit(report["repository_commit"], "repository_commit")
    for field in (
        "schema_implementation_sha256",
        "task_implementation_sha256",
        "statistics_implementation_sha256",
        "tests_sha256",
        "smoke_sha256",
        "task_inventory_sha256",
        "observations_sha256",
        "score_rows_sha256",
        "qualification_sha256",
    ):
        _sha256(report[field], field)
    expected_mode_count = sum(len(RESULT_MODES[item]) for item in DIMENSIONS)
    if (
        isinstance(report["task_count"], bool)
        or not isinstance(report["task_count"], int)
        or report["task_count"] < 1
        or isinstance(report["dimension_mode_count"], bool)
        or not isinstance(report["dimension_mode_count"], int)
        or report["dimension_mode_count"] != expected_mode_count
    ):
        raise ValueError("qualification coverage is incomplete")
    bootstrap = report["bootstrap"]
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {
        "confidence_level",
        "samples",
        "seed",
    }:
        raise ValueError("qualification bootstrap contract is invalid")
    if (
        bootstrap["confidence_level"] != 0.95
        or isinstance(bootstrap["samples"], bool)
        or not isinstance(bootstrap["samples"], int)
        or bootstrap["samples"] < 100
        or isinstance(bootstrap["seed"], bool)
        or not isinstance(bootstrap["seed"], int)
        or bootstrap["seed"] < 0
    ):
        raise ValueError("qualification bootstrap contract is invalid")
    summaries = report["summaries"]
    if not isinstance(summaries, list) or len(summaries) != expected_mode_count:
        raise ValueError("qualification summaries must be a list")
    coverage = {
        (summary.get("dimension"), summary.get("mode"))
        for summary in summaries
        if isinstance(summary, Mapping)
    }
    expected_coverage = {
        (dimension, mode)
        for dimension in DIMENSIONS
        for mode in RESULT_MODES[dimension]
    }
    if coverage != expected_coverage:
        raise ValueError("qualification summary coverage is incomplete")
    manual_packet = report["manual_review_packet"]
    if not isinstance(manual_packet, list) or not manual_packet:
        raise ValueError("manual review packet must not be empty")
    manual_ids: set[str] = set()
    for item in manual_packet:
        if not isinstance(item, Mapping) or set(item) != {
            "item_id",
            "task_sha256",
            "category",
            "response_sha256",
            "rubric_dimensions",
            "judgment",
        }:
            raise ValueError("manual review packet item is malformed")
        item_id = item["item_id"]
        if not isinstance(item_id, str) or not item_id or item_id in manual_ids:
            raise ValueError("manual review packet item identity is invalid")
        manual_ids.add(item_id)
        _sha256(item["task_sha256"], "manual task")
        _sha256(item["response_sha256"], "manual response")
        if not isinstance(item["category"], str) or not item["category"]:
            raise ValueError("manual review packet category is invalid")
        rubric = item["rubric_dimensions"]
        if (
            not isinstance(rubric, list)
            or len(rubric) < 3
            or len(rubric) != len(set(rubric))
            or any(not isinstance(value, str) or not value for value in rubric)
        ):
            raise ValueError("manual review packet rubric is invalid")
        if item["judgment"] is not None:
            raise ValueError("qualification cannot contain human judgments")
    manual_summary = next(
        summary
        for summary in summaries
        if summary["dimension"] == "manual_review"
    )
    if manual_summary.get("count") != len(manual_packet):
        raise ValueError("manual review packet coverage is incomplete")
    required_flags = {
        "fixture_only": True,
        "prompt_content_included": False,
        "held_out_opened": False,
        "model_invoked": False,
        "checkpoint_opened": False,
        "production_suite_frozen": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }
    if any(report[field] is not expected for field, expected in required_flags.items()):
        raise ValueError("qualification authority or scope flags are invalid")
    if report["qualification_sha256"] != qualification_identity(report):
        raise ValueError("qualification identity mismatch")


def qualification_evidence(report: Mapping[str, object]) -> dict[str, object]:
    """Extract a compact frozen identity that still binds the complete report."""

    validate_fixture_qualification(report)
    evidence: dict[str, object] = {
        "schema_id": EVIDENCE_SCHEMA_ID,
        "report_schema_id": report["schema_id"],
        "repository_commit": report["repository_commit"],
        "qualification_sha256": report["qualification_sha256"],
        "schema_implementation_sha256": report["schema_implementation_sha256"],
        "task_implementation_sha256": report["task_implementation_sha256"],
        "statistics_implementation_sha256": report[
            "statistics_implementation_sha256"
        ],
        "tests_sha256": report["tests_sha256"],
        "smoke_sha256": report["smoke_sha256"],
        "task_inventory_sha256": report["task_inventory_sha256"],
        "observations_sha256": report["observations_sha256"],
        "score_rows_sha256": report["score_rows_sha256"],
        "summaries_sha256": hashlib.sha256(
            canonical_json(report["summaries"])
        ).hexdigest(),
        "manual_review_packet_sha256": hashlib.sha256(
            canonical_json(report["manual_review_packet"])
        ).hexdigest(),
        "task_count": report["task_count"],
        "dimension_mode_count": report["dimension_mode_count"],
        "bootstrap": report["bootstrap"],
        "fixture_only": report["fixture_only"],
        "prompt_content_included": report["prompt_content_included"],
        "held_out_opened": report["held_out_opened"],
        "model_invoked": report["model_invoked"],
        "checkpoint_opened": report["checkpoint_opened"],
        "production_suite_frozen": report["production_suite_frozen"],
        "evaluation_run_authorized": report["evaluation_run_authorized"],
        "training_authorized": report["training_authorized"],
        "evidence_sha256": "0" * 64,
    }
    evidence["evidence_sha256"] = evidence_identity(evidence)
    return evidence


def validate_qualification_evidence(
    evidence: Mapping[str, object], report: Mapping[str, object]
) -> None:
    """Require exact compact-evidence reproduction from a complete report."""

    expected = qualification_evidence(report)
    if dict(evidence) != expected:
        raise ValueError("qualification evidence does not match the report")
