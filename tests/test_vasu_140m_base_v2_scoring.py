from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from evaluation.framework.vasu_140m_base_v2_statistics import (
    bootstrap_mean_ci,
    build_fixture_qualification,
    qualification_identity,
    qualification_evidence,
    select_manual_review,
    summarize_dimension_mode,
    validate_fixture_qualification,
    validate_qualification_evidence,
)
from evaluation.framework.vasu_140m_base_v2_tasks import (
    TASK_SCHEMA_ID,
    score_task,
    task_identity,
    validate_inventory,
    validate_task,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def task(dimension: str, index: int = 0) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": TASK_SCHEMA_ID,
        "item_id": f"{dimension}-{index}",
        "dimension": dimension,
        "split": "development",
        "strata": {},
        "input": {},
        "scoring": {},
    }
    if dimension == "likelihood":
        value["strata"] = {"source": f"source-{index % 2}"}
        value["input"] = {
            "text_sha256": digest(f"likelihood-text-{index}"),
            "target_token_count": 2,
        }
        value["scoring"] = {"kind": "token_log_likelihood"}
    elif dimension == "factuality":
        value["strata"] = {"task_family": "multiple_choice"}
        value["input"] = {
            "prompt_sha256": digest(f"factuality-prompt-{index}"),
            "choice_ids": ["a", "b", "c"],
        }
        value["scoring"] = {"correct_choice_id": "b"}
    elif dimension == "arithmetic":
        value["strata"] = {
            "operation": "addition",
            "difficulty": f"tier-{index % 2}",
            "template_family": f"template-{index % 2}",
        }
        value["input"] = {
            "prompt_sha256": digest(f"arithmetic-prompt-{index}")
        }
        value["scoring"] = {
            "answer_type": "integer",
            "expected_answer": "42",
        }
    elif dimension == "repetition":
        value["strata"] = {"prompt_family": f"continuation-{index % 2}"}
        value["input"] = {
            "prompt_sha256": digest(f"repetition-prompt-{index}")
        }
        value["scoring"] = {"loop_ngram_size": 2}
    elif dimension == "robustness":
        value["strata"] = {"variant_kind": f"whitespace-{index % 2}"}
        value["input"] = {
            "pair_id": f"pair-{index}",
            "baseline_prompt_sha256": digest(f"robust-baseline-{index}"),
            "variant_prompt_sha256": digest(f"robust-variant-{index}"),
            "variant_kind": f"whitespace-{index % 2}",
        }
        value["scoring"] = {"accepted_answers": ["Paris", "Paris, France"]}
    elif dimension == "manual_review":
        value["strata"] = {"category": f"category-{index % 2}"}
        value["input"] = {
            "prompt_sha256": digest(f"manual-prompt-{index}")
        }
        value["scoring"] = {
            "rubric_dimensions": [
                "coherence",
                "factual_support",
                "degeneration",
            ]
        }
    else:
        raise AssertionError(dimension)
    return value


def observation(
    dimension: str, index: int = 0, mode: str | None = None
) -> dict[str, object]:
    if dimension == "likelihood":
        return {
            "item_id": f"{dimension}-{index}",
            "mode": "direct_likelihood",
            "target_token_log_likelihoods": [-1.0 - index, -2.0],
        }
    if dimension == "factuality":
        return {
            "item_id": f"{dimension}-{index}",
            "mode": "direct_likelihood",
            "choice_scores": [
                {"choice_id": "a", "total_log_likelihood": -3.0, "token_count": 1},
                {"choice_id": "b", "total_log_likelihood": -2.0, "token_count": 2},
                {"choice_id": "c", "total_log_likelihood": -4.0, "token_count": 2},
            ],
        }
    if dimension == "arithmetic":
        return {
            "item_id": f"{dimension}-{index}",
            "mode": mode or "greedy",
            "response": "42",
            "generated_tokens": 1,
            "truncated": False,
            "prompt_leakage": False,
        }
    if dimension == "repetition":
        return {
            "item_id": f"{dimension}-{index}",
            "mode": mode or "greedy",
            "response_token_ids": [10, 11, 12, 13],
            "response_text_sha256": digest(f"repetition-response-{index}"),
            "eos_emitted": True,
            "truncated": False,
        }
    if dimension == "robustness":
        return {
            "item_id": f"{dimension}-{index}",
            "mode": mode or "greedy",
            "baseline_response": "Paris.",
            "variant_response": "paris",
        }
    if dimension == "manual_review":
        return {
            "item_id": f"{dimension}-{index}",
            "mode": "manual",
            "response_sha256": digest(f"manual-response-{index}"),
            "response_token_count": 12,
        }
    raise AssertionError(dimension)


def fixture_material() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tasks: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    for dimension in ("likelihood", "factuality"):
        for index in range(2):
            tasks.append(task(dimension, index))
            observations.append(observation(dimension, index))
    for dimension in ("arithmetic", "repetition", "robustness"):
        for index, mode in enumerate(("greedy", "sampled")):
            tasks.append(task(dimension, index))
            observations.append(observation(dimension, index, mode))
    for index in range(2):
        tasks.append(task("manual_review", index))
        observations.append(observation("manual_review", index))
    return tasks, observations


def qualification() -> dict[str, object]:
    tasks, observations = fixture_material()
    return build_fixture_qualification(
        tasks=tasks,
        observations=observations,
        repository_commit="a" * 40,
        schema_implementation_sha256="1" * 64,
        task_implementation_sha256="2" * 64,
        statistics_implementation_sha256="3" * 64,
        tests_sha256="4" * 64,
        smoke_sha256="5" * 64,
        bootstrap_samples=200,
        bootstrap_seed=140,
    )


def test_valid_tasks_and_inventory_are_hash_stable() -> None:
    tasks, _ = fixture_material()
    validate_inventory(tasks)
    assert task_identity(tasks[0]) == task_identity(deepcopy(tasks[0]))


@pytest.mark.parametrize("field", ["schema_id", "dimension", "split"])
def test_task_rejects_identity_mutations(field: str) -> None:
    value = task("likelihood")
    value[field] = "invalid"
    with pytest.raises(ValueError):
        validate_task(value)


def test_task_rejects_unknown_fields_and_missing_strata() -> None:
    value = task("arithmetic")
    value["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_task(value)
    value = task("arithmetic")
    value["strata"].pop("operation")
    with pytest.raises(ValueError, match="missing required"):
        validate_task(value)


def test_inventory_rejects_duplicate_ids_and_prompt_hashes() -> None:
    first = task("likelihood", 0)
    second = task("likelihood", 1)
    second["item_id"] = first["item_id"]
    with pytest.raises(ValueError, match="duplicate item IDs"):
        validate_inventory([first, second])
    second = task("likelihood", 1)
    second["input"]["text_sha256"] = first["input"]["text_sha256"]
    with pytest.raises(ValueError, match="reuses a prompt/text"):
        validate_inventory([first, second])


def test_factuality_and_arithmetic_contracts_fail_closed() -> None:
    value = task("factuality")
    value["scoring"]["correct_choice_id"] = "missing"
    with pytest.raises(ValueError, match="not in choice_ids"):
        validate_task(value)
    value = task("arithmetic")
    value["scoring"]["answer_type"] = "decimal"
    with pytest.raises(ValueError, match="unsupported"):
        validate_task(value)
    value = task("arithmetic")
    value["scoring"] = {
        "answer_type": "reduced_fraction",
        "expected_answer": "2/4",
    }
    with pytest.raises(ValueError, match="must be reduced"):
        validate_task(value)


def test_robustness_requires_distinct_prompt_identities() -> None:
    value = task("robustness")
    value["input"]["variant_prompt_sha256"] = value["input"][
        "baseline_prompt_sha256"
    ]
    with pytest.raises(ValueError, match="must differ"):
        validate_task(value)


def test_observation_identity_and_mode_must_match_task() -> None:
    value = observation("arithmetic")
    value["item_id"] = "other"
    with pytest.raises(ValueError, match="item identity"):
        score_task(task("arithmetic"), value)
    value = observation("arithmetic")
    value["mode"] = "direct_likelihood"
    with pytest.raises(ValueError, match="mode"):
        score_task(task("arithmetic"), value)


def test_likelihood_scoring_and_validation() -> None:
    score = score_task(task("likelihood"), observation("likelihood"))
    assert score["target_token_count"] == 2
    assert score["total_log_likelihood"] == -3.0
    assert score["loss"] == 1.5
    value = observation("likelihood")
    value["target_token_log_likelihoods"] = [-1.0]
    with pytest.raises(ValueError, match="count mismatch"):
        score_task(task("likelihood"), value)
    value = observation("likelihood")
    value["target_token_log_likelihoods"] = [0.1, -1.0]
    with pytest.raises(ValueError, match="cannot be positive"):
        score_task(task("likelihood"), value)


def test_factuality_reports_raw_and_length_normalized_predictions() -> None:
    score = score_task(task("factuality"), observation("factuality"))
    assert score["raw_prediction"] == "b"
    assert score["raw_correct"] is True
    assert score["normalized_prediction"] == "b"
    assert score["normalized_correct"] is True


def test_factuality_ties_are_not_silently_broken() -> None:
    value = observation("factuality")
    value["choice_scores"][0]["total_log_likelihood"] = -2.0
    score = score_task(task("factuality"), value)
    assert score["raw_tie"] is True
    assert score["raw_prediction"] is None
    assert score["raw_correct"] is False


@pytest.mark.parametrize(
    ("response", "leakage", "expected"),
    [
        ("42", False, "correct"),
        ("41", False, "incorrect"),
        ("answer is 42", False, "malformed"),
        ("", False, "unanswered"),
        ("42", True, "prompt_leakage"),
    ],
)
def test_arithmetic_outcome_taxonomy(
    response: str, leakage: bool, expected: str
) -> None:
    value = observation("arithmetic")
    value["response"] = response
    value["prompt_leakage"] = leakage
    score = score_task(task("arithmetic"), value)
    assert score["outcome"] == expected


def test_repetition_reports_token_level_degeneration() -> None:
    value = observation("repetition")
    value["response_token_ids"] = [1, 2, 1, 2, 1, 2]
    score = score_task(task("repetition"), value)
    assert score["terminal_loop"] is True
    assert score["repeated_bigram_rate"] > 0
    assert score["unique_token_ratio"] == pytest.approx(1 / 3)


def test_repetition_accepts_empty_output_as_visible_metric() -> None:
    value = observation("repetition")
    value["response_token_ids"] = []
    score = score_task(task("repetition"), value)
    assert score["empty"] is True
    assert score["unique_token_ratio"] == 0.0


def test_robustness_is_paired_and_normalized() -> None:
    score = score_task(task("robustness"), observation("robustness"))
    assert score["baseline_correct"] is True
    assert score["variant_correct"] is True
    assert score["consistent"] is True
    assert score["paired_correctness_delta"] == 0
    value = observation("robustness")
    value["variant_response"] = "London"
    score = score_task(task("robustness"), value)
    assert score["paired_correctness_delta"] == -1
    value["variant_response"] = ""
    score = score_task(task("robustness"), value)
    assert score["variant_empty"] is True


def test_manual_review_never_infers_a_judgment() -> None:
    score = score_task(task("manual_review"), observation("manual_review"))
    assert score["human_review_required"] is True
    assert score["human_judgment"] is None


def test_bootstrap_is_deterministic_and_rejects_bad_inputs() -> None:
    first = bootstrap_mean_ci([0.0, 1.0, 1.0], samples=200, seed=9)
    second = bootstrap_mean_ci([0.0, 1.0, 1.0], samples=200, seed=9)
    assert first == second
    assert first["mean"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="must not be empty"):
        bootstrap_mean_ci([], samples=200)
    with pytest.raises(ValueError, match="at least 100"):
        bootstrap_mean_ci([1.0], samples=99)


def test_dimension_summaries_remain_separate_and_stratified() -> None:
    scores = [
        score_task(task("likelihood", index), observation("likelihood", index))
        for index in range(2)
    ]
    summary = summarize_dimension_mode(scores, samples=200, seed=7)
    assert summary["dimension"] == "likelihood"
    assert summary["target_token_count"] == 4
    assert set(summary["strata"]["source"]) == {"source-0", "source-1"}
    mixed = scores + [score_task(task("factuality"), observation("factuality"))]
    with pytest.raises(ValueError, match="one exact dimension"):
        summarize_dimension_mode(mixed, samples=200)


def test_manual_sampling_is_deterministic_and_covers_categories() -> None:
    scores = [
        score_task(task("manual_review", index), observation("manual_review", index))
        for index in range(4)
    ]
    first = select_manual_review(scores, count=2, seed=140)
    assert first == select_manual_review(scores, count=2, seed=140)
    assert {item["category"] for item in first} == {"category-0", "category-1"}
    assert all(item["judgment"] is None for item in first)
    with pytest.raises(ValueError, match="cover every category"):
        select_manual_review(scores, count=1, seed=140)


def test_fixture_qualification_is_complete_and_non_authorizing() -> None:
    report = qualification()
    validate_fixture_qualification(report)
    assert report["dimension_mode_count"] == 9
    assert report["held_out_opened"] is False
    assert report["model_invoked"] is False
    assert report["checkpoint_opened"] is False
    assert report["production_suite_frozen"] is False
    assert report["training_authorized"] is False
    assert report["qualification_sha256"] == qualification_identity(report)


def test_fixture_qualification_is_order_and_identity_bound() -> None:
    first = qualification()
    second = qualification()
    assert first == second
    tasks, observations = fixture_material()
    observations.reverse()
    changed = build_fixture_qualification(
        tasks=tasks,
        observations=observations,
        repository_commit="a" * 40,
        schema_implementation_sha256="1" * 64,
        task_implementation_sha256="2" * 64,
        statistics_implementation_sha256="3" * 64,
        tests_sha256="4" * 64,
        smoke_sha256="5" * 64,
        bootstrap_samples=200,
    )
    assert first["observations_sha256"] != changed["observations_sha256"]


def test_fixture_qualification_rejects_held_out_or_incomplete_observations() -> None:
    tasks, observations = fixture_material()
    tasks[0]["split"] = "held_out"
    with pytest.raises(PermissionError, match="cannot open held-out"):
        build_fixture_qualification(
            tasks=tasks,
            observations=observations,
            repository_commit="a" * 40,
            schema_implementation_sha256="1" * 64,
            task_implementation_sha256="2" * 64,
            statistics_implementation_sha256="3" * 64,
            tests_sha256="4" * 64,
            smoke_sha256="5" * 64,
            bootstrap_samples=200,
        )
    tasks, observations = fixture_material()
    observations.pop()
    with pytest.raises(ValueError, match="one observation"):
        build_fixture_qualification(
            tasks=tasks,
            observations=observations,
            repository_commit="a" * 40,
            schema_implementation_sha256="1" * 64,
            task_implementation_sha256="2" * 64,
            statistics_implementation_sha256="3" * 64,
            tests_sha256="4" * 64,
            smoke_sha256="5" * 64,
            bootstrap_samples=200,
        )


def test_qualification_mutation_and_authority_fail_closed() -> None:
    report = qualification()
    report["training_authorized"] = True
    report["qualification_sha256"] = qualification_identity(report)
    with pytest.raises(ValueError, match="authority or scope"):
        validate_fixture_qualification(report)
    report = qualification()
    report["task_count"] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_fixture_qualification(report)
    report = qualification()
    report["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_fixture_qualification(report)
    report = qualification()
    report["manual_review_packet"][0]["judgment"] = "approved"
    report["qualification_sha256"] = qualification_identity(report)
    with pytest.raises(ValueError, match="cannot contain human judgments"):
        validate_fixture_qualification(report)
    report = qualification()
    report["bootstrap"]["samples"] = 99
    report["qualification_sha256"] = qualification_identity(report)
    with pytest.raises(ValueError, match="bootstrap contract"):
        validate_fixture_qualification(report)
    report = qualification()
    report["summaries"].append(deepcopy(report["summaries"][0]))
    report["qualification_sha256"] = qualification_identity(report)
    with pytest.raises(ValueError, match="summaries must be a list"):
        validate_fixture_qualification(report)


def test_compact_evidence_binds_the_complete_qualification() -> None:
    report = qualification()
    evidence = qualification_evidence(report)
    validate_qualification_evidence(evidence, report)
    evidence["summaries_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        validate_qualification_evidence(evidence, report)
