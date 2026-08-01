"""Strict task records and deterministic scorers for VASU-140M eval v2."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence

from evaluation.framework.vasu_140m_base_v2 import DIMENSIONS, RESULT_MODES


TASK_SCHEMA_ID = "vasu_140m_base_evaluation_task_v2"
SHA256_HEX_LENGTH = 64
ANSWER_PATTERNS = {
    "integer": re.compile(r"[+-]?\d+"),
    "reduced_fraction": re.compile(r"[+-]?\d+/\d+"),
    "comparison_symbol": re.compile(r"[<>=]"),
    "boolean": re.compile(r"(?:yes|no)", re.IGNORECASE),
}
COMMON_FIELDS = {
    "schema_id",
    "item_id",
    "dimension",
    "split",
    "strata",
    "input",
    "scoring",
}
REQUIRED_STRATA = {
    "likelihood": {"source"},
    "factuality": {"task_family"},
    "arithmetic": {"operation", "difficulty", "template_family"},
    "repetition": {"prompt_family"},
    "robustness": {"variant_kind"},
    "manual_review": {"category"},
}
INPUT_FIELDS = {
    "likelihood": {"text_sha256", "target_token_count"},
    "factuality": {"prompt_sha256", "choice_ids"},
    "arithmetic": {"prompt_sha256"},
    "repetition": {"prompt_sha256"},
    "robustness": {
        "pair_id",
        "baseline_prompt_sha256",
        "variant_prompt_sha256",
        "variant_kind",
    },
    "manual_review": {"prompt_sha256"},
}
SCORING_FIELDS = {
    "likelihood": {"kind"},
    "factuality": {"correct_choice_id"},
    "arithmetic": {"answer_type", "expected_answer"},
    "repetition": {"loop_ngram_size"},
    "robustness": {"accepted_answers"},
    "manual_review": {"rubric_dimensions"},
}
OBSERVATION_FIELDS = {
    "likelihood": {
        "item_id",
        "mode",
        "target_token_log_likelihoods",
    },
    "factuality": {"item_id", "mode", "choice_scores"},
    "arithmetic": {
        "item_id",
        "mode",
        "response",
        "generated_tokens",
        "truncated",
        "prompt_leakage",
    },
    "repetition": {
        "item_id",
        "mode",
        "response_token_ids",
        "response_text_sha256",
        "eos_emitted",
        "truncated",
    },
    "robustness": {
        "item_id",
        "mode",
        "baseline_response",
        "variant_response",
    },
    "manual_review": {
        "item_id",
        "mode",
        "response_sha256",
        "response_token_count",
    },
}


def canonical_json(value: object) -> bytes:
    """Encode identity material deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def task_identity(task: Mapping[str, object]) -> str:
    """Return the stable identity of one validated task record."""

    return hashlib.sha256(canonical_json(task)).hexdigest()


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _perplexity(loss: float) -> float:
    if loss > math.log(sys.float_info.max):
        raise ValueError("loss is too large for finite perplexity")
    return math.exp(loss)


def _string_list(value: object, label: str, *, minimum: int = 1) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} values")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must contain only non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    return value


def validate_task(task: Mapping[str, object]) -> None:
    """Validate one prompt-hash-only task definition."""

    _exact_keys(task, COMMON_FIELDS, "task")
    if task["schema_id"] != TASK_SCHEMA_ID:
        raise ValueError("task schema identity mismatch")
    _string(task["item_id"], "task.item_id")
    dimension = _string(task["dimension"], "task.dimension")
    if dimension not in DIMENSIONS:
        raise ValueError("task dimension is unsupported")
    if task["split"] not in {"development", "held_out"}:
        raise ValueError("task split is unsupported")

    strata = _mapping(task["strata"], "task.strata")
    if not strata:
        raise ValueError("task.strata must not be empty")
    for name, value in strata.items():
        _string(name, "task.strata key")
        _string(value, f"task.strata.{name}")
    missing_strata = sorted(REQUIRED_STRATA[dimension] - set(strata))
    if missing_strata:
        raise ValueError(f"task.strata is missing required fields: {missing_strata}")

    inputs = _mapping(task["input"], "task.input")
    _exact_keys(inputs, INPUT_FIELDS[dimension], "task.input")
    scoring = _mapping(task["scoring"], "task.scoring")
    _exact_keys(scoring, SCORING_FIELDS[dimension], "task.scoring")

    if dimension == "likelihood":
        _sha256(inputs["text_sha256"], "task.input.text_sha256")
        _positive_int(inputs["target_token_count"], "task.input.target_token_count")
        if scoring["kind"] != "token_log_likelihood":
            raise ValueError("likelihood scoring kind is unsupported")
    elif dimension == "factuality":
        _sha256(inputs["prompt_sha256"], "task.input.prompt_sha256")
        choices = _string_list(inputs["choice_ids"], "task.input.choice_ids", minimum=2)
        if scoring["correct_choice_id"] not in choices:
            raise ValueError("factuality correct choice is not in choice_ids")
    elif dimension == "arithmetic":
        _sha256(inputs["prompt_sha256"], "task.input.prompt_sha256")
        answer_type = _string(scoring["answer_type"], "task.scoring.answer_type")
        if answer_type not in ANSWER_PATTERNS:
            raise ValueError("arithmetic answer type is unsupported")
        expected = _string(scoring["expected_answer"], "task.scoring.expected_answer")
        if ANSWER_PATTERNS[answer_type].fullmatch(expected.strip()) is None:
            raise ValueError("arithmetic expected answer does not match its type")
        if answer_type == "reduced_fraction":
            numerator, denominator = (int(part) for part in expected.split("/"))
            if denominator <= 0 or math.gcd(numerator, denominator) != 1:
                raise ValueError("expected fraction must be reduced with positive denominator")
    elif dimension == "repetition":
        _sha256(inputs["prompt_sha256"], "task.input.prompt_sha256")
        width = _positive_int(scoring["loop_ngram_size"], "loop_ngram_size")
        if width < 2:
            raise ValueError("loop_ngram_size must be at least 2")
    elif dimension == "robustness":
        _string(inputs["pair_id"], "task.input.pair_id")
        _sha256(
            inputs["baseline_prompt_sha256"],
            "task.input.baseline_prompt_sha256",
        )
        _sha256(
            inputs["variant_prompt_sha256"],
            "task.input.variant_prompt_sha256",
        )
        if inputs["baseline_prompt_sha256"] == inputs["variant_prompt_sha256"]:
            raise ValueError("robustness baseline and variant must differ")
        _string(inputs["variant_kind"], "task.input.variant_kind")
        _string_list(scoring["accepted_answers"], "task.scoring.accepted_answers")
    else:
        _sha256(inputs["prompt_sha256"], "task.input.prompt_sha256")
        _string_list(
            scoring["rubric_dimensions"],
            "task.scoring.rubric_dimensions",
            minimum=3,
        )


def validate_inventory(tasks: Sequence[Mapping[str, object]]) -> None:
    """Validate unique development or sealed held-out task records."""

    if not tasks:
        raise ValueError("task inventory must not be empty")
    item_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    robustness_pairs: set[str] = set()
    for task in tasks:
        validate_task(task)
        item_id = str(task["item_id"])
        if item_id in item_ids:
            raise ValueError("task inventory contains duplicate item IDs")
        item_ids.add(item_id)
        inputs = _mapping(task["input"], "task.input")
        hashes = [
            str(value)
            for key, value in inputs.items()
            if key.endswith("sha256")
        ]
        if any(value in prompt_hashes for value in hashes):
            raise ValueError("task inventory reuses a prompt/text identity")
        prompt_hashes.update(hashes)
        if task["dimension"] == "robustness":
            pair_id = str(inputs["pair_id"])
            if pair_id in robustness_pairs:
                raise ValueError("task inventory contains duplicate robustness pairs")
            robustness_pairs.add(pair_id)


def _validate_observation(
    task: Mapping[str, object], observation: Mapping[str, object]
) -> str:
    dimension = str(task["dimension"])
    _exact_keys(observation, OBSERVATION_FIELDS[dimension], "observation")
    if observation["item_id"] != task["item_id"]:
        raise ValueError("observation item identity mismatch")
    mode = _string(observation["mode"], "observation.mode")
    if mode not in RESULT_MODES[dimension]:
        raise ValueError("observation mode does not match task dimension")
    return mode


def _normalize_answer(value: str) -> str:
    return " ".join(value.strip().casefold().split()).rstrip(".,!?;:")


def _ngram_repetition(token_ids: Sequence[int], width: int) -> float:
    groups = [tuple(token_ids[index : index + width]) for index in range(len(token_ids) - width + 1)]
    return 0.0 if not groups else 1.0 - (len(set(groups)) / len(groups))


def _terminal_loop(token_ids: Sequence[int], width: int) -> bool:
    if len(token_ids) < width * 3:
        return False
    tail = tuple(token_ids[-width:])
    return (
        tuple(token_ids[-2 * width : -width]) == tail
        and tuple(token_ids[-3 * width : -2 * width]) == tail
    )


def score_task(
    task: Mapping[str, object], observation: Mapping[str, object]
) -> dict[str, object]:
    """Score one validated task observation without model execution."""

    validate_task(task)
    mode = _validate_observation(task, observation)
    dimension = str(task["dimension"])
    scoring = _mapping(task["scoring"], "task.scoring")
    inputs = _mapping(task["input"], "task.input")
    result: dict[str, object] = {
        "item_id": task["item_id"],
        "task_sha256": task_identity(task),
        "dimension": dimension,
        "split": task["split"],
        "mode": mode,
        "strata": dict(_mapping(task["strata"], "task.strata")),
    }

    if dimension == "likelihood":
        values = observation["target_token_log_likelihoods"]
        if not isinstance(values, list):
            raise ValueError("target_token_log_likelihoods must be a list")
        expected_count = int(inputs["target_token_count"])
        if len(values) != expected_count:
            raise ValueError("target log-likelihood count mismatch")
        log_likelihoods = [
            _finite(value, "target token log-likelihood") for value in values
        ]
        if any(value > 0 for value in log_likelihoods):
            raise ValueError("token log-likelihood cannot be positive")
        total = math.fsum(log_likelihoods)
        mean = total / expected_count
        result.update(
            {
                "target_token_count": expected_count,
                "total_log_likelihood": total,
                "mean_log_likelihood": mean,
                "loss": -mean,
                "perplexity": _perplexity(-mean),
            }
        )
    elif dimension == "factuality":
        raw_scores = observation["choice_scores"]
        if not isinstance(raw_scores, list):
            raise ValueError("choice_scores must be a list")
        choice_ids = list(inputs["choice_ids"])
        if len(raw_scores) != len(choice_ids):
            raise ValueError("choice score count mismatch")
        scores: dict[str, tuple[float, float]] = {}
        for index, raw in enumerate(raw_scores):
            choice = _mapping(raw, f"choice score {index}")
            _exact_keys(
                choice,
                {"choice_id", "total_log_likelihood", "token_count"},
                f"choice score {index}",
            )
            choice_id = _string(choice["choice_id"], "choice_id")
            if choice_id in scores or choice_id not in choice_ids:
                raise ValueError("choice score identity is duplicate or unknown")
            total = _finite(choice["total_log_likelihood"], "choice total")
            count = _positive_int(choice["token_count"], "choice token_count")
            if total > 0:
                raise ValueError("choice log-likelihood cannot be positive")
            scores[choice_id] = (total, total / count)
        if set(scores) != set(choice_ids):
            raise ValueError("choice scores do not cover all choices")
        raw_order = sorted(scores, key=lambda key: (-scores[key][0], key))
        normalized_order = sorted(scores, key=lambda key: (-scores[key][1], key))
        raw_tie = len(raw_order) > 1 and scores[raw_order[0]][0] == scores[raw_order[1]][0]
        normalized_tie = (
            len(normalized_order) > 1
            and scores[normalized_order[0]][1] == scores[normalized_order[1]][1]
        )
        correct = str(scoring["correct_choice_id"])
        result.update(
            {
                "correct_choice_id": correct,
                "raw_prediction": None if raw_tie else raw_order[0],
                "normalized_prediction": (
                    None if normalized_tie else normalized_order[0]
                ),
                "raw_tie": raw_tie,
                "normalized_tie": normalized_tie,
                "raw_correct": not raw_tie and raw_order[0] == correct,
                "normalized_correct": (
                    not normalized_tie and normalized_order[0] == correct
                ),
                "raw_margin": scores[raw_order[0]][0] - scores[raw_order[1]][0],
                "normalized_margin": (
                    scores[normalized_order[0]][1]
                    - scores[normalized_order[1]][1]
                ),
            }
        )
    elif dimension == "arithmetic":
        response = observation["response"]
        if not isinstance(response, str):
            raise ValueError("observation.response must be a string")
        generated = _positive_int(
            observation["generated_tokens"],
            "observation.generated_tokens",
            allow_zero=True,
        )
        for field in ("truncated", "prompt_leakage"):
            if not isinstance(observation[field], bool):
                raise ValueError(f"observation.{field} must be boolean")
        answer_type = str(scoring["answer_type"])
        match = ANSWER_PATTERNS[answer_type].fullmatch(response.strip())
        parsed = match.group(0).casefold() if match else None
        expected = str(scoring["expected_answer"]).strip().casefold()
        if observation["prompt_leakage"]:
            outcome = "prompt_leakage"
        elif not response.strip():
            outcome = "unanswered"
        elif match is None:
            outcome = "malformed"
        elif parsed == expected:
            outcome = "correct"
        else:
            outcome = "incorrect"
        result.update(
            {
                "outcome": outcome,
                "correct": outcome == "correct",
                "parsed_answer": parsed,
                "expected_answer": expected,
                "generated_tokens": generated,
                "truncated": observation["truncated"],
                "prompt_leakage": observation["prompt_leakage"],
            }
        )
    elif dimension == "repetition":
        token_ids = observation["response_token_ids"]
        if not isinstance(token_ids, list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        ):
            raise ValueError("response_token_ids must be non-negative integers")
        _sha256(observation["response_text_sha256"], "response_text_sha256")
        for field in ("eos_emitted", "truncated"):
            if not isinstance(observation[field], bool):
                raise ValueError(f"observation.{field} must be boolean")
        width = int(scoring["loop_ngram_size"])
        result.update(
            {
                "generated_tokens": len(token_ids),
                "empty": not token_ids,
                "eos_emitted": observation["eos_emitted"],
                "truncated": observation["truncated"],
                "unique_token_ratio": (
                    len(set(token_ids)) / len(token_ids) if token_ids else 0.0
                ),
                "repeated_unigram_rate": _ngram_repetition(token_ids, 1),
                "repeated_bigram_rate": _ngram_repetition(token_ids, 2),
                "repeated_trigram_rate": _ngram_repetition(token_ids, 3),
                "terminal_loop": _terminal_loop(token_ids, width),
            }
        )
    elif dimension == "robustness":
        baseline = observation["baseline_response"]
        variant = observation["variant_response"]
        if not isinstance(baseline, str) or not isinstance(variant, str):
            raise ValueError("robustness responses must be strings")
        accepted = {
            _normalize_answer(value) for value in scoring["accepted_answers"]
        }
        baseline_normalized = _normalize_answer(baseline)
        variant_normalized = _normalize_answer(variant)
        baseline_correct = baseline_normalized in accepted
        variant_correct = variant_normalized in accepted
        result.update(
            {
                "pair_id": inputs["pair_id"],
                "variant_kind": inputs["variant_kind"],
                "baseline_correct": baseline_correct,
                "variant_correct": variant_correct,
                "baseline_empty": not bool(baseline_normalized),
                "variant_empty": not bool(variant_normalized),
                "consistent": baseline_normalized == variant_normalized,
                "paired_correctness_delta": (
                    int(variant_correct) - int(baseline_correct)
                ),
            }
        )
    else:
        _sha256(observation["response_sha256"], "observation.response_sha256")
        response_tokens = _positive_int(
            observation["response_token_count"],
            "observation.response_token_count",
            allow_zero=True,
        )
        result.update(
            {
                "category": task["strata"]["category"],
                "rubric_dimensions": list(scoring["rubric_dimensions"]),
                "response_sha256": observation["response_sha256"],
                "response_token_count": response_tokens,
                "human_review_required": True,
                "human_judgment": None,
            }
        )
    return result


def build_prompt_free_fixture_material(
    *, modes: Sequence[str] = ("greedy", "sampled")
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build scorer qualification material containing hashes, not prompts."""

    if tuple(modes) != ("greedy", "sampled"):
        raise ValueError("fixture modes must be exactly greedy then sampled")

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    tasks: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []

    for index in range(2):
        tasks.append(
            {
                "schema_id": TASK_SCHEMA_ID,
                "item_id": f"likelihood-{index}",
                "dimension": "likelihood",
                "split": "development",
                "strata": {"source": f"fixture-source-{index}"},
                "input": {
                    "text_sha256": digest(f"likelihood-text-{index}"),
                    "target_token_count": 3,
                },
                "scoring": {"kind": "token_log_likelihood"},
            }
        )
        observations.append(
            {
                "item_id": f"likelihood-{index}",
                "mode": "direct_likelihood",
                "target_token_log_likelihoods": [
                    -0.5 - index,
                    -1.0,
                    -1.5,
                ],
            }
        )
        tasks.append(
            {
                "schema_id": TASK_SCHEMA_ID,
                "item_id": f"factuality-{index}",
                "dimension": "factuality",
                "split": "development",
                "strata": {
                    "task_family": "multiple_choice" if index == 0 else "cloze"
                },
                "input": {
                    "prompt_sha256": digest(f"factuality-prompt-{index}"),
                    "choice_ids": ["a", "b", "c"],
                },
                "scoring": {"correct_choice_id": "b"},
            }
        )
        observations.append(
            {
                "item_id": f"factuality-{index}",
                "mode": "direct_likelihood",
                "choice_scores": [
                    {
                        "choice_id": "a",
                        "total_log_likelihood": -3.0,
                        "token_count": 1,
                    },
                    {
                        "choice_id": "b",
                        "total_log_likelihood": -2.0 - index,
                        "token_count": 2,
                    },
                    {
                        "choice_id": "c",
                        "total_log_likelihood": -4.0,
                        "token_count": 2,
                    },
                ],
            }
        )

    arithmetic_responses = ("42", "41", "answer 42", "42")
    repetition_tokens = (
        [10, 11, 12, 13],
        [1, 2, 1, 2, 1, 2],
        [1, 2, 3, 1, 2, 4],
        [],
    )
    robustness_responses = ("paris", "London", "Paris, France", "Paris.")
    for index in range(4):
        mode = modes[index % len(modes)]
        tasks.append(
            {
                "schema_id": TASK_SCHEMA_ID,
                "item_id": f"arithmetic-{index}",
                "dimension": "arithmetic",
                "split": "development",
                "strata": {
                    "operation": "addition" if index % 2 == 0 else "division",
                    "difficulty": f"tier-{index + 1}",
                    "template_family": f"fixture-template-{index}",
                },
                "input": {
                    "prompt_sha256": digest(f"arithmetic-prompt-{index}")
                },
                "scoring": {
                    "answer_type": "integer",
                    "expected_answer": "42",
                },
            }
        )
        observations.append(
            {
                "item_id": f"arithmetic-{index}",
                "mode": mode,
                "response": arithmetic_responses[index],
                "generated_tokens": 1,
                "truncated": False,
                "prompt_leakage": False,
            }
        )
        tasks.append(
            {
                "schema_id": TASK_SCHEMA_ID,
                "item_id": f"repetition-{index}",
                "dimension": "repetition",
                "split": "development",
                "strata": {"prompt_family": f"fixture-continuation-{index}"},
                "input": {
                    "prompt_sha256": digest(f"repetition-prompt-{index}")
                },
                "scoring": {"loop_ngram_size": 2},
            }
        )
        observations.append(
            {
                "item_id": f"repetition-{index}",
                "mode": mode,
                "response_token_ids": repetition_tokens[index],
                "response_text_sha256": digest(f"repetition-response-{index}"),
                "eos_emitted": index % 2 == 0,
                "truncated": index % 2 == 1,
            }
        )
        tasks.append(
            {
                "schema_id": TASK_SCHEMA_ID,
                "item_id": f"robustness-{index}",
                "dimension": "robustness",
                "split": "development",
                "strata": {
                    "variant_kind": "whitespace" if index % 2 == 0 else "case"
                },
                "input": {
                    "pair_id": f"fixture-pair-{index}",
                    "baseline_prompt_sha256": digest(
                        f"robustness-baseline-{index}"
                    ),
                    "variant_prompt_sha256": digest(
                        f"robustness-variant-{index}"
                    ),
                    "variant_kind": "whitespace" if index % 2 == 0 else "case",
                },
                "scoring": {"accepted_answers": ["Paris", "Paris, France"]},
            }
        )
        observations.append(
            {
                "item_id": f"robustness-{index}",
                "mode": mode,
                "baseline_response": "Paris.",
                "variant_response": robustness_responses[index],
            }
        )

    for index in range(2):
        tasks.append(
            {
                "schema_id": TASK_SCHEMA_ID,
                "item_id": f"manual-review-{index}",
                "dimension": "manual_review",
                "split": "development",
                "strata": {"category": f"fixture-category-{index}"},
                "input": {
                    "prompt_sha256": digest(f"manual-review-prompt-{index}")
                },
                "scoring": {
                    "rubric_dimensions": [
                        "coherence",
                        "factual_support",
                        "degeneration",
                    ]
                },
            }
        )
        observations.append(
            {
                "item_id": f"manual-review-{index}",
                "mode": "manual",
                "response_sha256": digest(f"manual-review-response-{index}"),
                "response_token_count": 12 + index,
            }
        )

    validate_inventory(tasks)
    return tasks, observations
