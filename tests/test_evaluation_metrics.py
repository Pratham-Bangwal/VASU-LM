import pytest

from evaluation.metrics import (
    evaluate_response,
    repetition_ratio,
    validate_checks,
)


def _evaluate(checks, response):
    return evaluate_response({"checks": checks}, response)


def test_prompt_without_checks_is_not_scored():
    assert evaluate_response({}, "anything") == {
        "checks": {},
        "passed_checks": 0,
        "total_checks": 0,
        "score": None,
    }


def test_non_empty_check():
    assert _evaluate({"non_empty": True}, "answer")["score"] == 1.0
    assert _evaluate({"non_empty": True}, "  ")["score"] == 0.0


def test_exact_answer_is_case_insensitive_and_rejects_extra_text():
    assert _evaluate({"exact_answer": "42"}, "42.")["score"] == 1.0
    assert _evaluate({"exact_answer": "42"}, "142")["score"] == 0.0
    assert _evaluate({"exact_answer": "42"}, "42 because...")["score"] == 0.0


def test_accepted_answer_requires_clear_answer_boundary():
    checks = {"accepted_answers": ["Jupiter", "New Delhi"]}
    assert _evaluate(checks, "Jupiter is the largest planet.")["score"] == 1.0
    assert _evaluate(checks, "Jupiterian weather")["score"] == 0.0


def test_required_keywords_support_all_and_any_modes():
    assert _evaluate(
        {"required_keywords": ["store", "data"]},
        "Databases store data.",
    )["score"] == 1.0
    assert _evaluate(
        {
            "required_keywords": ["data", "patterns", "learn"],
            "required_keywords_mode": "any",
        },
        "It can learn from examples.",
    )["score"] == 1.0


def test_forbidden_keywords():
    checks = {"forbidden_keywords": ["meat", "chicken", "egg"]}
    assert _evaluate(checks, "Oatmeal and fruit")["score"] == 1.0
    assert _evaluate(checks, "Chicken sandwich")["score"] == 0.0


def test_line_and_word_counts_ignore_blank_lines_and_punctuation():
    assert _evaluate({"exact_line_count": 2}, "one\n\n two")["score"] == 1.0
    assert _evaluate({"max_line_count": 1}, "one line")["score"] == 1.0
    assert _evaluate({"exact_word_count": 3}, "Blue, deep ocean!")["score"] == 1.0


def test_numbered_item_count_only_counts_list_prefixes():
    checks = {"numbered_item_count": 3}
    response = "1. Plan\n2) Build\n3. Test\nThe year 2026 is irrelevant."
    assert _evaluate(checks, response)["score"] == 1.0


def test_repetition_metric_is_canonical_and_thresholded():
    assert repetition_ratio("Hello, hello world") == pytest.approx(1 / 3)
    assert _evaluate(
        {"max_repetition_ratio": 0.34}, "Hello, hello world"
    )["score"] == 1.0


def test_uncertainty_rejects_confident_numeric_prediction():
    checks = {"uncertainty_required": True}
    assert _evaluate(
        checks, "I cannot predict the exact future price."
    )["score"] == 1.0
    assert _evaluate(
        checks, "It depends on the market, but it will be $100,000."
    )["score"] == 0.0
    assert _evaluate(
        checks, "It depends on many factors; last year it was $100."
    )["score"] == 1.0


def test_clarification_detection():
    checks = {"clarification_required": True}
    assert _evaluate(checks, "Could you provide more details?")["score"] == 1.0
    assert _evaluate(checks, "The thing is blue.")["score"] == 0.0


@pytest.mark.parametrize(
    "response",
    [
        'print("Hello")',
        "```python\nvalue = 3\n```",
        "```python\ndef greet():\n    return 'hi'\n```",
    ],
)
def test_python_code_detection_accepts_parseable_constructs(response):
    assert _evaluate({"python_code_required": True}, response)["score"] == 1.0


def test_python_code_detection_rejects_pseudocode():
    assert _evaluate(
        {"python_code_required": True}, "Tell Python to print Hello"
    )["score"] == 0.0


def test_aggregate_score_is_fraction_of_configured_checks():
    result = _evaluate(
        {
            "non_empty": True,
            "required_keywords": ["data"],
            "max_line_count": 1,
            "forbidden_keywords": ["wrong"],
        },
        "Data is wrong.",
    )
    assert result["passed_checks"] == 3
    assert result["total_checks"] == 4
    assert result["score"] == 0.75


@pytest.mark.parametrize(
    "checks, message",
    [
        ({"unknown": True}, "unknown automatic check"),
        ({"non_empty": False}, "non_empty must be true"),
        ({"exact_answer": ""}, "exact_answer"),
        ({"accepted_answers": []}, "accepted_answers"),
        ({"required_keywords": []}, "required_keywords"),
        ({"required_keywords_mode": "some"}, "required_keywords_mode"),
        ({"required_keywords_mode": "any"}, "requires required_keywords"),
        ({"exact_line_count": -1}, "exact_line_count"),
        ({"max_repetition_ratio": 1.1}, "max_repetition_ratio"),
        ({"python_code_required": False}, "python_code_required must be true"),
    ],
)
def test_invalid_check_configuration_fails_clearly(checks, message):
    with pytest.raises(ValueError, match=message):
        validate_checks(checks)
