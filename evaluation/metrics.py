"""Deterministic, task-specific response metrics for VASU evaluation."""

from __future__ import annotations

import ast
from collections.abc import Mapping
import re
import string
from typing import Any


PRIMARY_CHECKS = (
    "non_empty",
    "exact_answer",
    "accepted_answers",
    "required_keywords",
    "forbidden_keywords",
    "exact_line_count",
    "max_line_count",
    "exact_word_count",
    "numbered_item_count",
    "max_repetition_ratio",
    "uncertainty_required",
    "clarification_required",
    "python_code_required",
)
SUPPORTED_CHECK_KEYS = frozenset((*PRIMARY_CHECKS, "required_keywords_mode"))
TRAILING_ANSWER_PUNCTUATION = " \t\r\n.,!?;:"
SURROUNDING_WORD_PUNCTUATION = string.punctuation + "“”‘’"

UNCERTAINTY_PHRASES = (
    "cannot know",
    "can't know",
    "can not know",
    "cannot predict",
    "can't predict",
    "impossible to predict exactly",
    "not possible to know exactly",
    "do not know",
    "don't know",
    "no way to know",
    "uncertain",
    "depends on",
)
CLARIFICATION_PHRASES = (
    "please clarify",
    "what do you mean",
    "could you provide more details",
    "can you provide more details",
    "which thing",
    "can you specify",
    "could you specify",
)


def normalized_words(text: str) -> list[str]:
    """Return canonical case-folded words for counts and repetition."""

    words = []
    for raw_word in text.split():
        word = raw_word.strip(SURROUNDING_WORD_PUNCTUATION).casefold()
        if word:
            words.append(word)
    return words


def repetition_ratio(text: str) -> float:
    """Calculate the canonical repeated-word fraction used in all reports."""

    words = normalized_words(text)
    if not words:
        return 0.0
    return 1.0 - (len(set(words)) / len(words))


def _normalize_answer(value: str) -> str:
    collapsed = " ".join(value.strip().split())
    return collapsed.rstrip(TRAILING_ANSWER_PUNCTUATION).casefold()


def _accepted_answer_matches(response: str, accepted: str) -> bool:
    normalized_response = _normalize_answer(response)
    normalized_accepted = _normalize_answer(accepted)
    if normalized_response == normalized_accepted:
        return True
    if not normalized_response.startswith(normalized_accepted):
        return False
    boundary = normalized_response[len(normalized_accepted):len(normalized_accepted) + 1]
    return bool(boundary and (boundary.isspace() or boundary in ",.;:!?()-°"))


def _validate_string_list(name: str, value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain only non-empty strings")


def validate_checks(checks: dict[str, Any]) -> None:
    """Reject malformed or unsupported deterministic check definitions."""

    if not isinstance(checks, dict):
        raise ValueError("checks must be an object")
    unknown = sorted(set(checks) - SUPPORTED_CHECK_KEYS)
    if unknown:
        raise ValueError(f"unknown automatic check(s): {', '.join(unknown)}")

    if "non_empty" in checks and checks["non_empty"] is not True:
        raise ValueError("non_empty must be true when configured")
    if "exact_answer" in checks:
        value = checks["exact_answer"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("exact_answer must be a non-empty string")
    if "accepted_answers" in checks:
        _validate_string_list("accepted_answers", checks["accepted_answers"])
    if "required_keywords" in checks:
        _validate_string_list("required_keywords", checks["required_keywords"])
    if "forbidden_keywords" in checks:
        _validate_string_list("forbidden_keywords", checks["forbidden_keywords"])

    keyword_mode = checks.get("required_keywords_mode", "all")
    if keyword_mode not in ("all", "any"):
        raise ValueError("required_keywords_mode must be 'all' or 'any'")
    if "required_keywords_mode" in checks and "required_keywords" not in checks:
        raise ValueError("required_keywords_mode requires required_keywords")

    for name in (
        "exact_line_count",
        "max_line_count",
        "exact_word_count",
        "numbered_item_count",
    ):
        if name in checks:
            value = checks[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    if "max_repetition_ratio" in checks:
        value = checks["max_repetition_ratio"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            raise ValueError("max_repetition_ratio must be between 0 and 1")

    for name in (
        "uncertainty_required",
        "clarification_required",
        "python_code_required",
    ):
        if name in checks and checks[name] is not True:
            raise ValueError(f"{name} must be true when configured")


def _result(passed: bool, details: str) -> dict[str, Any]:
    return {"passed": passed, "score": 1.0 if passed else 0.0, "details": details}


def _contains_confident_numeric_prediction(text: str) -> bool:
    patterns = (
        r"\b(?:will|shall|is going to)\b[^.\n]{0,60}\b\d[\d,.]*",
        r"\bexact(?: price| value)?\s+(?:is|will be)\b[^.\n]{0,40}\d",
        r"(?:[$€£₹]\s*\d[\d,.]*)[^.\n]{0,30}\b(?:guaranteed|certain)\b",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _python_candidates(response: str) -> list[str]:
    fenced = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = [candidate.strip() for candidate in fenced if candidate.strip()]
    code_prefix = re.compile(
        r"^(?:print\s*\(|def\s+\w+\s*\(|for\s+.+:|while\s+.+:|"
        r"import\s+\w+|from\s+\w+\s+import\s+|[A-Za-z_]\w*\s*=)"
    )
    candidates.extend(
        line.strip()
        for line in response.splitlines()
        if code_prefix.match(line.strip())
    )
    return candidates


def _is_plausible_python(response: str) -> tuple[bool, str]:
    candidates = _python_candidates(response)
    accepted_nodes = (
        ast.Assign,
        ast.AnnAssign,
        ast.FunctionDef,
        ast.For,
        ast.While,
        ast.Import,
        ast.ImportFrom,
    )
    for candidate in candidates:
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, accepted_nodes):
                return True, f"parsed Python construct: {type(node).__name__}"
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "print"
            ):
                return True, "parsed Python print call"
    return False, "no supported, parseable Python construct found"


def evaluate_response(
    prompt_item: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    """Evaluate one cleaned response using only configured deterministic checks."""

    checks = prompt_item.get("checks", {})
    validate_checks(checks)
    if not checks:
        return {"checks": {}, "passed_checks": 0, "total_checks": 0, "score": None}

    results: dict[str, dict[str, Any]] = {}
    folded = response.casefold()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    words = normalized_words(response)

    if "non_empty" in checks:
        passed = bool(response.strip())
        results["non_empty"] = _result(passed, "response is non-empty" if passed else "response is empty")
    if "exact_answer" in checks:
        expected = checks["exact_answer"]
        passed = _normalize_answer(response) == _normalize_answer(expected)
        results["exact_answer"] = _result(
            passed, f"normalized response {'matches' if passed else 'does not match'} {expected!r}"
        )
    if "accepted_answers" in checks:
        accepted = checks["accepted_answers"]
        match = next((answer for answer in accepted if _accepted_answer_matches(response, answer)), None)
        results["accepted_answers"] = _result(
            match is not None,
            f"matched accepted answer {match!r}" if match is not None else "response did not equal or clearly begin with an accepted answer",
        )
    if "required_keywords" in checks:
        keywords = checks["required_keywords"]
        present = [keyword for keyword in keywords if keyword.casefold() in folded]
        mode = checks.get("required_keywords_mode", "all")
        passed = len(present) == len(keywords) if mode == "all" else bool(present)
        missing = [keyword for keyword in keywords if keyword not in present]
        results["required_keywords"] = _result(
            passed,
            f"mode={mode}; present={present}; missing={missing}",
        )
    if "forbidden_keywords" in checks:
        found = [
            keyword
            for keyword in checks["forbidden_keywords"]
            if keyword.casefold() in folded
        ]
        results["forbidden_keywords"] = _result(
            not found, f"forbidden terms found: {found}" if found else "no forbidden terms found"
        )
    if "exact_line_count" in checks:
        expected = checks["exact_line_count"]
        results["exact_line_count"] = _result(
            len(lines) == expected, f"non-empty lines={len(lines)}; expected={expected}"
        )
    if "max_line_count" in checks:
        maximum = checks["max_line_count"]
        results["max_line_count"] = _result(
            len(lines) <= maximum, f"non-empty lines={len(lines)}; maximum={maximum}"
        )
    if "exact_word_count" in checks:
        expected = checks["exact_word_count"]
        results["exact_word_count"] = _result(
            len(words) == expected, f"normalized words={len(words)}; expected={expected}"
        )
    if "numbered_item_count" in checks:
        count = len(re.findall(r"(?m)^\s*\d+[.)]\s+", response))
        expected = checks["numbered_item_count"]
        results["numbered_item_count"] = _result(
            count == expected, f"numbered items={count}; expected={expected}"
        )
    if "max_repetition_ratio" in checks:
        ratio = repetition_ratio(response)
        maximum = float(checks["max_repetition_ratio"])
        results["max_repetition_ratio"] = _result(
            ratio <= maximum, f"repetition_ratio={ratio:.4f}; maximum={maximum:.4f}"
        )
    if "uncertainty_required" in checks:
        phrases = [phrase for phrase in UNCERTAINTY_PHRASES if phrase in folded]
        confident_number = _contains_confident_numeric_prediction(response)
        passed = bool(phrases) and not confident_number
        details = f"uncertainty phrases={phrases}; confident numeric prediction={confident_number}"
        results["uncertainty_required"] = _result(passed, details)
    if "clarification_required" in checks:
        phrases = [phrase for phrase in CLARIFICATION_PHRASES if phrase in folded]
        results["clarification_required"] = _result(
            bool(phrases), f"clarification phrases={phrases}"
        )
    if "python_code_required" in checks:
        passed, details = _is_plausible_python(response)
        results["python_code_required"] = _result(passed, details)

    passed_checks = sum(int(result["passed"]) for result in results.values())
    total_checks = len(results)
    return {
        "checks": results,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "score": passed_checks / total_checks if total_checks else None,
    }
