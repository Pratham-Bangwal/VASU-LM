"""Transparent structural, exact, heuristic, and degeneration scorers."""

from __future__ import annotations

import json
import re
from typing import Any

from .schemas import CapabilityTask, HEURISTIC, HUMAN


NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
WORD = re.compile(r"\b\w+\b")


def parse_single_number(text: str) -> float | None:
    """Return a number only when the response contains exactly one value."""

    matches = NUMBER.findall(text.strip())
    return float(matches[0]) if len(matches) == 1 else None


def repetition_metrics(text: str) -> dict[str, Any]:
    words = [word.lower() for word in WORD.findall(text)]
    def ratio(width: int) -> float:
        groups = [tuple(words[index:index + width]) for index in range(len(words) - width + 1)]
        return 0.0 if not groups else 1.0 - len(set(groups)) / len(groups)
    sentences = [item.strip() for item in re.split(r"[.!?]+", text) if item.strip()]
    duplicate_sentences = len(sentences) - len(set(sentences))
    return {
        "words": len(words),
        "repeated_unigram_ratio": ratio(1),
        "repeated_bigram_ratio": ratio(2),
        "repeated_trigram_ratio": ratio(3),
        "duplicate_sentence_rate": duplicate_sentences / len(sentences) if sentences else 0.0,
        "empty": not bool(text.strip()),
        "prompt_leakage": "User:" in text or "Assistant:" in text,
    }


def _score_rule(text: str, scoring: dict[str, Any]) -> tuple[bool | None, dict[str, Any]]:
    rule = scoring["rule"]
    stripped = text.strip()
    if rule == "exact":
        answers = {answer.lower() for answer in scoring["answers"]}
        return stripped.lower() in answers, {"answers": sorted(answers)}
    if rule == "number":
        parsed = parse_single_number(stripped)
        expected = float(scoring["answer"])
        return parsed == expected, {"parsed": parsed, "expected": expected}
    if rule == "count_lines":
        lines = [line for line in stripped.splitlines() if line.strip()]
        return len(lines) == int(scoring["count"]), {"count": len(lines)}
    if rule == "numbered_steps":
        matches = re.findall(r"(?m)^\s*\d+[.)]\s+", stripped)
        return len(matches) == int(scoring["count"]), {"count": len(matches)}
    if rule == "bullet_count":
        matches = re.findall(r"(?m)^\s*[-*]\s+", stripped)
        return len(matches) == int(scoring["count"]), {"count": len(matches)}
    if rule == "json_keys":
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return False, {"valid_json": False}
        keys = set(value) if isinstance(value, dict) else set()
        required = set(scoring["keys"])
        return required <= keys, {"valid_json": True, "keys": sorted(keys)}
    if rule == "keywords":
        lower = stripped.lower()
        required = scoring.get("required", [])
        forbidden = scoring.get("forbidden", [])
        passed = all(word.lower() in lower for word in required) and not any(word.lower() in lower for word in forbidden)
        return passed, {"required_present": [word for word in required if word.lower() in lower], "forbidden_present": [word for word in forbidden if word.lower() in lower]}
    if rule == "human":
        return None, {"reason": "human review required"}
    raise ValueError(f"Unsupported scoring rule: {rule}")


def score_task(task: CapabilityTask, response: str) -> dict[str, Any]:
    """Score one response without collapsing heuristic/human metrics into accuracy."""

    passed, detail = _score_rule(response, task.scoring)
    result: dict[str, Any] = {"metric_kind": task.metric_kind, "rule": task.scoring["rule"], "passed": passed, "detail": detail, "repetition": repetition_metrics(response)}
    if task.metric_kind == HEURISTIC:
        result["heuristic_note"] = "Transparent rubric signal; not objective correctness."
    if task.metric_kind == HUMAN:
        result["human_review_required"] = True
    return result
