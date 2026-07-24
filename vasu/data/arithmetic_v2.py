"""Deterministic, metadata-verified arithmetic-v2 logical examples."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence


FORMAT_VERSION = "vasu_verified_arithmetic_v2"
DATASET_ID = "verified_arithmetic_v2"
TRAINING_TEXT_FORMAT = "Question: {prompt}\nAnswer: {answer}"
CATEGORIES = (
    "addition",
    "subtraction",
    "multiplication",
    "exact_division",
    "comparison",
    "sequence",
    "fraction",
    "percentage",
    "word_problem",
    "mixed_expression",
    "numeric_property",
)
TIERS = ("tier_1", "tier_2", "tier_3", "tier_4")
SPLIT_RANGES = {
    "train": (20, 999),
    "development": (1_000, 1_499),
    "evaluation": (1_500, 1_999),
}
TEMPLATES = {
    "train": (
        ("direct_v1", "{body}"),
        ("calculate_v1", "Calculate: {body}"),
        ("number_only_v1", "Answer exactly: {body}"),
    ),
    "development": (("heldout_exact_v1", "Find the exact answer: {body}"),),
    "evaluation": (("heldout_concise_v1", "Give only the answer: {body}"),),
}
FORBIDDEN_PROMPT_FRAGMENTS = (
    "7 + 8",
    "12 * 3",
    "12 × 3",
    "17 plus 25",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def normalized_expression(record: Mapping[str, Any]) -> str:
    """Return the canonical semantics identity used for split isolation."""

    return canonical_json(record["verification"])


def normalized_expression_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(normalized_expression(record).encode()).hexdigest()


def _integer(rng: random.Random, split: str, tier: str) -> int:
    lower, upper = SPLIT_RANGES[split]
    if split == "train" and tier == "tier_1":
        return rng.randint(1, 9)
    if split == "train" and tier == "tier_2":
        return rng.randint(10, 99)
    if split == "train" and tier == "tier_3":
        return rng.randint(100, 999)
    if split == "train":
        return rng.randint(2_000, 9_999)
    return rng.randint(lower, upper)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _build_problem(
    category: str,
    split: str,
    tier: str,
    rng: random.Random,
    variant: int,
) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    a = _integer(rng, split, tier)
    b = _integer(rng, split, tier)
    signed = tier in {"tier_3", "tier_4"} and variant % 2 == 0
    if signed:
        a *= -1 if variant % 4 == 0 else 1
        b *= -1 if variant % 3 == 0 else 1

    if category == "addition":
        values = [a, b]
        if variant % 5 == 0:
            values.append(_integer(rng, split, tier))
        answer = str(sum(values))
        return (
            " + ".join(map(str, values)) + " = ?",
            answer,
            "integer",
            {"kind": "sum", "values": values},
            {"values": values, "sign_pattern": "signed" if min(values) < 0 else "positive"},
        )
    if category == "subtraction":
        values = [a, b]
        if variant % 5 == 0:
            values.append(max(1, abs(_integer(rng, split, tier)) // 3))
        answer = str(values[0] - sum(values[1:]))
        return (
            " - ".join(map(str, values)) + " = ?",
            answer,
            "integer",
            {"kind": "subtract", "values": values},
            {"values": values, "sign_pattern": "signed" if min(values) < 0 else "positive"},
        )
    if category == "multiplication":
        if split != "train":
            a = _integer(rng, split, tier)
            b = rng.randint(2, 25)
        elif tier == "tier_1":
            a, b = rng.randint(1, 9), rng.randint(1, 9)
        elif tier == "tier_2":
            a, b = rng.randint(10, 99), rng.randint(2, 9)
        elif tier == "tier_3":
            a, b = rng.randint(10, 99), rng.randint(10, 99)
        else:
            a, b = rng.randint(100, 999), rng.randint(2, 9)
        if signed:
            a = -a
        return (
            f"{a} * {b} = ?",
            str(a * b),
            "integer",
            {"kind": "product", "values": [a, b]},
            {"values": [a, b], "sign_pattern": "signed" if a < 0 else "positive"},
        )
    if category == "exact_division":
        divisor = rng.randint(2, 20)
        quotient = _integer(rng, split, tier)
        if signed:
            quotient = -abs(quotient)
        dividend = divisor * quotient
        return (
            f"{dividend} / {divisor} = ?",
            str(quotient),
            "integer",
            {"kind": "exact_division", "dividend": dividend, "divisor": divisor},
            {"values": [dividend, divisor], "sign_pattern": "signed" if quotient < 0 else "positive"},
        )
    if category == "comparison":
        if variant % 3 == 0:
            answer = ">" if a > b else "<" if a < b else "="
            return (
                f"Which symbol makes this true: {a} ? {b}",
                answer,
                "comparison_symbol",
                {"kind": "compare", "left": a, "right": b},
                {"values": [a, b], "sign_pattern": "signed" if min(a, b) < 0 else "positive"},
            )
        third = _integer(rng, split, tier)
        values = [a, b, third]
        direction = "largest" if variant % 2 else "smallest"
        result = max(values) if direction == "largest" else min(values)
        return (
            f"What is the {direction} of {a}, {b}, and {third}?",
            str(result),
            "integer",
            {"kind": "extreme", "values": values, "direction": direction},
            {"values": values, "sign_pattern": "signed" if min(values) < 0 else "positive"},
        )
    if category == "sequence":
        start = _integer(rng, split, tier)
        step = rng.randint(1, 25) * (-1 if variant % 2 else 1)
        values = [start + step * index for index in range(4)]
        return (
            f"Complete the sequence: {', '.join(map(str, values))}, ?",
            str(start + step * 4),
            "integer",
            {"kind": "arithmetic_sequence", "start": start, "step": step, "terms": 4},
            {"values": values, "step": step, "sign_pattern": "decreasing" if step < 0 else "increasing"},
        )
    if category == "fraction":
        denominator_ranges = {
            "train": (2, 24),
            "development": (25, 40),
            "evaluation": (41, 56),
        }
        denominator_low, denominator_high = denominator_ranges[split]
        d1 = rng.randint(denominator_low, denominator_high)
        d2 = rng.randint(denominator_low, denominator_high)
        n1, n2 = rng.randint(1, d1 - 1), rng.randint(1, d2 - 1)
        first, second = Fraction(n1, d1), Fraction(n2, d2)
        operation = ("simplify", "compare", "add", "subtract", "multiply")[variant % 5]
        if operation == "simplify":
            scale = rng.randint(2, 9)
            body = f"Simplify {first.numerator * scale}/{first.denominator * scale}."
            result: str = _fraction_text(first)
        elif operation == "compare":
            body = f"Compare {n1}/{d1} and {n2}/{d2} using >, <, or =."
            result = ">" if first > second else "<" if first < second else "="
        else:
            symbol = {"add": "+", "subtract": "-", "multiply": "*"}[operation]
            body = f"{n1}/{d1} {symbol} {n2}/{d2} = ?"
            value = {"add": first + second, "subtract": first - second, "multiply": first * second}[operation]
            result = _fraction_text(value)
        return (
            body,
            result,
            "comparison_symbol" if operation == "compare" else "reduced_fraction",
            {"kind": "fraction", "operation": operation, "first": [n1, d1], "second": [n2, d2]},
            {"values": [n1, d1, n2, d2], "sign_pattern": "nonnegative"},
        )
    if category == "percentage":
        percent = (5, 10, 20, 25, 50, 75)[variant % 6]
        unit = max(1, abs(_integer(rng, split, tier)))
        number = unit * 100 // math.gcd(percent, 100)
        operation = ("of", "increase", "decrease")[variant % 3]
        if operation == "of":
            result = number * percent // 100
            body = f"What is {percent}% of {number}?"
        else:
            delta = number * percent // 100
            result = number + delta if operation == "increase" else number - delta
            body = f"{operation.title()} {number} by {percent}%."
        return (
            body,
            str(result),
            "integer",
            {"kind": "percentage", "operation": operation, "number": number, "percent": percent},
            {"values": [number, percent], "sign_pattern": "positive"},
        )
    if category == "word_problem":
        count = abs(_integer(rng, split, tier))
        change = max(1, abs(_integer(rng, split, tier)) // 4)
        operation = ("add", "subtract", "multiply", "divide")[variant % 4]
        if operation == "add":
            body, result = f"A box has {count} beads and receives {change} more. How many beads are there?", count + change
        elif operation == "subtract":
            total = count + change
            body, result = f"A box has {total} beads and loses {change}. How many remain?", count
        elif operation == "multiply":
            factor = rng.randint(2, 9)
            body, result = f"There are {factor} boxes with {count} beads each. How many beads total?", factor * count
            change = factor
        else:
            groups = rng.randint(2, 9)
            total = count * groups
            body, result = f"{total} beads are split equally among {groups} boxes. How many per box?", count
            change = groups
        return (
            body,
            str(result),
            "integer",
            {"kind": "word_problem", "operation": operation, "first": count, "second": change},
            {"values": [count, change], "sign_pattern": "positive"},
        )
    if category == "mixed_expression":
        c = rng.randint(2, 20)
        operation = "sum_then_multiply" if variant % 2 else "difference_then_add"
        if operation == "sum_then_multiply":
            result, body = (a + b) * c, f"({a} + {b}) * {c} = ?"
        else:
            result, body = a - b + c, f"({a} - {b}) + {c} = ?"
        return (
            body,
            str(result),
            "integer",
            {"kind": "mixed", "operation": operation, "values": [a, b, c]},
            {"values": [a, b, c], "sign_pattern": "signed" if min(a, b) < 0 else "positive"},
        )
    value = abs(_integer(rng, split, tier))
    property_name = ("even", "divisible", "prime", "factor")[variant % 4]
    if property_name == "even":
        body, result = f"Is {value} even? Answer yes or no.", "yes" if value % 2 == 0 else "no"
        verification = {"kind": "property", "property": "even", "value": value}
    elif property_name == "divisible":
        divisor = rng.randint(2, 12)
        body, result = f"Is {value} divisible by {divisor}? Answer yes or no.", "yes" if value % divisor == 0 else "no"
        verification = {"kind": "property", "property": "divisible", "value": value, "divisor": divisor}
    elif property_name == "prime":
        if split == "train":
            value = rng.randint(2, 997)
        else:
            value = rng.randint(*SPLIT_RANGES[split])
        body, result = f"Is {value} prime? Answer yes or no.", "yes" if _is_prime(value) else "no"
        verification = {"kind": "property", "property": "prime", "value": value}
    else:
        divisor = rng.randint(2, 12)
        body, result = f"Is {divisor} a factor of {value}? Answer yes or no.", "yes" if value % divisor == 0 else "no"
        verification = {"kind": "property", "property": "factor", "value": value, "divisor": divisor}
    return body, result, "boolean", verification, {"values": [value], "sign_pattern": "positive"}


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    return all(value % divisor for divisor in range(2, math.isqrt(value) + 1))


def recompute_answer(record: Mapping[str, Any]) -> str:
    data = record["verification"]
    kind = data["kind"]
    if kind == "sum":
        return str(sum(data["values"]))
    if kind == "subtract":
        return str(data["values"][0] - sum(data["values"][1:]))
    if kind == "product":
        return str(math.prod(data["values"]))
    if kind == "exact_division":
        return str(data["dividend"] // data["divisor"])
    if kind == "compare":
        return ">" if data["left"] > data["right"] else "<" if data["left"] < data["right"] else "="
    if kind == "extreme":
        function = max if data["direction"] == "largest" else min
        return str(function(data["values"]))
    if kind == "arithmetic_sequence":
        return str(data["start"] + data["step"] * data["terms"])
    if kind == "fraction":
        first = Fraction(*data["first"])
        second = Fraction(*data["second"])
        operation = data["operation"]
        if operation == "simplify":
            return _fraction_text(first)
        if operation == "compare":
            return ">" if first > second else "<" if first < second else "="
        value = {"add": first + second, "subtract": first - second, "multiply": first * second}[operation]
        return _fraction_text(value)
    if kind == "percentage":
        amount = data["number"] * data["percent"] // 100
        return str({"of": amount, "increase": data["number"] + amount, "decrease": data["number"] - amount}[data["operation"]])
    if kind == "word_problem":
        a, b, operation = data["first"], data["second"], data["operation"]
        return str({"add": a + b, "subtract": a, "multiply": a * b, "divide": a}[operation])
    if kind == "mixed":
        a, b, c = data["values"]
        return str((a + b) * c if data["operation"] == "sum_then_multiply" else a - b + c)
    if kind == "property":
        value = data["value"]
        prop = data["property"]
        truth = {
            "even": value % 2 == 0,
            "prime": _is_prime(value),
            "divisible": value % data.get("divisor", 1) == 0,
            "factor": value % data.get("divisor", 1) == 0,
        }[prop]
        return "yes" if truth else "no"
    raise ValueError(f"unsupported verification kind: {kind}")


def generate_records(split: str, count: int, seed: int) -> list[dict[str, Any]]:
    if split not in SPLIT_RANGES:
        raise ValueError(f"unknown arithmetic-v2 split: {split}")
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(f"{FORMAT_VERSION}:{seed}:{split}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(records) < count:
        if attempts > count * 100:
            raise RuntimeError(
                "arithmetic-v2 uniqueness search exhausted near "
                f"index={len(records)}, category={CATEGORIES[len(records) % len(CATEGORIES)]}"
            )
        index = len(records)
        category = CATEGORIES[index % len(CATEGORIES)]
        category_position = index // len(CATEGORIES)
        tier_bucket = category_position % 100
        tier = (
            "tier_1"
            if tier_bucket < 1
            else "tier_2"
            if tier_bucket < 15
            else "tier_3"
            if tier_bucket < 55
            else "tier_4"
        )
        body, answer, answer_type, verification, metadata = _build_problem(
            category, split, tier, rng, attempts
        )
        if any(fragment in body for fragment in FORBIDDEN_PROMPT_FRAGMENTS):
            attempts += 1
            continue
        normalized = canonical_json(verification)
        attempts += 1
        if normalized in seen:
            continue
        seen.add(normalized)
        template_id, template = TEMPLATES[split][
            (index // (len(CATEGORIES) * len(TIERS))) % len(TEMPLATES[split])
        ]
        prompt = template.format(body=body)
        record = {
            "id": f"v2:{split}:{index:07d}",
            "split": split,
            "operation": category,
            "difficulty_tier": tier,
            "prompt": prompt,
            "answer": answer,
            "answer_type": answer_type,
            "text": TRAINING_TEXT_FORMAT.format(prompt=prompt, answer=answer),
            "generator_version": FORMAT_VERSION,
            "template_id": template_id,
            "template_family": template_id.rsplit("_", 1)[0],
            "operand_metadata": metadata,
            "verification": verification,
            "normalized_expression_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "provenance": "synthetic_programmatically_verified_v2",
        }
        if recompute_answer(record) != answer:
            raise AssertionError("arithmetic-v2 answer verification failed")
        records.append(record)
    return records


def distribution(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))
