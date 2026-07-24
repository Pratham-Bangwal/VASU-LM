"""Build deterministic, exactly verifiable arithmetic logical examples.

This creates logical text records only. It does not tokenize data, build a
training mixture, or start training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any


FORMAT_VERSION = "vasu_verified_arithmetic_v1"
TEMPLATE_ID = "question_answer_v1"
SPLITS = {
    "train": (100, 999),
    "development": (1_000, 1_499),
    "evaluation": (1_500, 1_999),
}
OPERATIONS = (
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "comparison",
    "sequence",
    "fraction",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    """Create one verified record without using capability-suite wording."""

    lower, upper = SPLITS[split]
    operation = OPERATIONS[index % len(OPERATIONS)]
    left = rng.randint(lower, upper)
    right = rng.randint(lower, upper)
    operands: dict[str, Any] = {"left": left, "right": right}
    if operation == "addition":
        expression, answer = f"{left} + {right}", str(left + right)
    elif operation == "subtraction":
        expression, answer = f"{left} - {right}", str(left - right)
    elif operation == "multiplication":
        right = rng.randint(2, 12)
        operands["right"] = right
        expression, answer = f"{left} * {right}", str(left * right)
    elif operation == "division":
        divisor = rng.randint(2, 12)
        quotient = rng.randint(lower, upper)
        operands = {"divisor": divisor, "quotient": quotient}
        expression, answer = f"{divisor * quotient} / {divisor}", str(quotient)
    elif operation == "comparison":
        expression = f"{left} compared with {right}"
        answer = "greater" if left > right else "less" if left < right else "equal"
    elif operation == "sequence":
        step = rng.randint(2, 12)
        operands = {"start": left, "step": step, "terms": 4}
        expression = f"{left}, {left + step}, {left + 2 * step}, {left + 3 * step}, next"
        answer = str(left + 4 * step)
    else:
        denominator = rng.randint(2, 12)
        numerator = rng.randint(1, denominator - 1)
        operands = {
            "denominator": denominator,
            "first_numerator": numerator,
            "second_numerator": denominator - numerator,
        }
        expression, answer = f"{numerator}/{denominator} + {denominator - numerator}/{denominator}", "1"
    prompt = f"Solve this arithmetic exercise: {expression}"
    text = f"Question: {prompt}\nAnswer: {answer}"
    difficulty = {
        "train": "tier_1",
        "development": "tier_2",
        "evaluation": "tier_3",
    }[split]
    return {
        "id": f"{split}:{index:06d}",
        "split": split,
        "operation": operation,
        "difficulty_tier": difficulty,
        "prompt": prompt,
        "expression": expression,
        "answer": answer,
        "text": text,
        "generator_version": FORMAT_VERSION,
        "template_id": TEMPLATE_ID,
        "operand_metadata": operands,
        "provenance": "synthetic_verified_v1",
    }


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def verify_record(record: dict[str, Any]) -> bool:
    """Recompute each synthetic answer without evaluating arbitrary text."""

    expression = str(record["expression"])
    answer = str(record["answer"])
    operation = record["operation"]
    if operation in {"addition", "subtraction", "multiplication", "division"}:
        left, operator, right = expression.split()
        first, second = int(left), int(right)
        result = {
            "+": first + second,
            "-": first - second,
            "*": first * second,
            "/": first // second,
        }[operator]
        return str(result) == answer
    if operation == "comparison":
        left, _, _, right = expression.split()
        expected = "greater" if int(left) > int(right) else "less" if int(left) < int(right) else "equal"
        return answer == expected
    if operation == "sequence":
        values = [int(value.strip()) for value in expression.removesuffix(", next").split(",")]
        return len(values) == 4 and values[-1] + (values[1] - values[0]) == int(answer)
    if operation == "fraction":
        first, second = expression.split(" + ")
        numerator_a, denominator_a = (int(value) for value in first.split("/"))
        numerator_b, denominator_b = (int(value) for value in second.split("/"))
        return (
            denominator_a == denominator_b
            and numerator_a + numerator_b == denominator_a
            and answer == "1"
        )
    return False


def generate_records(split: str, count: int, seed: int) -> list[dict[str, Any]]:
    """Generate one deterministic logical split and verify every answer."""

    if split not in SPLITS:
        raise ValueError(f"unknown arithmetic split: {split!r}")
    if count < 1:
        raise ValueError("split count must be positive")
    rng = random.Random(f"{seed}:{split}")
    records = [_record(split, index, rng) for index in range(count)]
    if not all(verify_record(record) for record in records):
        raise AssertionError("arithmetic generator produced an invalid record")
    return records


def build_corpus(output_dir: Path, counts: dict[str, int], seed: int) -> dict[str, Any]:
    """Create deterministic non-overlapping splits and a hash-bound manifest."""

    if set(counts) != set(SPLITS) or any(value < 1 for value in counts.values()):
        raise ValueError("counts must provide a positive value for every split")
    files: dict[str, dict[str, Any]] = {}
    for split, count in counts.items():
        records = generate_records(split, count, seed)
        path = output_dir / f"{split}.jsonl"
        _write_atomic(
            path,
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        )
        files[split] = {"path": str(path), "count": count, "sha256": _sha256(path)}
    manifest = {
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "splits": files,
        "operation_types": list(OPERATIONS),
        "template_id": TEMPLATE_ID,
        "training_text_format": "Question: {prompt}\nAnswer: {answer}",
        "operand_ranges": {name: list(bounds) for name, bounds in SPLITS.items()},
        "leakage_policy": "No capability-v1 prompt wording or its 7+8 / 12*3 fixtures are emitted.",
        "training_authorized": False,
    }
    manifest_path = output_dir / "manifest.json"
    _write_atomic(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--development-count", type=int, default=20)
    parser.add_argument("--evaluation-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_corpus(
        args.output_dir,
        {"train": args.train_count, "development": args.development_count, "evaluation": args.evaluation_count},
        args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
