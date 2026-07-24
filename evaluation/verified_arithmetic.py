"""Deterministic exact-answer evaluation for verified arithmetic releases."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from evaluation.framework.scoring import repetition_metrics
from vasu.inference.generate import generate_token_ids


INTEGER = re.compile(r"[+-]?\d+")
FRACTION = re.compile(r"[+-]?\d+/\d+")
COMPARISON = re.compile(r"[<>=]")
BOOLEAN = re.compile(r"(?:yes|no)", re.IGNORECASE)
SUPPORTED_ANSWER_TYPES = {
    "integer": INTEGER,
    "reduced_fraction": FRACTION,
    "comparison_symbol": COMPARISON,
    "boolean": BOOLEAN,
}
EVALUATOR_VERSION = "vasu_verified_arithmetic_evaluator_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_split(
    manifest_path: Path,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Load and hash-validate one logical dev/eval split."""

    names = {"dev": "dev.jsonl", "eval": "eval.jsonl"}
    if split not in names:
        raise ValueError("split must be 'dev' or 'eval'")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != "verified_arithmetic_v2":
        raise ValueError("manifest is not verified_arithmetic_v2")
    artifact_name = names[split]
    artifact = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact, Mapping):
        raise ValueError(f"manifest has no {artifact_name} artifact")
    path = manifest_path.parent / str(artifact["path"])
    actual = sha256_file(path)
    if actual != artifact.get("sha256"):
        raise ValueError(f"{artifact_name} SHA-256 mismatch")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_split = "development" if split == "dev" else "evaluation"
    expected_count = int(manifest["logical_example_counts"][expected_split])
    if len(records) != expected_count:
        raise ValueError(f"{artifact_name} record count mismatch")
    identifiers = [record.get("id") for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{artifact_name} contains duplicate IDs")
    if any(record.get("split") != expected_split for record in records):
        raise ValueError(f"{artifact_name} contains a wrong split")
    return manifest, records, path


def select_proxy_records(
    records: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Select a stable evenly spread interval-validation proxy."""

    if count < 1 or count > len(records):
        raise ValueError("proxy count must be within the split size")
    if count == len(records):
        return list(records)
    indices = [(index * len(records)) // count for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError("proxy selection produced duplicate indices")
    return [records[index] for index in indices]


def format_arithmetic_prompt(record: Mapping[str, Any]) -> str:
    """Match the v2 base-CPT serialization boundary without revealing answer."""

    prompt = str(record.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("arithmetic record has an empty prompt")
    return f"Question: {prompt}\nAnswer:"


def classify_response(
    record: Mapping[str, Any],
    response: str,
    *,
    truncated: bool = False,
) -> dict[str, Any]:
    """Classify a strict exact-answer response without permissive extraction."""

    answer_type = str(record.get("answer_type"))
    pattern = SUPPORTED_ANSWER_TYPES.get(answer_type)
    if pattern is None:
        raise ValueError(f"unsupported arithmetic answer type: {answer_type}")
    stripped = response.strip()
    leakage = (
        "Question:" in response
        or "Answer:" in response
        or str(record.get("prompt", "")) in response
    )
    full_match = pattern.fullmatch(stripped)
    parsed = full_match.group(0).lower() if full_match else None
    expected = str(record["answer"]).strip().lower()
    if not stripped:
        outcome = "unanswered"
    elif leakage:
        outcome = "prompt_leakage"
    elif full_match is None:
        outcome = "malformed"
    elif parsed == expected:
        outcome = "correct"
    else:
        outcome = "incorrect"
    return {
        "outcome": outcome,
        "correct": outcome == "correct",
        "parsed_answer": parsed,
        "expected_answer": expected,
        "malformed": outcome == "malformed",
        "unanswered": outcome == "unanswered",
        "truncated": bool(truncated),
        "prompt_leakage": leakage,
        "repetition": repetition_metrics(response),
    }


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    generated_tokens: int
    truncated: bool
    duration_seconds: float


@torch.inference_mode()
def generate_answer(
    model: Any,
    tokenizer: Any,
    record: Mapping[str, Any],
    device: torch.device,
    *,
    max_new_tokens: int,
) -> GeneratedAnswer:
    """Generate one greedy base-model continuation and expose truncation."""

    prompt = format_arithmetic_prompt(record)
    started = time.perf_counter()
    token_ids = generate_token_ids(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        prompt_format="plain",
        use_kv_cache=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    truncated = len(token_ids) >= max_new_tokens and (
        not token_ids or token_ids[-1] != eos_id
    )
    return GeneratedAnswer(
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        generated_tokens=len(token_ids),
        truncated=truncated,
        duration_seconds=time.perf_counter() - started,
    )


def evaluate_records(
    records: Iterable[dict[str, Any]],
    generator: Callable[[Mapping[str, Any]], GeneratedAnswer],
    *,
    completed_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate logical records, skipping exactly completed IDs on resume."""

    completed = completed_ids or set()
    results = []
    for record in records:
        identifier = str(record["id"])
        if identifier in completed:
            continue
        generated = generator(record)
        score = classify_response(
            record,
            generated.text,
            truncated=generated.truncated,
        )
        results.append(
            {
                "id": identifier,
                "split": record["split"],
                "operation": record["operation"],
                "difficulty_tier": record["difficulty_tier"],
                "template_family": record["template_family"],
                "answer_type": record["answer_type"],
                "prompt": record["prompt"],
                "response": generated.text,
                "generated_tokens": generated.generated_tokens,
                "duration_seconds": generated.duration_seconds,
                **score,
            }
        )
    return results


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return transparent overall and category-level arithmetic metrics."""

    def summarize_group(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(items)
        outcomes: dict[str, int] = defaultdict(int)
        for item in items:
            outcomes[str(item["outcome"])] += 1
        return {
            "count": count,
            "correct": outcomes["correct"],
            "exact_accuracy": outcomes["correct"] / count if count else None,
            "incorrect": outcomes["incorrect"],
            "malformed": outcomes["malformed"],
            "malformed_rate": outcomes["malformed"] / count if count else None,
            "unanswered": outcomes["unanswered"],
            "unanswered_rate": outcomes["unanswered"] / count if count else None,
            "truncated": sum(bool(item["truncated"]) for item in items),
            "prompt_leakage": outcomes["prompt_leakage"],
            "generated_tokens": sum(int(item["generated_tokens"]) for item in items),
            "duration_seconds": sum(float(item["duration_seconds"]) for item in items),
        }

    dimensions: dict[str, dict[str, Any]] = {}
    for field in (
        "operation",
        "difficulty_tier",
        "template_family",
        "answer_type",
    ):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for result in results:
            groups[str(result[field])].append(result)
        dimensions[field] = {
            name: summarize_group(items) for name, items in sorted(groups.items())
        }
    overall = summarize_group(results)
    duration = float(overall["duration_seconds"])
    overall["throughput_tokens_per_second"] = (
        float(overall["generated_tokens"]) / duration if duration > 0 else None
    )
    return {"overall": overall, "by": dimensions}

