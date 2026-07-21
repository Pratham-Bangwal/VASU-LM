"""Build a reproducible Alpaca-v3 baseline from preserved 40-prompt reports.

This utility is read-only with respect to model checkpoints. It summarizes the
same greedy and sampled prompt suite already used for the UltraChat comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CHECKPOINT_ID = "vasu_60m_alpaca_masked_v3_from_200k"
CHECKPOINT_PATH = Path(
    "checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt"
)
CHECKPOINT_SHA256 = (
    "c5da8e1f95f84ad391338548ab777d2aabf3f931f7f6c5aff54c040caef63c43"
)
MASKED_VALIDATION_LOSS = 2.522585
GREEDY_REPORT = Path(
    "evaluation/checkpoint_comparison_expanded_40_greedy_auto.json"
)
SAMPLED_REPORT = Path(
    "evaluation/checkpoint_comparison_expanded_40_sampled_auto.json"
)
OUTPUT_PATH = Path(
    "evaluation/results/vasu_60m_ultrachat_parent_alpaca_v3_baseline.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"comparison report must be a list: {path}")
    matches = [row for row in payload if row.get("name") == CHECKPOINT_ID]
    if len(matches) != 1:
        raise ValueError(f"expected one {CHECKPOINT_ID!r} entry in {path}")
    return matches[0]


def aggregate(row: dict[str, Any]) -> dict[str, Any]:
    generations = row.get("generations", [])
    if not generations:
        raise ValueError("comparison entry contains no generations")
    categories = {
        "general_instruction_following": {
            "instruction_following", "planning", "conversation"
        },
        "short_factual_questions": {
            "factual_knowledge", "definition", "simple_explanation"
        },
        "format_following": {"formatting"},
    }

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        word_counts = [int(item["stats"]["words"]) for item in items]
        repetition = [float(item["stats"]["repetition_ratio"]) for item in items]
        automatic = [
            float(score)
            for item in items
            if (score := item["automatic_evaluation"].get("score")) is not None
        ]
        empty = sum(bool(item["stats"]["empty"]) for item in items)
        obvious_incoherence = sum(
            bool(item["stats"]["empty"])
            or float(item["stats"]["repetition_ratio"]) > 0.65
            or int(item["stats"]["words"]) < 3
            for item in items
        )
        count = len(items)
        return {
            "prompts": count,
            "response_relevance": {
                "value": (
                    sum(automatic) / len(automatic) if automatic else None
                ),
                "evaluated_prompts": len(automatic),
                "method": (
                    "automatic constraint score used as a conservative proxy; "
                    "not a semantic-correctness judgment"
                ),
            },
            "repetition": {
                "mean_ratio": sum(repetition) / count,
                "maximum_ratio": max(repetition),
            },
            "empty_output_rate": empty / count,
            "response_length_words": {
                "mean": sum(word_counts) / count,
                "minimum": min(word_counts),
                "maximum": max(word_counts),
            },
            "obvious_incoherence": {
                "count": obvious_incoherence,
                "rate": obvious_incoherence / count,
                "rule": "empty, fewer than 3 words, or repetition ratio above 0.65",
            },
        }

    result = {"all_prompts": summarize(generations)}
    for label, allowed in categories.items():
        subset = [item for item in generations if item["category"] in allowed]
        result[label] = summarize(subset)
    return result


def build_baseline(*, verify_checkpoint_hash: bool = True) -> dict[str, Any]:
    greedy = load_checkpoint_result(GREEDY_REPORT)
    sampled = load_checkpoint_result(SAMPLED_REPORT)
    greedy_ids = [item["prompt_id"] for item in greedy["generations"]]
    sampled_ids = [item["prompt_id"] for item in sampled["generations"]]
    if greedy_ids != sampled_ids:
        raise ValueError("greedy and sampled reports use different prompt suites")
    for row in (greedy, sampled):
        if Path(row["path"]).as_posix() != CHECKPOINT_PATH.as_posix():
            raise ValueError("comparison report checkpoint path mismatch")
        if row.get("model_config") != "vasu_60m":
            raise ValueError("comparison report architecture mismatch")
        if row.get("prompt_format") != "alpaca":
            raise ValueError("comparison report prompt format mismatch")
    if verify_checkpoint_hash and sha256(CHECKPOINT_PATH) != CHECKPOINT_SHA256:
        raise ValueError("Alpaca-v3 checkpoint SHA-256 mismatch")

    return {
        "format_version": "vasu_ultrachat_parent_baseline_v1",
        "status": "retrospectively verified preserved parent baseline",
        "checkpoint": str(CHECKPOINT_PATH).replace("\\", "/"),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_configuration": "vasu_60m",
        "prompt_format": "alpaca",
        "masked_validation_loss": MASKED_VALIDATION_LOSS,
        "prompt_count": len(greedy_ids),
        "prompt_ids": greedy_ids,
        "modes": {
            "greedy": {
                "generation_settings": greedy["generation_config"],
                "metrics": aggregate(greedy),
            },
            "sampled": {
                "generation_settings": sampled["generation_config"],
                "metrics": aggregate(sampled),
            },
        },
        "source_reports": {
            str(GREEDY_REPORT).replace("\\", "/"): sha256(GREEDY_REPORT),
            str(SAMPLED_REPORT).replace("\\", "/"): sha256(SAMPLED_REPORT),
        },
        "notes": (
            "This baseline is derived from the preserved pre/post UltraChat "
            "40-prompt reports. Metrics are heuristics for comparison, not "
            "claims of factual correctness or general reliability."
        ),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = build_baseline()
    atomic_write_json(args.output, payload)
    print(f"Baseline checkpoint: {payload['checkpoint']}")
    print(f"Prompt count: {payload['prompt_count']}")
    print(f"Masked validation loss: {payload['masked_validation_loss']:.6f}")
    print(f"Output: {args.output}")
    print("Optimizer updates: 0")


if __name__ == "__main__":
    main()
