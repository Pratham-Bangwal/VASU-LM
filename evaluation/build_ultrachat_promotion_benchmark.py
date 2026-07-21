"""Build the frozen, synthetic UltraChat-promotion instruction benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


OUTPUT = Path("evaluation/benchmarks/ultrachat_promotion_v1.json")
SEED = 42

TOPICS = [
    "rainbows", "batteries", "public libraries", "volcanoes", "recycling",
    "maps", "sleep", "bridges", "clouds", "gardening", "vaccines",
    "databases", "music", "photosynthesis", "budgeting", "rivers",
    "passwords", "democracy",
]


def _item(index: int, category: str, prompt: str, **constraints: Any) -> dict[str, Any]:
    return {
        "id": f"{category}_{index:03d}",
        "category": category,
        "prompt": prompt,
        "constraints": constraints,
    }


def build_prompts() -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for index, topic in enumerate(TOPICS, start=1):
        prompts.extend(
            [
                _item(index, "general_question_answering", f"What is one practical thing a beginner should know about {topic}? Answer directly."),
                _item(index, "short_factual_questions", f"State one widely accepted fact about {topic} in one sentence.", sentence_count=1),
                _item(index, "explanations", f"Explain {topic} to a curious twelve-year-old in two sentences.", sentence_count=2),
                _item(index, "summarization", f"Summarize this passage in under 30 words: A community studied {topic} for several months. Volunteers collected observations, checked them twice, and shared a short public report.", max_words=30),
                _item(index, "rewriting", f"Rewrite this without changing its meaning and do not use a list: 'Learning about {topic} can help people make informed choices.'", forbid_list=True),
                _item(index, "format_following", f'Return valid JSON with exactly the keys "topic" and "summary" about {topic}.', json_keys=["topic", "summary"]),
                _item(index, "list_generation", f"Give exactly three bullet points about {topic}.", bullet_count=3),
                _item(index, "step_by_step_instructions", f"Give exactly four numbered steps for safely beginning a small project about {topic}.", numbered_count=4),
                _item(index, "simple_reasoning", f"A learner studies {topic} for 20 minutes on Monday and 25 minutes on Tuesday. How many minutes total? Answer with one number.", exact_words=1),
                _item(index, "creative_writing", f"Write a two-sentence micro-story involving {topic} and a lost notebook.", sentence_count=2),
                _item(index, "conversational_responses", f"A friend says, 'I feel overwhelmed trying to understand {topic}.' Reply supportively in two sentences.", sentence_count=2),
                _item(index, "uncertainty_handling", f"A user asks for an exact prediction of what researchers will discover about {topic} in 2050. Respond cautiously in two sentences.", sentence_count=2, uncertainty_required=True),
            ]
        )
    return prompts


def build_payload() -> dict[str, Any]:
    prompts = build_prompts()
    return {
        "benchmark_version": "ultrachat_promotion_v1",
        "description": "Synthetic instruction benchmark written for checkpoint comparison; not copied from Alpaca or UltraChat.",
        "random_seed": SEED,
        "prompt_count": len(prompts),
        "categories": sorted({item["category"] for item in prompts}),
        "prompts": prompts,
    }


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = build_payload()
    write_atomic(OUTPUT, payload)
    print(f"Prompts: {payload['prompt_count']}")
    print(f"Output: {OUTPUT}")
    print(f"SHA-256: {sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
