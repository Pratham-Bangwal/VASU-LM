"""Prompt-suite contamination checks for candidate documents."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from .filters import comparison_normalize


WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class PromptEvidence:
    prompt_id: str
    normalized_prompt: str
    fragments: tuple[str, ...]


def load_prompt_evidence(path: Path, ngram_words: int) -> tuple[PromptEvidence, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Prompt suite must be a JSON list")
    evidence = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"Prompt suite item {index} lacks string id/prompt")
        normalized = comparison_normalize(item["prompt"])
        words = WORD_PATTERN.findall(normalized)
        fragments = tuple(
            " ".join(words[start : start + ngram_words])
            for start in range(max(0, len(words) - ngram_words + 1))
        )
        evidence.append(PromptEvidence(item["id"], normalized, fragments))
    return tuple(evidence)


def check_contamination(
    text: str,
    document_id: str,
    evidence: tuple[PromptEvidence, ...],
) -> tuple[bool, list[dict[str, str]]]:
    normalized = comparison_normalize(text)
    matches: list[dict[str, str]] = []
    exact = False
    for prompt in evidence:
        method = None
        if prompt.normalized_prompt and prompt.normalized_prompt in normalized:
            method = "full_prompt"
            exact = True
        elif prompt.fragments and any(fragment in normalized for fragment in prompt.fragments):
            method = "distinctive_ngram"
        if method:
            digest = __import__("hashlib").sha256(normalized.encode("utf-8")).hexdigest()
            matches.append(
                {
                    "document_id": document_id,
                    "prompt_id": prompt.prompt_id,
                    "matching_method": method,
                    "document_hash": digest,
                    "safe_excerpt": normalized[:160],
                }
            )
    return exact, matches

