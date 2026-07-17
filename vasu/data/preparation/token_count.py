"""Exact VASU-tokenizer measurement helpers."""

from __future__ import annotations

from pathlib import Path

from vasu.tokenizer.tokenizer import VASUTokenizer


def load_vasu_tokenizer(path: Path) -> VASUTokenizer:
    tokenizer = VASUTokenizer()
    tokenizer.load(str(path))
    if tokenizer.tokenizer.get_vocab_size() != 32_000:
        raise ValueError("VASU tokenizer vocabulary must remain exactly 32,000")
    return tokenizer


def count_tokens(tokenizer: VASUTokenizer, text: str) -> int:
    return len(tokenizer.encode(text))

