"""Run the fixture-only VASU-140M 513-token record qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.vasu_140m_records import (  # noqa: E402
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    TOKENIZER_SHA256,
    UNK_TOKEN_ID,
    VOCAB_SIZE,
    build_fixture_report,
    compile_text_example,
    validate_frozen_fixture_report,
)


FIXTURES = {
    "train": (
        ("fixture-train-001", "User: Add two and three.\nAssistant:\n", "Answer: 5"),
        ("fixture-train-002", "User: Name a primary color.\nAssistant:\n", "Answer: red"),
    ),
    "development": (
        ("fixture-dev-001", "User: Continue: A, B, C.\nAssistant:\n", "Answer: D"),
    ),
    "evaluation": (
        ("fixture-eval-001", "User: Is seven greater than four?\nAssistant:\n", "Answer: yes"),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(tokenizer_path: Path) -> dict[str, object]:
    if sha256_file(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("tokenizer identity does not match the frozen specification")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    expected_specials = {
        "[PAD]": PAD_TOKEN_ID,
        "[UNK]": UNK_TOKEN_ID,
        "[BOS]": BOS_TOKEN_ID,
        "[EOS]": EOS_TOKEN_ID,
    }
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError("tokenizer vocabulary size does not match the specification")
    if {name: tokenizer.token_to_id(name) for name in expected_specials} != expected_specials:
        raise ValueError("tokenizer special-token IDs do not match the specification")
    splits = {
        split: [
            compile_text_example(
                tokenizer=tokenizer,
                example_id=example_id,
                split=split,
                prompt=prompt,
                response=response,
            )
            for example_id, prompt, response in rows
        ]
        for split, rows in FIXTURES.items()
    }
    first = build_fixture_report(splits)
    second = build_fixture_report(splits)
    if first != second:
        raise RuntimeError("fixture rebuild is not deterministic")
    validate_frozen_fixture_report(first)
    return first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=REPOSITORY_ROOT / "assets" / "tokenizer.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.tokenizer), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
