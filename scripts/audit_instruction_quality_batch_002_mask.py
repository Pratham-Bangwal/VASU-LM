"""Audit Batch 002 response-mask boundaries using decoded tokens."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vasu.tokenizer.tokenizer import VASUTokenizer


TOKEN_PATH = Path(
    "data/processed/instruct/"
    "vasu_instruction_quality_v1_batch_002.bin"
)
MASK_PATH = Path(
    "data/processed/instruct/"
    "vasu_instruction_quality_v1_batch_002_mask.bin"
)
MANIFEST_PATH = Path(
    "data/manifests/instruct/"
    "vasu_instruction_quality_v1_batch_002_release.json"
)
TOKENIZER_PATH = Path("assets/tokenizer.json")

TOKENS_PER_RECORD = 257
SAMPLE_ROWS = [0, 1, 10, 25, 50, 75, 94]


def decode(tokenizer: VASUTokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids)
    except (AttributeError, TypeError):
        return tokenizer.tokenizer.decode(token_ids)


def contiguous_supervised_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None

    for index, value in enumerate(mask):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            spans.append((start, index))
            start = None

    if start is not None:
        spans.append((start, len(mask)))

    return spans


def main() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    tokens = np.fromfile(TOKEN_PATH, dtype=np.uint16)
    mask = np.fromfile(MASK_PATH, dtype=np.uint8)

    assert tokens.size == mask.size
    assert tokens.size % TOKENS_PER_RECORD == 0

    record_count = tokens.size // TOKENS_PER_RECORD

    assert record_count == manifest["records"]
    assert set(np.unique(mask)).issubset({0, 1})

    token_rows = tokens.reshape(record_count, TOKENS_PER_RECORD)
    mask_rows = mask.reshape(record_count, TOKENS_PER_RECORD)

    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER_PATH))

    invalid_rows: list[str] = []
    total_spans = 0

    for row_index, row_mask in enumerate(mask_rows):
        target_mask = row_mask[1:]
        spans = contiguous_supervised_spans(target_mask)

        if not spans:
            invalid_rows.append(
                f"row {row_index}: no supervised target tokens"
            )
            continue

        total_spans += len(spans)

        for start, end in spans:
            if end <= start:
                invalid_rows.append(
                    f"row {row_index}: empty supervised span"
                )

    if invalid_rows:
        raise ValueError(
            "Mask-boundary validation failed:\n"
            + "\n".join(invalid_rows[:20])
        )

    print(f"Packed records: {record_count}")
    print(f"Supervised spans: {total_spans}")
    print(f"Assistant-loss tokens: {int(mask.sum())}")
    print()

    for row_index in SAMPLE_ROWS:
        row_tokens = token_rows[row_index]
        target_tokens = row_tokens[1:]
        target_mask = mask_rows[row_index][1:]
        spans = contiguous_supervised_spans(target_mask)

        print("=" * 90)
        print(f"ROW {row_index}")
        print(f"Supervised spans: {len(spans)}")

        for span_index, (start, end) in enumerate(spans, start=1):
            context_start = max(0, start - 20)

            context_text = decode(
                tokenizer,
                target_tokens[context_start:start].tolist(),
            )
            response_text = decode(
                tokenizer,
                target_tokens[start:end].tolist(),
            )

            print(f"\nSpan {span_index}")
            print("Prompt tail:")
            print(repr(context_text))
            print("Supervised response:")
            print(repr(response_text))

    print()
    print("Structural mask audit: PASSED")
    print(
        "Manually confirm that each supervised span begins with "
        "the assistant response, not the user prompt."
    )
    print("Training authorized: False")


if __name__ == "__main__":
    main()
