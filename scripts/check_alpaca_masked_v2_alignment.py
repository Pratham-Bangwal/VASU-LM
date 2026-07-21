"""Validate and inspect the packed masked-Alpaca-v2 files."""

import json
from pathlib import Path
import random

import numpy as np

from vasu.data.alpaca_masked_v2 import validate_metadata_against_arrays
from vasu.tokenizer.tokenizer import VASUTokenizer


TOKEN_FILE = Path("data/processed/instruct/alpaca_masked_v2.bin")
MASK_FILE = Path("data/processed/instruct/alpaca_masked_v2_mask.bin")
METADATA_FILE = Path(
    "data/processed/instruct/alpaca_masked_v2_metadata.json"
)
TOKENIZER_FILE = Path("assets/tokenizer.json")
DISPLAY_SAMPLES = 3
SEED = 42


def visible_decode(
    tokenizer: VASUTokenizer,
    token_ids: list[int],
    pad_id: int,
    eos_id: int,
) -> str:
    parts: list[str] = []
    ordinary: list[int] = []

    def flush() -> None:
        if ordinary:
            parts.append(tokenizer.decode(ordinary))
            ordinary.clear()

    for token_id in token_ids:
        if token_id == eos_id:
            flush()
            parts.append("[EOS]")
        elif token_id == pad_id:
            flush()
            parts.append("[PAD]")
        else:
            ordinary.append(token_id)
    flush()
    return "".join(parts)


def print_record(
    record_index: int,
    tokens: np.ndarray,
    mask: np.ndarray,
    tokenizer: VASUTokenizer,
    pad_id: int,
    eos_id: int,
) -> None:
    print(f"\n--- Record {record_index} ---")
    start = 0
    while start < len(tokens):
        is_pad = int(tokens[start]) == pad_id
        value = int(mask[start])
        end = start + 1
        while end < len(tokens):
            if (int(tokens[end]) == pad_id) != is_pad:
                break
            if int(mask[end]) != value:
                break
            end += 1
        label = "PAD MASK=0" if is_pad else (
            "LOSS MASK=1" if value == 1 else "PROMPT MASK=0"
        )
        decoded = visible_decode(
            tokenizer,
            tokens[start:end].astype(int).tolist(),
            pad_id,
            eos_id,
        )
        print(f"[{label}] {decoded!r}")
        start = end


def main() -> None:
    for path in (TOKEN_FILE, MASK_FILE, METADATA_FILE, TOKENIZER_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    record_length = int(metadata["record_length"])
    flat_tokens = np.memmap(TOKEN_FILE, dtype=np.uint16, mode="r")
    flat_mask = np.memmap(MASK_FILE, dtype=np.uint8, mode="r")
    if len(flat_tokens) != len(flat_mask):
        raise ValueError("token count does not equal mask count")
    if len(flat_tokens) % record_length:
        raise ValueError("binary files contain an incomplete record")

    tokens = np.asarray(flat_tokens).reshape(-1, record_length)
    mask = np.asarray(flat_mask).reshape(-1, record_length)
    validate_metadata_against_arrays(tokens, mask, metadata)

    eos_id = int(metadata["eos_token_id"])
    pad_id = int(metadata["pad_token_id"])
    supervised_eos = (tokens == eos_id) & (mask == 1)
    if int(supervised_eos.sum()) != int(metadata["eos_tokens"]):
        raise ValueError("supervised EOS count does not match metadata")

    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER_FILE))

    incomplete_response_runs = 0
    for record_tokens, record_mask in zip(tokens, mask, strict=True):
        response_starts = np.flatnonzero(
            (record_mask == 1)
            & np.concatenate(([True], record_mask[:-1] == 0))
        )
        for response_start in response_starts:
            prior_eos = np.flatnonzero(
                (record_tokens[:response_start] == eos_id)
                & (record_mask[:response_start] == 1)
            )
            prompt_start = int(prior_eos[-1] + 1) if len(prior_eos) else 0
            prompt_ids = record_tokens[prompt_start:response_start]
            prompt_ids = prompt_ids[prompt_ids != pad_id]
            prompt_text = tokenizer.decode(prompt_ids.astype(int).tolist())
            if not prompt_text.rstrip().endswith("Assistant:"):
                raise ValueError(
                    "supervised response does not follow an Assistant: header"
                )

            response_end = response_start
            while (
                response_end + 1 < len(record_mask)
                and record_mask[response_end + 1] == 1
            ):
                response_end += 1
            if record_tokens[response_end] != eos_id:
                incomplete_response_runs += 1

    if incomplete_response_runs != int(metadata["truncated_examples"]):
        raise ValueError(
            "response runs without EOS do not match truncated_examples"
        )

    # Stored masks describe current tokens; the loader must use mask[1:] for y.
    first_tokens = tokens[0]
    first_mask = mask[0]
    if len(first_tokens[:-1]) != len(first_mask[1:]):
        raise AssertionError("shifted target-mask alignment failed")
    sample_count = min(DISPLAY_SAMPLES, len(tokens))
    indices = random.Random(SEED).sample(range(len(tokens)), sample_count)
    for index in indices:
        print_record(
            index,
            tokens[index],
            mask[index],
            tokenizer,
            pad_id,
            eos_id,
        )

    print("\nMask validation passed.")
    print(f"Records: {len(tokens):,}")
    print(f"Tokens/masks: {len(flat_tokens):,}")
    print(f"Assistant target tokens: {int(mask.sum()):,}")
    print(f"Supervised EOS tokens: {int(supervised_eos.sum()):,}")
    print(f"Padding tokens: {int((tokens == pad_id).sum()):,}")


if __name__ == "__main__":
    main()
