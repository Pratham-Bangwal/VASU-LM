"""Prepare boundary-preserving, assistant-only Alpaca records."""

import json
from pathlib import Path

import numpy as np

from vasu.data.alpaca_masked_v2 import (
    atomic_write_array,
    atomic_write_json,
    build_metadata,
    build_packed_records,
    validate_metadata_against_arrays,
)
from vasu.tokenizer.tokenizer import VASUTokenizer


SOURCE_FILE = Path("data/raw/instruct/alpaca.jsonl")
TOKENIZER_FILE = Path("assets/tokenizer.json")
TOKEN_FILE = Path("data/processed/instruct/alpaca_masked_v2.bin")
MASK_FILE = Path("data/processed/instruct/alpaca_masked_v2_mask.bin")
METADATA_FILE = Path(
    "data/processed/instruct/alpaca_masked_v2_metadata.json"
)
SEQUENCE_LENGTH = 256
SEED = 42
MAX_MALFORMED_FRACTION = 0.01


def load_source(path: Path) -> list[dict]:
    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON at {path}:{line_number}: {error}"
                ) from error
            examples.append(item)
    return examples


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(SOURCE_FILE)
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(TOKENIZER_FILE)

    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER_FILE))
    pad_id = tokenizer.tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    if pad_id is None:
        raise RuntimeError("tokenizer does not define [PAD]")
    if eos_id is None:
        raise RuntimeError("tokenizer does not define [EOS]")

    examples = load_source(SOURCE_FILE)
    if not examples:
        raise ValueError("source dataset contains no examples")

    token_records, mask_records, stats = build_packed_records(
        examples=examples,
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    malformed_fraction = stats.malformed_examples / len(examples)
    if malformed_fraction > MAX_MALFORMED_FRACTION:
        raise ValueError(
            f"Malformed example rate {malformed_fraction:.2%} exceeds "
            f"the allowed {MAX_MALFORMED_FRACTION:.2%}."
        )
    if stats.assistant_loss_tokens <= 0:
        raise ValueError("dataset contains no assistant supervision")

    metadata = build_metadata(
        token_records,
        mask_records,
        stats,
        tokenizer_path=str(TOKENIZER_FILE),
        tokenizer_vocab_size=tokenizer.tokenizer.get_vocab_size(),
        sequence_length=SEQUENCE_LENGTH,
        source_dataset_path=str(SOURCE_FILE),
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        seed=SEED,
    )
    validate_metadata_against_arrays(token_records, mask_records, metadata)

    # Publish both binary files first and metadata last. Metadata therefore acts
    # as the completion marker for a fully prepared dataset generation.
    atomic_write_array(TOKEN_FILE, token_records.reshape(-1))
    atomic_write_array(MASK_FILE, mask_records.reshape(-1))
    atomic_write_json(METADATA_FILE, metadata)

    actual_tokens = np.memmap(TOKEN_FILE, dtype=np.uint16, mode="r")
    actual_mask = np.memmap(MASK_FILE, dtype=np.uint8, mode="r")
    if len(actual_tokens) != len(actual_mask):
        raise ValueError("published token and mask files differ in length")

    print(json.dumps(metadata, indent=2))
    print(f"Saved tokens: {TOKEN_FILE}")
    print(f"Saved masks: {MASK_FILE}")
    print(f"Saved metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
