"""Build the boundary-aware, EOS-supervised UltraChat masked-v2 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from vasu.data.alpaca_masked_v2 import atomic_write_array, atomic_write_json
from vasu.data.ultrachat_masked_v2 import (
    TurnRecordPacker,
    build_metadata,
    encode_conversation,
    sha256_file,
    validate_records,
)
from vasu.tokenizer.tokenizer import VASUTokenizer


SOURCE_FILE = Path("data/raw/instruct/ultrachat.jsonl")
TOKENIZER_FILE = Path("assets/tokenizer.json")
TOKEN_FILE = Path("data/processed/instruct/ultrachat_masked_v2.bin")
MASK_FILE = Path("data/processed/instruct/ultrachat_masked_v2_mask.bin")
METADATA_FILE = Path(
    "data/processed/instruct/ultrachat_masked_v2_metadata.json"
)
SEQUENCE_LENGTH = 256
SPLIT_SEED = 42
MIN_TOTAL_TOKENS = 4_900_000
MAX_TOTAL_TOKENS = 5_100_000
MAX_MALFORMED_EXAMPLES = 100


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"source row at {path}:{line_number} is not an object"
                )
            yield line_number, value


def prepare(*, force: bool = False) -> dict[str, Any]:
    for required in (SOURCE_FILE, TOKENIZER_FILE):
        if not required.is_file():
            raise FileNotFoundError(required)
    outputs = (TOKEN_FILE, MASK_FILE, METADATA_FILE)
    if not force and any(path.exists() for path in outputs):
        raise FileExistsError(
            "masked-v2 output already exists; pass --force to replace only "
            "these dedicated v2 files"
        )

    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER_FILE))
    ids = {
        name: tokenizer.tokenizer.token_to_id(f"[{name.upper()}]")
        for name in ("pad", "unk", "bos", "eos")
    }
    if any(value is None for value in ids.values()):
        raise ValueError("tokenizer must expose PAD, UNK, BOS, and EOS IDs")
    pad_id, unk_id, bos_id, eos_id = (
        int(ids[name]) for name in ("pad", "unk", "bos", "eos")
    )
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    if vocab_size > 65_536:
        raise ValueError("uint16 output cannot represent this vocabulary")

    packer = TurnRecordPacker(
        sequence_length=SEQUENCE_LENGTH,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    inspected = 0
    excluded_at_limit = 0
    forbidden = frozenset((pad_id, bos_id, eos_id))
    progress = tqdm(iter_jsonl(SOURCE_FILE), desc="UltraChat conversations")
    for _, sample in progress:
        inspected += 1
        snapshot = packer.snapshot()
        try:
            turns = encode_conversation(
                sample, tokenizer, forbidden_token_ids=forbidden
            )
        except ValueError:
            packer.stats.malformed_examples += 1
            packer.stats.examples_dropped += 1
            if packer.stats.malformed_examples > MAX_MALFORMED_EXAMPLES:
                raise ValueError(
                    "malformed UltraChat examples exceeded the explicit "
                    f"limit of {MAX_MALFORMED_EXAMPLES}"
                )
            continue
        packer.add_conversation(turns)
        projected_tokens = packer.projected_record_count() * (
            SEQUENCE_LENGTH + 1
        )
        if projected_tokens > MAX_TOTAL_TOKENS:
            packer.restore(snapshot)
            excluded_at_limit = 1
            break
        if inspected % 1_000 == 0:
            progress.set_postfix(tokens=f"{projected_tokens:,}")

    token_records, mask_records = packer.finalize()
    if not MIN_TOTAL_TOKENS <= token_records.size <= MAX_TOTAL_TOKENS:
        raise ValueError(
            f"selected token count {token_records.size:,} is outside "
            f"[{MIN_TOTAL_TOKENS:,}, {MAX_TOTAL_TOKENS:,}]"
        )

    atomic_write_array(TOKEN_FILE, token_records.reshape(-1))
    atomic_write_array(MASK_FILE, mask_records.reshape(-1))
    metadata = build_metadata(
        tokens=token_records,
        mask=mask_records,
        stats=packer.stats,
        source_path=SOURCE_FILE,
        source_sha256=sha256_file(SOURCE_FILE),
        source_examples_inspected=inspected,
        selection_stop_example_excluded=excluded_at_limit,
        tokenizer_path=TOKENIZER_FILE,
        tokenizer_sha256=sha256_file(TOKENIZER_FILE),
        vocabulary_size=vocab_size,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        unk_token_id=unk_id,
        sequence_length=SEQUENCE_LENGTH,
        split_seed=SPLIT_SEED,
        token_file_sha256=sha256_file(TOKEN_FILE),
        mask_file_sha256=sha256_file(MASK_FILE),
    )
    metadata["malformed_example_limit"] = MAX_MALFORMED_EXAMPLES
    metadata["split_strategy"] = (
        "contiguous 95/5 fixed-record split; seed 42 controls the "
        "deterministic training-record permutation"
    )
    validate_records(token_records, mask_records, metadata)
    atomic_write_json(METADATA_FILE, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    metadata = prepare(force=args.force)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
