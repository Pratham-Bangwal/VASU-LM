"""Read-only structural and semantic validation for UltraChat masked v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vasu.data.ultrachat_masked_v2 import (
    FORMAT_VERSION,
    sha256_file,
    validate_records,
)
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.instruction_dataset import PackedInstructionDataset


DEFAULT_METADATA = Path(
    "data/processed/instruct/ultrachat_masked_v2_metadata.json"
)


def validate(metadata_path: Path, *, samples: int = 3) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported UltraChat masked-v2 format")
    root = metadata_path.parents[3]
    token_path = root / "data/processed/instruct/ultrachat_masked_v2.bin"
    mask_path = root / "data/processed/instruct/ultrachat_masked_v2_mask.bin"
    tokenizer_path = root / str(metadata["tokenizer_path"])
    for path in (token_path, mask_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(token_path) != metadata["token_file_sha256"]:
        raise ValueError("token file SHA-256 mismatch")
    if sha256_file(mask_path) != metadata["mask_file_sha256"]:
        raise ValueError("mask file SHA-256 mismatch")
    if sha256_file(tokenizer_path) != metadata["tokenizer_sha256"]:
        raise ValueError("tokenizer SHA-256 mismatch")

    record_length = int(metadata["record_length"])
    tokens_flat = np.memmap(token_path, dtype=np.uint16, mode="r")
    mask_flat = np.memmap(mask_path, dtype=np.uint8, mode="r")
    if len(tokens_flat) != len(mask_flat):
        raise ValueError("token and mask element counts differ")
    if len(tokens_flat) % record_length:
        raise ValueError("files do not contain complete records")
    tokens = np.asarray(tokens_flat).reshape(-1, record_length)
    mask = np.asarray(mask_flat).reshape(-1, record_length)
    validate_records(tokens, mask, metadata)

    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_path))
    eos_id = int(metadata["eos_token_id"])
    pad_id = int(metadata["pad_token_id"])
    decoded: list[str] = []
    inspected_runs = 0
    for record_index, (record_tokens, record_mask) in enumerate(
        zip(tokens, mask, strict=True)
    ):
        starts = np.flatnonzero(
            (record_mask == 1)
            & np.concatenate(([True], record_mask[:-1] == 0))
        )
        prior_boundary = 0
        for start in starts:
            prompt = record_tokens[prior_boundary:int(start)].tolist()
            prompt = [int(value) for value in prompt if value != pad_id]
            prompt_text = tokenizer.decode(prompt)
            if not prompt_text.rstrip().endswith("Assistant:"):
                raise ValueError(
                    f"record {record_index} supervised run lacks the exact "
                    "Assistant: response boundary"
                )
            end = int(start)
            while end + 1 < record_length and record_mask[end + 1] == 1:
                end += 1
            prior_boundary = end + 1 if record_tokens[end] == eos_id else end + 1
            inspected_runs += 1
        if len(decoded) < samples:
            nonpad = [int(value) for value in record_tokens if value != pad_id]
            decoded.append(tokenizer.decode(nonpad))

    dataset = PackedInstructionDataset(
        str(token_path),
        str(mask_path),
        seq_len=int(metadata["sequence_length"]),
    )
    x, y, target_mask = dataset[0]
    if not np.array_equal(x.numpy(), tokens[0, :-1]):
        raise ValueError("dataset input shift mismatch")
    if not np.array_equal(y.numpy(), tokens[0, 1:]):
        raise ValueError("dataset target shift mismatch")
    if not np.array_equal(target_mask.numpy(), mask[0, 1:]):
        raise ValueError("dataset loss-mask shift mismatch")

    print(f"Format: {metadata['format_version']}")
    print(f"Records: {len(tokens):,}")
    print(f"Total tokens: {tokens.size:,}")
    print(f"Supervised tokens: {int(mask.sum()):,}")
    print(f"Supervised EOS: {metadata['supervised_eos_count']:,}")
    print(f"Truncated turns without EOS: {metadata['truncated_turns']:,}")
    print(f"Semantic response runs inspected: {inspected_runs:,}")
    print("Split overlap: none (contiguous record ranges)")
    for index, text in enumerate(decoded, start=1):
        print(f"\nDecoded record {index}:\n{text}")
    print("Validation: PASSED")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    validate(args.metadata.resolve(), samples=args.samples)


if __name__ == "__main__":
    main()
