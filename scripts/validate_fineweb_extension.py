"""Validate a finalized FineWeb extension shard and optional manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vasu.data.fineweb_extension import (
    ORIGINAL_DATA_SHA256,
    ORIGINAL_TRAIN_END,
    REQUIRED_METADATA_FIELDS,
    VOCAB_SIZE,
    build_manifest,
    sha256_file,
    validate_manifest,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_extension(
    output_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = load_json(metadata_path)
    missing = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    if missing:
        raise ValueError(f"Metadata fields missing: {', '.join(missing)}")
    if output_path.stat().st_size % 2:
        raise ValueError("Output byte size is not divisible by two.")
    if metadata["dtype"] != "uint16" or metadata["bytes_per_token"] != 2:
        raise ValueError("Extension must use uint16 with two bytes per token.")
    actual_tokens = output_path.stat().st_size // 2
    if actual_tokens != int(metadata["actual_written_tokens"]):
        raise ValueError("Metadata token count does not match file size.")
    if int(metadata["file_size_bytes"]) != output_path.stat().st_size:
        raise ValueError("Metadata byte size does not match output.")
    if actual_tokens < int(metadata["requested_target_tokens"]):
        raise ValueError("Finalized shard is below its requested target.")
    tokens = np.memmap(output_path, dtype=np.uint16, mode="r")
    minimum = int(tokens.min())
    maximum = int(tokens.max())
    if minimum != int(metadata["minimum_token_id"]) or maximum != int(metadata["maximum_token_id"]):
        raise ValueError("Metadata token range does not match the shard.")
    if minimum < 0 or maximum >= VOCAB_SIZE:
        raise ValueError("Shard contains an out-of-vocabulary token ID.")
    tokenizer_path = Path(metadata["tokenizer_path"])
    if sha256_file(tokenizer_path) != metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer SHA-256 mismatch.")
    if sha256_file(output_path) != metadata["output_sha256"]:
        raise ValueError("Output SHA-256 mismatch.")
    original_path = Path(metadata["original_data_path"])
    original_hash = sha256_file(original_path)
    if original_hash != metadata["original_data_sha256"]:
        raise ValueError("Original FineWeb binary was modified.")
    if (
        original_path.resolve()
        == Path("data/processed/pretrain/fineweb_1m.bin").resolve()
        and original_hash != ORIGINAL_DATA_SHA256
    ):
        raise ValueError("Production FineWeb binary hash is unexpected.")
    if int(metadata["original_train_end"]) != ORIGINAL_TRAIN_END:
        raise ValueError("Original validation boundary changed.")
    if not metadata["dataset_revision"] or len(metadata["dataset_revision"]) != 40:
        raise ValueError("A full pinned dataset revision is required.")
    if not metadata["separator_policy"]:
        raise ValueError("Separator policy is missing.")
    for field in (
        "documents_seen",
        "documents_written",
        "documents_dropped",
        "duplicates_within_extension",
    ):
        if not isinstance(metadata[field], int):
            raise ValueError(f"{field} must be an integer.")
    state_path = output_path.with_suffix(output_path.suffix + ".state.json")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if state_path.exists() or temporary_path.exists():
        raise ValueError("Finalized output still has temporary/resume files.")
    if metadata.get("resume_state_finalized") is not True:
        raise ValueError("Resume state is not marked finalized.")
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--create-manifest", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = validate_extension(args.output, args.metadata)
    manifest_status = "not requested"
    if args.manifest:
        if args.create_manifest:
            metadata_with_path = dict(metadata)
            metadata_with_path["output_path"] = args.output.as_posix()
            build_manifest(
                extension_metadata=metadata_with_path,
                manifest_path=args.manifest,
            )
        manifest = load_json(args.manifest)
        validate_manifest(manifest)
        manifest_status = "valid"
    print("FineWeb extension validation: PASS")
    print(f"Output: {args.output}")
    print(f"Tokens: {metadata['actual_written_tokens']:,}")
    print(
        f"Token range: {metadata['minimum_token_id']}.."
        f"{metadata['maximum_token_id']}"
    )
    print(f"Source: {metadata['dataset_repository']}/{metadata['dataset_config']}")
    print(f"Revision: {metadata['dataset_revision']}")
    print(f"Deduplicated: {metadata['duplicates_within_extension']:,} within extension")
    print(f"Manifest: {manifest_status}")


if __name__ == "__main__":
    main()
