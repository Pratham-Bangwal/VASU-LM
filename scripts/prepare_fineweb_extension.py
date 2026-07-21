"""Stream and tokenize a pinned FineWeb-Edu extension shard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from datasets import load_dataset

from vasu.data.fineweb_extension import (
    MIN_FREE_DISK_GIB,
    prepare_extension_from_rows,
    validate_source_selection,
)
from vasu.tokenizer.tokenizer import VASUTokenizer


DEFAULT_REPOSITORY = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "CC-MAIN-2025-26"
DEFAULT_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a compatible, resumable FineWeb-Edu extension."
    )
    parser.add_argument("--dataset-repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--dataset-config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--buffer-token-limit", type=int, default=1_000_000)
    parser.add_argument(
        "--stop-after-documents",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_source_selection(
        args.dataset_repository,
        args.dataset_config,
        args.dataset_revision,
        args.split,
    )
    if not args.tokenizer.exists():
        raise FileNotFoundError(args.tokenizer)

    metadata_output = args.metadata_output or args.output.with_name(
        f"{args.output.stem}_metadata.json"
    )
    tokenizer = VASUTokenizer()
    tokenizer.load(str(args.tokenizer))
    if tokenizer.tokenizer.get_vocab_size() != 32_000:
        raise ValueError("Tokenizer vocabulary must remain exactly 32,000.")

    dataset = load_dataset(
        args.dataset_repository,
        name=args.dataset_config,
        revision=args.dataset_revision,
        split=args.split,
        streaming=True,
    )
    try:
        metadata = prepare_extension_from_rows(
            rows=dataset,
            tokenizer=tokenizer,
            tokenizer_path=args.tokenizer,
            output_path=args.output,
            metadata_path=metadata_output,
            dataset_repository=args.dataset_repository,
            dataset_config=args.dataset_config,
            dataset_revision=args.dataset_revision,
            split=args.split,
            target_tokens=args.target_tokens,
            buffer_token_limit=args.buffer_token_limit,
            seed=args.seed,
            resume=args.resume,
            max_documents=args.max_documents,
            minimum_free_disk_gib=MIN_FREE_DISK_GIB,
            stop_after_documents=args.stop_after_documents,
        )
    except (KeyboardInterrupt, InterruptedError) as error:
        raise SystemExit(
            "Preparation interrupted; temporary output, SQLite deduplication, "
            "and resume state were preserved. Re-run with --resume."
        ) from error

    print(f"Prepared: {args.output}")
    print(f"Metadata: {metadata_output}")
    print(f"Actual tokens: {metadata['actual_written_tokens']:,}")


if __name__ == "__main__":
    main()
