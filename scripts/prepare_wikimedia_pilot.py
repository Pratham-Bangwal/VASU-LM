"""Prepare one strictly bounded Wikimedia factual pilot shard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

# Make direct execution independent of the caller's working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.preparation.deduplication import fineweb_document_index_status  # noqa: E402
from vasu.data.preparation.wikimedia import (  # noqa: E402
    acquire_pinned_shard,
    load_preparation_config,
    prepare_wikimedia_pilot,
    resolve_paths,
    validate_preparation_config,
    validate_preparation_output,
    validate_registry_approval,
)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--validate-output", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Cap execution at 100 rows, 20 parents, 20 chunks, and 20,000 tokens.",
    )
    parser.add_argument(
        "--review-sample",
        action="store_true",
        help="Sample at most 500 shard-spanning rows and retain 50 review chunks.",
    )
    args = parser.parse_args(argv)
    selected = sum((args.dry_run, args.validate_output))
    if selected > 1:
        parser.error("--dry-run and --validate-output are mutually exclusive")
    if args.resume and args.restart:
        parser.error("--resume and --restart are mutually exclusive")
    if args.smoke_test and args.review_sample:
        parser.error("--smoke-test and --review-sample are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    config = load_preparation_config(config_path)
    if args.smoke_test:
        config = config.for_smoke_test()
        validate_preparation_config(config)
    elif args.review_sample:
        config = config.for_review_sample()
        validate_preparation_config(config)
    validate_registry_approval(config, REPOSITORY_ROOT)
    paths = resolve_paths(config, REPOSITORY_ROOT)

    if args.dry_run:
        print("Wikimedia pilot dry-run: validation passed")
        print(f"Dataset: {config.dataset_name}")
        print(f"Subset/split: {config.subset}/{config.split}")
        print(f"Pinned revision: {config.pinned_revision}")
        print(f"Selected shard: {config.shard_identifier}")
        print(f"Maximum download bytes: {config.max_download_bytes:,}")
        print(f"Maximum raw examples: {config.max_raw_examples:,}")
        print(
            "Maximum accepted parent documents: "
            f"{config.max_accepted_parent_documents:,}"
        )
        print(f"Maximum accepted chunks: {config.max_accepted_chunks:,}")
        print(f"Maximum output tokens: {config.max_output_tokens:,}")
        print(
            "Row selection: "
            f"{'deterministic broad review' if config.review_sampling_enabled else 'sequential'}"
        )
        print(f"Output JSONL: {paths['output_jsonl'].relative_to(REPOSITORY_ROOT)}")
        print(
            "FineWeb cross-deduplication: "
            f"{fineweb_document_index_status(REPOSITORY_ROOT, config.fineweb_index_path)['status']}"
        )
        status = fineweb_document_index_status(REPOSITORY_ROOT, config.fineweb_index_path)
        if config.fineweb_index_required and not status.get("training_ready", False):
            print("Factual training preparation status: blocked (FineWeb index required)")
        print("Download performed: no")
        return 0

    if args.validate_output:
        result = validate_preparation_output(config, repository_root=REPOSITORY_ROOT)
        print(
            f"Output valid: parent_documents={result['parent_documents']:,}, "
            f"chunks={result['chunks']:,}, "
            f"tokens={result['tokens']:,}, sha256={result['sha256']}"
        )
        return 0

    shard_path, acquisition = acquire_pinned_shard(config, REPOSITORY_ROOT)
    manifest = prepare_wikimedia_pilot(
        config,
        repository_root=REPOSITORY_ROOT,
        input_parquet=shard_path,
        acquisition_metadata=acquisition,
        resume=args.resume,
        restart=args.restart,
    )
    print(f"Completion status: {manifest['completion_status']}")
    print(f"Raw examples: {manifest['raw_examples']:,}")
    print(
        "Accepted parent documents: "
        f"{manifest['accepted_parent_documents']:,}"
    )
    print(f"Accepted chunks: {manifest['accepted_chunks']:,}")
    print(f"VASU tokens: {manifest['total_vasu_tokens']:,}")
    print(f"Reviewable JSONL: {config.output_paths.output_jsonl}")
    print(f"Manifest: {config.output_paths.manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
