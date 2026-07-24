"""Build and validate the canonical verified-arithmetic v2 release."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.serialize_vasu_verified_arithmetic_v1 import (
    MASK_DTYPE,
    RECORD_WIDTH,
    TOKEN_DTYPE,
    _artifact_metadata,
    _record_provenance,
    _source_metadata,
    _write_array,
    _write_json,
    _write_jsonl,
    sha256_file,
    sha256_json,
    tokenize_splits,
)
from vasu.data.arithmetic_packing import (
    PACKED_ARITHMETIC_FORMAT,
    pack_arithmetic_unique_pass,
    summarize_packed_records,
)
from vasu.data.arithmetic_v2 import (
    CATEGORIES,
    DATASET_ID,
    FORMAT_VERSION,
    FORBIDDEN_PROMPT_FRAGMENTS,
    SPLIT_RANGES,
    TEMPLATES,
    TRAINING_TEXT_FORMAT,
    distribution,
    generate_records,
    recompute_answer,
)


SCHEMA_VERSION = "vasu_verified_arithmetic_release_v2"
DEFAULT_OUTPUT = Path("data/processed/capability/verified_arithmetic_v2")
DEFAULT_TOKENIZER = Path("assets/tokenizer.json")
DEFAULT_COUNTS = {"train": 32_000, "development": 1_000, "evaluation": 1_000}
DEFAULT_SEED = 42


def _overlap_checks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, bool]:
    ids: set[str] = set()
    expressions: set[str] = set()
    prompts: set[str] = set()
    id_clear = expression_clear = prompt_clear = True
    for records in splits.values():
        current_ids = {str(record["id"]) for record in records}
        current_expressions = {
            str(record["normalized_expression_sha256"]) for record in records
        }
        current_prompts = {str(record["prompt"]) for record in records}
        id_clear &= not bool(ids & current_ids)
        expression_clear &= not bool(expressions & current_expressions)
        prompt_clear &= not bool(prompts & current_prompts)
        ids.update(current_ids)
        expressions.update(current_expressions)
        prompts.update(current_prompts)
    return {
        "stable_id_overlap_clear": id_clear,
        "normalized_expression_overlap_clear": expression_clear,
        "prompt_overlap_clear": prompt_clear,
    }


def _contamination_clear(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    return all(
        fragment not in str(record["prompt"])
        for records in splits.values()
        for record in records
        for fragment in FORBIDDEN_PROMPT_FRAGMENTS
    )


def _reuse_projection(unique_records: int, allocated: int) -> dict[str, Any]:
    quotient, remainder = divmod(allocated, unique_records)
    distribution_values = {str(quotient): unique_records - remainder}
    if remainder:
        distribution_values[str(quotient + 1)] = remainder
    return {
        "allocated_records": allocated,
        "available_unique_records": unique_records,
        "effective_passes": allocated / unique_records,
        "reused_records": max(allocated - unique_records, 0),
        "maximum_reuse_count": quotient + int(remainder > 0),
        "minimum_reuse_count": quotient,
        "mean_reuse_count": allocated / unique_records,
        "reuse_count_distribution": distribution_values,
        "unique_coverage_ratio": min(allocated, unique_records) / unique_records,
        "warning_threshold": 5.0,
        "hard_limit": 10.0,
        "status": (
            "fail"
            if allocated / unique_records > 10
            else "warning"
            if allocated / unique_records > 5
            else "pass"
        ),
    }


def _validate_logical_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, bool]:
    checks = _overlap_checks(splits)
    checks["all_answers_verified"] = all(
        recompute_answer(record) == record["answer"]
        for records in splits.values()
        for record in records
    )
    checks["capability_fixture_exclusion"] = _contamination_clear(splits)
    checks["heldout_template_check"] = not (
        {
            record["template_id"] for record in splits["train"]
        }
        & {
            record["template_id"]
            for split in ("development", "evaluation")
            for record in splits[split]
        }
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("arithmetic-v2 logical validation failed: " + ", ".join(failed))
    return checks


def build_manifest(
    *,
    staging: Path,
    tokenizer_path: Path,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    token_lengths: Mapping[str, Mapping[str, Any]],
    special_ids: Mapping[str, int | None],
    vocab_size: int,
    packed_records: Sequence[Any],
    seed: int,
    created_at: str,
    checks: Mapping[str, bool],
    minimum_packed_records: int,
) -> dict[str, Any]:
    stats = summarize_packed_records(packed_records)
    artifacts = (
        "train_tokens.bin",
        "train_loss_mask.bin",
        "train_records.jsonl",
        "dev.jsonl",
        "eval.jsonl",
    )
    logical_hashes = {
        split: sha256_json(
            sorted((dict(record) for record in records), key=lambda item: item["id"])
        )
        for split, records in splits.items()
    }
    expression_hashes = {
        split: sha256_json(
            sorted(str(record["normalized_expression_sha256"]) for record in records)
        )
        for split, records in splits.items()
    }
    replay = {
        "candidate_a_5_percent": _reuse_projection(stats.packed_records, 3_907),
        "candidate_b_15_percent": _reuse_projection(stats.packed_records, 11_722),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "release_status": "validated_candidate",
        "created_at": created_at,
        "generator": _source_metadata(Path("vasu/data/arithmetic_v2.py")),
        "serializer": _source_metadata(Path(__file__).relative_to(Path.cwd())),
        "generator_configuration": {
            "format_version": FORMAT_VERSION,
            "seed": seed,
            "counts": {split: len(records) for split, records in splits.items()},
            "categories": list(CATEGORIES),
            "split_operand_ranges": {
                split: list(bounds) for split, bounds in SPLIT_RANGES.items()
            },
            "templates": {
                split: [identifier for identifier, _ in templates]
                for split, templates in TEMPLATES.items()
            },
            "training_text_format": TRAINING_TEXT_FORMAT,
        },
        "tokenizer": {
            **_source_metadata(tokenizer_path),
            "vocabulary_size": vocab_size,
            "special_token_ids": dict(special_ids),
        },
        "record_width": RECORD_WIDTH,
        "context_length": RECORD_WIDTH - 1,
        "token_dtype": TOKEN_DTYPE.name,
        "mask_dtype": MASK_DTYPE.name,
        "mask_storage": {
            "alignment": "stored_token_positions",
            "effective_alignment": "target_positions",
            "conversion": "stored_mask[1:]",
        },
        "packing": {
            "algorithm_version": PACKED_ARITHMETIC_FORMAT,
            "algorithm": "deterministic_greedy_complete_examples",
            "seed": seed,
            "record_count_policy": "one_unique_training_pass",
            "packed_records": stats.packed_records,
            "unique_examples_consumed": stats.unique_examples_consumed,
            "total_logical_examples_consumed": stats.total_logical_examples_consumed,
            "replay_epochs": stats.replay_epochs,
            "real_tokens": stats.real_tokens,
            "padding_tokens": stats.padding_tokens,
            "mean_utilization": stats.mean_utilization,
            "minimum_utilization": stats.minimum_utilization,
            "maximum_utilization": stats.maximum_utilization,
            "padding_percentage": stats.padding_percentage,
            "minimum_required_records": minimum_packed_records,
        },
        "logical_example_counts": {
            split: len(records) for split, records in splits.items()
        },
        "operation_counts": {
            split: distribution(records, "operation")
            for split, records in splits.items()
        },
        "difficulty_counts": {
            split: distribution(records, "difficulty_tier")
            for split, records in splits.items()
        },
        "template_counts": {
            split: distribution(records, "template_id")
            for split, records in splits.items()
        },
        "answer_type_counts": {
            split: distribution(records, "answer_type")
            for split, records in splits.items()
        },
        "sign_pattern_counts": {
            split: dict(
                sorted(
                    Counter(
                        str(record["operand_metadata"]["sign_pattern"])
                        for record in records
                    ).items()
                )
            )
            for split, records in splits.items()
        },
        "token_length_statistics": dict(token_lengths),
        "logical_split_sha256": logical_hashes,
        "normalized_expression_set_sha256": expression_hashes,
        "replay_safety_projection": replay,
        "artifacts": {
            name: _artifact_metadata(staging / name) for name in artifacts
        },
        "validation": {
            **dict(checks),
            "round_trip_check": True,
            "terminal_eos_check": True,
            "pad_absent_from_logical_examples": True,
            "answer_verification_count": sum(len(records) for records in splits.values()),
        },
        "training_authorized": False,
        "future_use_status": "requires schedule and experiment authorization",
    }


def validate_release(
    directory: Path,
    *,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported arithmetic-v2 manifest schema")
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected arithmetic-v2 dataset ID")
    if manifest.get("training_authorized") is not False:
        raise ValueError("arithmetic-v2 must remain unauthorized")
    tokenizer = tokenizer_path or Path(manifest["tokenizer"]["path"])
    if sha256_file(tokenizer) != manifest["tokenizer"]["sha256"]:
        raise ValueError("tokenizer hash mismatch")
    for metadata in manifest["artifacts"].values():
        path = directory / metadata["path"]
        if path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"artifact byte-size mismatch: {path.name}")
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path.name}")
    record_count = int(manifest["packing"]["packed_records"])
    tokens = np.fromfile(directory / "train_tokens.bin", dtype=TOKEN_DTYPE)
    masks = np.fromfile(directory / "train_loss_mask.bin", dtype=MASK_DTYPE)
    if len(tokens) != record_count * RECORD_WIDTH or len(masks) != len(tokens):
        raise ValueError("packed arithmetic-v2 shapes are invalid")
    tokens = tokens.reshape(record_count, RECORD_WIDTH)
    masks = masks.reshape(record_count, RECORD_WIDTH)
    pad = int(manifest["tokenizer"]["special_token_ids"]["pad"])
    eos = int(manifest["tokenizer"]["special_token_ids"]["eos"])
    if not np.isin(masks, (0, 1)).all() or np.any(masks[:, 0]):
        raise ValueError("arithmetic-v2 masks are invalid")
    if np.any(masks[:, 1:][tokens[:, 1:] == pad]):
        raise ValueError("PAD target contributes to arithmetic-v2 loss")
    provenance = [
        json.loads(line)
        for line in (directory / "train_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row, token_row in zip(provenance, tokens, strict=True):
        for span, logical in zip(row["example_spans"], row["logical_examples"], strict=True):
            if logical["id"] in seen:
                raise ValueError("arithmetic-v2 one-pass release repeats an example")
            seen.add(logical["id"])
            if recompute_answer(logical) != logical["answer"]:
                raise ValueError("arithmetic-v2 answer verification failed")
            if token_row[int(span["end"]) - 1] != eos:
                raise ValueError("arithmetic-v2 example lacks terminal EOS")
            train.append(logical)
    splits: dict[str, list[dict[str, Any]]] = {"train": train}
    for filename, split in (("dev.jsonl", "development"), ("eval.jsonl", "evaluation")):
        splits[split] = [
            json.loads(line)
            for line in (directory / filename).read_text(encoding="utf-8").splitlines()
        ]
    _validate_logical_splits(splits)
    for split, records in splits.items():
        if len(records) != manifest["logical_example_counts"][split]:
            raise ValueError(f"{split} arithmetic-v2 count mismatch")
        if sha256_json(sorted(records, key=lambda item: item["id"])) != manifest["logical_split_sha256"][split]:
            raise ValueError(f"{split} arithmetic-v2 logical hash mismatch")
    if record_count < int(manifest["packing"]["minimum_required_records"]):
        raise ValueError("arithmetic-v2 has fewer than required packed records")
    return manifest


def serialize_release(
    *,
    tokenizer_path: Path,
    output_dir: Path,
    counts: Mapping[str, int] = DEFAULT_COUNTS,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
    created_at: str | None = None,
    dry_run: bool = False,
    minimum_packed_records: int = 2_500,
) -> dict[str, Any]:
    if set(counts) != {"train", "development", "evaluation"}:
        raise ValueError("v2 counts must define train/development/evaluation")
    if output_dir.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    backup: Path | None = None
    try:
        splits = {
            split: generate_records(split, int(count), seed)
            for split, count in counts.items()
        }
        checks = _validate_logical_splits(splits)
        tokenized, lengths, special_ids, vocab_size = tokenize_splits(
            splits, tokenizer_path
        )
        packed = pack_arithmetic_unique_pass(
            tokenized["train"],
            eos_token_id=int(special_ids["eos"]),
            pad_token_id=int(special_ids["pad"]),
            seed=seed,
            utilization_warning_threshold=0.75,
            strict_utilization=True,
        )
        _write_array(staging / "train_tokens.bin", np.stack([row.tokens for row in packed]))
        _write_array(staging / "train_loss_mask.bin", np.stack([row.stored_mask for row in packed]))
        _write_jsonl(
            staging / "train_records.jsonl",
            _record_provenance(
                packed,
                {str(record["id"]): record for record in splits["train"]},
            ),
        )
        _write_jsonl(staging / "dev.jsonl", splits["development"])
        _write_jsonl(staging / "eval.jsonl", splits["evaluation"])
        manifest = build_manifest(
            staging=staging,
            tokenizer_path=tokenizer_path,
            splits=splits,
            token_lengths=lengths,
            special_ids=special_ids,
            vocab_size=vocab_size,
            packed_records=packed,
            seed=seed,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            checks=checks,
            minimum_packed_records=minimum_packed_records,
        )
        _write_json(staging / "manifest.json", manifest)
        validate_release(staging, tokenizer_path=tokenizer_path)
        if dry_run:
            return manifest
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                raise FileExistsError(backup)
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if backup and backup.exists():
                os.replace(backup, output_dir)
            raise
        if backup:
            shutil.rmtree(backup)
        return validate_release(output_dir, tokenizer_path=tokenizer_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-count", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--development-count", type=int, default=1_000)
    parser.add_argument("--evaluation-count", type=int, default=1_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        manifest = validate_release(args.output_dir, tokenizer_path=args.tokenizer)
    else:
        manifest = serialize_release(
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            counts={
                "train": args.train_count,
                "development": args.development_count,
                "evaluation": args.evaluation_count,
            },
            seed=args.seed,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
