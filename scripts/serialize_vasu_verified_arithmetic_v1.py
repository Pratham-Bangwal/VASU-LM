"""Serialize the canonical verified-arithmetic v1 capability dataset.

The serializer creates one immutable, unique pass of arithmetic training
examples. It does not construct a multi-source schedule and never starts
training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

if __package__:
    from scripts.prepare_vasu_verified_arithmetic_v1 import (
        FORMAT_VERSION,
        OPERATIONS,
        SPLITS,
        TEMPLATE_ID,
        generate_records,
        verify_record,
    )
else:
    from prepare_vasu_verified_arithmetic_v1 import (
        FORMAT_VERSION,
        OPERATIONS,
        SPLITS,
        TEMPLATE_ID,
        generate_records,
        verify_record,
    )
from vasu.data.arithmetic_packing import (
    DEFAULT_SEQUENCE_LENGTH,
    PACKED_ARITHMETIC_FORMAT,
    PackedArithmeticRecord,
    TokenizedArithmeticExample,
    pack_arithmetic_unique_pass,
    summarize_packed_records,
)
from vasu.tokenizer.tokenizer import VASUTokenizer


SCHEMA_VERSION = "verified_arithmetic_release_v1"
DATASET_ID = "verified_arithmetic_v1"
DEFAULT_OUTPUT_DIR = Path("data/processed/capability/verified_arithmetic_v1")
DEFAULT_TOKENIZER = Path("assets/tokenizer.json")
DEFAULT_COUNTS = {"train": 80, "development": 20, "evaluation": 20}
DEFAULT_SEED = 42
TOKEN_DTYPE = np.dtype(np.uint16)
MASK_DTYPE = np.dtype(np.uint8)
RECORD_WIDTH = DEFAULT_SEQUENCE_LENGTH + 1
CAPABILITY_EXCLUSIONS = ("7 + 8", "12 * 3", "12 × 3")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    """Return stable compact JSON used for logical and configuration hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for record in records:
            handle.write(canonical_json(record).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_array(path: Path, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    with path.open("wb") as handle:
        contiguous.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("wb") as handle:
        handle.write(content.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_metadata(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def _special_token_id(tokenizer: VASUTokenizer, token: str) -> int:
    token_id = tokenizer.tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"authoritative tokenizer has no {token} token")
    return int(token_id)


def _round_trip_matches(text: str, decoded: str) -> bool:
    # VASU's byte-level tokenizer deterministically adds one prefix space.
    return decoded == text or decoded == f" {text}"


def tokenize_splits(
    logical_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer_path: Path,
) -> tuple[
    dict[str, list[TokenizedArithmeticExample]],
    dict[str, dict[str, Any]],
    dict[str, int | None],
    int,
]:
    """Tokenize all splits and validate special-token and round-trip invariants."""

    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_path))
    pad_id = _special_token_id(tokenizer, "[PAD]")
    eos_id = _special_token_id(tokenizer, "[EOS]")
    bos_id = tokenizer.tokenizer.token_to_id("[BOS]")
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    if vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError("uint16 cannot represent the tokenizer vocabulary")

    tokenized: dict[str, list[TokenizedArithmeticExample]] = {}
    distributions: dict[str, dict[str, Any]] = {}
    for split, records in logical_splits.items():
        examples: list[TokenizedArithmeticExample] = []
        lengths: list[int] = []
        for record in records:
            text = str(record["text"])
            answer = str(record["answer"])
            encoded = tokenizer.encode(text)
            if pad_id in encoded:
                raise ValueError(f"logical example {record['id']} contains PAD")
            if eos_id in encoded:
                raise ValueError(
                    f"logical example {record['id']} contains embedded EOS"
                )
            decoded = tokenizer.decode(encoded)
            if not _round_trip_matches(text, decoded) or answer not in decoded:
                raise ValueError(
                    f"logical example {record['id']} failed tokenizer round-trip"
                )
            token_ids = (*encoded, eos_id)
            if len(token_ids) > RECORD_WIDTH:
                raise ValueError(
                    f"logical example {record['id']} exceeds {RECORD_WIDTH} tokens"
                )
            if token_ids.count(eos_id) != 1 or token_ids[-1] != eos_id:
                raise ValueError(
                    f"logical example {record['id']} lacks exactly one terminal EOS"
                )
            examples.append(
                TokenizedArithmeticExample(
                    source_id=str(record["id"]),
                    token_ids=tuple(token_ids),
                )
            )
            lengths.append(len(token_ids))
        values, counts = np.unique(np.asarray(lengths), return_counts=True)
        distributions[split] = {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": float(np.median(lengths)),
            "histogram": {
                str(int(value)): int(count)
                for value, count in zip(values, counts, strict=True)
            },
        }
        tokenized[split] = examples
    return (
        tokenized,
        distributions,
        {"pad": pad_id, "eos": eos_id, "bos": bos_id},
        vocab_size,
    )


def _record_provenance(
    records: Sequence[PackedArithmeticRecord],
    logical_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "record_index": index,
            "example_ids": list(record.example_ids),
            "example_spans": [
                {
                    "example_id": span.source_id,
                    "start": span.start,
                    "end": span.end,
                    "replay_epoch": span.replay_epoch,
                }
                for span in record.spans
            ],
            "replay_epoch": record.replay_epoch,
            "used_token_count": record.used_token_count,
            "padding_token_count": record.padding_token_count,
            "utilization_ratio": record.utilization_ratio,
            "logical_examples": [
                dict(logical_records[example_id])
                for example_id in record.example_ids
            ],
        }
        for index, record in enumerate(records)
    ]


def _count_field(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def _split_overlap_is_clear(
    logical_splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for records in logical_splits.values():
        ids = {str(record["id"]) for record in records}
        prompts = {str(record["prompt"]) for record in records}
        if seen_ids & ids or seen_prompts & prompts:
            return False
        seen_ids.update(ids)
        seen_prompts.update(prompts)
    return True


def _capability_exclusions_clear(
    logical_splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    return all(
        excluded not in str(record["text"])
        for records in logical_splits.values()
        for record in records
        for excluded in CAPABILITY_EXCLUSIONS
    )


def build_manifest(
    *,
    staging_dir: Path,
    tokenizer_path: Path,
    logical_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    token_lengths: Mapping[str, Mapping[str, Any]],
    special_ids: Mapping[str, int | None],
    vocab_size: int,
    packed_records: Sequence[PackedArithmeticRecord],
    generator_config: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    """Build the release manifest after all data artifacts are finalized."""

    generator_path = Path("scripts/prepare_vasu_verified_arithmetic_v1.py")
    serializer_path = Path(__file__).relative_to(Path.cwd())
    stats = summarize_packed_records(packed_records)
    artifact_names = (
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
        for split, records in logical_splits.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "release_status": "validated_candidate",
        "created_at": created_at,
        "generator": _source_metadata(generator_path),
        "generator_configuration": dict(generator_config),
        "generator_configuration_sha256": sha256_json(generator_config),
        "serializer": _source_metadata(serializer_path),
        "tokenizer": {
            **_source_metadata(tokenizer_path),
            "vocabulary_size": vocab_size,
            "special_token_ids": dict(special_ids),
        },
        "record_width": RECORD_WIDTH,
        "context_length": DEFAULT_SEQUENCE_LENGTH,
        "token_dtype": TOKEN_DTYPE.name,
        "mask_dtype": MASK_DTYPE.name,
        "mask_storage": {
            "alignment": "stored_token_positions",
            "record_width": RECORD_WIDTH,
            "effective_alignment": "target_positions",
            "effective_width": DEFAULT_SEQUENCE_LENGTH,
            "conversion": "stored_mask[1:]",
        },
        "packing": {
            "algorithm_version": PACKED_ARITHMETIC_FORMAT,
            "algorithm": "deterministic_greedy_complete_examples",
            "seed": generator_config["seed"],
            "record_count_policy": "one_unique_training_pass",
            "cross_example_masking": "EOS-to-next-example target disabled",
            "pad_tail_policy": "explicit PAD tail; all PAD-involved targets disabled",
            "packed_records": stats.packed_records,
            "total_logical_examples_consumed": (
                stats.total_logical_examples_consumed
            ),
            "unique_examples_consumed": stats.unique_examples_consumed,
            "replay_epochs": stats.replay_epochs,
            "real_tokens": stats.real_tokens,
            "padding_tokens": stats.padding_tokens,
            "mean_utilization": stats.mean_utilization,
            "minimum_utilization": stats.minimum_utilization,
            "maximum_utilization": stats.maximum_utilization,
            "padding_percentage": stats.padding_percentage,
            "examples_per_record": {
                "minimum": min(len(record.spans) for record in packed_records),
                "maximum": max(len(record.spans) for record in packed_records),
                "mean": sum(len(record.spans) for record in packed_records)
                / len(packed_records),
            },
        },
        "split_definitions": {
            split: {"operand_range": list(bounds)}
            for split, bounds in SPLITS.items()
        },
        "logical_example_counts": {
            split: len(records) for split, records in logical_splits.items()
        },
        "operation_counts": {
            split: _count_field(records, "operation")
            for split, records in logical_splits.items()
        },
        "difficulty_counts": {
            split: _count_field(records, "difficulty_tier")
            for split, records in logical_splits.items()
        },
        "token_length_statistics": dict(token_lengths),
        "logical_split_sha256": logical_hashes,
        "artifacts": {
            name: _artifact_metadata(staging_dir / name) for name in artifact_names
        },
        "validation": {
            "all_answers_verified": all(
                verify_record(dict(record))
                for records in logical_splits.values()
                for record in records
            ),
            "capability_v1_exclusion_check": _capability_exclusions_clear(
                logical_splits
            ),
            "split_overlap_check": _split_overlap_is_clear(logical_splits),
            "tokenizer_round_trip_check": True,
            "terminal_eos_check": True,
            "pad_absent_from_logical_examples": True,
        },
        "training_authorized": False,
        "future_use_status": "requires separate mixture and training authorization",
    }


def validate_release(
    directory: Path,
    *,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a completed or staged arithmetic release without modification."""

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported arithmetic manifest schema")
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected arithmetic dataset ID")
    if manifest.get("training_authorized") is not False:
        raise ValueError("arithmetic release must not authorize training")
    required_sections = {
        "artifacts",
        "logical_example_counts",
        "logical_split_sha256",
        "packing",
        "tokenizer",
        "validation",
    }
    missing_sections = required_sections - manifest.keys()
    if missing_sections:
        raise ValueError(
            "arithmetic manifest is missing sections: "
            + ", ".join(sorted(missing_sections))
        )

    expected_tokenizer = tokenizer_path or Path(manifest["tokenizer"]["path"])
    if not expected_tokenizer.is_file():
        raise FileNotFoundError(expected_tokenizer)
    if sha256_file(expected_tokenizer) != manifest["tokenizer"]["sha256"]:
        raise ValueError("tokenizer hash mismatch")

    for metadata in manifest["artifacts"].values():
        path = directory / metadata["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"artifact byte-size mismatch: {path.name}")
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path.name}")

    record_count = int(manifest["packing"]["packed_records"])
    tokens = np.fromfile(directory / "train_tokens.bin", dtype=TOKEN_DTYPE)
    masks = np.fromfile(directory / "train_loss_mask.bin", dtype=MASK_DTYPE)
    expected_values = record_count * RECORD_WIDTH
    if len(tokens) != expected_values or len(masks) != expected_values:
        raise ValueError("packed token/mask shapes do not match manifest")
    tokens = tokens.reshape(record_count, RECORD_WIDTH)
    masks = masks.reshape(record_count, RECORD_WIDTH)
    if not np.isin(masks, (0, 1)).all() or np.any(masks[:, 0] != 0):
        raise ValueError("stored masks are invalid")
    pad_id = int(manifest["tokenizer"]["special_token_ids"]["pad"])
    eos_id = int(manifest["tokenizer"]["special_token_ids"]["eos"])
    effective = masks[:, 1:]
    if np.any(effective[tokens[:, 1:] == pad_id] != 0):
        raise ValueError("PAD target contributes to loss")
    if np.any(effective[tokens[:, :-1] == pad_id] != 0):
        raise ValueError("PAD input contributes to loss")

    provenance = [
        json.loads(line)
        for line in (directory / "train_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(provenance) != record_count:
        raise ValueError("train provenance record count mismatch")
    if [record["record_index"] for record in provenance] != list(
        range(record_count)
    ):
        raise ValueError("train provenance record indexes are not contiguous")

    train_records: list[dict[str, Any]] = []
    train_ids: set[str] = set()
    for index, record in enumerate(provenance):
        used = int(record["used_token_count"])
        padding = int(record["padding_token_count"])
        if used < 1 or used + padding != RECORD_WIDTH:
            raise ValueError(f"invalid packed accounting in record {index}")
        if np.any(tokens[index, used:] != pad_id):
            raise ValueError(f"padding is not a contiguous tail in record {index}")
        if np.any(tokens[index, :used] == pad_id):
            raise ValueError(f"real token region contains PAD in record {index}")
        if np.any(effective[index, max(used - 1, 0) :] != 0):
            raise ValueError(f"padding transition is supervised in record {index}")

        spans = record["example_spans"]
        examples = record.get("logical_examples", [])
        if len(spans) != len(examples) or len(spans) != len(
            record["example_ids"]
        ):
            raise ValueError(f"provenance example count mismatch in record {index}")
        cursor = 0
        for span_index, (span, logical) in enumerate(zip(spans, examples, strict=True)):
            start, end = int(span["start"]), int(span["end"])
            if start != cursor or not start < end <= used:
                raise ValueError(f"invalid example span in record {index}")
            if tokens[index, end - 1] != eos_id:
                raise ValueError(f"example lacks terminal EOS in record {index}")
            if end - start < 2 or effective[index, end - 2] != 1:
                raise ValueError(f"answer-to-EOS target is masked in record {index}")
            if span_index + 1 < len(spans) and effective[index, end - 1] != 0:
                raise ValueError(
                    f"cross-example transition is supervised in record {index}"
                )
            if logical["id"] != span["example_id"]:
                raise ValueError(f"logical provenance ID mismatch in record {index}")
            if logical["id"] in train_ids:
                raise ValueError(f"training example repeated: {logical['id']}")
            if not verify_record(logical):
                raise ValueError(f"training example has invalid answer: {logical['id']}")
            train_ids.add(logical["id"])
            train_records.append(logical)
            cursor = end
        if cursor != used:
            raise ValueError(f"unaccounted real tokens in record {index}")

    if len(train_records) != manifest["logical_example_counts"]["train"]:
        raise ValueError("training logical count mismatch")
    if (
        sha256_json(sorted(train_records, key=lambda item: item["id"]))
        != manifest["logical_split_sha256"]["train"]
    ):
        raise ValueError("training logical hash mismatch")

    logical_splits: dict[str, list[dict[str, Any]]] = {"train": train_records}
    for artifact_name, split in (
        ("dev.jsonl", "development"),
        ("eval.jsonl", "evaluation"),
    ):
        records = [
            json.loads(line)
            for line in (directory / artifact_name)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if len(records) != manifest["logical_example_counts"][split]:
            raise ValueError(f"{split} logical count mismatch")
        if (
            sha256_json(sorted(records, key=lambda item: item["id"]))
            != manifest["logical_split_sha256"][split]
        ):
            raise ValueError(f"{split} logical hash mismatch")
        if not all(verify_record(record) for record in records):
            raise ValueError(f"{split} contains an invalid exact answer")
        logical_splits[split] = records
    if not _split_overlap_is_clear(logical_splits):
        raise ValueError("logical train/development/evaluation splits overlap")
    if not _capability_exclusions_clear(logical_splits):
        raise ValueError("release contains a capability-v1 arithmetic fixture")
    return manifest


def serialize_release(
    *,
    tokenizer_path: Path,
    output_dir: Path,
    counts: Mapping[str, int],
    seed: int,
    overwrite: bool = False,
    utilization_warning_threshold: float | None = 0.75,
    strict_utilization: bool = False,
    created_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build, validate, and atomically publish the arithmetic release."""

    if set(counts) != set(SPLITS) or any(value < 1 for value in counts.values()):
        raise ValueError("counts must provide every split with a positive value")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if output_dir.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"{output_dir} exists; pass --overwrite to replace it transactionally"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    backup_dir: Path | None = None
    try:
        logical_splits = {
            split: generate_records(split, int(counts[split]), seed)
            for split in SPLITS
        }
        if not _split_overlap_is_clear(logical_splits):
            raise ValueError("arithmetic logical splits overlap")
        if not _capability_exclusions_clear(logical_splits):
            raise ValueError("arithmetic corpus contains capability-v1 fixtures")
        tokenized, token_lengths, special_ids, vocab_size = tokenize_splits(
            logical_splits, tokenizer_path
        )
        records = pack_arithmetic_unique_pass(
            tokenized["train"],
            eos_token_id=int(special_ids["eos"]),
            pad_token_id=int(special_ids["pad"]),
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
            seed=seed,
            utilization_warning_threshold=utilization_warning_threshold,
            strict_utilization=strict_utilization,
        )
        token_array = np.stack([record.tokens for record in records])
        mask_array = np.stack([record.stored_mask for record in records])
        if token_array.shape != mask_array.shape:
            raise ValueError("packed token and stored-mask shapes differ")
        _write_array(staging_dir / "train_tokens.bin", token_array)
        _write_array(staging_dir / "train_loss_mask.bin", mask_array)
        _write_jsonl(
            staging_dir / "train_records.jsonl",
            _record_provenance(
                records,
                {
                    str(record["id"]): record
                    for record in logical_splits["train"]
                },
            ),
        )
        _write_jsonl(staging_dir / "dev.jsonl", logical_splits["development"])
        _write_jsonl(staging_dir / "eval.jsonl", logical_splits["evaluation"])

        generator_config = {
            "format_version": FORMAT_VERSION,
            "template_id": TEMPLATE_ID,
            "training_text_format": "Question: {prompt}\nAnswer: {answer}",
            "seed": seed,
            "counts": dict(counts),
            "operations": list(OPERATIONS),
            "split_operand_ranges": {
                split: list(bounds) for split, bounds in SPLITS.items()
            },
        }
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        manifest = build_manifest(
            staging_dir=staging_dir,
            tokenizer_path=tokenizer_path,
            logical_splits=logical_splits,
            token_lengths=token_lengths,
            special_ids=special_ids,
            vocab_size=vocab_size,
            packed_records=records,
            generator_config=generator_config,
            created_at=timestamp,
        )
        _write_json(staging_dir / "manifest.json", manifest)
        validate_release(staging_dir, tokenizer_path=tokenizer_path)
        if dry_run:
            return manifest

        if output_dir.exists():
            backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
            if backup_dir.exists():
                raise FileExistsError(
                    f"transaction backup path already exists: {backup_dir}"
                )
            os.replace(output_dir, backup_dir)
        try:
            os.replace(staging_dir, output_dir)
        except BaseException:
            if backup_dir is not None and backup_dir.exists():
                os.replace(backup_dir, output_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
        return validate_release(output_dir, tokenizer_path=tokenizer_path)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-count", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument(
        "--development-count",
        type=int,
        default=DEFAULT_COUNTS["development"],
    )
    parser.add_argument(
        "--evaluation-count",
        type=int,
        default=DEFAULT_COUNTS["evaluation"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compact-json-summary", action="store_true")
    parser.add_argument("--strict-utilization", action="store_true")
    parser.add_argument("--strict-utilization-threshold", type=float)
    parser.add_argument(
        "--utilization-warning-threshold",
        type=float,
        default=0.75,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only and args.dry_run:
        raise SystemExit("--validate-only and --dry-run are mutually exclusive")
    strict_threshold = args.strict_utilization_threshold
    warning_threshold = (
        strict_threshold
        if strict_threshold is not None
        else args.utilization_warning_threshold
    )
    if args.validate_only:
        manifest = validate_release(
            args.output_dir,
            tokenizer_path=args.tokenizer,
        )
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
            utilization_warning_threshold=warning_threshold,
            strict_utilization=(
                args.strict_utilization or strict_threshold is not None
            ),
            dry_run=args.dry_run,
        )
    if args.compact_json_summary:
        summary = {
            "dataset_id": manifest["dataset_id"],
            "release_status": manifest["release_status"],
            "logical_example_counts": manifest["logical_example_counts"],
            "packing": manifest["packing"],
            "training_authorized": manifest["training_authorized"],
            "dry_run": args.dry_run,
        }
        print(json.dumps(summary, separators=(",", ":"), ensure_ascii=False))
    else:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
