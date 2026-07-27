"""Immutable operation-aware views and schedules for verified arithmetic data.

The canonical arithmetic-v2 release packs different operations into the same
physical record.  A schedule over those records cannot control operation
exposure.  This module builds a separately versioned, homogeneous-operation
packed *view* from the authoritative train-only provenance, then schedules
that view without changing the canonical release.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from vasu.data.arithmetic_packing import (
    TokenizedArithmeticExample,
    pack_arithmetic_unique_pass,
)
from vasu.tokenizer.tokenizer import VASUTokenizer


INDEX_FORMAT = "vasu_arithmetic_operation_index_v1"
VIEW_FORMAT = "vasu_operation_aware_arithmetic_view_v1"
SCHEDULE_FORMAT = "vasu_operation_aware_arithmetic_schedule_v1"
RECORD_WIDTH = 257
SCHEDULE_DTYPE = np.dtype([("record", "<u8"), ("stage", "<u4")])
SUPPORTED_OPERATIONS = frozenset(
    {
        "addition",
        "subtraction",
        "multiplication",
        "exact_division",
        "percentage",
        "fraction",
        "mixed_expression",
        "sequence",
        "word_problem",
        "comparison",
        "numeric_property",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(path, b"".join((canonical_json(row) + "\n").encode() for row in rows))


@dataclass(frozen=True)
class OperationIndexEntry:
    source_example_id: str
    split: str
    operation: str
    difficulty_tier: str
    answer_type: str
    template_family: str
    packed_record_index: int
    start: int
    end: int


def build_operation_index(train_records_path: Path) -> list[OperationIndexEntry]:
    """Read explicit, authoritative train provenance; reject non-train rows."""
    entries: list[OperationIndexEntry] = []
    for record_index, line in enumerate(
        train_records_path.read_text(encoding="utf-8").splitlines()
    ):
        row = json.loads(line)
        for span, example in zip(
            row["example_spans"], row["logical_examples"], strict=True
        ):
            if example.get("split") != "train":
                raise ValueError("operation index may contain train examples only")
            operation = str(example["operation"])
            if operation not in SUPPORTED_OPERATIONS:
                raise ValueError(f"unsupported operation in provenance: {operation}")
            entries.append(
                OperationIndexEntry(
                    source_example_id=str(example["id"]),
                    split="train",
                    operation=operation,
                    difficulty_tier=str(example["difficulty_tier"]),
                    answer_type=str(example["answer_type"]),
                    template_family=str(example["template_family"]),
                    packed_record_index=record_index,
                    start=int(span["start"]),
                    end=int(span["end"]),
                )
            )
    if not entries or len({item.source_example_id for item in entries}) != len(entries):
        raise ValueError("operation index requires unique train example IDs")
    return entries


def write_operation_index(
    *, train_records_path: Path, output_path: Path
) -> dict[str, Any]:
    entries = build_operation_index(train_records_path)
    rows = [asdict(item) for item in entries]
    _write_jsonl(output_path, rows)
    return {
        "schema_version": INDEX_FORMAT,
        "source_train_records": {
            "path": train_records_path.as_posix(),
            "sha256": sha256_file(train_records_path),
        },
        "index_path": output_path.name,
        "index_sha256": sha256_file(output_path),
        "entry_count": len(entries),
        "operation_counts": dict(
            sorted(Counter(item.operation for item in entries).items())
        ),
        "split_counts": {"train": len(entries)},
    }


def _load_train_examples(train_records_path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for line in train_records_path.read_text(encoding="utf-8").splitlines():
        examples.extend(json.loads(line)["logical_examples"])
    if any(item.get("split") != "train" for item in examples):
        raise ValueError("derived operation-aware view must be train-only")
    return examples


def _pack_group(
    examples: Sequence[Mapping[str, Any]],
    tokenizer: VASUTokenizer,
    eos: int,
    pad: int,
    seed: int,
) -> list[Any]:
    tokenized = [
        TokenizedArithmeticExample(
            str(item["id"]), tuple((*tokenizer.encode(str(item["text"])), eos))
        )
        for item in examples
    ]
    return pack_arithmetic_unique_pass(
        tokenized, eos_token_id=eos, pad_token_id=pad, seed=seed
    )


def build_operation_view(
    *, train_records_path: Path, tokenizer_path: Path, output_dir: Path, seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a new homogeneous-operation packed view without touching v2."""
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite operation-aware view: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    examples = _load_train_examples(train_records_path)
    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_path))
    eos = tokenizer.tokenizer.token_to_id("[EOS]")
    pad = tokenizer.tokenizer.token_to_id("[PAD]")
    if eos is None or pad is None:
        raise ValueError("tokenizer special tokens are missing")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in examples:
        operation = str(item["operation"])
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"unsupported operation: {operation}")
        grouped[(operation, str(item["difficulty_tier"]))].append(item)
    index = write_operation_index(
        train_records_path=train_records_path,
        output_path=output_dir.parent / f".{output_dir.name}.index.tmp",
    )
    temporary_index = output_dir.parent / f".{output_dir.name}.index.tmp"
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        shutil.move(str(temporary_index), staging / "source_operation_index.jsonl")
        rows: list[dict[str, Any]] = []
        token_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        source_by_id = {str(item["id"]): item for item in examples}
        for group_number, ((operation, tier), members) in enumerate(
            sorted(grouped.items())
        ):
            for packed in _pack_group(
                members, tokenizer, int(eos), int(pad), seed + group_number
            ):
                record_index = len(rows)
                token_rows.append(packed.tokens)
                mask_rows.append(packed.stored_mask)
                rows.append(
                    {
                        "record_index": record_index,
                        "operation": operation,
                        "difficulty_tier": tier,
                        "source_example_ids": list(packed.example_ids),
                        "source_examples": [
                            source_by_id[item] for item in packed.example_ids
                        ],
                        "source_record_mapping": "authoritative_train_records_jsonl",
                        "used_token_count": packed.used_token_count,
                        "padding_token_count": packed.padding_token_count,
                    }
                )
        tokens = np.stack(token_rows)
        masks = np.stack(mask_rows)
        _atomic_bytes(
            staging / "train_tokens.bin", tokens.astype(np.uint16, copy=False).tobytes()
        )
        _atomic_bytes(
            staging / "train_loss_mask.bin", masks.astype(np.uint8, copy=False).tobytes()
        )
        _write_jsonl(staging / "records.jsonl", rows)
        manifest = {
            "schema_version": VIEW_FORMAT,
            "dataset_id": "verified_arithmetic_v2_operation_view_v1",
            "seed": seed,
            "record_width": RECORD_WIDTH,
            "context_length": 256,
            "training_authorized": False,
            "source": {
                "train_records_path": train_records_path.as_posix(),
                "train_records_sha256": sha256_file(train_records_path),
                "canonical_dataset_mutated": False,
                "split": "train",
            },
            "tokenizer": {
                "path": tokenizer_path.as_posix(),
                "sha256": sha256_file(tokenizer_path),
            },
            "source_operation_index": {
                **index,
                "index_path": "source_operation_index.jsonl",
            },
            "artifacts": {},
            "operation_counts": dict(
                sorted(Counter(row["operation"] for row in rows).items())
            ),
            "difficulty_counts": dict(
                sorted(Counter(row["difficulty_tier"] for row in rows).items())
            ),
            "logical_example_count": len(examples),
            "packed_record_count": len(rows),
            "mask_alignment": "stored_mask[1:] targets; EOS boundaries and PAD transitions masked",
        }
        for name in (
            "train_tokens.bin",
            "train_loss_mask.bin",
            "records.jsonl",
            "source_operation_index.jsonl",
        ):
            path = staging / name
            manifest["artifacts"][name] = {
                "path": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        _write_json(staging / "manifest.json", manifest)
        validate_operation_view(staging)
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                raise FileExistsError(f"operation-aware view backup exists: {backup}")
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, output_dir)
    finally:
        if temporary_index.exists():
            temporary_index.unlink()
        if staging.exists():
            shutil.rmtree(staging)
    return validate_operation_view(output_dir)


def validate_operation_view(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != VIEW_FORMAT
        or manifest.get("training_authorized") is not False
    ):
        raise ValueError("invalid operation-aware view manifest")
    if (
        sha256_file(Path(manifest["source"]["train_records_path"]))
        != manifest["source"]["train_records_sha256"]
    ):
        raise ValueError("operation-aware view source hash mismatch")
    for metadata in manifest["artifacts"].values():
        path = directory / metadata["path"]
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError("operation-aware view artifact hash mismatch")
    tokens = np.fromfile(directory / "train_tokens.bin", dtype=np.uint16).reshape(
        -1, RECORD_WIDTH
    )
    masks = np.fromfile(directory / "train_loss_mask.bin", dtype=np.uint8).reshape(
        -1, RECORD_WIDTH
    )
    if (
        len(tokens) != manifest["packed_record_count"]
        or np.any(masks[:, 0])
        or not np.isin(masks, (0, 1)).all()
    ):
        raise ValueError("operation-aware view shape or mask invalid")
    if np.any(masks[:, 1:][tokens[:, 1:] == 0]):
        raise ValueError("PAD target is supervised")
    authoritative_examples: dict[str, dict[str, Any]] = {}
    for line in Path(manifest["source"]["train_records_path"]).read_text(
        encoding="utf-8"
    ).splitlines():
        for example in json.loads(line)["logical_examples"]:
            source_id = str(example["id"])
            if source_id in authoritative_examples:
                raise ValueError("operation-aware view source has duplicate example IDs")
            authoritative_examples[source_id] = example
    rows = [
        json.loads(line)
        for line in (directory / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(rows) != manifest["packed_record_count"]:
        raise ValueError("operation-aware view record count mismatch")
    for index, item in enumerate(rows):
        if item.get("record_index") != index:
            raise ValueError("operation-aware view record indices are not contiguous")
        operation = item.get("operation")
        examples = item.get("source_examples", [])
        if (
            operation not in SUPPORTED_OPERATIONS
            or not examples
            or any(example.get("split") != "train" for example in examples)
            or any(example.get("operation") != operation for example in examples)
        ):
            raise ValueError("operation-aware view has invalid provenance")
        if item.get("source_example_ids") != [str(example["id"]) for example in examples]:
            raise ValueError("operation-aware view source ID mapping mismatch")
        for example in examples:
            authoritative = authoritative_examples.get(str(example["id"]))
            if authoritative != example:
                raise ValueError("operation-aware view source mapping is not authoritative")
    return manifest


def _largest_remainder(
    total: int,
    weights: Mapping[str, float],
    *,
    require_supported_operations: bool = True,
) -> dict[str, int]:
    if (
        total < 1
        or not weights
        or any(not math.isfinite(value) or value < 0 for value in weights.values())
    ):
        raise ValueError("schedule weights must be finite, non-negative, and non-empty")
    if (
        require_supported_operations and set(weights) - SUPPORTED_OPERATIONS
    ) or not math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError(
            "schedule weights must cover supported operations and sum to one"
        )
    raw = {key: total * value for key, value in weights.items()}
    allocated = {key: math.floor(value) for key, value in raw.items()}
    for key in sorted(weights, key=lambda key: (-(raw[key] - allocated[key]), key))[
        : total - sum(allocated.values())
    ]:
        allocated[key] += 1
    return allocated


def build_operation_schedule(
    *,
    view_dir: Path,
    output_dir: Path,
    total_records: int,
    operation_weights: Mapping[str, float],
    seed: int,
    with_replacement: bool,
    stages: Sequence[Mapping[str, Any]] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Sample homogeneous view records with deterministic largest-remainder counts."""
    validate_operation_view(view_dir)
    rows = [
        json.loads(line)
        for line in (view_dir / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    pools: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        pools[row["operation"]].append(int(row["record_index"]))
    counts = _largest_remainder(total_records, operation_weights)
    selected: list[tuple[int, int]] = []
    used_by_operation: dict[str, set[int]] = defaultdict(set)
    stage_rows = list(stages) or [
        {"id": "uniform", "weight": 1.0, "operations": sorted(operation_weights)}
    ]
    if not math.isclose(
        math.fsum(float(item["weight"]) for item in stage_rows), 1.0, abs_tol=1e-12
    ):
        raise ValueError("stage weights must sum to one")
    remaining = dict(counts)
    stage_counts = _largest_remainder(
        total_records,
        {str(index): float(item["weight"]) for index, item in enumerate(stage_rows)},
        require_supported_operations=False,
    )
    for stage_index, stage in enumerate(stage_rows):
        allowed = [item for item in stage["operations"] if item in remaining]
        local_weights = {item: operation_weights[item] for item in allowed}
        normalizer = math.fsum(local_weights.values())
        if normalizer <= 0:
            raise ValueError("stage must allow at least one requested operation")
        local = _largest_remainder(
            stage_counts[str(stage_index)],
            {item: value / normalizer for item, value in local_weights.items()},
        )
        for operation, amount in local.items():
            amount = min(amount, remaining[operation])
            remaining[operation] -= amount
            pool = list(pools[operation])
            random.Random(f"{seed}:{stage_index}:{operation}").shuffle(pool)
            available = pool if with_replacement else [item for item in pool if item not in used_by_operation[operation]]
            if not with_replacement and amount > len(available):
                raise ValueError(
                    "no-replacement operation request exceeds available records"
                )
            selected.extend(
                (available[index % len(available)], stage_index) for index in range(amount)
            )
            if not with_replacement:
                used_by_operation[operation].update(available[:amount])
    # Remaining allocations are permitted only when the final stage declares the
    # operation.  Silently placing them in a disallowed stage would invalidate a
    # curriculum's scientific identity.
    final_allowed = set(stage_rows[-1]["operations"])
    for operation, amount in remaining.items():
        if amount:
            if operation not in final_allowed:
                raise ValueError(
                    "stage constraints cannot satisfy requested operation allocation"
                )
            pool = list(pools[operation])
            random.Random(f"{seed}:remainder:{operation}").shuffle(pool)
            available = pool if with_replacement else [item for item in pool if item not in used_by_operation[operation]]
            if not with_replacement and amount > len(available):
                raise ValueError(
                    "no-replacement operation request exceeds available records"
                )
            selected.extend(
                (available[index % len(available)], len(stage_rows) - 1)
                for index in range(amount)
            )
            if not with_replacement:
                used_by_operation[operation].update(available[:amount])
    if len(selected) != total_records:
        raise RuntimeError("operation schedule accounting did not close")
    schedule = np.asarray(selected, dtype=SCHEDULE_DTYPE)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite operation schedule: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        _atomic_bytes(staging / "schedule.bin", schedule.tobytes())
        operation_by_record = {
            int(row["record_index"]): row["operation"] for row in rows
        }
        observed = Counter(
            operation_by_record[int(item["record"])] for item in schedule
        )
        duplicates = total_records - len(set(int(item["record"]) for item in schedule))
        manifest = {
            "schema_version": SCHEDULE_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_authorized": False,
            "view_manifest_sha256": sha256_file(view_dir / "manifest.json"),
            "view_manifest_path": (view_dir / "manifest.json").as_posix(),
            "seed": seed,
            "with_replacement": with_replacement,
            "total_records": total_records,
            "total_tokens": total_records * 256,
            "operation_weights": dict(operation_weights),
            "operation_counts": dict(sorted(observed.items())),
            "duplicate_records": duplicates,
            "stages": [dict(item) for item in stage_rows],
            "stage_counts": dict(
                sorted(Counter(int(item["stage"]) for item in schedule).items())
            ),
            "schedule": {
                "path": "schedule.bin",
                "sha256": sha256_file(staging / "schedule.bin"),
                "dtype": [list(item) for item in SCHEDULE_DTYPE.descr],
                "entry_bytes": SCHEDULE_DTYPE.itemsize,
            },
        }
        _write_json(staging / "manifest.json", manifest)
        validate_operation_schedule(staging)
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                raise FileExistsError(f"operation schedule backup exists: {backup}")
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
            except BaseException:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validate_operation_schedule(output_dir)


def validate_operation_schedule(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    schedule_path = directory / manifest["schedule"]["path"]
    if (
        manifest.get("schema_version") != SCHEDULE_FORMAT
        or manifest.get("training_authorized") is not False
    ):
        raise ValueError("invalid operation schedule manifest")
    view_path = Path(manifest["view_manifest_path"])
    if (
        not view_path.is_file()
        or sha256_file(view_path) != manifest["view_manifest_sha256"]
    ):
        raise ValueError("operation schedule source-view hash mismatch")
    if (
        sha256_file(schedule_path) != manifest["schedule"]["sha256"]
        or schedule_path.stat().st_size
        != int(manifest["total_records"]) * SCHEDULE_DTYPE.itemsize
    ):
        raise ValueError("operation schedule identity mismatch")
    schedule = np.fromfile(schedule_path, dtype=SCHEDULE_DTYPE)
    if (
        len(schedule) != int(manifest["total_records"])
        or int(manifest["total_tokens"]) != len(schedule) * 256
    ):
        raise ValueError("operation schedule accounting invalid")
    return manifest
