"""Immutable, input-only paired release construction for Candidate E.

The constructor intentionally has no arithmetic generator or default source
paths. Callers must supply independently reviewed logical splits. Building a
release never authorizes a schedule or training.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from vasu.data.arithmetic_step_supervision import (
    SupervisedArithmeticExample,
    compile_supervised_example,
    pack_supervised_examples,
    validate_matched_logical_splits,
)
from vasu.data.arithmetic_v2 import recompute_answer
from vasu.tokenizer.tokenizer import VASUTokenizer


SCHEMA_VERSION = "vasu_candidate_e_paired_release_v1"
RECORD_WIDTH = 257
TOKEN_DTYPE = np.dtype(np.uint16)
MASK_DTYPE = np.dtype(np.uint8)
VARIANTS = ("final_answer", "verified_steps")


def validate_matched_budget(
    *,
    control_scheduled_records: int,
    treatment_scheduled_records: int,
    sequence_length: int,
    microbatch_size: int,
    gradient_accumulation: int,
    optimizer_updates: int,
) -> dict[str, int]:
    """Validate Candidate E's equal processed-token/update contract."""

    values = (
        control_scheduled_records,
        treatment_scheduled_records,
        sequence_length,
        microbatch_size,
        gradient_accumulation,
        optimizer_updates,
    )
    if any(not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("matched-budget quantities must be positive integers")
    if control_scheduled_records != treatment_scheduled_records:
        raise ValueError("control and treatment scheduled record counts differ")
    expected_records = microbatch_size * gradient_accumulation * optimizer_updates
    if control_scheduled_records != expected_records:
        raise ValueError("scheduled records do not match batch/update accounting")
    return {
        "scheduled_records_per_arm": control_scheduled_records,
        "processed_tokens_per_arm": control_scheduled_records * sequence_length,
        "optimizer_updates": optimizer_updates,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(_canonical_json(record) + "\n" for record in records), encoding="utf-8"
    )


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _special_id(tokenizer: VASUTokenizer, token: str) -> int:
    value = tokenizer.tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"authoritative tokenizer has no {token} token")
    return int(value)


def _provenance(
    records: Sequence[Mapping[str, Any]],
    examples: Sequence[SupervisedArithmeticExample],
) -> list[dict[str, Any]]:
    by_id = {str(record["id"]): dict(record) for record in records}
    rows: list[dict[str, Any]] = []
    offset = 0
    current: list[dict[str, Any]] = []
    current_ids: list[str] = []
    for example in examples:
        if offset and offset + len(example.token_ids) > RECORD_WIDTH:
            rows.append(
                {
                    "record_index": len(rows),
                    "used_token_count": offset,
                    "examples": current,
                    "example_ids": current_ids,
                }
            )
            offset, current, current_ids = 0, [], []
        end = offset + len(example.token_ids)
        current.append(
            {
                "source_id": example.source_id,
                "start": offset,
                "end": end,
                "target_start": offset + example.target_start,
                "logical_example": by_id[example.source_id],
            }
        )
        current_ids.append(example.source_id)
        offset = end
        if offset == RECORD_WIDTH:
            rows.append(
                {
                    "record_index": len(rows),
                    "used_token_count": offset,
                    "examples": current,
                    "example_ids": current_ids,
                }
            )
            offset, current, current_ids = 0, [], []
    if current:
        rows.append(
            {
                "record_index": len(rows),
                "used_token_count": offset,
                "examples": current,
                "example_ids": current_ids,
            }
        )
    return rows


def _build_arm(
    *,
    staging: Path,
    variant: str,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer_path: Path,
    created_at: str,
) -> dict[str, Any]:
    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_path))
    pad, eos = _special_id(tokenizer, "[PAD]"), _special_id(tokenizer, "[EOS]")
    if pad == eos:
        raise ValueError("PAD and EOS must differ")
    compiled = {
        split: [
            compile_supervised_example(
                record, tokenizer=tokenizer, eos_token_id=eos, variant=variant
            )
            for record in records
        ]
        for split, records in splits.items()
    }
    packed = pack_supervised_examples(compiled["train"], pad_token_id=pad)
    arm = staging / variant
    arm.mkdir()
    np.stack([row.tokens for row in packed]).astype(TOKEN_DTYPE).tofile(
        arm / "train_tokens.bin"
    )
    np.stack([row.stored_mask for row in packed]).astype(MASK_DTYPE).tofile(
        arm / "train_loss_mask.bin"
    )
    provenance = _provenance(splits["train"], compiled["train"])
    _write_jsonl(arm / "train_records.jsonl", provenance)
    _write_jsonl(arm / "dev.jsonl", [dict(item) for item in splits["development"]])
    _write_jsonl(arm / "eval.jsonl", [dict(item) for item in splits["evaluation"]])
    artifacts = {
        name: _artifact(arm / name)
        for name in (
            "train_tokens.bin",
            "train_loss_mask.bin",
            "train_records.jsonl",
            "dev.jsonl",
            "eval.jsonl",
        )
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "created_at": created_at,
        "training_authorized": False,
        "record_width": RECORD_WIDTH,
        "tokenizer": {
            "path": tokenizer_path.as_posix(),
            "sha256": sha256_file(tokenizer_path),
            "special_token_ids": {"pad": pad, "eos": eos},
        },
        "prompt_prefix": "Question: {prompt}\\nResponse:\\n",
        "target_serialization": "Answer: {answer}"
        if variant == "final_answer"
        else "verified symbolic states + Answer: {answer}",
        "logical_split_sha256": {
            split: _sha256_json(
                sorted((dict(x) for x in records), key=lambda x: x["id"])
            )
            for split, records in splits.items()
        },
        "logical_example_counts": {
            split: len(records) for split, records in splits.items()
        },
        "packing": {
            "algorithm": "complete_examples_with_explicit_pad",
            "packed_record_count": len(packed),
            "supervised_token_count": int(
                sum(int(row.stored_mask.sum()) for row in packed)
            ),
        },
        "artifacts": artifacts,
        "validation": {
            "paired_source_validation": True,
            "prompt_unsupervised": True,
            "response_and_eos_supervised": True,
            "pad_and_cross_example_unsupervised": True,
        },
    }
    _write_json(arm / "manifest.json", manifest)
    validate_arm(arm, tokenizer_path=tokenizer_path)
    return manifest


def build_paired_release(
    *,
    logical_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer_path: Path,
    output_dir: Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build both arms atomically from reviewed records; no training side effects."""
    validate_matched_logical_splits(logical_splits, logical_splits)
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        for variant in VARIANTS:
            _build_arm(
                staging=staging,
                variant=variant,
                splits=logical_splits,
                tokenizer_path=tokenizer_path,
                created_at=timestamp,
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": timestamp,
            "training_authorized": False,
            "source_split_counts": {
                split: len(records) for split, records in logical_splits.items()
            },
            "arms": {
                variant: {
                    "manifest": f"{variant}/manifest.json",
                    "sha256": sha256_file(staging / variant / "manifest.json"),
                }
                for variant in VARIANTS
            },
            "matched_source_sha256": _sha256_json(
                {
                    split: sorted(str(x["id"]) for x in records)
                    for split, records in logical_splits.items()
                }
            ),
        }
        _write_json(staging / "manifest.json", manifest)
        validate_paired_release(staging, tokenizer_path=tokenizer_path)
        os.replace(staging, output_dir)
        return validate_paired_release(output_dir, tokenizer_path=tokenizer_path)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_arm(directory: Path, *, tokenizer_path: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("training_authorized") is not False
    ):
        raise ValueError("invalid or authorizing Candidate E arm manifest")
    if sha256_file(tokenizer_path) != manifest["tokenizer"]["sha256"]:
        raise ValueError("tokenizer hash mismatch")
    for metadata in manifest["artifacts"].values():
        path = directory / metadata["path"]
        if (
            not path.is_file()
            or path.stat().st_size != metadata["bytes"]
            or sha256_file(path) != metadata["sha256"]
        ):
            raise ValueError(f"artifact integrity failure: {metadata['path']}")
    tokens = np.fromfile(directory / "train_tokens.bin", dtype=TOKEN_DTYPE).reshape(
        -1, RECORD_WIDTH
    )
    masks = np.fromfile(directory / "train_loss_mask.bin", dtype=MASK_DTYPE).reshape(
        -1, RECORD_WIDTH
    )
    pad = int(manifest["tokenizer"]["special_token_ids"]["pad"])
    if (
        not np.isin(masks, (0, 1)).all()
        or np.any(masks[:, 0])
        or np.any(masks[tokens == pad])
    ):
        raise ValueError("invalid Candidate E mask")
    provenance = [
        json.loads(line)
        for line in (directory / "train_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(provenance) != len(tokens):
        raise ValueError("provenance count mismatch")
    for index, row in enumerate(provenance):
        for item in row["examples"]:
            start, end, target = (
                int(item["start"]),
                int(item["end"]),
                int(item["target_start"]),
            )
            if not start < target < end <= int(row["used_token_count"]):
                raise ValueError("invalid target boundary")
            if np.any(masks[index, start:target]) or not np.all(
                masks[index, target:end]
            ):
                raise ValueError("prompt/response mask mismatch")
            if (
                recompute_answer(item["logical_example"])
                != item["logical_example"]["answer"]
            ):
                raise ValueError("unverified arithmetic answer")
    return manifest


def validate_paired_release(directory: Path, *, tokenizer_path: Path) -> dict[str, Any]:
    root = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        root.get("schema_version") != SCHEMA_VERSION
        or root.get("training_authorized") is not False
    ):
        raise ValueError("invalid or authorizing Candidate E release")
    for variant in VARIANTS:
        arm = directory / variant
        if sha256_file(arm / "manifest.json") != root["arms"][variant]["sha256"]:
            raise ValueError("arm manifest hash mismatch")
        validate_arm(arm, tokenizer_path=tokenizer_path)
    return root
