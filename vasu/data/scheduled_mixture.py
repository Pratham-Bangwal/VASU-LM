"""Additive deterministic N-source fixed-record schedule support.

The legacy physical FineWeb/Wikimedia mixture remains implemented separately
in :mod:`vasu.data.pretraining_mixture`.
"""

from __future__ import annotations

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
import torch
from torch.utils.data import Dataset


SCHEDULE_FORMAT = "vasu_scheduled_mixture_v1"
SCHEDULE_DTYPE = np.dtype([("source", "<u4"), ("record", "<u8")])
SOURCE_KINDS = frozenset({"standard_token_stream", "packed_masked"})
TRAIN_SPLIT_ROLE = "train"
TOKEN_DTYPE = "uint16"
MASK_DTYPE = "uint8"


def sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ScheduledSource:
    """A hash-bound source exposed as deterministic fixed logical records."""

    identifier: str
    source_kind: str
    token_path: Path
    mask_path: Path | None
    manifest_path: Path
    token_sha256: str
    mask_sha256: str | None
    manifest_sha256: str
    split_role: str
    record_width: int
    context_length: int
    token_dtype: str
    mask_dtype: str | None
    available_records: int
    weight: float
    order: int
    replay_policy: str
    masked: bool
    token_offset: int = 0
    record_stride: int = 256

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_path"] = self.token_path.as_posix()
        payload["mask_path"] = (
            None if self.mask_path is None else self.mask_path.as_posix()
        )
        payload["manifest_path"] = self.manifest_path.as_posix()
        return payload

    @classmethod
    def from_manifest_dict(cls, payload: Mapping[str, Any]) -> ScheduledSource:
        values = dict(payload)
        values["token_path"] = Path(values["token_path"])
        values["mask_path"] = (
            None
            if values.get("mask_path") is None
            else Path(values["mask_path"])
        )
        values["manifest_path"] = Path(values["manifest_path"])
        return cls(**values)


@dataclass(frozen=True)
class SourceAllocation:
    source_id: str
    source_index: int
    requested_weight: float
    allocated_records: int
    allocated_tokens: int
    actual_weight: float
    available_unique_records: int
    unique_records_consumed: int
    replay_epochs: int
    reused_records: int
    effective_source_passes: float


def _validate_hash(value: str, field: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be hexadecimal SHA-256") from error


def validate_source(source: ScheduledSource, *, check_hashes: bool = True) -> None:
    """Validate source metadata, physical capacity, and optional masks."""

    if not source.identifier:
        raise ValueError("scheduled source ID must not be empty")
    if source.source_kind not in SOURCE_KINDS:
        raise ValueError(f"unsupported scheduled source kind: {source.source_kind}")
    if source.split_role != TRAIN_SPLIT_ROLE:
        raise ValueError(
            f"scheduled source {source.identifier!r} is not a training split"
        )
    if source.record_width != source.context_length + 1:
        raise ValueError("record width must equal context length plus one")
    if source.context_length < 1 or source.record_stride < 1:
        raise ValueError("context length and record stride must be positive")
    if source.token_offset < 0:
        raise ValueError("source token offset must be non-negative")
    if source.token_dtype != TOKEN_DTYPE:
        raise ValueError("scheduled token dtype must be uint16")
    if source.available_records < 1:
        raise ValueError("scheduled source must expose at least one record")
    if not math.isfinite(source.weight) or source.weight <= 0:
        raise ValueError("scheduled source weight must be finite and positive")
    if (
        not isinstance(source.order, int)
        or isinstance(source.order, bool)
        or source.order < 0
    ):
        raise ValueError("source order must be a non-negative integer")
    if source.replay_policy != "deterministic_permutation_epochs":
        raise ValueError("unsupported source replay policy")
    if source.masked != (source.source_kind == "packed_masked"):
        raise ValueError("masked flag does not match source kind")
    if source.masked:
        if source.mask_path is None or source.mask_dtype != MASK_DTYPE:
            raise ValueError("packed masked source requires a uint8 mask")
        if source.mask_sha256 is None:
            raise ValueError("packed masked source requires a mask SHA-256")
        if source.record_stride != source.record_width:
            raise ValueError("packed masked records require record-width stride")
    elif (
        source.mask_path is not None
        or source.mask_sha256 is not None
        or source.mask_dtype is not None
    ):
        raise ValueError("standard source must not declare a mask")

    for path, label in (
        (source.token_path, "token"),
        (source.manifest_path, "source manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{source.identifier} {label} file: {path}")
    if source.mask_path is not None and not source.mask_path.is_file():
        raise FileNotFoundError(f"{source.identifier} mask file: {source.mask_path}")
    _validate_hash(source.token_sha256, "token_sha256")
    _validate_hash(source.manifest_sha256, "manifest_sha256")
    if source.mask_sha256 is not None:
        _validate_hash(source.mask_sha256, "mask_sha256")
    if check_hashes:
        if sha256_file(source.token_path) != source.token_sha256:
            raise ValueError(f"source token hash mismatch: {source.identifier}")
        if sha256_file(source.manifest_path) != source.manifest_sha256:
            raise ValueError(f"source manifest hash mismatch: {source.identifier}")
        if (
            source.mask_path is not None
            and sha256_file(source.mask_path) != source.mask_sha256
        ):
            raise ValueError(f"source mask hash mismatch: {source.identifier}")

    token_count = source.token_path.stat().st_size // np.dtype(np.uint16).itemsize
    if source.token_path.stat().st_size % np.dtype(np.uint16).itemsize:
        raise ValueError(f"source token file is not uint16 aligned: {source.identifier}")
    required_tokens = (
        source.token_offset
        + (source.available_records - 1) * source.record_stride
        + source.record_width
    )
    if required_tokens > token_count:
        raise ValueError(
            f"available record count exceeds token file: {source.identifier}"
        )
    if source.mask_path is not None:
        mask_count = source.mask_path.stat().st_size
        required_mask = (
            (source.available_records - 1) * source.record_stride
            + source.record_width
        )
        if required_mask > mask_count:
            raise ValueError(
                f"available record count exceeds mask file: {source.identifier}"
            )
    source_manifest = json.loads(source.manifest_path.read_text(encoding="utf-8"))
    training_paths: set[Path] = set()
    for shard in source_manifest.get("training_shards", []):
        if "train" in str(shard.get("role", "")).lower():
            training_paths.add(Path(shard["path"]).resolve())
    train_section = source_manifest.get("train")
    if isinstance(train_section, dict) and train_section.get("path"):
        training_paths.add(Path(train_section["path"]).resolve())
    artifacts = source_manifest.get("artifacts")
    if isinstance(artifacts, dict) and "train_tokens.bin" in artifacts:
        training_paths.add(
            (
                source.manifest_path.parent
                / artifacts["train_tokens.bin"]["path"]
            ).resolve()
        )
    if source.token_path.resolve() not in training_paths:
        raise ValueError(
            f"source manifest does not declare a training artifact: "
            f"{source.identifier}"
        )


def validate_sources(
    sources: Sequence[ScheduledSource], *, check_hashes: bool = True
) -> None:
    if not sources:
        raise ValueError("at least one scheduled source is required")
    identifiers = [source.identifier for source in sources]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate scheduled source IDs")
    orders = [source.order for source in sources]
    if len(set(orders)) != len(orders):
        raise ValueError("scheduled source orders must be unique")
    for source in sources:
        validate_source(source, check_hashes=check_hashes)
    total = math.fsum(source.weight for source in sources)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("scheduled source weights must sum to one")


def allocate_records(
    total_records: int, sources: Sequence[ScheduledSource]
) -> list[int]:
    """Allocate records by deterministic largest remainder."""

    if total_records < 1:
        raise ValueError("total_records must be positive")
    validate_sources(sources, check_hashes=False)
    raw = [total_records * source.weight for source in sources]
    allocated = [math.floor(value) for value in raw]
    remainder = total_records - sum(allocated)
    ranked = sorted(
        range(len(sources)),
        key=lambda index: (
            -(raw[index] - allocated[index]),
            sources[index].order,
        ),
    )
    for index in ranked[:remainder]:
        allocated[index] += 1
    if sum(allocated) != total_records:
        raise RuntimeError("largest-remainder allocation did not close")
    if total_records >= len(sources) and any(value == 0 for value in allocated):
        raise ValueError("positive source weight received zero records")
    return allocated


def _source_epoch_order(
    source: ScheduledSource, *, seed: int, replay_epoch: int
) -> list[int]:
    order = list(range(source.available_records))
    random.Random(
        f"{seed}:{source.identifier}:{replay_epoch}"
    ).shuffle(order)
    return order


def build_schedule(
    sources: Sequence[ScheduledSource],
    total_records: int,
    seed: int,
) -> tuple[np.ndarray, list[SourceAllocation]]:
    """Build a deterministic interleaved source/local-record schedule."""

    allocations = allocate_records(total_records, sources)
    entries: list[tuple[int, int]] = []
    accounting: list[SourceAllocation] = []
    for source_index, (source, count) in enumerate(
        zip(sources, allocations, strict=True)
    ):
        selected: list[int] = []
        replay_epoch = 0
        while len(selected) < count:
            order = _source_epoch_order(
                source,
                seed=seed,
                replay_epoch=replay_epoch,
            )
            take = min(len(order), count - len(selected))
            selected.extend(order[:take])
            replay_epoch += 1
        entries.extend((source_index, local) for local in selected)
        accounting.append(
            SourceAllocation(
                source_id=source.identifier,
                source_index=source_index,
                requested_weight=source.weight,
                allocated_records=count,
                allocated_tokens=count * source.context_length,
                actual_weight=count / total_records,
                available_unique_records=source.available_records,
                unique_records_consumed=min(count, source.available_records),
                replay_epochs=replay_epoch,
                reused_records=max(count - source.available_records, 0),
                effective_source_passes=count / source.available_records,
            )
        )
    random.Random(f"{seed}:global_schedule").shuffle(entries)
    schedule = np.asarray(entries, dtype=SCHEDULE_DTYPE)
    if len(schedule) != total_records:
        raise RuntimeError("schedule length does not match requested records")
    if schedule.dtype != SCHEDULE_DTYPE:
        raise RuntimeError("schedule dtype changed unexpectedly")
    return schedule, accounting


def _write_schedule(path: Path, schedule: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.ascontiguousarray(schedule, dtype=SCHEDULE_DTYPE).tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("wb") as handle:
        handle.write(content.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def build_resolved_manifest(
    *,
    candidate_id: str,
    requested_plan_path: Path,
    sources: Sequence[ScheduledSource],
    schedule: np.ndarray,
    accounting: Sequence[SourceAllocation],
    output_dir: Path,
    seed: int,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    validation_sources: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEDULE_FORMAT,
        "candidate_id": candidate_id,
        "requested_plan": {
            "path": requested_plan_path.as_posix(),
            "sha256": sha256_file(requested_plan_path),
        },
        "seed": seed,
        "total_records": len(schedule),
        "total_tokens": len(schedule) * sources[0].context_length,
        "context_length": sources[0].context_length,
        "record_width": sources[0].record_width,
        "schedule": {
            "path": "schedule.bin",
            "sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "dtype": SCHEDULE_DTYPE.descr,
            "entry_bytes": SCHEDULE_DTYPE.itemsize,
        },
        "source_order": [source.identifier for source in sources],
        "sources": [source.to_manifest_dict() for source in sources],
        "allocations": [asdict(item) for item in accounting],
        "tokenizer": {
            "path": tokenizer_path.as_posix(),
            "sha256": tokenizer_sha256,
        },
        "validation_sources": dict(validation_sources),
        "generator": {
            "path": "vasu/data/scheduled_mixture.py",
            "sha256": sha256_file(Path("vasu/data/scheduled_mixture.py")),
            "version": SCHEDULE_FORMAT,
        },
        "output_directory": output_dir.as_posix(),
        "created_at": created_at,
        "training_authorized": False,
    }


def validate_schedule_release(
    directory: Path,
    *,
    check_source_hashes: bool = True,
) -> dict[str, Any]:
    manifest_path = directory / "resolved_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEDULE_FORMAT:
        raise ValueError("unsupported scheduled-mixture schema")
    if manifest.get("training_authorized") is not False:
        raise ValueError("scheduled mixture must not authorize training")
    sources = [
        ScheduledSource.from_manifest_dict(item) for item in manifest["sources"]
    ]
    validate_sources(sources, check_hashes=check_source_hashes)
    schedule_path = directory / manifest["schedule"]["path"]
    if not schedule_path.is_file():
        raise FileNotFoundError(schedule_path)
    if schedule_path.stat().st_size != (
        int(manifest["total_records"]) * SCHEDULE_DTYPE.itemsize
    ):
        raise ValueError("schedule byte size does not match manifest")
    if sha256_file(schedule_path) != manifest["schedule"]["sha256"]:
        raise ValueError("schedule hash mismatch")
    if manifest["schedule"].get("dtype") != [
        list(field) for field in SCHEDULE_DTYPE.descr
    ]:
        raise ValueError("schedule dtype declaration is invalid")
    if manifest["schedule"].get("entry_bytes") != SCHEDULE_DTYPE.itemsize:
        raise ValueError("schedule entry size declaration is invalid")
    schedule = np.memmap(schedule_path, dtype=SCHEDULE_DTYPE, mode="r")
    if np.any(schedule["source"] >= len(sources)):
        raise ValueError("schedule contains an invalid source index")
    counts = np.bincount(
        np.asarray(schedule["source"], dtype=np.int64),
        minlength=len(sources),
    )
    for index, source in enumerate(sources):
        local = schedule["record"][schedule["source"] == index]
        if len(local) and int(local.max()) >= source.available_records:
            raise ValueError(f"schedule local index exceeds {source.identifier}")
        expected = int(manifest["allocations"][index]["allocated_records"])
        if int(counts[index]) != expected:
            raise ValueError(f"schedule source count mismatch: {source.identifier}")
    if sum(int(value) for value in counts) != int(manifest["total_records"]):
        raise ValueError("schedule source counts do not close")
    if int(manifest["total_tokens"]) != (
        int(manifest["total_records"]) * int(manifest["context_length"])
    ):
        raise ValueError("scheduled token accounting is inconsistent")
    return manifest


def write_schedule_release(
    *,
    output_dir: Path,
    candidate_id: str,
    requested_plan_path: Path,
    sources: Sequence[ScheduledSource],
    total_records: int,
    seed: int,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    validation_sources: Mapping[str, Any],
    overwrite: bool = False,
    dry_run: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Transactionally build and validate a schedule release."""

    validate_sources(sources)
    if sha256_file(tokenizer_path) != tokenizer_sha256:
        raise ValueError("scheduled-mixture tokenizer hash mismatch")
    if output_dir.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{output_dir} exists; pass overwrite explicitly")
    schedule, accounting = build_schedule(sources, total_records, seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    backup: Path | None = None
    try:
        _write_schedule(staging / "schedule.bin", schedule)
        manifest = build_resolved_manifest(
            candidate_id=candidate_id,
            requested_plan_path=requested_plan_path,
            sources=sources,
            schedule=schedule,
            accounting=accounting,
            output_dir=output_dir,
            seed=seed,
            tokenizer_path=tokenizer_path,
            tokenizer_sha256=tokenizer_sha256,
            validation_sources=validation_sources,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
        )
        _write_json(staging / "resolved_manifest.json", manifest)
        validate_schedule_release(staging)
        if dry_run:
            return manifest
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                raise FileExistsError(f"schedule backup path exists: {backup}")
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if backup is not None and backup.exists():
                os.replace(backup, output_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return validate_schedule_release(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class ScheduledPretrainingDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Map a compact schedule to standard or packed-masked source records."""

    def __init__(
        self,
        sources: Sequence[ScheduledSource],
        schedule_path: Path,
        *,
        schedule_sha256: str,
        sequence_length: int = 256,
        validate_hashes: bool = True,
    ) -> None:
        self.sources = tuple(sources)
        self.schedule_path = Path(schedule_path)
        self.schedule_sha256 = schedule_sha256
        self.sequence_length = sequence_length
        validate_sources(self.sources, check_hashes=validate_hashes)
        if any(source.context_length != sequence_length for source in self.sources):
            raise ValueError("source context length differs from dataset")
        if sha256_file(self.schedule_path) != schedule_sha256:
            raise ValueError("scheduled dataset schedule hash mismatch")
        if self.schedule_path.stat().st_size % SCHEDULE_DTYPE.itemsize:
            raise ValueError("schedule file is not entry-aligned")
        self._schedule: np.memmap | None = None
        self._tokens: list[np.memmap] | None = None
        self._masks: list[np.memmap | None] | None = None
        self._pid: int | None = None

    @classmethod
    def from_resolved_manifest(
        cls,
        manifest_path: Path,
        *,
        validate_hashes: bool = True,
    ) -> ScheduledPretrainingDataset:
        manifest = validate_schedule_release(
            manifest_path.parent,
            check_source_hashes=validate_hashes,
        )
        sources = [
            ScheduledSource.from_manifest_dict(item)
            for item in manifest["sources"]
        ]
        return cls(
            sources,
            manifest_path.parent / manifest["schedule"]["path"],
            schedule_sha256=manifest["schedule"]["sha256"],
            sequence_length=int(manifest["context_length"]),
            validate_hashes=False,
        )

    def _open(self) -> None:
        pid = os.getpid()
        if self._schedule is not None and self._pid == pid:
            return
        self._schedule = np.memmap(
            self.schedule_path, dtype=SCHEDULE_DTYPE, mode="r"
        )
        self._tokens = [
            np.memmap(source.token_path, dtype=np.uint16, mode="r")
            for source in self.sources
        ]
        self._masks = [
            (
                None
                if source.mask_path is None
                else np.memmap(source.mask_path, dtype=np.uint8, mode="r")
            )
            for source in self.sources
        ]
        self._pid = pid

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_schedule"] = None
        state["_tokens"] = None
        state["_masks"] = None
        state["_pid"] = None
        return state

    def __len__(self) -> int:
        return self.schedule_path.stat().st_size // SCHEDULE_DTYPE.itemsize

    def _entry(self, index: int) -> tuple[int, int]:
        if not isinstance(index, int) or isinstance(index, bool):
            raise IndexError("scheduled index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError("scheduled index is out of range")
        self._open()
        assert self._schedule is not None
        entry = self._schedule[index]
        return int(entry["source"]), int(entry["record"])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_index, local = self._entry(index)
        assert self._tokens is not None and self._masks is not None
        source = self.sources[source_index]
        start = source.token_offset + local * source.record_stride
        end = start + source.record_width
        row = np.asarray(self._tokens[source_index][start:end], dtype=np.int64)
        if len(row) != source.record_width:
            raise IndexError("schedule entry has an incomplete token record")
        mask_memmap = self._masks[source_index]
        if mask_memmap is None:
            mask = np.ones(self.sequence_length, dtype=np.float32)
        else:
            mask_start = local * source.record_stride
            stored = np.asarray(
                mask_memmap[mask_start : mask_start + source.record_width],
                dtype=np.float32,
            )
            if len(stored) != source.record_width:
                raise IndexError("schedule entry has an incomplete mask record")
            mask = stored[1:].copy()
        return (
            torch.from_numpy(row[:-1].copy()),
            torch.from_numpy(row[1:].copy()),
            torch.from_numpy(mask),
        )

    def source_identity(self, index: int) -> tuple[str, int]:
        source_index, local = self._entry(index)
        return self.sources[source_index].identifier, local

    def resume_identity(self) -> dict[str, Any]:
        return {
            "kind": SCHEDULE_FORMAT,
            "schedule_sha256": self.schedule_sha256,
            "source_token_sha256": [
                source.token_sha256 for source in self.sources
            ],
            "source_mask_sha256": [
                source.mask_sha256 for source in self.sources
            ],
        }
