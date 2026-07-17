"""Atomic preparation progress and resume reconciliation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .reporting import atomic_write_json
from .schemas import PreparationProgress


def save_progress(path: Path, progress: PreparationProgress) -> None:
    atomic_write_json(path, asdict(progress))


def load_progress(path: Path) -> PreparationProgress:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Preparation progress does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Preparation progress is invalid JSON: {path}") from error
    return PreparationProgress(**payload)


def validate_resume_identity(
    progress: PreparationProgress,
    *,
    configuration_hash: str,
    source_id: str,
    pinned_revision: str,
    shard_identifier: str,
) -> None:
    expected = {
        "configuration_hash": configuration_hash,
        "source_id": source_id,
        "pinned_revision": pinned_revision,
        "shard_identifier": shard_identifier,
    }
    mismatches = [
        f"{field}: progress={getattr(progress, field)!r}, expected={value!r}"
        for field, value in expected.items()
        if getattr(progress, field) != value
    ]
    if mismatches:
        raise ValueError("Resume configuration mismatch: " + "; ".join(mismatches))


def reconcile_output(path: Path, committed_size: int) -> None:
    actual_size = path.stat().st_size if path.exists() else 0
    if actual_size < committed_size:
        raise ValueError(
            f"Output JSONL is shorter than committed progress: {actual_size} < "
            f"{committed_size} bytes"
        )
    if actual_size > committed_size:
        with path.open("r+b") as handle:
            handle.truncate(committed_size)

