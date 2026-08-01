"""Integrity boundary for temporary VASU-140M resume qualification checkpoints.

The module is deliberately qualification-only.  It can serialize and verify a
provided state payload, but it cannot construct a model, optimizer, dataset, or
training loop.  Persistent checkpoints remain owned by the general checkpoint
implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO

import torch

from vasu.training.vasu_140m_real_data_resume import (
    FAMILY_ID,
    STATE_COMPONENTS,
    canonical_json,
    sha256_bytes,
    state_sha256,
)


CHECKPOINT_SCHEMA_ID = "vasu_140m_real_data_resume_checkpoint_v1"
SIDECAR_SCHEMA_ID = "vasu_140m_real_data_resume_checkpoint_sidecar_v1"
CREATION_PHASES = frozenset({"partial_accumulation", "source_boundary"})
_PAYLOAD_FIELDS = frozenset(
    {
        "schema_id",
        "qualification_id",
        "specification_sha256",
        "repository_commit",
        "family_id",
        "creation_phase",
        "identities",
        "progress",
        "state",
        "state_sha256",
    }
)
_PROGRESS_FIELDS = frozenset(
    {
        "consumed_record_ids",
        "source_index",
        "record_index",
        "microbatch_count",
        "optimizer_update_count",
        "accumulated_microbatches",
        "supervised_target_count",
    }
)
_SIDECAR_FIELDS = frozenset(
    {
        "schema_id",
        "qualification_id",
        "specification_sha256",
        "repository_commit",
        "family_id",
        "creation_phase",
        "checkpoint_filename",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "state_sha256",
        "sidecar_sha256",
    }
)


def _exact(value: Mapping[str, object], expected: set[str] | frozenset[str], label: str) -> None:
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _commit(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase Git commit")
    return text


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _is_junction(path: Path) -> bool:
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _reject_link_components(path: Path, stop: Path) -> None:
    current = path
    while current != stop:
        if current.exists() and (current.is_symlink() or _is_junction(current)):
            raise ValueError(f"qualification path traverses a link or junction: {current}")
        current = current.parent


def _validate_qualification_root(root: Path) -> Path:
    resolved = root.resolve()
    system_temp = Path(tempfile.gettempdir()).resolve()
    if not resolved.is_relative_to(system_temp):
        raise ValueError("qualification checkpoints must remain under system temporary storage")
    _reject_link_components(root, system_temp)
    if not root.is_dir():
        raise ValueError("qualification checkpoint root must already exist")
    return resolved


def _checkpoint_filename(value: str) -> str:
    candidate = Path(value)
    if (
        candidate.name != value
        or candidate.suffix != ".pt"
        or value in {".pt", "..pt"}
    ):
        raise ValueError("checkpoint filename must be one local .pt filename")
    return value


def _state_identity(state: Mapping[str, object]) -> str:
    if set(state) != STATE_COMPONENTS:
        raise ValueError("checkpoint state must contain every exact-resume component")
    return state_sha256(state)


def validate_checkpoint_payload(payload: Mapping[str, object]) -> None:
    """Validate an in-memory payload without mutating runtime state."""

    _exact(payload, _PAYLOAD_FIELDS, "checkpoint payload")
    if payload["schema_id"] != CHECKPOINT_SCHEMA_ID:
        raise ValueError("checkpoint payload schema mismatch")
    _string(payload["qualification_id"], "qualification_id")
    _sha(payload["specification_sha256"], "specification_sha256")
    _commit(payload["repository_commit"], "repository_commit")
    if payload["family_id"] != FAMILY_ID:
        raise ValueError("checkpoint payload family mismatch")
    if payload["creation_phase"] not in CREATION_PHASES:
        raise ValueError("checkpoint creation phase mismatch")

    identities = _mapping(payload["identities"], "identities")
    if not identities:
        raise ValueError("checkpoint identities may not be empty")
    for name, digest in identities.items():
        _string(name, "identity name")
        _sha(digest, f"identities.{name}")

    progress = _mapping(payload["progress"], "progress")
    _exact(progress, _PROGRESS_FIELDS, "progress")
    records = progress["consumed_record_ids"]
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(item, str) or not item for item in records)
        or len(records) != len(set(records))
    ):
        raise ValueError("consumed record IDs must be non-empty and unique")
    for field in _PROGRESS_FIELDS - {"consumed_record_ids"}:
        _nonnegative_int(progress[field], f"progress.{field}")
    if progress["microbatch_count"] < len(records):
        raise ValueError("microbatch count cannot trail consumed record count")
    accumulated = progress["accumulated_microbatches"]
    if payload["creation_phase"] == "partial_accumulation" and accumulated == 0:
        raise ValueError("partial-accumulation checkpoint must retain partial work")
    if payload["creation_phase"] == "source_boundary" and accumulated != 0:
        raise ValueError("source-boundary checkpoint must be at an update boundary")

    state = _mapping(payload["state"], "state")
    gradients = _mapping(state.get("gradients"), "state.gradients")
    if payload["creation_phase"] == "partial_accumulation" and not gradients:
        raise ValueError("partial-accumulation checkpoint is missing gradients")
    if payload["creation_phase"] == "source_boundary" and gradients:
        raise ValueError("source-boundary checkpoint must not retain gradients")
    observed_state = _state_identity(state)
    if _sha(payload["state_sha256"], "state_sha256") != observed_state:
        raise ValueError("checkpoint state identity mismatch")


def build_checkpoint_payload(
    *,
    qualification_id: str,
    specification_sha256: str,
    repository_commit: str,
    creation_phase: str,
    identities: Mapping[str, str],
    progress: Mapping[str, object],
    state: Mapping[str, object],
) -> dict[str, object]:
    """Build a strict payload from already-created qualification state."""

    payload: dict[str, object] = {
        "schema_id": CHECKPOINT_SCHEMA_ID,
        "qualification_id": qualification_id,
        "specification_sha256": specification_sha256,
        "repository_commit": repository_commit,
        "family_id": FAMILY_ID,
        "creation_phase": creation_phase,
        "identities": dict(identities),
        "progress": dict(progress),
        "state": dict(state),
        "state_sha256": _state_identity(state),
    }
    validate_checkpoint_payload(payload)
    return payload


def sidecar_sha256(sidecar: Mapping[str, object]) -> str:
    body = dict(sidecar)
    body.pop("sidecar_sha256", None)
    return sha256_bytes(canonical_json(body))


def validate_sidecar(sidecar: Mapping[str, object]) -> None:
    _exact(sidecar, _SIDECAR_FIELDS, "checkpoint sidecar")
    if sidecar["schema_id"] != SIDECAR_SCHEMA_ID:
        raise ValueError("checkpoint sidecar schema mismatch")
    _string(sidecar["qualification_id"], "qualification_id")
    _sha(sidecar["specification_sha256"], "specification_sha256")
    _commit(sidecar["repository_commit"], "repository_commit")
    if sidecar["family_id"] != FAMILY_ID:
        raise ValueError("checkpoint sidecar family mismatch")
    if sidecar["creation_phase"] not in CREATION_PHASES:
        raise ValueError("checkpoint sidecar creation phase mismatch")
    _checkpoint_filename(_string(sidecar["checkpoint_filename"], "checkpoint_filename"))
    _sha(sidecar["checkpoint_sha256"], "checkpoint_sha256")
    if _nonnegative_int(sidecar["checkpoint_bytes"], "checkpoint_bytes") == 0:
        raise ValueError("checkpoint may not be empty")
    _sha(sidecar["state_sha256"], "state_sha256")
    if _sha(sidecar["sidecar_sha256"], "sidecar_sha256") != sidecar_sha256(sidecar):
        raise ValueError("checkpoint sidecar identity mismatch")


def _sha256_handle(handle: BinaryIO) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    byte_count = 0
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        byte_count += len(chunk)
    handle.seek(0)
    return digest.hexdigest(), byte_count


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def write_qualification_checkpoint(
    qualification_root: Path,
    checkpoint_filename: str,
    payload: Mapping[str, object],
    *,
    minimum_free_bytes: int = 1,
    replace: Callable[[str | bytes | os.PathLike[str] | os.PathLike[bytes], str | bytes | os.PathLike[str] | os.PathLike[bytes]], None] = os.replace,
) -> dict[str, object]:
    """Atomically write one non-overwriting checkpoint and hash sidecar."""

    validate_checkpoint_payload(payload)
    root = _validate_qualification_root(qualification_root)
    filename = _checkpoint_filename(checkpoint_filename)
    if isinstance(minimum_free_bytes, bool) or minimum_free_bytes < 1:
        raise ValueError("minimum_free_bytes must be positive")
    if shutil.disk_usage(root).free < minimum_free_bytes:
        raise OSError("insufficient disk for qualification checkpoint")

    destination = root / filename
    sidecar_path = destination.with_suffix(destination.suffix + ".sha256.json")
    checkpoint_temp = destination.with_suffix(destination.suffix + ".tmp")
    sidecar_temp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    for path in (destination, sidecar_path, checkpoint_temp, sidecar_temp):
        if path.exists():
            raise FileExistsError(f"qualification artifact already exists: {path.name}")

    checkpoint_promoted = False
    try:
        with checkpoint_temp.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        with checkpoint_temp.open("rb") as handle:
            checkpoint_sha, checkpoint_bytes = _sha256_handle(handle)
        sidecar: dict[str, object] = {
            "schema_id": SIDECAR_SCHEMA_ID,
            "qualification_id": payload["qualification_id"],
            "specification_sha256": payload["specification_sha256"],
            "repository_commit": payload["repository_commit"],
            "family_id": payload["family_id"],
            "creation_phase": payload["creation_phase"],
            "checkpoint_filename": filename,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": checkpoint_bytes,
            "state_sha256": payload["state_sha256"],
        }
        sidecar["sidecar_sha256"] = sidecar_sha256(sidecar)
        validate_sidecar(sidecar)
        _write_fsynced(sidecar_temp, canonical_json(sidecar) + b"\n")
        replace(checkpoint_temp, destination)
        checkpoint_promoted = True
        replace(sidecar_temp, sidecar_path)
        return dict(sidecar)
    except Exception:
        checkpoint_temp.unlink(missing_ok=True)
        sidecar_temp.unlink(missing_ok=True)
        # If checkpoint promotion succeeded but sidecar promotion failed, retain
        # the checkpoint as visible failure evidence in the isolated temp root.
        if not checkpoint_promoted:
            destination.unlink(missing_ok=True)
        raise


def load_verified_qualification_checkpoint(
    qualification_root: Path,
    checkpoint_filename: str,
    *,
    expected_specification_sha256: str,
    expected_identities: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify bytes and payload fully before returning any state to a caller."""

    root = _validate_qualification_root(qualification_root)
    filename = _checkpoint_filename(checkpoint_filename)
    destination = root / filename
    sidecar_path = destination.with_suffix(destination.suffix + ".sha256.json")
    _reject_link_components(destination, root)
    _reject_link_components(sidecar_path, root)
    if not destination.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("checkpoint and sidecar must both exist")

    raw_sidecar = sidecar_path.read_bytes()
    try:
        sidecar = json.loads(raw_sidecar)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint sidecar is not canonical JSON") from error
    if not isinstance(sidecar, Mapping) or raw_sidecar != canonical_json(sidecar) + b"\n":
        raise ValueError("checkpoint sidecar bytes are not canonical")
    validate_sidecar(sidecar)
    expected_spec = _sha(expected_specification_sha256, "expected specification")
    if sidecar["specification_sha256"] != expected_spec:
        raise ValueError("checkpoint specification identity mismatch")

    with destination.open("rb") as handle:
        observed_sha, observed_bytes = _sha256_handle(handle)
        if (
            observed_sha != sidecar["checkpoint_sha256"]
            or observed_bytes != sidecar["checkpoint_bytes"]
        ):
            raise ValueError("checkpoint byte identity mismatch")
        payload = torch.load(handle, map_location="cpu", weights_only=False)
        second_sha, second_bytes = _sha256_handle(handle)
        if second_sha != observed_sha or second_bytes != observed_bytes:
            raise ValueError("checkpoint mutated during validation")

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be an object")
    payload_copy = dict(payload)
    validate_checkpoint_payload(payload_copy)
    if payload_copy["specification_sha256"] != expected_spec:
        raise ValueError("checkpoint payload specification mismatch")
    if payload_copy["state_sha256"] != sidecar["state_sha256"]:
        raise ValueError("checkpoint and sidecar state identities differ")
    if payload_copy["qualification_id"] != sidecar["qualification_id"]:
        raise ValueError("checkpoint and sidecar qualification IDs differ")
    if payload_copy["repository_commit"] != sidecar["repository_commit"]:
        raise ValueError("checkpoint and sidecar repository commits differ")
    if payload_copy["creation_phase"] != sidecar["creation_phase"]:
        raise ValueError("checkpoint and sidecar phases differ")
    if dict(payload_copy["identities"]) != dict(expected_identities):
        raise ValueError("checkpoint bound identities mismatch")
    return payload_copy, dict(sidecar)
