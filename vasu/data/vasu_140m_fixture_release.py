"""Transactional fixture-only release construction for the VASU-140M contract.

This module intentionally cannot construct the planned production release. It
accepts only a small caller-supplied fixture, rejects production output paths,
and publishes one complete directory atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from vasu.data.vasu_140m_records import (
    EXPECTED_SPLITS,
    FAMILY_ID,
    MODEL_CONFIG_SHA256,
    RECORD_WIDTH,
    SPECIFICATION_SHA256,
    TOKENIZER_SHA256,
    LogicalExample,
    PackedRecord,
    pack_split,
    sha256_json,
    validate_packed_records,
    validate_split_isolation,
)


SCHEMA_ID = "vasu.model-family-fixture-release.v1"
PLAN_ID = "vasu_140m_instruction_seed_v1"
PLAN_SHA256 = "8da8cc92ef913bb567c96ec47986f849b84eb85ac2d11f3ad245fb1639361e16"
DECISION_SHA256 = (
    "4522d1c36a73700da9f3eec7cdd87561fded1a8f8661fa0ef88ab18a89058890"
)
MAX_FIXTURE_EXAMPLES = 30
PRODUCTION_RELEASE_PARTS = ("data", "processed", "vasu_140m", "instruction_seed_v1")
PRODUCTION_MANIFEST_PARTS = (
    "data",
    "manifests",
    "vasu_140m",
    "instruction_seed_v1.json",
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        .encode("utf-8")
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_fixture_destination(output_dir: Path, repository_root: Path) -> Path:
    root = repository_root.resolve()
    destination = output_dir.resolve()
    production_release = root.joinpath(*PRODUCTION_RELEASE_PARTS).resolve()
    production_manifest = root.joinpath(*PRODUCTION_MANIFEST_PARTS).resolve()
    if destination == root or _is_relative_to(destination, production_release):
        raise ValueError("fixture output must not target the production release path")
    if destination == production_manifest or _is_relative_to(
        production_manifest, destination
    ):
        raise ValueError("fixture output must not contain the production manifest")
    if "fixture" not in destination.name.casefold():
        raise ValueError("fixture output directory name must contain 'fixture'")
    if destination.exists():
        raise FileExistsError(f"fixture output already exists: {destination}")
    if not destination.parent.exists():
        raise ValueError("fixture output parent must already exist")
    return destination


def _logical_payload(examples: Sequence[LogicalExample]) -> list[dict[str, object]]:
    return [
        {
            "example_id": example.example_id,
            "semantic_sha256": example.semantic_sha256,
            "target_start": example.target_start,
            "token_count": len(example.token_ids),
        }
        for example in examples
    ]


def _packed_bytes(records: Sequence[PackedRecord]) -> tuple[bytes, bytes]:
    validate_packed_records(records)
    return (
        b"".join(record.tokens.tobytes() for record in records),
        b"".join(record.stored_mask.tobytes() for record in records),
    )


def validate_fixture_release(output_dir: Path) -> dict[str, object]:
    """Validate a published fixture manifest and every bound artifact."""

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("fixture manifest is missing")
    manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("fixture manifest must be an object")
    required = {
        "schema_id",
        "fixture_only",
        "plan_id",
        "plan_sha256",
        "independent_decision_sha256",
        "family_id",
        "model_config_sha256",
        "tokenizer_sha256",
        "record_specification_sha256",
        "record_width",
        "logical_input_sha256",
        "example_count",
        "splits",
        "publication",
        "production_release_created",
        "training_authorized",
        "training_permitted",
        "manifest_sha256",
    }
    if set(manifest) != required:
        raise ValueError("fixture manifest fields are invalid")
    expected = {
        "schema_id": SCHEMA_ID,
        "fixture_only": True,
        "plan_id": PLAN_ID,
        "plan_sha256": PLAN_SHA256,
        "independent_decision_sha256": DECISION_SHA256,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "record_specification_sha256": SPECIFICATION_SHA256,
        "record_width": RECORD_WIDTH,
        "production_release_created": False,
        "training_authorized": False,
        "training_permitted": False,
    }
    for field, value in expected.items():
        if manifest[field] != value:
            raise ValueError(f"fixture manifest {field} is invalid")
    body = dict(manifest)
    reported_hash = body.pop("manifest_sha256")
    if sha256_json(body) != reported_hash:
        raise ValueError("fixture manifest identity mismatch")
    splits = manifest["splits"]
    if not isinstance(splits, dict) or set(splits) != set(EXPECTED_SPLITS):
        raise ValueError("fixture manifest splits are invalid")
    expected_names = {"manifest.json"}
    for split in EXPECTED_SPLITS:
        split_report = splits[split]
        if not isinstance(split_report, dict):
            raise ValueError(f"fixture manifest {split} report is invalid")
        for artifact_name in ("tokens", "stored_mask"):
            artifact = split_report.get(artifact_name)
            if not isinstance(artifact, dict):
                raise ValueError(f"fixture manifest {split} {artifact_name} is invalid")
            relative = artifact.get("path")
            if not isinstance(relative, str) or Path(relative).name != relative:
                raise ValueError("fixture artifact path must be a local filename")
            path = output_dir / relative
            if not path.is_file():
                raise ValueError(f"fixture artifact is missing: {relative}")
            payload = path.read_bytes()
            if len(payload) != artifact.get("bytes"):
                raise ValueError(f"fixture artifact size mismatch: {relative}")
            if _sha256_bytes(payload) != artifact.get("sha256"):
                raise ValueError(f"fixture artifact hash mismatch: {relative}")
            expected_names.add(relative)
        token_bytes = int(split_report["tokens"]["bytes"])
        mask_bytes = int(split_report["stored_mask"]["bytes"])
        if token_bytes // 2 != mask_bytes or mask_bytes % RECORD_WIDTH:
            raise ValueError(f"fixture {split} token/mask layout mismatch")
    observed_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise ValueError("fixture output contains unbound files")
    return manifest


def build_fixture_release(
    *,
    splits: Mapping[str, Sequence[LogicalExample]],
    output_dir: Path,
    repository_root: Path,
    plan_sha256: str = PLAN_SHA256,
    decision_sha256: str = DECISION_SHA256,
    inject_failure_after_files: int | None = None,
) -> dict[str, object]:
    """Build and atomically publish a small non-production fixture release."""

    if plan_sha256 != PLAN_SHA256:
        raise ValueError("release plan identity mismatch")
    if decision_sha256 != DECISION_SHA256:
        raise ValueError("independent decision identity mismatch")
    total_examples = sum(len(examples) for examples in splits.values())
    if total_examples > MAX_FIXTURE_EXAMPLES:
        raise ValueError("fixture exceeds the hard example-count limit")
    validate_split_isolation(splits)
    destination = _validate_fixture_destination(output_dir, repository_root)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    written_files = 0

    def write(relative: str, payload: bytes) -> dict[str, object]:
        nonlocal written_files
        path = staging / relative
        path.write_bytes(payload)
        written_files += 1
        if inject_failure_after_files == written_files:
            raise RuntimeError("injected fixture publication failure")
        return {"path": relative, "bytes": len(payload), "sha256": _sha256_bytes(payload)}

    try:
        split_artifacts: dict[str, object] = {}
        logical_identity: dict[str, object] = {}
        for split in EXPECTED_SPLITS:
            examples = splits[split]
            records = pack_split(examples, split=split)
            token_bytes, mask_bytes = _packed_bytes(records)
            logical = _logical_payload(examples)
            logical_identity[split] = logical
            split_artifacts[split] = {
                "logical_example_count": len(examples),
                "packed_record_count": len(records),
                "used_token_count": sum(record.used_token_count for record in records),
                "supervised_token_count": sum(
                    int(record.stored_mask.sum()) for record in records
                ),
                "logical_sha256": sha256_json(logical),
                "tokens": write(f"{split}.tokens.bin", token_bytes),
                "stored_mask": write(f"{split}.mask.bin", mask_bytes),
            }
        manifest_body: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "fixture_only": True,
            "plan_id": PLAN_ID,
            "plan_sha256": plan_sha256,
            "independent_decision_sha256": decision_sha256,
            "family_id": FAMILY_ID,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "record_specification_sha256": SPECIFICATION_SHA256,
            "record_width": RECORD_WIDTH,
            "logical_input_sha256": sha256_json(logical_identity),
            "example_count": total_examples,
            "splits": split_artifacts,
            "publication": {
                "atomic_directory_replace": True,
                "overwrite_allowed": False,
                "production_path_rejected": True,
            },
            "production_release_created": False,
            "training_authorized": False,
            "training_permitted": False,
        }
        manifest_body["manifest_sha256"] = sha256_json(manifest_body)
        manifest_bytes = json.dumps(
            manifest_body, indent=2, sort_keys=True, ensure_ascii=True
        ).encode("utf-8") + b"\n"
        write("manifest.json", manifest_bytes)
        os.replace(staging, destination)
        return validate_fixture_release(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
