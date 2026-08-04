"""Atomic byte-preserving promotion of reviewed development inventories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from evaluation.framework.vasu_140m_base_v2 import canonical_json, sha256_file
from evaluation.framework.vasu_140m_base_v2_inventory import (
    inventory_identity,
    validate_inventory_manifest,
    validate_inventory_manifest_files,
)

DIMENSIONS = ("arithmetic", "factuality", "manual_review", "repetition", "robustness")
SOURCE_ROOT = Path("evaluation/fixtures/vasu_140m_assistant_authored_internal_v1")
OUTPUT_ROOT = Path("evaluation/candidates/vasu_140m_base_v2_development_v1")
SUITE_ID = "vasu-140m-base-evaluation-v2-development-candidate-v1"
RECEIPT_SCHEMA_ID = "vasu_140m_development_inventory_promotion_receipt_v1"
DEPENDENCIES = {
    Path("docs/VASU_140M_PROMPT_MATRIX_REJECTION_REMEDIATION_"
         "INDEPENDENT_REVIEW_DECISION_20260804.md"):
        "8626ded952d055633b089ee535869ba1dbbe183e19ff4bc5f01bba53b6f7e0c6",
    Path("docs/VASU_140M_SOURCE_ADMISSION_V4_CANONICAL_IDENTITY_REMEDIATION_"
         "INDEPENDENT_REVIEW_DECISION_20260804.md"):
        "f1fa40847613bdaccbaa353252334cd7b7fb61c24f4918cc4ea671b4ba43279d",
}


def _junction(path: Path) -> bool:
    predicate = getattr(os.path, "isjunction", None)
    return bool(predicate and predicate(path))


def _reject_links(path: Path, root: Path) -> None:
    current = path
    while current != root:
        if current.exists() and (current.is_symlink() or _junction(current)):
            raise ValueError("promotion path may not traverse a link or junction")
        current = current.parent


def _commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("repository commit must be a lowercase Git commit")
    return value


def promote_development_inventories(
    repository_root: Path, repository_commit: str
) -> dict[str, object]:
    """Promote all five public development bundles or none."""

    root = repository_root.resolve()
    commit = _commit(repository_commit)
    destination = (root / OUTPUT_ROOT).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("promotion output escapes repository")
    _reject_links(destination, root)
    if destination.exists():
        raise FileExistsError("development promotion output already exists")
    if not destination.parent.is_dir():
        raise ValueError("development promotion parent must already exist")
    staging = destination.with_name(f".{destination.name}.tmp")
    if staging.exists():
        raise FileExistsError("stale development promotion staging exists")

    for path, expected in DEPENDENCIES.items():
        absolute = root / path
        if not absolute.is_file() or sha256_file(absolute) != expected:
            raise ValueError(f"accepted dependency identity mismatch: {path}")

    sources: dict[str, dict[str, object]] = {}
    for dimension in DIMENSIONS:
        path = root / SOURCE_ROOT / dimension / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_inventory_manifest_files(manifest, root)
        if manifest["dimension"] != dimension or manifest["split"] != "development":
            raise ValueError("development source dimension/split mismatch")
        if manifest["fixture_only"] is not True:
            raise ValueError("development promotion source must be a fixture")
        sources[dimension] = manifest

    staging.mkdir()
    manifests: dict[str, dict[str, object]] = {}
    try:
        for dimension, source in sources.items():
            target_dir = staging / dimension
            target_dir.mkdir()
            for filename in ("payload.jsonl", "provenance.jsonl", "contamination.jsonl"):
                source_file = root / SOURCE_ROOT / dimension / filename
                shutil.copyfile(source_file, target_dir / filename)
            manifest = dict(source)
            manifest.update(
                inventory_id=f"{SUITE_ID}-{dimension}",
                suite_id=SUITE_ID,
                repository_commit=commit,
                fixture_only=False,
                production_suite_frozen=False,
                evaluation_run_authorized=False,
                training_authorized=False,
            )
            for field, filename in (
                ("payload", "payload.jsonl"),
                ("provenance_index", "provenance.jsonl"),
                ("contamination_index", "contamination.jsonl"),
            ):
                binding = dict(manifest[field])
                binding["path"] = (OUTPUT_ROOT / dimension / filename).as_posix()
                binding["sha256"] = sha256_file(target_dir / filename)
                binding["byte_count"] = (target_dir / filename).stat().st_size
                manifest[field] = binding
            manifest["inventory_sha256"] = inventory_identity(manifest)
            validate_inventory_manifest(manifest)
            (target_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            manifests[dimension] = manifest
        os.replace(staging, destination)
        for manifest in manifests.values():
            validate_inventory_manifest_files(manifest, root)
        receipt: dict[str, object] = {
            "schema_id": RECEIPT_SCHEMA_ID,
            "repository_commit": commit,
            "accepted_dependencies": {
                path.as_posix(): digest for path, digest in DEPENDENCIES.items()
            },
            "source_inventory_sha256s": {
                dimension: source["inventory_sha256"] for dimension, source in sources.items()
            },
            "promoted_inventory_sha256s": {
                dimension: manifest["inventory_sha256"]
                for dimension, manifest in manifests.items()
            },
            "payload_bytes_preserved": True,
            "production_suite_frozen": False,
            "evaluation_run_authorized": False,
            "training_authorized": False,
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
        with (destination / "promotion_receipt.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return receipt
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
