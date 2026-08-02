"""Fail-closed evaluation-parent exclusions for future VASU-140M data builds."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from evaluation.framework.vasu_140m_base_v2 import canonical_json, sha256_file
from evaluation.framework.vasu_140m_base_v2_inventory import (
    validate_inventory_manifest_files,
)


SCHEMA_ID = "vasu_140m_evaluation_parent_exclusions_v1"


def registry_identity(registry: Mapping[str, object]) -> str:
    body = dict(registry)
    body.pop("registry_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _provenance_ids(root: Path, manifest: Mapping[str, object]) -> set[str]:
    binding = manifest["provenance_index"]
    if not isinstance(binding, Mapping):
        raise ValueError("provenance binding must be an object")
    path = root / str(binding["path"])
    values: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        parent_id = str(record["parent_document_id"])
        if parent_id in values:
            raise ValueError("provenance contains duplicate parent documents")
        values.add(parent_id)
    return values


def validate_exclusion_registry_files(
    registry: Mapping[str, object], repository_root: Path
) -> None:
    expected = {
        "schema_id",
        "registry_id",
        "suite_id",
        "repository_commit",
        "sources",
        "fixture_only",
        "independently_curated",
        "production_suite_frozen",
        "training_data_release_created",
        "training_authorized",
        "registry_sha256",
    }
    if set(registry) != expected:
        raise ValueError("exclusion registry fields mismatch")
    if registry["schema_id"] != SCHEMA_ID:
        raise ValueError("exclusion registry schema mismatch")
    commit = registry["repository_commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("repository commit is invalid")
    if registry["fixture_only"] is not True or registry["independently_curated"] is not False:
        raise ValueError("self-curated exclusion flags are invalid")
    for field in (
        "production_suite_frozen",
        "training_data_release_created",
        "training_authorized",
    ):
        if registry[field] is not False:
            raise ValueError(f"{field} must remain false")
    if registry["registry_sha256"] != registry_identity(registry):
        raise ValueError("exclusion registry identity mismatch")

    root = repository_root.resolve()
    sources = registry["sources"]
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("exactly two source exclusions are required")
    observed_sources: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("source exclusion must be an object")
        required = {
            "source_id",
            "pending_admission",
            "development_manifest",
            "held_out_manifest",
            "excluded_parent_document_ids",
            "excluded_parent_count",
            "exclusion_sha256",
        }
        if set(source) != required:
            raise ValueError("source exclusion fields mismatch")
        source_id = str(source["source_id"])
        if source_id in observed_sources:
            raise ValueError("source exclusions must be unique")
        observed_sources.add(source_id)

        admission_binding = source["pending_admission"]
        if not isinstance(admission_binding, Mapping):
            raise ValueError("pending admission binding must be an object")
        admission_path = root / str(admission_binding["path"])
        if sha256_file(admission_path) != admission_binding["sha256"]:
            raise ValueError("pending admission identity mismatch")
        admission = _load_json(admission_path)
        policy = admission["evaluation_isolation"]["likelihood_policy"]
        if policy != {
            "created_after_acquisition": True,
            "document_level_isolation": True,
            "manifest_binding_required": True,
            "train_exclusion_required": True,
        }:
            raise ValueError("pending admission likelihood policy mismatch")
        if admission["decision"]["state"] != "pending" or admission["training_authorized"] is not False:
            raise ValueError("source admission must remain pending and non-authorizing")

        split_ids: dict[str, set[str]] = {}
        for split in ("development", "held_out"):
            binding = source[f"{split}_manifest"]
            if not isinstance(binding, Mapping):
                raise ValueError("manifest binding must be an object")
            manifest_path = root / str(binding["path"])
            if sha256_file(manifest_path) != binding["sha256"]:
                raise ValueError("likelihood manifest identity mismatch")
            manifest = _load_json(manifest_path)
            validate_inventory_manifest_files(manifest, root)
            if manifest["dimension"] != "likelihood" or manifest["split"] != split:
                raise ValueError("likelihood manifest split mismatch")
            if manifest["fixture_only"] is not True:
                raise ValueError("exclusions must remain qualification-only")
            split_ids[split] = _provenance_ids(root, manifest)
            if len(split_ids[split]) != 512:
                raise ValueError("each likelihood split must reserve 512 parents")
        if split_ids["development"] & split_ids["held_out"]:
            raise ValueError("development and held-out parents overlap")
        expected_ids = sorted(split_ids["development"] | split_ids["held_out"])
        if source["excluded_parent_document_ids"] != expected_ids:
            raise ValueError("excluded parent IDs do not match bound provenance")
        if source["excluded_parent_count"] != len(expected_ids):
            raise ValueError("excluded parent count mismatch")
        expected_sha = hashlib.sha256(canonical_json(expected_ids)).hexdigest()
        if source["exclusion_sha256"] != expected_sha:
            raise ValueError("source exclusion identity mismatch")
