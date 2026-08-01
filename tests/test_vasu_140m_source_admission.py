from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from vasu.data.vasu_140m_source_admission import (
    EVALUATION_SPLITS,
    PROMPT_EVALUATION_DIMENSIONS,
    FAMILY_ID,
    LEGACY_SCHEMA_ID,
    MODEL_CONFIG_SHA256,
    SCHEMA_ID,
    TOKENIZER_SHA256,
    package_identity,
    validate_admission_package,
    validate_admission_package_files,
)
from vasu.data.sources import load_source_records


ROOT = Path(__file__).resolve().parents[1]


def package() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "package_id": "synthetic-source-admission-v1",
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "source_record": {"path": "configs/data/sources/synthetic.json", "sha256": "a" * 64, "source_id": "synthetic", "registry_approval_status": "approved"},
        "legal_evidence": {
            "license_name": "CC0-1.0",
            "license_url": "https://example.test/license",
            "terms_url": "https://example.test/terms",
            "terms_revision": "v1",
            "commercial_use_allowed": True,
            "attribution_required": True,
            "redistribution_allowed": True,
            "gated_access": False,
            "requires_authentication": False,
            "obligations": ["retain provenance"],
            "unresolved_items": ["independent review pending"],
        },
        "acquisition": {"immutable_revision": "revision-1", "access_method": "HTTPS", "shard_inventory_sha256": "b" * 64, "raw_hash_algorithm": "sha256", "authorized": False},
        "document_lineage": {"stable_id_fields": ["document_id"], "revision_field": "revision", "shard_field": "shard", "transformation_id": "source_filter_v1"},
        "quality_policy": {"normalization_version": "nfc_v1", "filter_version": "filter_v1", "rejection_reasons": ["empty", "malformed"]},
        "evaluation_isolation": {
            "inventories": [
                {
                    "inventory_id": "fixture-factuality-development",
                    "dimension": "factuality",
                    "split": "development",
                    "path": "evaluation/benchmarks/example.json",
                    "sha256": "c" * 64,
                }
            ],
            "likelihood_policy": {
                "created_after_acquisition": True,
                "document_level_isolation": True,
                "train_exclusion_required": True,
                "manifest_binding_required": True,
            },
            "exact_method": "normalized substring",
            "ngram_words": 8,
            "scan_before_split": True,
        },
        "deduplication": {"normalization_version": "nfc_v1", "exact_method": "sha256", "near_method": "minhash_lsh_v1", "cross_source_indexes": ["fineweb_combined_v1"], "before_split": True},
        "decision": {"state": "pending", "reviewed_by": "", "reviewed_at": "", "notes": "Pending independent review."},
        "training_authorized": False,
        "package_sha256": "0" * 64,
    }
    value["package_sha256"] = package_identity(value)
    return value


def rehash(value: dict[str, object]) -> None:
    value["package_sha256"] = package_identity(value)


def set_complete_inventory_matrix(value: dict[str, object]) -> None:
    value["evaluation_isolation"]["inventories"] = [
        {
            "inventory_id": f"fixture-{dimension}-{split}",
            "dimension": dimension,
            "split": split,
            "path": f"evaluation/inventories/{dimension}-{split}.json",
            "sha256": hashlib.sha256(f"{dimension}-{split}".encode()).hexdigest(),
        }
        for dimension in sorted(PROMPT_EVALUATION_DIMENSIONS)
        for split in sorted(EVALUATION_SPLITS)
    ]


def test_pending_package_is_valid_and_non_authorizing() -> None:
    value = package()
    validate_admission_package(value)
    assert value["training_authorized"] is False
    assert value["acquisition"]["authorized"] is False
    assert value["source_record"]["registry_approval_status"] == "approved"
    assert value["decision"]["state"] == "pending"


def test_legacy_schema_and_registry_state_substitution_fail_closed() -> None:
    value = package()
    value["schema_id"] = LEGACY_SCHEMA_ID
    rehash(value)
    with pytest.raises(ValueError, match="schema identity mismatch"):
        validate_admission_package(value)

    value = package()
    value["source_record"]["registry_approval_status"] = "pending"
    value["decision"] = {
        "state": "approved",
        "reviewed_by": "Reviewer",
        "reviewed_at": "2026-08-01T12:00:00+05:30",
        "notes": "Synthetic invalid approval.",
    }
    value["legal_evidence"]["unresolved_items"] = []
    rehash(value)
    with pytest.raises(ValueError, match="approved generic source record"):
        validate_admission_package(value)


@pytest.mark.parametrize("field", ["family_id", "model_config_sha256", "tokenizer_sha256"])
def test_frozen_identity_mutations_fail(field: str) -> None:
    value = package()
    value[field] = "0" * 64
    rehash(value)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_admission_package(value)


def test_unknown_fields_and_identity_mutation_fail() -> None:
    value = package()
    value["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_admission_package(value)
    value = package()
    value["package_id"] = "changed"
    with pytest.raises(ValueError, match="package identity mismatch"):
        validate_admission_package(value)


def test_acquisition_and_training_authority_fail_closed() -> None:
    value = package()
    value["acquisition"]["authorized"] = True
    rehash(value)
    with pytest.raises(ValueError, match="acquisition.authorized must be false"):
        validate_admission_package(value)
    value = package()
    value["training_authorized"] = True
    rehash(value)
    with pytest.raises(ValueError, match="training_authorized must be false"):
        validate_admission_package(value)


def test_unsafe_path_and_missing_evaluation_inventory_fail() -> None:
    value = package()
    value["source_record"]["path"] = "../outside.json"
    rehash(value)
    with pytest.raises(ValueError, match="repository-relative"):
        validate_admission_package(value)
    value = package()
    value["evaluation_isolation"]["inventories"] = []
    rehash(value)
    with pytest.raises(ValueError, match="must be non-empty"):
        validate_admission_package(value)


def test_split_ordering_and_deduplication_policy_fail_closed() -> None:
    value = package()
    value["evaluation_isolation"]["scan_before_split"] = False
    rehash(value)
    with pytest.raises(ValueError, match="before split"):
        validate_admission_package(value)
    value = package()
    value["deduplication"]["before_split"] = False
    rehash(value)
    with pytest.raises(ValueError, match="before split"):
        validate_admission_package(value)


def test_approved_package_requires_review_and_resolved_legal_items() -> None:
    value = package()
    value["decision"]["state"] = "approved"
    rehash(value)
    with pytest.raises(ValueError, match="reviewed_by"):
        validate_admission_package(value)
    value["decision"]["reviewed_by"] = "Reviewer"
    value["decision"]["reviewed_at"] = "2026-08-01"
    rehash(value)
    with pytest.raises(ValueError, match="include a timezone"):
        validate_admission_package(value)
    value["decision"]["reviewed_at"] = "2026-08-01T12:00:00+05:30"
    rehash(value)
    with pytest.raises(ValueError, match="unresolved legal"):
        validate_admission_package(value)
    value["legal_evidence"]["unresolved_items"] = []
    set_complete_inventory_matrix(value)
    rehash(value)
    validate_admission_package(value)


def test_approved_package_requires_complete_evaluation_matrix() -> None:
    value = package()
    value["decision"] = {
        "state": "approved",
        "reviewed_by": "Reviewer",
        "reviewed_at": "2026-08-01T12:00:00+05:30",
        "notes": "Synthetic approval test.",
    }
    value["legal_evidence"]["unresolved_items"] = []
    rehash(value)
    with pytest.raises(ValueError, match="all 10 prompt inventories"):
        validate_admission_package(value)


def test_repository_bound_evidence_is_verified() -> None:
    value = package()
    source_path = ROOT / "configs/data/sources/fineweb_edu.json"
    inventory_path = ROOT / "evaluation/benchmarks/ultrachat_promotion_v1.json"
    value["source_record"] = {
        "path": source_path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_id": "fineweb_edu_extension_2025_26",
        "registry_approval_status": "approved",
    }
    record = next(
        item
        for item in load_source_records(source_path)
        if item.source_id == "fineweb_edu_extension_2025_26"
    )
    value["legal_evidence"].update(
        {
            "license_name": record.license_name,
            "license_url": record.license_url,
            "commercial_use_allowed": record.commercial_use_allowed,
            "attribution_required": record.attribution_required,
            "redistribution_allowed": record.redistribution_allowed,
            "gated_access": record.gated_access,
            "requires_authentication": record.requires_authentication,
        }
    )
    value["acquisition"].update(
        {
            "immutable_revision": record.pinned_revision,
            "access_method": record.access_method,
        }
    )
    value["evaluation_isolation"]["inventories"] = [
        {
            "inventory_id": "ultrachat-development",
            "dimension": "factuality",
            "split": "development",
            "path": inventory_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        }
    ]
    rehash(value)
    validate_admission_package_files(value, ROOT)

    value["source_record"]["sha256"] = "0" * 64
    rehash(value)
    with pytest.raises(ValueError, match="source registry file identity mismatch"):
        validate_admission_package_files(value, ROOT)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("legal_evidence", "license_name", "CC0-1.0", "license_name"),
        ("legal_evidence", "attribution_required", False, "attribution_required"),
        ("legal_evidence", "gated_access", True, "gated_access"),
        ("acquisition", "immutable_revision", "mutable-main", "immutable_revision"),
        ("acquisition", "access_method", "unbound download", "access_method"),
    ],
)
def test_repository_binding_rejects_semantic_registry_drift(
    section: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    value = package()
    source_path = ROOT / "configs/data/sources/wikimedia.json"
    inventory_path = ROOT / "evaluation/benchmarks/ultrachat_promotion_v1.json"
    record = load_source_records(source_path)[0]
    value["source_record"] = {
        "path": source_path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_id": record.source_id,
        "registry_approval_status": record.approval_status,
    }
    value["legal_evidence"].update(
        {
            "license_name": record.license_name,
            "license_url": record.license_url,
            "commercial_use_allowed": record.commercial_use_allowed,
            "attribution_required": record.attribution_required,
            "redistribution_allowed": record.redistribution_allowed,
            "gated_access": record.gated_access,
            "requires_authentication": record.requires_authentication,
        }
    )
    value["acquisition"].update(
        {
            "immutable_revision": record.pinned_revision,
            "access_method": record.access_method,
        }
    )
    value["evaluation_isolation"]["inventories"] = [
        {
            "inventory_id": "ultrachat-development",
            "dimension": "factuality",
            "split": "development",
            "path": inventory_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        }
    ]
    value[section][field] = replacement
    rehash(value)
    with pytest.raises(ValueError, match=message):
        validate_admission_package_files(value, ROOT)


def test_mutating_nested_evidence_without_rehash_fails() -> None:
    value = deepcopy(package())
    value["quality_policy"]["filter_version"] = "filter_v2"
    with pytest.raises(ValueError, match="package identity mismatch"):
        validate_admission_package(value)


def test_approved_repository_package_requires_real_inventory_manifests(
    tmp_path: Path,
) -> None:
    root = tmp_path
    registry_path = root / "configs/data/sources/wikimedia.json"
    registry_path.parent.mkdir(parents=True)
    original = ROOT / "configs/data/sources/wikimedia.json"
    registry_path.write_bytes(original.read_bytes())
    record = load_source_records(registry_path)[0]

    value = package()
    value["source_record"] = {
        "path": "configs/data/sources/wikimedia.json",
        "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "source_id": record.source_id,
        "registry_approval_status": record.approval_status,
    }
    value["legal_evidence"].update(
        {
            "license_name": record.license_name,
            "license_url": record.license_url,
            "commercial_use_allowed": record.commercial_use_allowed,
            "attribution_required": record.attribution_required,
            "redistribution_allowed": record.redistribution_allowed,
            "gated_access": record.gated_access,
            "requires_authentication": record.requires_authentication,
            "unresolved_items": [],
        }
    )
    value["acquisition"].update(
        {
            "immutable_revision": record.pinned_revision,
            "access_method": record.access_method,
        }
    )
    set_complete_inventory_matrix(value)
    for binding in value["evaluation_isolation"]["inventories"]:
        path = root / binding["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "inventory_id": binding["inventory_id"],
            "dimension": binding["dimension"],
            "split": binding["split"],
            "fixture_only": False,
        }
        path.write_text(json.dumps(manifest), encoding="utf-8")
        binding["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    value["decision"] = {
        "state": "approved",
        "reviewed_by": "Synthetic Test Reviewer",
        "reviewed_at": "2026-08-01T12:00:00+05:30",
        "notes": "Synthetic file-binding test only.",
    }
    rehash(value)
    validate_admission_package_files(value, root)

    first = value["evaluation_isolation"]["inventories"][0]
    path = root / first["path"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["fixture_only"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    first["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rehash(value)
    with pytest.raises(ValueError, match="not production-candidate"):
        validate_admission_package_files(value, root)
