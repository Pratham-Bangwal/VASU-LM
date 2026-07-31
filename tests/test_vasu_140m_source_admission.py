from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from vasu.data.vasu_140m_source_admission import (
    FAMILY_ID,
    LEGACY_SCHEMA_ID,
    MODEL_CONFIG_SHA256,
    SCHEMA_ID,
    TOKENIZER_SHA256,
    package_identity,
    validate_admission_package,
    validate_admission_package_files,
)


ROOT = Path(__file__).resolve().parents[1]


def package() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "package_id": "synthetic-source-admission-v1",
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "source_record": {"path": "configs/data/sources/synthetic.json", "sha256": "a" * 64, "source_id": "synthetic", "registry_approval_status": "approved"},
        "legal_evidence": {"license_name": "CC0-1.0", "license_url": "https://example.test/license", "terms_url": "https://example.test/terms", "terms_revision": "v1", "obligations": ["retain provenance"], "unresolved_items": ["independent review pending"]},
        "acquisition": {"immutable_revision": "revision-1", "access_method": "HTTPS", "shard_inventory_sha256": "b" * 64, "raw_hash_algorithm": "sha256", "authorized": False},
        "document_lineage": {"stable_id_fields": ["document_id"], "revision_field": "revision", "shard_field": "shard", "transformation_id": "source_filter_v1"},
        "quality_policy": {"normalization_version": "nfc_v1", "filter_version": "filter_v1", "rejection_reasons": ["empty", "malformed"]},
        "evaluation_isolation": {"inventories": [{"path": "evaluation/benchmarks/example.json", "sha256": "c" * 64}], "exact_method": "normalized substring", "ngram_words": 8, "scan_before_split": True},
        "deduplication": {"normalization_version": "nfc_v1", "exact_method": "sha256", "near_method": "minhash_lsh_v1", "cross_source_indexes": ["fineweb_combined_v1"], "before_split": True},
        "decision": {"state": "pending", "reviewed_by": "", "reviewed_at": "", "notes": "Pending independent review."},
        "training_authorized": False,
        "package_sha256": "0" * 64,
    }
    value["package_sha256"] = package_identity(value)
    return value


def rehash(value: dict[str, object]) -> None:
    value["package_sha256"] = package_identity(value)


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
    rehash(value)
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
    value["evaluation_isolation"]["inventories"] = [
        {
            "path": inventory_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        }
    ]
    rehash(value)
    validate_admission_package_files(value, ROOT)

    value["legal_evidence"]["unresolved_items"] = []
    value["decision"] = {
        "state": "approved",
        "reviewed_by": "Synthetic Test Reviewer",
        "reviewed_at": "2026-08-01T12:00:00+05:30",
        "notes": "Synthetic repository-binding test only.",
    }
    rehash(value)
    validate_admission_package_files(value, ROOT)

    value["source_record"]["sha256"] = "0" * 64
    rehash(value)
    with pytest.raises(ValueError, match="source registry file identity mismatch"):
        validate_admission_package_files(value, ROOT)


def test_mutating_nested_evidence_without_rehash_fails() -> None:
    value = deepcopy(package())
    value["quality_policy"]["filter_version"] = "filter_v2"
    with pytest.raises(ValueError, match="package identity mismatch"):
        validate_admission_package(value)
