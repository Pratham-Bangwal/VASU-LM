from __future__ import annotations

import copy
import hashlib

import pytest

from vasu.data.vasu_140m_source_admission import package_identity
from vasu.data.vasu_140m_source_admission_v4 import validate_admission_package_v4


def _package() -> dict[str, object]:
    from scripts.build_vasu_140m_pending_source_admissions import build_package

    package = build_package(
        registry="configs/data/sources/fineweb_edu.json",
        source_id="fineweb_edu_extension_2025_26",
        terms_url="https://example.com/terms",
        terms_revision="revision",
        obligations=["attribute"],
        unresolved=["independent source-specific review"],
        stable_ids=["source_document_id"],
        revision_field="dump",
        shard_field="shard",
        filter_version="v1",
        rejection_reasons=["contamination"],
        cross_indexes=["cross-source-v1"],
    )
    package["schema_id"] = "vasu_140m_base_source_admission_v4"
    package["prompt_matrix_decision"] = {
        "decision": "accepted",
        "path": "docs/matrix.md",
        "sha256": "a" * 64,
    }
    package["mandatory_exclusions"] = [
        {
            "canonical_sha256": "b" * 64,
            "path": "configs/data/exclusions/quarantine.json",
            "required": True,
            "sha256": "c" * 64,
        }
    ]
    package["package_sha256"] = package_identity(package)
    return package


def test_v4_requires_fineweb_quarantine() -> None:
    package = _package()
    validate_admission_package_v4(package)
    package["mandatory_exclusions"] = []
    package["package_sha256"] = package_identity(package)
    with pytest.raises(ValueError, match="requires its reviewed quarantine"):
        validate_admission_package_v4(package)


def test_v4_requires_accepted_matrix() -> None:
    package = _package()
    package["prompt_matrix_decision"]["decision"] = "rejected"
    package["package_sha256"] = package_identity(package)
    with pytest.raises(ValueError, match="must be accepted"):
        validate_admission_package_v4(package)


def test_v4_rejects_mutation() -> None:
    package = _package()
    mutated = copy.deepcopy(package)
    mutated["mandatory_exclusions"][0]["canonical_sha256"] = hashlib.sha256(
        b"mutation"
    ).hexdigest()
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_admission_package_v4(mutated)
