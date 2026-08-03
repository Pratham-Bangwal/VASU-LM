from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from vasu.data.vasu_140m_source_admission import package_identity
from vasu.data.vasu_140m_source_admission_v4 import (
    validate_admission_package_v4,
    validate_admission_package_v4_files,
)


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


def _real_package() -> dict[str, object]:
    root = Path.cwd()
    package = json.loads(
        (root / "configs/data/admissions/fineweb_edu_extension_2025_26.pending.json")
        .read_text(encoding="utf-8")
    )
    decision_path = Path(
        "docs/"
        "VASU_140M_PROMPT_MATRIX_REJECTION_REMEDIATION_"
        "INDEPENDENT_REVIEW_DECISION_20260804.md"
    )
    exclusion_path = Path(
        "configs/data/exclusions/"
        "vasu_140m_fineweb_semantic_quarantine_20260804.json"
    )
    package["schema_id"] = "vasu_140m_base_source_admission_v4"
    package["prompt_matrix_decision"] = {
        "decision": "accepted",
        "path": decision_path.as_posix(),
        "sha256": hashlib.sha256((root / decision_path).read_bytes()).hexdigest(),
    }
    package["mandatory_exclusions"] = [
        {
            "canonical_sha256": (
                "cfd31cf8ea68d27994b1d85caba163de1141a9ead1a8bc87c90ed39c7e67837b"
            ),
            "path": exclusion_path.as_posix(),
            "required": True,
            "sha256": hashlib.sha256((root / exclusion_path).read_bytes()).hexdigest(),
        }
    ]
    package["package_sha256"] = package_identity(package)
    return package


def test_v4_files_verify_embedded_canonical_identity() -> None:
    package = _real_package()
    validate_admission_package_v4_files(package, Path.cwd())
    package["mandatory_exclusions"][0]["canonical_sha256"] = "0" * 64
    package["package_sha256"] = package_identity(package)
    with pytest.raises(ValueError, match="canonical identity mismatch"):
        validate_admission_package_v4_files(package, Path.cwd())
