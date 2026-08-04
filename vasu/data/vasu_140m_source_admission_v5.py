"""Final source admission binding an accepted source-specific review."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from vasu.data.vasu_140m_source_admission import package_identity
from vasu.data.vasu_140m_source_admission_v4 import (
    SCHEMA_ID as V4_SCHEMA_ID,
    validate_admission_package_v4,
    validate_admission_package_v4_files,
)

SCHEMA_ID = "vasu_140m_base_source_admission_v5"


def _v4_view(package: Mapping[str, object]) -> dict[str, object]:
    value = dict(package)
    value.pop("source_admission_review", None)
    value["schema_id"] = V4_SCHEMA_ID
    value["package_sha256"] = package_identity(value)
    return value


def validate_admission_package_v5(package: Mapping[str, object]) -> None:
    if package.get("schema_id") != SCHEMA_ID:
        raise ValueError("source admission v5 schema mismatch")
    if set(package) != set(_v4_view(package)) | {"source_admission_review"}:
        raise ValueError("source admission v5 fields mismatch")
    validate_admission_package_v4(_v4_view(package))
    review = package["source_admission_review"]
    if not isinstance(review, Mapping) or set(review) != {
        "path", "sha256", "decision", "source_id"
    }:
        raise ValueError("source_admission_review fields mismatch")
    path = Path(str(review["path"]).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("source admission review path is unsafe")
    digest = review["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("source admission review SHA-256 is invalid")
    if review["decision"] != "accepted":
        raise ValueError("source admission review must be accepted")
    source_id = package["source_record"]["source_id"]
    if review["source_id"] != source_id:
        raise ValueError("source admission review source mismatch")
    if package["decision"]["state"] != "approved":
        raise ValueError("v5 final admission decision must be approved")
    if package["package_sha256"] != package_identity(package):
        raise ValueError("source admission v5 identity mismatch")


def validate_admission_package_v5_files(
    package: Mapping[str, object], repository_root: Path
) -> None:
    validate_admission_package_v5(package)
    root = repository_root.resolve()
    validate_admission_package_v4_files(_v4_view(package), root)
    review = package["source_admission_review"]
    path = (root / review["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("source admission review is missing or unsafe")
    if hashlib.sha256(path.read_bytes()).hexdigest() != review["sha256"]:
        raise ValueError("source admission review identity mismatch")
