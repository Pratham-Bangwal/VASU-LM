"""Source-admission v4 bindings for reviewed prompt matrices and exclusions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from vasu.data.vasu_140m_source_admission import (
    SCHEMA_ID as V3_SCHEMA_ID,
    package_identity,
    validate_admission_package,
    validate_admission_package_files,
)

SCHEMA_ID = "vasu_140m_base_source_admission_v4"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative")
    return path.as_posix()


def _v3_view(package: Mapping[str, object]) -> dict[str, object]:
    legacy = dict(package)
    legacy.pop("prompt_matrix_decision", None)
    legacy.pop("mandatory_exclusions", None)
    legacy["schema_id"] = V3_SCHEMA_ID
    legacy["package_sha256"] = package_identity(legacy)
    return legacy


def validate_admission_package_v4(package: Mapping[str, object]) -> None:
    """Validate v3 semantics plus immutable review and exclusion bindings."""

    if package.get("schema_id") != SCHEMA_ID:
        raise ValueError("source admission v4 schema mismatch")
    expected = set(_v3_view(package)) | {
        "prompt_matrix_decision",
        "mandatory_exclusions",
    }
    if set(package) != expected:
        raise ValueError("source admission v4 fields mismatch")
    validate_admission_package(_v3_view(package))

    decision = _mapping(package["prompt_matrix_decision"], "prompt_matrix_decision")
    if set(decision) != {"path", "sha256", "decision"}:
        raise ValueError("prompt_matrix_decision fields mismatch")
    _relative_path(decision["path"], "prompt_matrix_decision.path")
    _sha256(decision["sha256"], "prompt_matrix_decision.sha256")
    if decision["decision"] != "accepted":
        raise ValueError("prompt matrix decision must be accepted")

    exclusions = package["mandatory_exclusions"]
    if not isinstance(exclusions, list):
        raise ValueError("mandatory_exclusions must be a list")
    paths: set[str] = set()
    for index, raw in enumerate(exclusions):
        exclusion = _mapping(raw, f"mandatory exclusion {index}")
        if set(exclusion) != {"path", "sha256", "canonical_sha256", "required"}:
            raise ValueError(f"mandatory exclusion {index} fields mismatch")
        path = _relative_path(exclusion["path"], f"mandatory exclusion {index}.path")
        if path in paths:
            raise ValueError("mandatory exclusion paths must be unique")
        paths.add(path)
        _sha256(exclusion["sha256"], f"mandatory exclusion {index}.sha256")
        _sha256(
            exclusion["canonical_sha256"],
            f"mandatory exclusion {index}.canonical_sha256",
        )
        if exclusion["required"] is not True:
            raise ValueError("mandatory exclusions must be required")

    source_id = _mapping(package["source_record"], "source_record")["source_id"]
    if source_id == "fineweb_edu_extension_2025_26" and not exclusions:
        raise ValueError("FineWeb admission requires its reviewed quarantine")
    if package["package_sha256"] != package_identity(package):
        raise ValueError("source admission v4 identity mismatch")


def validate_admission_package_v4_files(
    package: Mapping[str, object], repository_root: Path
) -> None:
    """Validate every v4 binding against repository bytes."""

    validate_admission_package_v4(package)
    root = repository_root.resolve()
    validate_admission_package_files(_v3_view(package), root)
    bindings = [package["prompt_matrix_decision"], *package["mandatory_exclusions"]]
    for index, raw in enumerate(bindings):
        binding = _mapping(raw, f"binding {index}")
        relative = _relative_path(binding["path"], f"binding {index}.path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"binding {index} is missing or unsafe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
            raise ValueError(f"binding {index} identity mismatch")
