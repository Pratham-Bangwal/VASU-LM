"""Construct review-ready VASU-140M source-admission v4 candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vasu.data.vasu_140m_source_admission import package_identity
from vasu.data.vasu_140m_source_admission_v4 import (
    SCHEMA_ID,
    validate_admission_package_v4_files,
)

DIMENSIONS = ("arithmetic", "factuality", "manual_review", "repetition", "robustness")
DEVELOPMENT_ROOT = Path("evaluation/candidates/vasu_140m_base_v2_development_v1")
HELD_OUT_ROOT = Path("evaluation/candidates/vasu_140m_base_v2_heldout_curator_v1")
MATRIX_DECISION = Path(
    "docs/VASU_140M_PROMPT_MATRIX_REJECTION_REMEDIATION_"
    "INDEPENDENT_REVIEW_DECISION_20260804.md"
)
FINEWEB_QUARANTINE = Path(
    "configs/data/exclusions/vasu_140m_fineweb_semantic_quarantine_20260804.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_binding(root: Path, path: Path) -> dict[str, object]:
    manifest = json.loads((root / path).read_text(encoding="utf-8"))
    return {
        "inventory_id": manifest["inventory_id"],
        "dimension": manifest["dimension"],
        "split": manifest["split"],
        "path": path.as_posix(),
        "sha256": _sha(root / path),
    }


def prompt_inventories(root: Path) -> list[dict[str, object]]:
    values = []
    for dimension in DIMENSIONS:
        values.append(_inventory_binding(root, DEVELOPMENT_ROOT / dimension / "manifest.json"))
        values.append(_inventory_binding(root, HELD_OUT_ROOT / dimension / "manifest.json"))
    return values


def build_candidate(
    root: Path, pending_path: Path, *, require_fineweb_quarantine: bool
) -> dict[str, object]:
    """Upgrade one v3 pending package into a non-authorizing v4 candidate."""

    package = json.loads((root / pending_path).read_text(encoding="utf-8"))
    package["schema_id"] = SCHEMA_ID
    package["package_id"] = str(package["package_id"]).replace("pending-v1", "review-candidate-v4")
    package["evaluation_isolation"]["inventories"] = prompt_inventories(root)
    package["legal_evidence"]["unresolved_items"] = []
    package["decision"] = {
        "state": "pending",
        "reviewed_by": "",
        "reviewed_at": "",
        "notes": "Pending independent source-specific admission review; no source is admitted.",
    }
    package["prompt_matrix_decision"] = {
        "decision": "accepted",
        "path": MATRIX_DECISION.as_posix(),
        "sha256": _sha(root / MATRIX_DECISION),
    }
    exclusions: list[dict[str, object]] = []
    if require_fineweb_quarantine:
        artifact = json.loads((root / FINEWEB_QUARANTINE).read_text(encoding="utf-8"))
        exclusions.append(
            {
                "path": FINEWEB_QUARANTINE.as_posix(),
                "sha256": _sha(root / FINEWEB_QUARANTINE),
                "canonical_sha256": artifact["quarantine_sha256"],
                "required": True,
            }
        )
    package["mandatory_exclusions"] = exclusions
    package["training_authorized"] = False
    package["package_sha256"] = package_identity(package)
    validate_admission_package_v4_files(package, root)
    return package


def write_new(package: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(package, handle, indent=2, sort_keys=True)
        handle.write("\n")
