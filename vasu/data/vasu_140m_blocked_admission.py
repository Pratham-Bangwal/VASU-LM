"""Evidence-bound blocked source-admission records for VASU-140M."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from evaluation.framework.vasu_140m_base_v2 import canonical_json, sha256_file
from vasu.data.vasu_140m_contamination_scan import _sha
from vasu.data.vasu_140m_source_admission import validate_admission_package_files
from vasu.data.vasu_140m_streaming_contamination import validate_streaming_report


SCHEMA_ID = "vasu_140m_blocked_source_admission_evidence_v1"


def evidence_identity(value: Mapping[str, object]) -> str:
    body = dict(value)
    body.pop("evidence_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def validate_blocked_admission_files(
    value: Mapping[str, object], root: Path, *, require_external_artifacts: bool = True
) -> None:
    expected = {
        "schema_id", "evidence_id", "source_id", "pending_admission",
        "source_artifact", "source_manifest", "evaluation_exclusions",
        "exact_contamination_scan", "remaining_blockers", "decision",
        "source_admission_approved", "release_build_permitted",
        "training_authorized", "evidence_sha256",
    }
    if set(value) != expected or value["schema_id"] != SCHEMA_ID:
        raise ValueError("blocked admission evidence schema mismatch")
    repository = root.resolve()
    for field in (
        "pending_admission", "source_artifact", "source_manifest",
        "evaluation_exclusions", "exact_contamination_scan",
    ):
        binding = value[field]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"{field} binding mismatch")
        path = (repository / str(binding["path"])).resolve()
        if not path.is_relative_to(repository):
            raise ValueError(f"{field} path is unsafe")
        if field in {"source_artifact", "source_manifest"} and not require_external_artifacts:
            _sha(binding["sha256"], f"{field}.sha256")
            continue
        if not path.is_file():
            raise ValueError(f"{field} path is missing or unsafe")
        if sha256_file(path) != _sha(binding["sha256"], f"{field}.sha256"):
            raise ValueError(f"{field} identity mismatch")
    pending = json.loads((repository / value["pending_admission"]["path"]).read_text(encoding="utf-8"))
    validate_admission_package_files(pending, repository)
    if pending["decision"]["state"] != "pending":
        raise ValueError("base admission package must remain pending")
    scan = json.loads((repository / value["exact_contamination_scan"]["path"]).read_text(encoding="utf-8"))
    validate_streaming_report(scan)
    if scan["source_id"] != value["source_id"] or scan["semantic_scan_complete"] is not False:
        raise ValueError("scan source or semantic state mismatch")
    blockers = value["remaining_blockers"]
    if blockers != [
        "independently curated production prompt inventory matrix",
        "independent semantic candidate search and review",
    ]:
        raise ValueError("remaining blockers are incomplete or substituted")
    if value["decision"] != "blocked":
        raise ValueError("evidence decision must be blocked")
    for field in (
        "source_admission_approved", "release_build_permitted", "training_authorized"
    ):
        if value[field] is not False:
            raise ValueError(f"{field} must remain false")
    if value["evidence_sha256"] != evidence_identity(value):
        raise ValueError("blocked admission evidence identity mismatch")
