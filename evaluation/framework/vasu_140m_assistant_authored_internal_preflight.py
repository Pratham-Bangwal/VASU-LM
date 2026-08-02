"""Read-only integrity preflight for the assistant-authored internal suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.framework.vasu_140m_base_v2 import canonical_json, sha256_file
from evaluation.framework.vasu_140m_base_v2_inventory import (
    validate_inventory_manifest_files,
)
from evaluation.framework.vasu_140m_assistant_authored_internal_suite import (
    COUNTS,
    SUITE_ID,
)


QUALIFICATION_SCHEMA_ID = "vasu_140m_assistant_authored_internal_preflight_v1"
DEFAULT_SUITE_DIRECTORY = "evaluation/fixtures/vasu_140m_assistant_authored_internal_v1"


def qualification_identity(report: dict[str, object]) -> str:
    body = dict(report)
    body.pop("qualification_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def qualify_internal_suite(repository_root: Path) -> dict[str, object]:
    """Validate all fixture bytes without opening a checkpoint or a model."""

    root = repository_root.resolve()
    suite = root / DEFAULT_SUITE_DIRECTORY
    report_path = suite / "suite_report.json"
    if not report_path.is_file():
        raise ValueError("internal suite report is missing")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_authoring = {
        "provenance": "assistant_authored_internal_nonindependent",
        "independently_curated": False,
        "held_out_content_present": False,
        "training_data_eligible": False,
    }
    if source_report.get("suite_id") != SUITE_ID:
        raise ValueError("internal suite identity mismatch")
    if source_report.get("authoring") != expected_authoring:
        raise ValueError("internal suite authoring boundary mismatch")
    if source_report.get("record_counts") != COUNTS:
        raise ValueError("internal suite record counts mismatch")
    inventory_sha256s: dict[str, str] = {}
    for dimension, count in COUNTS.items():
        path = suite / dimension / "manifest.json"
        if not path.is_file():
            raise ValueError(f"internal suite manifest is missing: {dimension}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_inventory_manifest_files(manifest, root)
        if manifest["fixture_only"] is not True or manifest["split"] != "development":
            raise ValueError("internal suite manifest is not development-only")
        if int(manifest["payload"]["record_count"]) != count:
            raise ValueError("internal suite manifest count mismatch")
        inventory_sha256s[dimension] = str(manifest["inventory_sha256"])
    if source_report.get("inventory_sha256s") != inventory_sha256s:
        raise ValueError("internal suite inventory identities mismatch")
    result: dict[str, object] = {
        "schema_id": QUALIFICATION_SCHEMA_ID,
        "suite_report_path": f"{DEFAULT_SUITE_DIRECTORY}/suite_report.json",
        "suite_report_sha256": sha256_file(report_path),
        "suite_report_identity": source_report["report_sha256"],
        "inventory_sha256s": inventory_sha256s,
        "record_counts": dict(COUNTS),
        "held_out_content_present": False,
        "checkpoint_opened": False,
        "model_invoked": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }
    result["qualification_sha256"] = qualification_identity(result)
    return result
