"""Current-phase repository guard for the VASU-140M base lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vasu.training.vasu_140m_readiness import (
    ACTUAL_AUTHORIZATION_PATH,
    ACTUAL_PLAN_PATH,
    build_readiness_report,
    validate_readiness_report,
)


SCHEMA_ID = "vasu_140m_repository_policy_guard_v1"
FROZEN_READINESS = Path(
    "evaluation/fixtures/vasu_140m_base_pretraining_readiness_20260803.json"
)
FORBIDDEN_CURRENT_PHASE_PATHS = (
    Path("data/processed/vasu_140m/base_pretraining"),
    Path("data/manifests/vasu_140m/base_pretraining"),
    Path("checkpoints/vasu_140m"),
    ACTUAL_PLAN_PATH,
    ACTUAL_AUTHORIZATION_PATH,
)
EXPECTED_GATE_STATES = {
    "cuda_amp": "complete",
    "base_data_release": "blocked",
    "base_evaluation_v2": "blocked",
    "real_data_exact_resume": "blocked",
    "scientific_experiment_plan": "blocked",
    "training_authorization": "blocked",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _relative_entries(root: Path, path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.is_file():
        return [path.relative_to(root).as_posix()]
    return sorted(
        item.relative_to(root).as_posix()
        for item in path.rglob("*")
        if item.is_file()
    ) or [path.relative_to(root).as_posix() + "/"]


def build_policy_report(repository_root: Path) -> dict[str, object]:
    """Return violations of the frozen, explicitly non-authorizing phase."""

    root = repository_root.resolve()
    observed = build_readiness_report(root)
    validate_readiness_report(observed)
    violations: list[dict[str, object]] = []

    frozen_path = root / FROZEN_READINESS
    if not frozen_path.is_file():
        violations.append(
            {
                "code": "missing_frozen_readiness",
                "path": FROZEN_READINESS.as_posix(),
                "details": [],
            }
        )
    else:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        try:
            validate_readiness_report(frozen)
        except ValueError as error:
            violations.append(
                {
                    "code": "invalid_frozen_readiness",
                    "path": FROZEN_READINESS.as_posix(),
                    "details": [str(error)],
                }
            )
        else:
            if observed != frozen:
                violations.append(
                    {
                        "code": "readiness_identity_drift",
                        "path": FROZEN_READINESS.as_posix(),
                        "details": [
                            str(frozen.get("report_sha256")),
                            str(observed.get("report_sha256")),
                        ],
                    }
                )

    observed_states = {
        name: gate["status"] for name, gate in observed["gates"].items()
    }
    if observed_states != EXPECTED_GATE_STATES:
        violations.append(
            {
                "code": "unexpected_gate_transition",
                "path": FROZEN_READINESS.as_posix(),
                "details": [json.dumps(observed_states, sort_keys=True)],
            }
        )

    for relative in FORBIDDEN_CURRENT_PHASE_PATHS:
        entries = _relative_entries(root, root / relative)
        if entries:
            violations.append(
                {
                    "code": "protected_artifact_present",
                    "path": relative.as_posix(),
                    "details": entries,
                }
            )

    report: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "family_id": "vasu_140m_v1",
        "readiness_report_sha256": observed["report_sha256"],
        "expected_gate_states": EXPECTED_GATE_STATES,
        "checked_protected_paths": [
            path.as_posix() for path in FORBIDDEN_CURRENT_PHASE_PATHS
        ],
        "violations": violations,
        "passed": not violations,
        "eligible_for_experiment_plan": False,
        "eligible_for_authorization_review": False,
        "training_authorized": False,
    }
    report["report_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def validate_policy_report(report: dict[str, object]) -> None:
    """Validate the policy report and its permanent non-authorization fields."""

    expected = {
        "schema_id",
        "family_id",
        "readiness_report_sha256",
        "expected_gate_states",
        "checked_protected_paths",
        "violations",
        "passed",
        "eligible_for_experiment_plan",
        "eligible_for_authorization_review",
        "training_authorized",
        "report_sha256",
    }
    if set(report) != expected or report.get("schema_id") != SCHEMA_ID:
        raise ValueError("policy report schema mismatch")
    if report.get("expected_gate_states") != EXPECTED_GATE_STATES:
        raise ValueError("policy report gate contract mismatch")
    violations = report.get("violations")
    if not isinstance(violations, list) or report.get("passed") is not (not violations):
        raise ValueError("policy report pass state mismatch")
    if any(
        report.get(field) is not False
        for field in (
            "eligible_for_experiment_plan",
            "eligible_for_authorization_review",
            "training_authorized",
        )
    ):
        raise ValueError("policy report must remain non-authorizing")
    body = dict(report)
    observed_sha = body.pop("report_sha256", None)
    expected_sha = hashlib.sha256(_canonical_json(body)).hexdigest()
    if observed_sha != expected_sha:
        raise ValueError("policy report identity mismatch")
