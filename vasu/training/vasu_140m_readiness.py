"""Read-only VASU-140M base-pretraining readiness aggregation.

This module reports repository evidence; it is deliberately not an
authorization validator and has no route to launch an experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_ID = "vasu_140m_base_pretraining_readiness_report_v1"

CUDA_RESULT = Path("evaluation/results/vasu_140m_cuda_amp_qualification_20260731.json")
CUDA_DECISION = Path(
    "docs/VASU_140M_CUDA_AMP_QUALIFICATION_EXECUTION_"
    "INDEPENDENT_REVIEW_DECISION_20260731.md"
)
BLOCKED_ADMISSIONS = (
    Path(
        "configs/data/admissions/blocked_evidence/"
        "fineweb_edu_extension_2025_26.blocked.json"
    ),
    Path(
        "configs/data/admissions/blocked_evidence/"
        "wikipedia_en_20231101.blocked.json"
    ),
)
PRODUCTION_BASE_PATHS = (
    Path("data/processed/vasu_140m/base_pretraining"),
    Path("data/manifests/vasu_140m/base_pretraining"),
)
EVALUATION_FOUNDATION = (
    Path("evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json"),
    Path("evaluation/fixtures/vasu_140m_base_v2_inventory_builder_qualification_v1.json"),
    Path("evaluation/fixtures/vasu_140m_contamination_scan_qualification_v1.json"),
)
PRODUCTION_EVALUATION_PATH = Path("evaluation/benchmarks/vasu_140m_base_v2")
RESUME_FOUNDATION = (
    Path("evaluation/fixtures/vasu_140m_real_data_resume_contract_qualification_v1.json"),
    Path("evaluation/fixtures/vasu_140m_resume_checkpoint_qualification_v1.json"),
)
REAL_RESUME_RESULT = Path(
    "evaluation/results/vasu_140m/real_data_exact_resume/result.json"
)
ACTUAL_PLAN_PATH = Path("configs/training/vasu_140m_base_pretraining_v1.json")
ACTUAL_AUTHORIZATION_PATH = Path(
    "configs/training/authorizations/vasu_140m_base_pretraining_v1.json"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedded_report_sha256(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("report_sha256", None)
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _evidence(repository_root: Path, relative: Path) -> dict[str, object]:
    path = repository_root / relative
    return {
        "path": relative.as_posix(),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _load_object(repository_root: Path, relative: Path) -> dict[str, Any] | None:
    path = repository_root / relative
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{relative.as_posix()} must contain a JSON object")
    return value


def _gate(
    status: str,
    *,
    evidence: list[dict[str, object]],
    blockers: list[str],
) -> dict[str, object]:
    if status not in {"complete", "blocked", "invalid"}:
        raise ValueError(f"unsupported readiness status: {status}")
    if status == "complete" and blockers:
        raise ValueError("a complete gate cannot retain blockers")
    return {"status": status, "evidence": evidence, "blockers": blockers}


def build_readiness_report(repository_root: Path) -> dict[str, object]:
    """Aggregate exact repository evidence without mutating repository state."""

    root = repository_root.resolve()

    cuda_evidence = [_evidence(root, CUDA_RESULT), _evidence(root, CUDA_DECISION)]
    cuda_result = _load_object(root, CUDA_RESULT)
    cuda_decision_path = root / CUDA_DECISION
    cuda_decision_text = (
        cuda_decision_path.read_text(encoding="utf-8")
        if cuda_decision_path.is_file()
        else ""
    )
    cuda_result_file_sha = cuda_evidence[0]["sha256"]
    cuda_ok = bool(
        cuda_result
        and cuda_result.get("passed") is True
        and cuda_result.get("training_authorized") is False
        and cuda_result.get("family_id") == "vasu_140m_v1"
        and cuda_result.get("workload", {}).get("optimizer_updates") == 0
        and cuda_result.get("report_sha256") == _embedded_report_sha256(cuda_result)
        and isinstance(cuda_result_file_sha, str)
        and cuda_result_file_sha in cuda_decision_text
        and "Status: accepted; non-authorizing." in cuda_decision_text
        and "Accept the completed VASU-140M CUDA/AMP qualification execution."
        in cuda_decision_text
    )
    cuda_gate = _gate(
        "complete" if cuda_ok and all(item["exists"] for item in cuda_evidence) else "invalid",
        evidence=cuda_evidence,
        blockers=[] if cuda_ok and all(item["exists"] for item in cuda_evidence) else [
            "accepted finite no-update CUDA/AMP evidence is missing or inconsistent"
        ],
    )

    admission_evidence = [_evidence(root, path) for path in BLOCKED_ADMISSIONS]
    admission_payloads = [_load_object(root, path) for path in BLOCKED_ADMISSIONS]
    admission_fail_closed = all(
        payload
        and payload.get("source_admission_approved") is False
        and payload.get("release_build_permitted") is False
        and payload.get("training_authorized") is False
        for payload in admission_payloads
    )
    base_outputs_absent = all(not (root / path).exists() for path in PRODUCTION_BASE_PATHS)
    base_status = "blocked" if admission_fail_closed and base_outputs_absent else "invalid"
    base_gate = _gate(
        base_status,
        evidence=admission_evidence
        + [
            {"path": path.as_posix(), "exists": (root / path).exists(), "sha256": None}
            for path in PRODUCTION_BASE_PATHS
        ],
        blockers=[
            "independently curated production prompt inventory matrix",
            "independent semantic candidate search and review",
            "accepted source admissions and immutable base-pretraining release",
        ],
    )

    evaluation_evidence = [_evidence(root, path) for path in EVALUATION_FOUNDATION]
    evaluation_foundation_ok = all(item["exists"] for item in evaluation_evidence)
    evaluation_production_exists = (root / PRODUCTION_EVALUATION_PATH).exists()
    evaluation_gate = _gate(
        "blocked" if evaluation_foundation_ok and not evaluation_production_exists else "invalid",
        evidence=evaluation_evidence
        + [{
            "path": PRODUCTION_EVALUATION_PATH.as_posix(),
            "exists": evaluation_production_exists,
            "sha256": None,
        }],
        blockers=[
            "production inventories for all six dimensions",
            "independently curated and sealed held-out split",
            "independent contamination and semantic-review decisions",
            "frozen production suite identity",
        ],
    )

    resume_evidence = [_evidence(root, path) for path in RESUME_FOUNDATION]
    resume_foundation_ok = all(item["exists"] for item in resume_evidence)
    resume_result_exists = (root / REAL_RESUME_RESULT).is_file()
    resume_gate = _gate(
        "blocked" if resume_foundation_ok and not resume_result_exists else "invalid",
        evidence=resume_evidence + [_evidence(root, REAL_RESUME_RESULT)],
        blockers=[
            "accepted immutable base-pretraining release and source schedule",
            "accepted production evaluation development contract",
            "independently accepted execution implementation and identity",
            "separately authorized bounded real-data exact-resume execution",
        ],
    )

    plan_exists = (root / ACTUAL_PLAN_PATH).is_file()
    authorization_exists = (root / ACTUAL_AUTHORIZATION_PATH).is_file()
    plan_gate = _gate(
        "blocked" if not plan_exists else "invalid",
        evidence=[_evidence(root, ACTUAL_PLAN_PATH)],
        blockers=["all four readiness gates must be independently accepted first"],
    )
    authorization_gate = _gate(
        "blocked" if not authorization_exists else "invalid",
        evidence=[_evidence(root, ACTUAL_AUTHORIZATION_PATH)],
        blockers=[
            "accepted immutable experiment plan",
            "successful final operational preflight",
            "separate exact hash-bound human authorization",
        ],
    )

    gates = {
        "cuda_amp": cuda_gate,
        "base_data_release": base_gate,
        "base_evaluation_v2": evaluation_gate,
        "real_data_exact_resume": resume_gate,
        "scientific_experiment_plan": plan_gate,
        "training_authorization": authorization_gate,
    }
    report: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "family_id": "vasu_140m_v1",
        "gates": gates,
        "completed_gate_count": sum(
            gate["status"] == "complete" for gate in gates.values()
        ),
        "total_gate_count": len(gates),
        "eligible_for_experiment_plan": False,
        "eligible_for_authorization_review": False,
        "training_authorized": False,
    }
    report["report_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def validate_readiness_report(report: dict[str, object]) -> None:
    """Reject reports that overstate eligibility or have identity drift."""

    expected_fields = {
        "schema_id",
        "family_id",
        "gates",
        "completed_gate_count",
        "total_gate_count",
        "eligible_for_experiment_plan",
        "eligible_for_authorization_review",
        "training_authorized",
        "report_sha256",
    }
    if set(report) != expected_fields or report.get("schema_id") != SCHEMA_ID:
        raise ValueError("readiness report schema mismatch")
    gates = report.get("gates")
    if not isinstance(gates, dict) or len(gates) != 6:
        raise ValueError("readiness report must contain exactly six gates")
    completed = 0
    for name, gate in gates.items():
        if not isinstance(gate, dict) or set(gate) != {"status", "evidence", "blockers"}:
            raise ValueError(f"gate {name} schema mismatch")
        if gate["status"] not in {"complete", "blocked", "invalid"}:
            raise ValueError(f"gate {name} status mismatch")
        if gate["status"] == "complete":
            completed += 1
            if gate["blockers"]:
                raise ValueError(f"complete gate {name} retains blockers")
    if report["completed_gate_count"] != completed or report["total_gate_count"] != 6:
        raise ValueError("readiness gate counts mismatch")
    if any(
        report[field] is not False
        for field in (
            "eligible_for_experiment_plan",
            "eligible_for_authorization_review",
            "training_authorized",
        )
    ):
        raise ValueError("readiness report must remain non-authorizing")
    body = dict(report)
    observed_sha = body.pop("report_sha256", None)
    expected_sha = hashlib.sha256(_canonical_json(body)).hexdigest()
    if observed_sha != expected_sha:
        raise ValueError("readiness report identity mismatch")
