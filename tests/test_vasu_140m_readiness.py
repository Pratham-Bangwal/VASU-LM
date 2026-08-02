from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from vasu.training.vasu_140m_readiness import (
    BLOCKED_ADMISSIONS,
    CUDA_DECISION,
    CUDA_RESULT,
    EVALUATION_FOUNDATION,
    RESUME_FOUNDATION,
    build_readiness_report,
    validate_readiness_report,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_REPORT = Path(
    "evaluation/fixtures/vasu_140m_base_pretraining_readiness_20260803.json"
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


def _report_sha(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _fixture_repository(tmp_path: Path) -> Path:
    cuda_result = {
        "passed": True,
        "training_authorized": False,
        "family_id": "vasu_140m_v1",
        "workload": {"optimizer_updates": 0},
    }
    cuda_result["report_sha256"] = _report_sha(cuda_result)
    _write(
        tmp_path / CUDA_RESULT,
        cuda_result,
    )
    result_file_sha = hashlib.sha256((tmp_path / CUDA_RESULT).read_bytes()).hexdigest()
    _write(
        tmp_path / CUDA_DECISION,
        "Status: accepted; non-authorizing.\n\n"
        "Accept the completed VASU-140M CUDA/AMP qualification execution.\n\n"
        f"Result file SHA-256: {result_file_sha}\n",
    )
    for path in BLOCKED_ADMISSIONS:
        _write(
            tmp_path / path,
            {
                "source_admission_approved": False,
                "release_build_permitted": False,
                "training_authorized": False,
            },
        )
    for path in (*EVALUATION_FOUNDATION, *RESUME_FOUNDATION):
        _write(tmp_path / path, {"fixture": True})
    return tmp_path


def test_current_readiness_is_fail_closed(tmp_path: Path) -> None:
    report = build_readiness_report(_fixture_repository(tmp_path))
    validate_readiness_report(report)
    assert report["completed_gate_count"] == 1
    assert report["gates"]["cuda_amp"]["status"] == "complete"
    assert all(
        report["gates"][name]["status"] == "blocked"
        for name in report["gates"]
        if name != "cuda_amp"
    )
    assert report["training_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(training_authorized=True),
        lambda value: value.update(eligible_for_experiment_plan=True),
        lambda value: value["gates"]["base_data_release"].update(status="complete"),
        lambda value: value["gates"].pop("real_data_exact_resume"),
    ],
)
def test_overstated_or_incomplete_reports_are_rejected(tmp_path: Path, mutation) -> None:
    report = build_readiness_report(_fixture_repository(tmp_path))
    mutation(report)
    with pytest.raises(ValueError):
        validate_readiness_report(report)


def test_contradictory_admission_evidence_is_invalid(tmp_path: Path) -> None:
    root = _fixture_repository(tmp_path)
    payload = json.loads((root / BLOCKED_ADMISSIONS[0]).read_text(encoding="utf-8"))
    payload["source_admission_approved"] = True
    _write(root / BLOCKED_ADMISSIONS[0], payload)
    report = build_readiness_report(root)
    assert report["gates"]["base_data_release"]["status"] == "invalid"
    assert report["eligible_for_experiment_plan"] is False


def test_missing_cuda_decision_cannot_complete_gate(tmp_path: Path) -> None:
    root = _fixture_repository(tmp_path)
    (root / CUDA_DECISION).unlink()
    report = build_readiness_report(root)
    assert report["gates"]["cuda_amp"]["status"] == "invalid"


def test_cuda_decision_must_bind_exact_result_bytes(tmp_path: Path) -> None:
    root = _fixture_repository(tmp_path)
    (root / CUDA_DECISION).write_text(
        "Status: accepted; non-authorizing.\n\n"
        "Accept the completed VASU-140M CUDA/AMP qualification execution.\n\n"
        f"Result file SHA-256: {'0' * 64}\n",
        encoding="utf-8",
    )
    report = build_readiness_report(root)
    assert report["gates"]["cuda_amp"]["status"] == "invalid"


def test_report_identity_detects_mutation(tmp_path: Path) -> None:
    report = build_readiness_report(_fixture_repository(tmp_path))
    changed = copy.deepcopy(report)
    changed["gates"]["base_data_release"]["blockers"].append("mutated")
    with pytest.raises(ValueError, match="identity"):
        validate_readiness_report(changed)


def test_repository_readiness_matches_frozen_evidence() -> None:
    observed = build_readiness_report(REPOSITORY_ROOT)
    frozen = json.loads((REPOSITORY_ROOT / FROZEN_REPORT).read_text(encoding="utf-8"))
    assert observed == frozen
    validate_readiness_report(frozen)
