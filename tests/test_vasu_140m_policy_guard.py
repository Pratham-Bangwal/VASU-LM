from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from vasu.training.vasu_140m_policy_guard import (
    ACTUAL_AUTHORIZATION_PATH,
    ACTUAL_PLAN_PATH,
    FROZEN_READINESS,
    build_policy_report,
    validate_policy_report,
)
from vasu.training.vasu_140m_readiness import (
    BLOCKED_ADMISSIONS,
    CUDA_DECISION,
    CUDA_RESULT,
    EVALUATION_FOUNDATION,
    RESUME_FOUNDATION,
)


ROOT = Path(__file__).resolve().parents[1]


def _copy_guard_fixture(tmp_path: Path) -> Path:
    for relative in (
        CUDA_RESULT,
        CUDA_DECISION,
        *BLOCKED_ADMISSIONS,
        *EVALUATION_FOUNDATION,
        *RESUME_FOUNDATION,
        FROZEN_READINESS,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return tmp_path


def test_current_repository_passes_frozen_policy_guard() -> None:
    report = build_policy_report(ROOT)
    validate_policy_report(report)
    assert report["passed"] is True
    assert report["violations"] == []
    assert report["training_authorized"] is False


@pytest.mark.parametrize(
    "relative",
    [
        Path("data/processed/vasu_140m/base_pretraining/train.tokens.bin"),
        Path("data/manifests/vasu_140m/base_pretraining/v1.json"),
        Path("checkpoints/vasu_140m/base_pretraining/step_1.pt"),
        ACTUAL_PLAN_PATH,
        ACTUAL_AUTHORIZATION_PATH,
    ],
)
def test_protected_artifact_presence_fails_closed(
    tmp_path: Path, relative: Path
) -> None:
    root = _copy_guard_fixture(tmp_path)
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"unauthorized")
    report = build_policy_report(root)
    assert report["passed"] is False
    assert any(
        violation["code"] == "protected_artifact_present"
        and violation["path"] in relative.as_posix()
        for violation in report["violations"]
    )


def test_authorization_mutation_is_rejected() -> None:
    report = build_policy_report(ROOT)
    changed = copy.deepcopy(report)
    changed["training_authorized"] = True
    with pytest.raises(ValueError, match="non-authorizing"):
        validate_policy_report(changed)


def test_policy_report_identity_detects_mutation() -> None:
    report = build_policy_report(ROOT)
    changed = copy.deepcopy(report)
    changed["checked_protected_paths"].append("mutated")
    with pytest.raises(ValueError, match="identity"):
        validate_policy_report(changed)
