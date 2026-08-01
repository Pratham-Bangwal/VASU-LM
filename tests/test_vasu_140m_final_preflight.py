from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from test_vasu_140m_base_plan import COMMIT, plan
from test_vasu_140m_real_data_resume import valid_result
import scripts.preflight_vasu_140m_base_package as preflight_script
from vasu.training.vasu_140m_final_preflight import (
    FinalPreflightObservation,
    build_final_preflight_report,
    report_identity,
    sha256_file,
    validate_final_preflight_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _binding_paths(value: dict[str, object]) -> set[str]:
    paths = {value["tokenizer"]["path"]}
    for gate in value["gate_evidence"].values():
        paths.add(gate["artifact"]["path"])
        paths.add(gate["decision"]["path"])
    paths.add(value["initialization"]["implementation"]["path"])
    paths.add(value["data"]["release"]["path"])
    paths.add(value["data"]["schedule"]["path"])
    paths.add(value["training"]["optimizer"]["implementation"]["path"])
    paths.add(value["evaluation"]["suite"]["path"])
    return paths


def package(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    value = plan()
    for relative in _binding_paths(value):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plan_path = tmp_path / "configs/training/synthetic-plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(value), encoding="utf-8")
    decision_path = tmp_path / "docs/synthetic-plan-decision.md"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        "# Independent Plan Decision\n\n"
        "Status: accepted; non-authorizing.\n\n"
        f"Plan file SHA-256: {sha256_file(plan_path)}\n\n"
        f"Plan identity SHA-256: {value['plan_sha256']}\n",
        encoding="utf-8",
    )
    return plan_path, decision_path, value


def observation(**changes: object) -> FinalPreflightObservation:
    value = FinalPreflightObservation(
        worktree_clean=True,
        runtime_commit=COMMIT,
        cuda_available=True,
        cuda_device_index=0,
        amp_dtype="bf16",
        free_disk_bytes=20_000_000_000,
        telemetry_available=True,
        telemetry_provider="synthetic-test",
        temperature_celsius=50.0,
        atomic_replace_supported=True,
        checkpoint_sidecar_supported=True,
        explicit_resume_enforced=True,
    )
    return replace(value, **changes)


def run(tmp_path: Path, observed: FinalPreflightObservation | None = None):
    plan_path, decision_path, _ = package(tmp_path)
    return build_final_preflight_report(
        plan_path=plan_path,
        plan_sha256=sha256_file(plan_path),
        plan_decision_path=decision_path.relative_to(tmp_path).as_posix(),
        plan_decision_sha256=sha256_file(decision_path),
        repository_root=tmp_path,
        observation=observed or observation(),
    )


def test_complete_synthetic_package_is_eligible_only_for_authorization_review(
    tmp_path: Path,
) -> None:
    report = run(tmp_path)
    assert report["eligible_for_authorization_review"] is True
    assert all(report["checks"].values())
    assert report["model_created"] is False
    assert report["optimizer_created"] is False
    assert report["training_started"] is False
    assert report["training_authorized"] is False
    validate_final_preflight_report(report)


@pytest.mark.parametrize(
    ("changes", "failed_check"),
    [
        ({"worktree_clean": False}, "clean_worktree"),
        ({"cuda_available": False}, "cuda_available"),
        ({"amp_dtype": "fp16"}, "amp_dtype_exact"),
        ({"free_disk_bytes": 1}, "disk_reserve_satisfied"),
        ({"telemetry_available": False}, "thermal_telemetry_available"),
        ({"temperature_celsius": 83.0}, "temperature_below_warning"),
        ({"atomic_replace_supported": False}, "same_volume_atomic_replace"),
        ({"checkpoint_sidecar_supported": False}, "checkpoint_sidecars"),
        ({"explicit_resume_enforced": False}, "explicit_resume"),
    ],
)
def test_runtime_failures_remain_visible_and_non_authorizing(
    tmp_path: Path,
    changes: dict[str, object],
    failed_check: str,
) -> None:
    report = run(tmp_path, observation(**changes))
    assert report["eligible_for_authorization_review"] is False
    assert report["checks"][failed_check] is False
    assert report["training_authorized"] is False


def test_plan_and_decision_hashes_fail_closed(tmp_path: Path) -> None:
    plan_path, decision_path, _ = package(tmp_path)
    with pytest.raises(ValueError, match="plan file identity"):
        build_final_preflight_report(
            plan_path=plan_path,
            plan_sha256="0" * 64,
            plan_decision_path=decision_path.relative_to(tmp_path).as_posix(),
            plan_decision_sha256=sha256_file(decision_path),
            repository_root=tmp_path,
            observation=observation(),
        )
    with pytest.raises(ValueError, match="plan decision identity"):
        build_final_preflight_report(
            plan_path=plan_path,
            plan_sha256=sha256_file(plan_path),
            plan_decision_path=decision_path.relative_to(tmp_path).as_posix(),
            plan_decision_sha256="0" * 64,
            repository_root=tmp_path,
            observation=observation(),
        )


def test_unaccepted_decision_and_existing_output_fail_closed(tmp_path: Path) -> None:
    plan_path, decision_path, value = package(tmp_path)
    decision_path.write_text("Decision: reject\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not independently accepted"):
        build_final_preflight_report(
            plan_path=plan_path,
            plan_sha256=sha256_file(plan_path),
            plan_decision_path=decision_path.relative_to(tmp_path).as_posix(),
            plan_decision_sha256=sha256_file(decision_path),
            repository_root=tmp_path,
            observation=observation(),
        )
    decision_path.write_text(
        "Status: accepted\n"
        f"Plan file SHA-256: {sha256_file(plan_path)}\n"
        f"Plan identity SHA-256: {value['plan_sha256']}\n",
        encoding="utf-8",
    )
    output = tmp_path / value["outputs"]["checkpoint_directory"]
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="must be absent"):
        build_final_preflight_report(
            plan_path=plan_path,
            plan_sha256=sha256_file(plan_path),
            plan_decision_path=decision_path.relative_to(tmp_path).as_posix(),
            plan_decision_sha256=sha256_file(decision_path),
            repository_root=tmp_path,
            observation=observation(),
        )


def test_report_identity_and_authorization_flags_are_immutable(tmp_path: Path) -> None:
    report = run(tmp_path)
    report["optimizer_created"] = True
    report["preflight_sha256"] = report_identity(report)
    with pytest.raises(ValueError, match="optimizer_created must be false"):
        validate_final_preflight_report(report)


def test_unrelated_accepted_decision_cannot_be_substituted(tmp_path: Path) -> None:
    plan_path, decision_path, _ = package(tmp_path)
    decision_path.write_text(
        "Status: accepted\nPlan file SHA-256: " + "0" * 64 + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bound to the plan"):
        build_final_preflight_report(
            plan_path=plan_path,
            plan_sha256=sha256_file(plan_path),
            plan_decision_path=decision_path.relative_to(tmp_path).as_posix(),
            plan_decision_sha256=sha256_file(decision_path),
            repository_root=tmp_path,
            observation=observation(),
        )


def test_cli_derives_resume_capabilities_from_bound_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "evaluation/resume-result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(valid_result()), encoding="utf-8")
    value = {
        "gate_evidence": {
            "real_data_exact_resume": {
                "artifact": {
                    "path": "evaluation/resume-result.json",
                    "sha256": sha256_file(result_path),
                }
            }
        }
    }
    monkeypatch.setattr(preflight_script, "REPOSITORY_ROOT", tmp_path)
    assert preflight_script._resume_capabilities(value) == (True, True)

    value["gate_evidence"]["real_data_exact_resume"]["artifact"]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        preflight_script._resume_capabilities(value)
