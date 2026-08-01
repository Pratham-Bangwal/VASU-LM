"""Fail-closed, non-authorizing final preflight for VASU-140M base plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from vasu.training.vasu_140m_base_plan import (
    validate_base_pretraining_plan_files,
)


SCHEMA_ID = "vasu_140m_base_pretraining_final_preflight_v1"


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_identity(report: Mapping[str, object]) -> str:
    body = dict(report)
    body.pop("preflight_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _commit(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character Git commit")
    return value


def _repository_file(root: Path, relative: str, label: str) -> Path:
    path_value = Path(relative.replace("\\", "/"))
    if path_value.is_absolute() or ".." in path_value.parts:
        raise ValueError(f"{label} must be repository-relative")
    path = (root / path_value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} is missing or escapes the repository")
    return path


@dataclass(frozen=True)
class FinalPreflightObservation:
    """Runtime facts gathered before any model or optimizer construction."""

    worktree_clean: bool
    runtime_commit: str
    cuda_available: bool
    cuda_device_index: int
    amp_dtype: str
    free_disk_bytes: int
    telemetry_available: bool
    telemetry_provider: str
    temperature_celsius: float | None
    atomic_replace_supported: bool
    checkpoint_sidecar_supported: bool
    explicit_resume_enforced: bool


def _decision_is_accepted(
    text: str, *, plan_file_sha256: str, plan_identity_sha256: str
) -> bool:
    normalized = " ".join(text.casefold().split())
    markers = (
        "status: accepted",
        "decision: accept",
        "decision **accept**",
        "decision accept",
    )
    return (
        any(marker in normalized for marker in markers)
        and plan_file_sha256 in normalized
        and plan_identity_sha256 in normalized
    )


def build_final_preflight_report(
    *,
    plan_path: Path,
    plan_sha256: str,
    plan_decision_path: str,
    plan_decision_sha256: str,
    repository_root: Path,
    observation: FinalPreflightObservation,
) -> dict[str, object]:
    """Validate a complete plan package without constructing training state."""

    root = repository_root.resolve()
    resolved_plan = plan_path.resolve()
    if not resolved_plan.is_relative_to(root) or not resolved_plan.is_file():
        raise ValueError("plan path must be an existing repository file")
    expected_plan_sha = _sha(plan_sha256, "plan_sha256")
    observed_plan_sha = sha256_file(resolved_plan)
    if observed_plan_sha != expected_plan_sha:
        raise ValueError("plan file identity mismatch")
    plan = json.loads(resolved_plan.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("plan file must contain a JSON object")

    runtime_commit = _commit(observation.runtime_commit, "runtime_commit")
    validate_base_pretraining_plan_files(
        plan,
        root,
        runtime_commit=runtime_commit,
    )
    decision = _repository_file(root, plan_decision_path, "plan decision")
    expected_decision_sha = _sha(plan_decision_sha256, "plan_decision_sha256")
    observed_decision_sha = sha256_file(decision)
    if observed_decision_sha != expected_decision_sha:
        raise ValueError("plan decision identity mismatch")
    if not _decision_is_accepted(
        decision.read_text(encoding="utf-8"),
        plan_file_sha256=observed_plan_sha,
        plan_identity_sha256=plan["plan_sha256"],
    ):
        raise ValueError(
            "plan decision is not independently accepted and bound to the plan"
        )

    training = plan["training"]
    safety = plan["safety"]
    disk = safety["disk"]
    thermal = safety["thermal"]
    checks = {
        "clean_worktree": observation.worktree_clean is True,
        "runtime_commit_exact": runtime_commit == plan["repository_commit"],
        "plan_identity_exact": observed_plan_sha == expected_plan_sha,
        "plan_decision_exact": observed_decision_sha == expected_decision_sha,
        "plan_independently_accepted": True,
        "all_bound_files_exact": True,
        "outputs_absent": True,
        "cuda_available": observation.cuda_available is True,
        "cuda_device_valid": (
            isinstance(observation.cuda_device_index, int)
            and not isinstance(observation.cuda_device_index, bool)
            and observation.cuda_device_index >= 0
        ),
        "amp_dtype_exact": observation.amp_dtype == training["precision"],
        "disk_reserve_satisfied": (
            isinstance(observation.free_disk_bytes, int)
            and not isinstance(observation.free_disk_bytes, bool)
            and observation.free_disk_bytes >= disk["minimum_free_bytes"]
        ),
        "thermal_telemetry_available": (
            observation.telemetry_available is True
            and bool(observation.telemetry_provider.strip())
        ),
        "temperature_below_warning": (
            observation.temperature_celsius is not None
            and observation.temperature_celsius < thermal["warning_celsius"]
        ),
        "same_volume_atomic_replace": observation.atomic_replace_supported is True,
        "checkpoint_sidecars": observation.checkpoint_sidecar_supported is True,
        "explicit_resume": observation.explicit_resume_enforced is True,
        "training_authorized_false": plan["training_authorized"] is False,
    }
    eligible = all(checks.values())
    report: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "experiment_id": plan["experiment_id"],
        "repository_commit": runtime_commit,
        "plan": {
            "path": resolved_plan.relative_to(root).as_posix(),
            "sha256": observed_plan_sha,
            "plan_sha256": plan["plan_sha256"],
        },
        "plan_decision": {
            "path": decision.relative_to(root).as_posix(),
            "sha256": observed_decision_sha,
        },
        "observation": asdict(observation),
        "checks": checks,
        "eligible_for_authorization_review": eligible,
        "model_created": False,
        "optimizer_created": False,
        "checkpoint_created": False,
        "training_started": False,
        "authorization_created": False,
        "training_authorized": False,
    }
    report["preflight_sha256"] = report_identity(report)
    validate_final_preflight_report(report)
    return report


def validate_final_preflight_report(report: Mapping[str, object]) -> None:
    """Validate immutable eligibility evidence; failed reports remain valid evidence."""

    required = {
        "schema_id",
        "experiment_id",
        "repository_commit",
        "plan",
        "plan_decision",
        "observation",
        "checks",
        "eligible_for_authorization_review",
        "model_created",
        "optimizer_created",
        "checkpoint_created",
        "training_started",
        "authorization_created",
        "training_authorized",
        "preflight_sha256",
    }
    if set(report) != required:
        raise ValueError("final preflight report fields mismatch")
    if report["schema_id"] != SCHEMA_ID:
        raise ValueError("final preflight report schema mismatch")
    _commit(report["repository_commit"], "repository_commit")
    for label in ("plan", "plan_decision"):
        value = report[label]
        if not isinstance(value, Mapping) or "sha256" not in value:
            raise ValueError(f"{label} evidence is invalid")
        _sha(value["sha256"], f"{label}.sha256")
    checks = report["checks"]
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("final preflight checks are missing")
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ValueError("final preflight checks must be boolean")
    if report["eligible_for_authorization_review"] is not all(checks.values()):
        raise ValueError("final preflight eligibility does not match checks")
    for field in (
        "model_created",
        "optimizer_created",
        "checkpoint_created",
        "training_started",
        "authorization_created",
        "training_authorized",
    ):
        if report[field] is not False:
            raise ValueError(f"final preflight {field} must be false")
    reported = _sha(report["preflight_sha256"], "preflight_sha256")
    if reported != report_identity(report):
        raise ValueError("final preflight report identity mismatch")
