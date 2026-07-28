from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

import train_vasu_60m_capability_cpt as launcher
import vasu.training.capability_cpt as cpt
from vasu.data.scheduled_mixture import sha256_file
from vasu.training.capability_runtime import build_capability_identity, validation_configuration_hash


def _synthetic_d_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path, Path]:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "configs/training/capability_cpt_d_control_10m_from_a_v1.json"
    path.parent.mkdir(parents=True)
    config = json.loads(Path("D:/VASU/configs/training/capability_cpt_d_control_10m_from_a_v1.json").read_text())
    config["training_authorized"] = True
    path.write_text(json.dumps(config, separators=(",", ":")), encoding="utf-8")
    raw = path.read_text(encoding="utf-8")
    unauthorized = raw.replace('"training_authorized":true', '"training_authorized":false')
    record = {
        "format_version": cpt.AUTHORIZATION_FORMAT, "status": "approved", "decision": "authorized",
        "candidate_id": config["experiment_id"], "approver": "test", "authorized_at": "2026-07-27",
        "repository_commit": "test", "clean_tree_requirement": True,
        "experiment_config": {"path": "configs/training/capability_cpt_d_control_10m_from_a_v1.json", "preauthorization_sha256": cpt.sha256_file_from_text(unauthorized), "expected_authorized_sha256_after_single_boolean_edit": sha256_file(path)},
        "parent_checkpoint": config["parent_checkpoint"], "tokenizer": config["tokenizer"],
        "resolved_manifest": {"sha256": config["resolved_mixture_manifest_sha256"]},
        "schedule": {"sha256": config["expected_schedule_sha256"]},
        "validation_configuration": {"sha256": validation_configuration_hash(config)},
        "capability_runtime_identity": {"sha256": build_capability_identity(config)["sha256"]},
        "authorization_scope": {"authorized_candidate": config["experiment_id"], "candidate_a_authorized": False, "candidate_b_authorized": False, "authorized_token_budget": config["total_tokens"], "authorized_optimizer_updates": config["step_accounting"]["optimizer_updates"]},
    }
    auth = tmp_path / "configs/authorization/capability_cpt_d_control_10m_from_a_v1.authorization.json"
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(cpt, "AUTHORIZATION_SCOPES", MappingProxyType({config["experiment_id"]: cpt.AuthorizationScope(config["experiment_id"], record["experiment_config"]["path"], False)}))
    config["_authorization_config_path"] = str(path)
    return config, path, auth


def test_registry_is_immutable() -> None:
    with pytest.raises(TypeError):
        cpt.AUTHORIZATION_SCOPES["x"] = None  # type: ignore[index]


def test_unknown_and_candidate_b_fail_closed() -> None:
    for experiment_id in ("unknown", "capability_cpt_b_balanced_20m_v2"):
        with pytest.raises(PermissionError, match="unsupported"):
            cpt.require_training_authorization({"experiment_id": experiment_id, "training_authorized": True, "technical_gates": {"replay_safety": "passed", "cuda_smoke": "passed", "cuda_exact_resume": "passed"}, "production_runtime": {}})


def test_d_control_false_is_blocked() -> None:
    with pytest.raises(PermissionError, match="training_authorized is false"):
        cpt.require_training_authorization({"experiment_id": "capability_cpt_d_control_10m_from_a_v1", "training_authorized": False})


def test_synthetic_registered_control_passes_without_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, path, auth = _synthetic_d_control(tmp_path, monkeypatch)
    before = (path.read_bytes(), auth.read_bytes())
    cpt.require_training_authorization(config)
    assert before == (path.read_bytes(), auth.read_bytes())


@pytest.mark.parametrize("field", ["format_version", "approver", "authorized_at", "parent_checkpoint", "tokenizer", "resolved_manifest", "schedule", "validation_configuration", "capability_runtime_identity"])
def test_registered_control_rejects_bound_record_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    config, _, auth = _synthetic_d_control(tmp_path, monkeypatch)
    record = json.loads(auth.read_text())
    if field == "format_version":
        record[field] = "bad"
    elif field in {"approver", "authorized_at"}:
        record[field] = None
    else:
        record[field]["sha256"] = "0" * 64
    auth.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(PermissionError):
        cpt.require_training_authorization(config)


def test_extra_config_mutation_rejects_transition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, path, _ = _synthetic_d_control(tmp_path, monkeypatch)
    config["learning_rate"] = 2e-5
    path.write_text(json.dumps(config, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(PermissionError):
        cpt.require_training_authorization(config)


def test_real_a_and_c_legacy_authorizations_remain_valid() -> None:
    for name in ("capability_cpt_a_factual_20m_v2", "capability_cpt_c_control_20m_v2"):
        config = json.loads((Path("configs/training") / f"{name}.json").read_text())
        config["_authorization_config_path"] = f"configs/training/{name}.json"
        cpt.require_training_authorization(config)


def test_launch_path_preserves_registry_authorization_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "control.json"
    config_path.write_text("{}", encoding="utf-8")
    config = {"experiment_id": "capability_cpt_d_control_10m_from_a_v1", "training_authorized": True, "checkpoint_directory": str(tmp_path / "checkpoints")}
    monkeypatch.setattr(
        launcher,
        "validate_capability_config",
        lambda _: {"config": config, "resolved_manifest": {"schedule": {"sha256": "x"}}, "step_accounting": {}, "capability_identity": {}},
    )
    monkeypatch.setattr(launcher, "git_state", lambda: {"clean": True})
    monkeypatch.setattr(
        launcher,
        "require_training_authorization",
        lambda _: (_ for _ in ()).throw(PermissionError("registry-specific failure")),
    )
    monkeypatch.setattr("sys.argv", ["launcher", "--config", str(config_path)])
    with pytest.raises(SystemExit, match="registry-specific failure"):
        launcher.main()


def test_launch_path_never_relabels_registry_failure_as_boolean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "control.json"
    config_path.write_text("{}", encoding="utf-8")
    config = {"experiment_id": "capability_cpt_d_control_10m_from_a_v1", "training_authorized": True, "checkpoint_directory": str(tmp_path / "checkpoints")}
    monkeypatch.setattr(
        launcher,
        "validate_capability_config",
        lambda _: {"config": config, "resolved_manifest": {"schedule": {"sha256": "x"}}, "step_accounting": {}, "capability_identity": {}},
    )
    monkeypatch.setattr(launcher, "git_state", lambda: {"clean": True})
    monkeypatch.setattr(
        launcher,
        "require_training_authorization",
        lambda _: (_ for _ in ()).throw(PermissionError("authorization record is missing")),
    )
    monkeypatch.setattr("sys.argv", ["launcher", "--config", str(config_path)])
    with pytest.raises(SystemExit, match="authorization record is missing") as error:
        launcher.main()
    assert "training_authorized is false" not in str(error.value)
