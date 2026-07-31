from __future__ import annotations

from pathlib import Path

import pytest

from vasu.training.vasu_140m_base_plan import (
    FAMILY_ID,
    FAMILY_SHA256,
    MODEL_CONFIG_SHA256,
    REQUIRED_DIMENSIONS,
    SCHEMA_ID,
    plan_identity,
    sha256_file,
    validate_base_pretraining_plan,
    validate_base_pretraining_plan_files,
)


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def binding(path: str) -> dict[str, str]:
    return {"path": path, "sha256": sha256_file(ROOT / path)}


def plan() -> dict[str, object]:
    cuda_artifact = "evaluation/results/vasu_140m_cuda_amp_qualification_20260731.json"
    cuda_decision = (
        "docs/VASU_140M_CUDA_AMP_QUALIFICATION_EXECUTION_"
        "INDEPENDENT_REVIEW_DECISION_20260731.md"
    )
    base_artifact = "evaluation/fixtures/vasu_140m_base_text_record_fixture_v1.json"
    base_decision = (
        "docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN_"
        "INDEPENDENT_REVIEW_DECISION_20260731.md"
    )
    evaluation_artifact = (
        "evaluation/fixtures/vasu_140m_base_evaluation_v2_schema_"
        "qualification_precommit.json"
    )
    evaluation_decision = (
        "docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_"
        "INDEPENDENT_REVIEW_DECISION_20260801.md"
    )
    resume_artifact = (
        "evaluation/results/vasu_140m_exact_resume_qualification_v2_20260730.json"
    )
    resume_decision = "docs/VASU_140M_EXACT_RESUME_QUALIFICATION_20260730.md"
    schedule_artifact = "evaluation/fixtures/vasu_140m_513_record_spec_v1.json"
    sources = [
        {
            "source_id": "synthetic-source-a",
            "manifest_sha256": "0" * 64,
            "token_sha256": "1" * 64,
            "mask_sha256": "2" * 64,
            "lineage_sha256": "3" * 64,
            "train_record_count": 16,
            "supervised_target_count": 8_000,
        },
        {
            "source_id": "synthetic-source-b",
            "manifest_sha256": "7" * 64,
            "token_sha256": "4" * 64,
            "mask_sha256": "5" * 64,
            "lineage_sha256": "6" * 64,
            "train_record_count": 16,
            "supervised_target_count": 8_000,
        },
    ]
    value: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "experiment_id": "vasu-140m-base-pretraining-synthetic-plan",
        "repository_commit": COMMIT,
        "family_id": FAMILY_ID,
        "family_sha256": FAMILY_SHA256,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer": binding("assets/tokenizer.json"),
        "gate_evidence": {
            "cuda_amp": {
                "artifact": binding(cuda_artifact),
                "decision": binding(cuda_decision),
                "status": "accepted",
            },
            "base_data_release": {
                "artifact": binding(base_artifact),
                "decision": binding(base_decision),
                "status": "accepted",
            },
            "base_evaluation_v2": {
                "artifact": binding(evaluation_artifact),
                "decision": binding(evaluation_decision),
                "status": "accepted",
            },
            "real_data_exact_resume": {
                "artifact": binding(resume_artifact),
                "decision": binding(resume_decision),
                "status": "accepted",
            },
        },
        "hypothesis": {
            "question": "Can the frozen family learn a reproducible base-text objective?",
            "prediction": "Development likelihood improves without failed safeguards.",
            "falsifier": "Loss is non-finite or frozen rejection criteria trigger.",
            "comparison": {
                "kind": "pre_update_random_initialization",
                "reference": "same initialized model before the first optimizer update",
                "scientific_role": "within-lineage pre-update calibration",
                "matched_parent": False,
            },
        },
        "initialization": {
            "kind": "fresh_random",
            "parent_checkpoint": None,
            "seed": 140,
            "implementation": binding("vasu/model/model.py"),
        },
        "data": {
            "release": binding(base_artifact),
            "schedule": {
                **binding(schedule_artifact),
                "record_count": 32,
            },
            "sources": sources,
            "no_replacement": True,
            "record_width": 513,
            "sequence_length": 512,
        },
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "microbatch_count": 32,
            "optimizer_updates": 4,
            "processed_records": 32,
            "processed_input_positions": 16_384,
            "supervised_target_budget": 16_000,
            "precision": "bf16",
            "seed": 140,
            "gradient_clip": 1.0,
            "optimizer": {
                "name": "adamw",
                "learning_rate": 0.0003,
                "betas": [0.9, 0.95],
                "epsilon": 1e-8,
                "weight_decay": 0.1,
                "implementation": binding("vasu/training/trainer.py"),
            },
            "scheduler": {
                "name": "cosine",
                "total_updates": 4,
                "warmup_updates": 1,
                "minimum_learning_rate": 0.00003,
            },
            "validation_interval_updates": 1,
            "checkpoint_interval_updates": 2,
            "exact_resume": True,
            "final_validation": True,
            "final_checkpoint": True,
        },
        "runtime_envelope": {
            "minimum_input_positions_per_second": 1_000,
            "maximum_input_positions_per_second": 100_000,
            "maximum_wall_time_seconds": 60,
            "maximum_optimizer_updates": 4,
            "maximum_input_positions": 16_384,
            "stop_at_budget": True,
        },
        "evaluation": {
            "suite": binding(evaluation_artifact),
            "development_dimensions": sorted(REQUIRED_DIMENSIONS),
            "held_out_dimensions": sorted(REQUIRED_DIMENSIONS),
            "development_before_training": True,
            "held_out_before_training": False,
            "held_out_opening_point": "after final checkpoint selection",
            "promotion_rules": [
                {
                    "metric": "development_loss",
                    "scope": "all_sources",
                    "operator": "<",
                    "threshold": 10.0,
                }
            ],
            "rejection_rules": [
                {
                    "metric": "nonfinite_updates",
                    "scope": "training",
                    "operator": ">",
                    "threshold": 0,
                }
            ],
        },
        "safety": {
            "disk": {
                "minimum_free_bytes": 10_000_000_000,
                "before_launch": True,
                "before_checkpoint": True,
            },
            "thermal": {
                "required": True,
                "warning_celsius": 82,
                "abort_celsius": 87,
                "critical_celsius": 90,
                "consecutive_abort_readings": 2,
            },
            "checkpoint": {
                "atomic": True,
                "retention_count": 2,
                "integrity_sidecars": True,
            },
            "abort_on": [
                "checkpoint_failure",
                "cuda_oom",
                "disk_limit",
                "nonfinite",
                "resume_divergence",
                "thermal_limit",
                "validation_failure",
            ],
        },
        "outputs": {
            "checkpoint_directory": "checkpoints/vasu_140m/synthetic-plan",
            "log_directory": "logs/vasu_140m/synthetic-plan",
            "result_directory": "evaluation/results/vasu_140m/synthetic-plan",
            "must_be_absent": True,
        },
        "review": {
            "status": "pending_independent_review",
            "reviewer": "",
            "decision_path": None,
        },
        "training_authorized": False,
        "plan_sha256": "0" * 64,
    }
    value["plan_sha256"] = plan_identity(value)
    return value


def rehash(value: dict[str, object]) -> None:
    value["plan_sha256"] = plan_identity(value)


def test_valid_plan_is_review_only_and_file_bound() -> None:
    value = plan()
    validate_base_pretraining_plan(value)
    validate_base_pretraining_plan_files(value, ROOT, runtime_commit=COMMIT)
    assert value["training_authorized"] is False
    assert value["review"]["status"] == "pending_independent_review"


@pytest.mark.parametrize("field", ["family_id", "family_sha256", "model_config_sha256"])
def test_family_identity_mutations_fail(field: str) -> None:
    value = plan()
    value[field] = "0" * 64
    rehash(value)
    with pytest.raises(ValueError, match="identity mismatch|fingerprint|configuration"):
        validate_base_pretraining_plan(value)


def test_all_four_gates_must_be_independently_accepted() -> None:
    value = plan()
    del value["gate_evidence"]["real_data_exact_resume"]
    rehash(value)
    with pytest.raises(ValueError, match="exactly the four"):
        validate_base_pretraining_plan(value)
    value = plan()
    value["gate_evidence"]["base_data_release"]["status"] = "pending"
    rehash(value)
    with pytest.raises(ValueError, match="independently accepted"):
        validate_base_pretraining_plan(value)


def test_gate_paths_and_selected_artifacts_cannot_be_substituted() -> None:
    value = plan()
    value["gate_evidence"]["cuda_amp"]["decision"] = value["gate_evidence"]["cuda_amp"][
        "artifact"
    ]
    rehash(value)
    with pytest.raises(ValueError, match="artifact and decision must be distinct"):
        validate_base_pretraining_plan(value)

    value = plan()
    artifact = value["gate_evidence"]["cuda_amp"]["artifact"]
    value["gate_evidence"]["cuda_amp"]["decision"] = {
        "path": artifact["path"].upper(),
        "sha256": artifact["sha256"],
    }
    rehash(value)
    with pytest.raises(ValueError, match="artifact and decision must be distinct"):
        validate_base_pretraining_plan(value)

    value = plan()
    value["gate_evidence"]["base_data_release"]["artifact"] = value["gate_evidence"][
        "cuda_amp"
    ]["decision"]
    rehash(value)
    with pytest.raises(ValueError, match="paths must be unique"):
        validate_base_pretraining_plan(value)

    value = plan()
    value["data"]["release"] = binding("assets/tokenizer.json")
    rehash(value)
    with pytest.raises(ValueError, match="selected data release"):
        validate_base_pretraining_plan(value)

    value = plan()
    value["evaluation"]["suite"] = binding("assets/tokenizer.json")
    rehash(value)
    with pytest.raises(ValueError, match="selected evaluation suite"):
        validate_base_pretraining_plan(value)

    value = plan()
    value["data"]["schedule"] = {
        **value["data"]["release"],
        "record_count": 32,
    }
    rehash(value)
    with pytest.raises(ValueError, match="schedule must be distinct"):
        validate_base_pretraining_plan(value)


def test_first_lineage_requires_fresh_initialization() -> None:
    value = plan()
    value["initialization"]["kind"] = "checkpoint"
    value["initialization"]["parent_checkpoint"] = "candidate-a.pt"
    rehash(value)
    with pytest.raises(ValueError, match="fresh random"):
        validate_base_pretraining_plan(value)


def test_first_lineage_requires_an_explicit_honest_comparison() -> None:
    value = plan()
    value["hypothesis"]["comparison"]["matched_parent"] = True
    rehash(value)
    with pytest.raises(ValueError, match="no matched trained parent"):
        validate_base_pretraining_plan(value)

    value = plan()
    value["hypothesis"]["comparison"]["kind"] = "candidate_a"
    rehash(value)
    with pytest.raises(ValueError, match="pre-update model"):
        validate_base_pretraining_plan(value)


def test_source_schedule_and_supervised_budget_must_reconcile() -> None:
    value = plan()
    value["data"]["schedule"]["record_count"] = 31
    rehash(value)
    with pytest.raises(ValueError, match="schedule and source"):
        validate_base_pretraining_plan(value)
    value = plan()
    value["training"]["supervised_target_budget"] += 1
    rehash(value)
    with pytest.raises(ValueError, match="supervised-target"):
        validate_base_pretraining_plan(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("microbatch_count", 31, "microbatch accounting"),
        ("optimizer_updates", 3, "optimizer-update accounting"),
        ("processed_input_positions", 16_383, "input-position budget"),
    ],
)
def test_step_accounting_is_exact(field: str, value: int, message: str) -> None:
    candidate = plan()
    candidate["training"][field] = value
    if field == "optimizer_updates":
        candidate["training"]["scheduler"]["total_updates"] = value
    rehash(candidate)
    with pytest.raises(ValueError, match=message):
        validate_base_pretraining_plan(candidate)


def test_optimizer_scheduler_and_resume_are_frozen() -> None:
    value = plan()
    value["training"]["optimizer"]["name"] = "sgd"
    rehash(value)
    with pytest.raises(ValueError, match="optimizer must be adamw"):
        validate_base_pretraining_plan(value)
    value = plan()
    value["training"]["scheduler"]["warmup_updates"] = 4
    rehash(value)
    with pytest.raises(ValueError, match="warmup"):
        validate_base_pretraining_plan(value)
    value = plan()
    value["training"]["exact_resume"] = False
    rehash(value)
    with pytest.raises(ValueError, match="exact resume"):
        validate_base_pretraining_plan(value)

    value = plan()
    value["training"]["validation_interval_updates"] = 5
    rehash(value)
    with pytest.raises(ValueError, match="validation interval"):
        validate_base_pretraining_plan(value)

    value = plan()
    value["training"]["final_checkpoint"] = False
    rehash(value)
    with pytest.raises(ValueError, match="final validation and checkpoint"):
        validate_base_pretraining_plan(value)


def test_evaluation_dimensions_and_held_out_policy_fail_closed() -> None:
    value = plan()
    value["evaluation"]["held_out_dimensions"].pop()
    rehash(value)
    with pytest.raises(ValueError, match="every dimension"):
        validate_base_pretraining_plan(value)
    value = plan()
    value["evaluation"]["held_out_before_training"] = True
    rehash(value)
    with pytest.raises(ValueError, match="split-opening"):
        validate_base_pretraining_plan(value)


def test_safety_outputs_review_and_authorization_fail_closed() -> None:
    value = plan()
    value["safety"]["thermal"]["abort_celsius"] = 92
    rehash(value)
    with pytest.raises(ValueError, match="ordered thresholds"):
        validate_base_pretraining_plan(value)
    value = plan()
    value["safety"]["disk"]["minimum_free_bytes"] = 1_000_000
    rehash(value)
    with pytest.raises(ValueError, match="at least 10 GB"):
        validate_base_pretraining_plan(value)
    value = plan()
    value["outputs"]["checkpoint_directory"] = "tmp/synthetic-plan/checkpoints"
    rehash(value)
    with pytest.raises(ValueError, match="isolated root"):
        validate_base_pretraining_plan(value)
    value = plan()
    value["review"]["status"] = "accepted"
    rehash(value)
    with pytest.raises(ValueError, match="pending"):
        validate_base_pretraining_plan(value)
    value = plan()
    value["training_authorized"] = True
    rehash(value)
    with pytest.raises(ValueError, match="must be false"):
        validate_base_pretraining_plan(value)


def test_runtime_envelope_is_bounded_and_matches_training() -> None:
    value = plan()
    value["runtime_envelope"]["maximum_optimizer_updates"] = 5
    rehash(value)
    with pytest.raises(ValueError, match="optimizer-update cap"):
        validate_base_pretraining_plan(value)

    value = plan()
    value["runtime_envelope"]["maximum_wall_time_seconds"] = 1
    rehash(value)
    with pytest.raises(ValueError, match="slow-bound estimate"):
        validate_base_pretraining_plan(value)

    value = plan()
    value["runtime_envelope"]["maximum_input_positions_per_second"] = 100
    rehash(value)
    with pytest.raises(ValueError, match="throughput envelope is reversed"):
        validate_base_pretraining_plan(value)


def test_plan_identity_runtime_commit_and_file_hash_fail_closed() -> None:
    value = plan()
    value["hypothesis"]["prediction"] = "changed"
    with pytest.raises(ValueError, match="plan identity"):
        validate_base_pretraining_plan(value)
    value = plan()
    with pytest.raises(ValueError, match="runtime commit"):
        validate_base_pretraining_plan_files(value, ROOT, runtime_commit="b" * 40)
    value = plan()
    value["tokenizer"]["sha256"] = "0" * 64
    rehash(value)
    with pytest.raises(ValueError, match="tokenizer identity"):
        validate_base_pretraining_plan_files(value, ROOT, runtime_commit=COMMIT)
