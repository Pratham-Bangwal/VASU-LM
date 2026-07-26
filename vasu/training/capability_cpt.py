"""Validation and authorization gate for scheduled capability-CPT configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from vasu.data.scheduled_mixture import (
    ScheduledPretrainingDataset,
    sha256_file,
    validate_schedule_release,
)
from vasu.training.accumulation import build_accumulation_plan
from vasu.training.capability_runtime import (
    CAPABILITY_RUNTIME_VERSION,
    build_capability_identity,
    sha256_file as runtime_sha256_file,
)


CONFIG_FORMAT = "vasu_capability_cpt_experiment_v1"
BLOCKED_MESSAGE = "Training is blocked because training_authorized is false."


@dataclass(frozen=True)
class CapabilityStepAccounting:
    records: int
    dataloader_microbatches: int
    records_per_microbatch: int
    tokens_per_microbatch: int
    accumulation_steps: int
    optimizer_updates: int
    final_group_complete: bool
    processed_tokens: int
    warmup_updates: int
    scheduler_total_steps: int


def calculate_step_accounting(
    *,
    records: int,
    batch_size: int,
    accumulation_steps: int,
    sequence_length: int,
    warmup_fraction: float,
) -> CapabilityStepAccounting:
    plan = build_accumulation_plan(
        record_count=records,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        sequence_length=sequence_length,
    )
    warmup = math.ceil(plan.optimizer_steps * warmup_fraction)
    return CapabilityStepAccounting(
        records=records,
        dataloader_microbatches=plan.microbatches,
        records_per_microbatch=batch_size,
        tokens_per_microbatch=batch_size * sequence_length,
        accumulation_steps=accumulation_steps,
        optimizer_updates=plan.optimizer_steps,
        final_group_complete=(
            plan.final_microbatches == accumulation_steps
            and plan.final_records == batch_size * accumulation_steps
        ),
        processed_tokens=plan.trained_tokens,
        warmup_updates=warmup,
        scheduler_total_steps=plan.optimizer_steps,
    )


def load_capability_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != CONFIG_FORMAT:
        raise ValueError("unsupported capability-CPT configuration")
    if not isinstance(payload.get("training_authorized"), bool):
        raise ValueError("training_authorized must be boolean")
    required = {
        "model_configuration": "vasu_60m",
        "sequence_length": 256,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-5,
        "optimizer_backend": "standard",
        "weight_decay": 0.1,
        "seed": 42,
        "exact_resume": True,
        "shuffle": False,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise ValueError(f"{field} must remain {expected!r}")
    return payload


def validate_capability_config(path: Path) -> dict[str, Any]:
    config = load_capability_config(path)
    resolved_path = Path(config["resolved_mixture_manifest"])
    if sha256_file(resolved_path) != config["resolved_mixture_manifest_sha256"]:
        raise ValueError("resolved mixture manifest hash mismatch")
    manifest = validate_schedule_release(resolved_path.parent)
    if manifest["candidate_id"] != config["experiment_id"]:
        raise ValueError("candidate config and resolved manifest IDs differ")
    if manifest["schedule"]["sha256"] != config["expected_schedule_sha256"]:
        raise ValueError("candidate schedule SHA-256 mismatch")
    expected_sources = {
        item["identifier"]: {
            "token_sha256": item["token_sha256"],
            "mask_sha256": item["mask_sha256"],
            "manifest_sha256": item["manifest_sha256"],
        }
        for item in manifest["sources"]
    }
    if config["expected_source_hashes"] != expected_sources:
        raise ValueError("candidate source hashes differ from resolved manifest")
    if config["tokenizer"]["sha256"] != manifest["tokenizer"]["sha256"]:
        raise ValueError("candidate tokenizer hash mismatch")
    if config["total_records"] != manifest["total_records"]:
        raise ValueError("candidate record count mismatch")
    if config["total_tokens"] != manifest["total_tokens"]:
        raise ValueError("candidate token count mismatch")

    accounting = calculate_step_accounting(
        records=config["total_records"],
        batch_size=config["batch_size"],
        accumulation_steps=config["gradient_accumulation_steps"],
        sequence_length=config["sequence_length"],
        warmup_fraction=config["warmup_fraction"],
    )
    expected_accounting = config["step_accounting"]
    if expected_accounting != asdict(accounting):
        raise ValueError("candidate optimizer/token accounting is inconsistent")
    if config["scheduler"]["total_steps"] != accounting.scheduler_total_steps:
        raise ValueError("scheduler total steps are inconsistent")
    if config["scheduler"]["warmup_steps"] != accounting.warmup_updates:
        raise ValueError("scheduler warmup steps are inconsistent")
    if config["validation_sources"] != manifest["validation_sources"]:
        raise ValueError("candidate validation references changed")
    required_replay = config.get("replay_safety")
    if required_replay is not None:
        if manifest.get("replay_safety") != required_replay:
            raise ValueError("candidate replay-safety configuration changed")
        failed = [
            item["source_id"]
            for item in manifest["allocations"]
            if item.get("replay_safety_status") not in {"pass", "override"}
        ]
        if failed:
            raise ValueError(
                "candidate replay-safety gate failed: " + ", ".join(failed)
            )
    gates = config.get("technical_gates", {})
    if gates:
        for name in ("replay_safety", "cuda_smoke", "cuda_exact_resume"):
            if gates.get(name) not in {"passed", "not_completed", "failed"}:
                raise ValueError(f"invalid technical gate status: {name}")
        artifact = config.get("technical_validation_artifact", {})
        artifact_path = Path(artifact.get("path", ""))
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError("technical validation artifact identity mismatch")
    parent = config.get("parent_checkpoint", {})
    parent_path = Path(parent.get("path", ""))
    if not parent_path.is_file() or sha256_file(parent_path) != parent.get(
        "sha256"
    ):
        raise ValueError("candidate parent checkpoint identity mismatch")

    runtime = config.get("production_runtime")
    capability_identity = None
    if runtime is not None:
        if runtime.get("format_version") != CAPABILITY_RUNTIME_VERSION:
            raise ValueError("unsupported capability production runtime")
        if runtime.get("explicit_resume_only") is not True:
            raise ValueError("capability runtime requires explicit resume selection")
        if runtime.get("require_clean_git_for_authorized_launch") is not True:
            raise ValueError("authorized capability runs must require a clean Git tree")
        validation = config.get("validation")
        if not isinstance(validation, dict):
            raise ValueError("production capability runtime requires validation config")
        wiki = validation.get("wikimedia", {})
        wiki_path = Path(wiki.get("path", ""))
        if (
            not wiki_path.is_file()
            or runtime_sha256_file(wiki_path) != wiki.get("sha256")
        ):
            raise ValueError("Wikimedia validation identity mismatch")
        arithmetic = validation.get("arithmetic_proxy", {})
        arithmetic_manifest = Path(arithmetic.get("manifest", ""))
        if (
            not arithmetic_manifest.is_file()
            or runtime_sha256_file(arithmetic_manifest)
            != arithmetic.get("manifest_sha256")
        ):
            raise ValueError("arithmetic proxy manifest identity mismatch")
        if arithmetic.get("split") != "dev":
            raise ValueError("interval arithmetic validation must use development")
        if arithmetic.get("do_sample") is not False:
            raise ValueError("arithmetic proxy validation must be deterministic")
        if not 1 <= int(arithmetic.get("record_count", 0)) <= 1000:
            raise ValueError("arithmetic proxy record_count is invalid")
        if config.get("best_checkpoint_policy", {}).get("mixed_score") is not False:
            raise ValueError("opaque mixed best-checkpoint score is forbidden")
        retention = config.get("checkpoint_retention", {})
        if int(retention.get("periodic_keep", 0)) < 2:
            raise ValueError("checkpoint retention must keep two periodic files")
        thermal = config.get("thermal_safety", {})
        if thermal.get("required_for_authorized_run") is not True:
            raise ValueError("authorized capability runtime requires thermal monitoring")
        if config.get("disk_safety", {}).get("check_before_every_checkpoint") is not True:
            raise ValueError("runtime disk check must run before every checkpoint")
        capability_identity = build_capability_identity(config)

    dataset = ScheduledPretrainingDataset.from_resolved_manifest(resolved_path)
    if len(dataset) != config["total_records"]:
        raise ValueError("scheduled dataset length differs from candidate config")
    return {
        "config": config,
        "resolved_manifest": manifest,
        "step_accounting": asdict(accounting),
        "dataset": dataset,
        "capability_identity": capability_identity,
    }


def require_training_authorization(config: dict[str, Any]) -> None:
    if config.get("training_authorized") is not True:
        raise PermissionError(BLOCKED_MESSAGE)
    gates = config.get("technical_gates", {})
    incomplete = [
        name
        for name in ("replay_safety", "cuda_smoke", "cuda_exact_resume")
        if gates.get(name) != "passed"
    ]
    if incomplete:
        raise PermissionError(
            "Training is blocked because technical gates are incomplete: "
            + ", ".join(incomplete)
        )
    if config.get("production_runtime") is None:
        raise PermissionError(
            "Training is blocked because the production runtime is not configured."
        )

    config_path_value = config.get("_authorization_config_path")
    if not config_path_value:
        raise PermissionError(
            "Training is blocked because the authorization-bound config path "
            "was not provided."
        )
    config_path = Path(config_path_value)
    authorization_path = Path("configs/authorization") / (
        f"{config['experiment_id']}.authorization.json"
    )
    if not authorization_path.is_file():
        raise PermissionError(
            "Training is blocked because the authorization record is missing: "
            f"{authorization_path}"
        )
    try:
        record = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError(
            "Training is blocked because the authorization record is invalid."
        ) from error
    if record.get("status") != "approved" or record.get("decision") != "authorized":
        raise PermissionError(
            "Training is blocked because the authorization record is not approved."
        )
    if record.get("candidate_id") != config.get("experiment_id"):
        raise PermissionError("Training is blocked because the authorization candidate differs.")
    experiment_config = record.get("experiment_config", {})
    expected_path = str(experiment_config.get("path", "")).replace("\\", "/")
    try:
        actual_path = config_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        actual_path = config_path.as_posix()
    if expected_path != actual_path:
        raise PermissionError("Training is blocked because the authorization config path differs.")
    try:
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError("Training is blocked because the config cannot be read.") from error
    if on_disk != {key: value for key, value in config.items() if not key.startswith("_")}:
        raise PermissionError("Training is blocked because the in-memory config differs from disk.")
    expected_hash = experiment_config.get("expected_authorized_sha256_after_single_boolean_edit")
    if sha256_file(config_path) != expected_hash:
        raise PermissionError("Training is blocked because the authorized config hash differs.")
    if record.get("parent_checkpoint", {}).get("sha256") != config.get("parent_checkpoint", {}).get("sha256"):
        raise PermissionError("Training is blocked because the parent checkpoint identity differs.")
    if record.get("tokenizer", {}).get("sha256") != config.get("tokenizer", {}).get("sha256"):
        raise PermissionError("Training is blocked because the tokenizer identity differs.")
    if record.get("resolved_manifest", {}).get("sha256") != config.get("resolved_mixture_manifest_sha256"):
        raise PermissionError("Training is blocked because the mixture identity differs.")
    if record.get("schedule", {}).get("sha256") != config.get("expected_schedule_sha256"):
        raise PermissionError("Training is blocked because the schedule identity differs.")
    scope = record.get("authorization_scope", {})
    if scope.get("authorized_candidate") != config.get("experiment_id"):
        raise PermissionError("Training is blocked because the authorization scope differs.")
    if scope.get("candidate_a_authorized") is not False or scope.get("candidate_b_authorized") is not False:
        raise PermissionError("Training is blocked because the authorization scope is unsafe.")
    if scope.get("authorized_token_budget") != config.get("total_tokens"):
        raise PermissionError("Training is blocked because the token budget differs.")
    if scope.get("authorized_optimizer_updates") != config.get("step_accounting", {}).get("optimizer_updates"):
        raise PermissionError("Training is blocked because the optimizer budget differs.")
