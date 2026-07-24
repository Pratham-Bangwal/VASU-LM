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

    dataset = ScheduledPretrainingDataset.from_resolved_manifest(resolved_path)
    if len(dataset) != config["total_records"]:
        raise ValueError("scheduled dataset length differs from candidate config")
    return {
        "config": config,
        "resolved_manifest": manifest,
        "step_accounting": asdict(accounting),
        "dataset": dataset,
    }


def require_training_authorization(config: dict[str, Any]) -> None:
    if config.get("training_authorized") is not True:
        raise PermissionError(BLOCKED_MESSAGE)
