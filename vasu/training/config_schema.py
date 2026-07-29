"""Minimal fail-closed schema for proposed capability experiment configs."""

from __future__ import annotations

from typing import Any, Mapping


def validate_proposed_capability_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a review-stage config without making it launchable."""

    if config.get("training_authorized") is not False:
        raise ValueError("proposed configs must set training_authorized to false")
    experiment_id = config.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be a non-empty string")
    for field in ("sequence_length", "batch_size", "gradient_accumulation_steps"):
        value = config.get(field)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config.get("exact_resume") is not True:
        raise ValueError("exact_resume must be true")
    return {
        "format_version": "vasu_proposed_capability_config_schema_v1",
        "experiment_id": experiment_id,
        "training_authorized": False,
        "review_only": True,
    }
