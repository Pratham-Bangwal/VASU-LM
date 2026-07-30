"""Fail-closed model-family identity for new VASU checkpoint workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vasu.config import ModelConfig

from .families import (
    ModelFamilySpec,
    get_model_family,
    validate_family_config,
)

MODEL_FAMILY_CHECKPOINT_SCHEMA = "vasu.model-family-checkpoint-identity.v1"
MODEL_FAMILY_IDENTITY_KEY = "model_family_identity"


def build_model_family_identity(
    family_id: str,
    config: ModelConfig,
) -> dict[str, object]:
    """Build identity metadata after validating the exact family config."""
    family = validate_family_config(family_id, config)
    return {
        "schema": MODEL_FAMILY_CHECKPOINT_SCHEMA,
        "family_id": family.family_id,
        "family_fingerprint": family.family_fingerprint,
        "config_fingerprint": family.config_fingerprint,
        "parameter_count": family.expected_parameter_count,
    }


def validate_checkpoint_family_identity(
    checkpoint: Mapping[str, Any],
    expected_family_id: str,
) -> ModelFamilySpec:
    """Validate checkpoint metadata without loading it into a model."""
    family = get_model_family(expected_family_id)
    identity = checkpoint.get(MODEL_FAMILY_IDENTITY_KEY)
    if not isinstance(identity, Mapping):
        raise ValueError(
            f"checkpoint is missing {MODEL_FAMILY_IDENTITY_KEY!r} metadata"
        )

    expected: dict[str, object] = {
        "schema": MODEL_FAMILY_CHECKPOINT_SCHEMA,
        "family_id": family.family_id,
        "family_fingerprint": family.family_fingerprint,
        "config_fingerprint": family.config_fingerprint,
        "parameter_count": family.expected_parameter_count,
    }
    for name, expected_value in expected.items():
        actual_value = identity.get(name)
        if actual_value != expected_value:
            raise ValueError(
                f"checkpoint model-family {name} mismatch: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )

    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model state must be a mapping")
    if not state:
        raise ValueError("checkpoint model state must not be empty")
    if not all(isinstance(name, str) for name in state):
        raise ValueError("checkpoint model-state keys must be strings")
    return family


def load_family_model_state(
    model: Any,
    checkpoint: Mapping[str, Any],
    expected_family_id: str,
) -> Any:
    """Validate family and destination config before strict state loading."""
    family = validate_checkpoint_family_identity(
        checkpoint,
        expected_family_id,
    )
    validate_family_model(model, family.family_id)
    return model.load_state_dict(checkpoint["model"], strict=True)


def validate_family_model(
    model: Any,
    expected_family_id: str,
) -> ModelFamilySpec:
    """Validate a live model's config and unique parameter count."""
    family = get_model_family(expected_family_id)
    config = getattr(model, "config", None)
    if not isinstance(config, ModelConfig):
        raise TypeError("destination model must expose a ModelConfig")
    validate_family_config(family.family_id, config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != family.expected_parameter_count:
        raise ValueError(
            f"model parameter count does not match family {family.family_id}: "
            f"expected {family.expected_parameter_count}, got {parameter_count}"
        )
    return family
