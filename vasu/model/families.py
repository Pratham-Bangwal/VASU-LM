"""Immutable identities for supported VASU model families."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

from vasu.config import (
    ModelConfig,
    get_vasu_140m_config,
    get_vasu_60m_config,
)

FAMILY_IDENTITY_SCHEMA = "vasu.model-family-identity.v1"


@dataclass(frozen=True)
class ModelFamilySpec:
    """A versioned model shape and its expected parameter count."""

    family_id: str
    config_values: tuple[tuple[str, object], ...]
    expected_parameter_count: int

    @classmethod
    def from_config(
        cls,
        family_id: str,
        config: ModelConfig,
        expected_parameter_count: int,
    ) -> ModelFamilySpec:
        return cls(
            family_id=family_id,
            config_values=tuple(model_config_payload(config).items()),
            expected_parameter_count=expected_parameter_count,
        )

    @property
    def config(self) -> ModelConfig:
        """Return a fresh config for callers expecting a config attribute."""
        return self.build_config()

    @property
    def config_fingerprint(self) -> str:
        return model_config_fingerprint(self.build_config())

    @property
    def family_fingerprint(self) -> str:
        payload = {
            "schema": FAMILY_IDENTITY_SCHEMA,
            "family_id": self.family_id,
            "model_config": dict(self.config_values),
        }
        return _canonical_sha256(payload)

    def build_config(self) -> ModelConfig:
        """Return a fresh mutable config rather than the registry instance."""
        return ModelConfig(**dict(self.config_values))


def model_config_payload(config: ModelConfig) -> dict[str, object]:
    """Return the complete canonical persistent shape of a model config."""
    payload = asdict(config)
    validate_model_config(config)
    return payload


def model_config_fingerprint(config: ModelConfig) -> str:
    """Hash every model-config field using canonical JSON."""
    return _canonical_sha256(model_config_payload(config))


def expected_parameter_count(config: ModelConfig) -> int:
    """Calculate parameters for the current tied-embedding VASU architecture."""
    validate_model_config(config)
    embedding = config.vocab_size * config.dim
    attention = 4 * config.dim * config.dim
    mlp = 3 * config.dim * config.hidden_dim
    norms = 2 * config.dim
    biases = 4 * config.dim + 2 * config.hidden_dim + config.dim
    per_layer = attention + mlp + norms
    if config.bias:
        per_layer += biases
    return embedding + config.n_layers * per_layer + config.dim


def validate_model_config(config: ModelConfig) -> None:
    """Fail closed on invalid dimensions before model construction."""
    integer_fields = {
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "dim": config.dim,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "hidden_dim": config.hidden_dim,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.dim % config.n_heads:
        raise ValueError("dim must be divisible by n_heads")
    if (
        isinstance(config.dropout, bool)
        or not isinstance(config.dropout, (int, float))
        or not math.isfinite(config.dropout)
        or config.dropout < 0.0
        or config.dropout >= 1.0
    ):
        raise ValueError("dropout must be in [0, 1)")
    if (
        isinstance(config.rope_theta, bool)
        or not isinstance(config.rope_theta, (int, float))
        or not math.isfinite(config.rope_theta)
        or config.rope_theta <= 0.0
    ):
        raise ValueError("rope_theta must be positive")
    if not isinstance(config.bias, bool):
        raise ValueError("bias must be boolean")


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_FAMILIES = {
    "vasu_31m_v1": ModelFamilySpec.from_config(
        family_id="vasu_31m_v1",
        config=ModelConfig(),
        expected_parameter_count=31_168_896,
    ),
    "vasu_60m_v1": ModelFamilySpec.from_config(
        family_id="vasu_60m_v1",
        config=get_vasu_60m_config(),
        expected_parameter_count=58_337_792,
    ),
    "vasu_140m_v1": ModelFamilySpec.from_config(
        family_id="vasu_140m_v1",
        config=get_vasu_140m_config(),
        expected_parameter_count=137_841_408,
    ),
}
MODEL_FAMILIES: Mapping[str, ModelFamilySpec] = MappingProxyType(_FAMILIES)


def get_model_family(family_id: str) -> ModelFamilySpec:
    """Resolve one exact versioned family ID."""
    try:
        return MODEL_FAMILIES[family_id]
    except KeyError as error:
        known = ", ".join(sorted(MODEL_FAMILIES))
        raise ValueError(
            f"unknown model family {family_id!r}; expected one of: {known}"
        ) from error


def validate_family_config(
    family_id: str,
    config: ModelConfig,
) -> ModelFamilySpec:
    """Require an exact config match for the declared family."""
    family = get_model_family(family_id)
    actual = model_config_fingerprint(config)
    if actual != family.config_fingerprint:
        raise ValueError(
            f"model config does not match family {family_id}: "
            f"expected {family.config_fingerprint}, got {actual}"
        )
    calculated = expected_parameter_count(config)
    if calculated != family.expected_parameter_count:
        raise ValueError(
            f"parameter contract does not match family {family_id}: "
            f"expected {family.expected_parameter_count}, got {calculated}"
        )
    return family
