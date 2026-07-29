from dataclasses import replace

import pytest
import torch

from scripts.preflight_vasu_140m import build_preflight_report
from vasu.config import (
    ModelConfig,
    get_vasu_140m_config,
    get_vasu_60m_config,
)
from vasu.model import (
    MODEL_FAMILIES,
    VASUModel,
    expected_parameter_count,
    get_model_family,
    model_config_fingerprint,
    validate_family_config,
    validate_model_config,
)


@pytest.mark.parametrize(
    ("family_id", "expected_count"),
    [
        ("vasu_31m_v1", 31_168_896),
        ("vasu_60m_v1", 58_337_792),
        ("vasu_140m_v1", 137_841_408),
    ],
)
def test_family_contracts_match_meta_model(
    family_id: str,
    expected_count: int,
) -> None:
    family = get_model_family(family_id)
    config = family.build_config()
    with torch.device("meta"):
        model = VASUModel(config)

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        expected_count
    )
    assert expected_parameter_count(config) == expected_count
    assert model.lm_head.weight is model.embedding.embedding.weight
    assert validate_family_config(family_id, config) is family


def test_vasu_140m_configuration_is_exact_and_isolated() -> None:
    config = get_vasu_140m_config()

    assert (
        config.vocab_size,
        config.max_seq_len,
        config.dim,
        config.n_heads,
        config.n_layers,
        config.hidden_dim,
        config.dropout,
        config.rope_theta,
        config.bias,
    ) == (32000, 512, 768, 12, 12, 3072, 0.1, 10000.0, False)
    assert get_vasu_60m_config().max_seq_len == 256
    assert ModelConfig().dim == 384


def test_family_registry_and_configs_cannot_be_mutated_by_callers() -> None:
    family = get_model_family("vasu_140m_v1")
    first = family.build_config()
    first.dim = 1
    family.config.dim = 2

    assert family.build_config().dim == 768
    with pytest.raises(TypeError):
        MODEL_FAMILIES["other"] = family  # type: ignore[index]


def test_family_validation_rejects_shape_drift() -> None:
    config = replace(get_vasu_140m_config(), max_seq_len=256)

    with pytest.raises(ValueError, match="does not match family"):
        validate_family_config("vasu_140m_v1", config)


def test_unknown_family_lists_valid_ids() -> None:
    with pytest.raises(ValueError, match="vasu_140m_v1"):
        get_model_family("vasu_140m")


@pytest.mark.parametrize(
    "config",
    [
        replace(ModelConfig(), dim=0),
        replace(ModelConfig(), n_heads=5),
        replace(ModelConfig(), dropout=1.0),
        replace(ModelConfig(), dropout=float("nan")),
        replace(ModelConfig(), rope_theta=0.0),
        replace(ModelConfig(), rope_theta=float("inf")),
        replace(ModelConfig(), bias=1),
    ],
)
def test_invalid_model_config_fails_closed(config: ModelConfig) -> None:
    with pytest.raises(ValueError):
        validate_model_config(config)


def test_config_fingerprint_is_stable_and_complete() -> None:
    baseline = get_vasu_140m_config()

    assert model_config_fingerprint(baseline) == model_config_fingerprint(
        get_vasu_140m_config()
    )
    assert model_config_fingerprint(baseline) != model_config_fingerprint(
        replace(baseline, dropout=0.2)
    )


def test_read_only_preflight_closes_training_gates() -> None:
    report = build_preflight_report()

    assert report["passed"] is True
    assert report["training_authorized"] is False
    assert report["parameter_count"] == 137_841_408
    assert all(report["checks"].values())
    assert "exact_resume" in report["remaining_gates"]
    assert "513_token_data_release" in report["remaining_gates"]
