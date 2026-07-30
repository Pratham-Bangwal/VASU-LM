from dataclasses import replace

import pytest

from vasu.config import get_vasu_140m_config, get_vasu_60m_config
from vasu.model import (
    MODEL_FAMILY_CHECKPOINT_SCHEMA,
    build_model_family_identity,
    load_family_model_state,
    validate_checkpoint_family_identity,
    validate_family_model,
)


def _checkpoint() -> dict[str, object]:
    return {
        "model": {"embedding.embedding.weight": object()},
        "model_family_identity": build_model_family_identity(
            "vasu_140m_v1",
            get_vasu_140m_config(),
        ),
    }


def test_identity_binds_every_family_contract_field() -> None:
    identity = build_model_family_identity(
        "vasu_140m_v1",
        get_vasu_140m_config(),
    )

    assert identity == {
        "schema": MODEL_FAMILY_CHECKPOINT_SCHEMA,
        "family_id": "vasu_140m_v1",
        "family_fingerprint": (
            "72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b"
        ),
        "config_fingerprint": (
            "29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059"
        ),
        "parameter_count": 137_841_408,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "other"),
        ("family_id", "vasu_60m_v1"),
        ("family_fingerprint", "0" * 64),
        ("config_fingerprint", "0" * 64),
        ("parameter_count", 1),
    ],
)
def test_identity_drift_is_rejected(field: str, value: object) -> None:
    checkpoint = _checkpoint()
    checkpoint["model_family_identity"][field] = value

    with pytest.raises(ValueError, match=field):
        validate_checkpoint_family_identity(
            checkpoint,
            "vasu_140m_v1",
        )


def test_missing_identity_and_model_state_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_checkpoint_family_identity(
            {"model": {"weight": object()}},
            "vasu_140m_v1",
        )
    checkpoint = _checkpoint()
    checkpoint["model"] = {}
    with pytest.raises(ValueError, match="must not be empty"):
        validate_checkpoint_family_identity(
            checkpoint,
            "vasu_140m_v1",
        )


def test_wrong_expected_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="family_id mismatch"):
        validate_checkpoint_family_identity(
            _checkpoint(),
            "vasu_60m_v1",
        )


class _RecordingModel:
    def __init__(self, config) -> None:
        self.config = config
        self.load_calls = 0

    def parameters(self):
        class _Count:
            @staticmethod
            def numel() -> int:
                return 137_841_408

        return iter((_Count(),))

    def load_state_dict(self, state, strict):
        self.load_calls += 1
        return state, strict


def test_wrong_destination_config_is_rejected_before_state_loading() -> None:
    model = _RecordingModel(get_vasu_60m_config())

    with pytest.raises(ValueError, match="does not match family"):
        load_family_model_state(model, _checkpoint(), "vasu_140m_v1")

    assert model.load_calls == 0


def test_matching_destination_loads_strictly() -> None:
    model = _RecordingModel(get_vasu_140m_config())
    checkpoint = _checkpoint()

    state, strict = load_family_model_state(
        model,
        checkpoint,
        "vasu_140m_v1",
    )

    assert state == checkpoint["model"]
    assert strict is True
    assert model.load_calls == 1


def test_identity_builder_rejects_config_drift() -> None:
    config = replace(get_vasu_140m_config(), max_seq_len=256)

    with pytest.raises(ValueError, match="does not match family"):
        build_model_family_identity("vasu_140m_v1", config)


def test_live_model_parameter_count_is_part_of_family_contract() -> None:
    model = _RecordingModel(get_vasu_140m_config())
    model.parameters = lambda: iter(())

    with pytest.raises(ValueError, match="parameter count"):
        validate_family_model(model, "vasu_140m_v1")
