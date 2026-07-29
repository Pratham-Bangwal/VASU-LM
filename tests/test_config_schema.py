import pytest

from vasu.training.config_schema import validate_proposed_capability_config


def test_proposed_config_schema_is_review_only() -> None:
    assert (
        validate_proposed_capability_config(
            {
                "training_authorized": False,
                "experiment_id": "candidate_e",
                "sequence_length": 256,
                "batch_size": 2,
                "gradient_accumulation_steps": 16,
                "exact_resume": True,
            }
        )["review_only"]
        is True
    )


def test_proposed_config_schema_rejects_authorizing_input() -> None:
    with pytest.raises(ValueError, match="false"):
        validate_proposed_capability_config({"training_authorized": True})
