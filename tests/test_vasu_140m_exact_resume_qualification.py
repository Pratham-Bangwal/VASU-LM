from pathlib import Path

import pytest
import torch

from scripts.qualify_vasu_140m_exact_resume import (
    SyntheticResumeDataset,
    state_sha256,
)
from vasu.config import get_vasu_140m_config
from vasu.model import build_model_family_identity
from vasu.training.resumable_sampler import ResumableBatchSampler


def test_state_digest_is_stable_and_tensor_sensitive() -> None:
    first = {"tensor": torch.tensor([1.0, 2.0]), "step": 1}
    second = {"step": 1, "tensor": torch.tensor([1.0, 2.0])}
    changed = {"tensor": torch.tensor([1.0, 3.0]), "step": 1}

    assert state_sha256(first) == state_sha256(second)
    assert state_sha256(first) != state_sha256(changed)


def test_synthetic_dataset_is_deterministic_and_identity_bound() -> None:
    first = SyntheticResumeDataset([])
    second = SyntheticResumeDataset([])

    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)
    assert first.resume_identity() == second.resume_identity()


def test_sampler_mismatch_fails_before_position_restore() -> None:
    source = ResumableBatchSampler(
        4,
        1,
        shuffle=True,
        seed=140_044,
        drop_last=True,
    )
    source.mark_batch_consumed()
    destination = ResumableBatchSampler(
        4,
        2,
        shuffle=True,
        seed=140_044,
        drop_last=True,
    )

    with pytest.raises(ValueError, match="batch_size"):
        destination.load_state_dict(source.state_dict())


def test_family_identity_rejects_drift_before_qualification() -> None:
    config = get_vasu_140m_config()
    identity = build_model_family_identity("vasu_140m_v1", config)

    assert identity["parameter_count"] == 137_841_408


def test_worker_paths_are_isolated(tmp_path: Path) -> None:
    assert not (tmp_path / "control.json").exists()
    assert not (tmp_path / "resumed.json").exists()
