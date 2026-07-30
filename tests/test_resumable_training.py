"""Regression tests for deterministic sampler and mid-accumulation resume."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import torch
import pytest
from torch import nn
from torch.utils.data import Dataset

import vasu.training.trainer as trainer_module
from vasu.training.resumable_sampler import ResumableBatchSampler
from vasu.training.trainer import Trainer


class RecordingLanguageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, seen: list[int]) -> None:
        self.seen = seen
        self.inputs = torch.tensor([[index, index + 1] for index in range(8)])
        self.targets = torch.tensor([[index + 1, index + 2] for index in range(8)])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen.append(index)
        return self.inputs[index] % 12, self.targets[index] % 12


class IdentityRecordingDataset(RecordingLanguageDataset):
    def __init__(self, seen: list[int], identity: str) -> None:
        super().__init__(seen)
        self.identity = identity

    def resume_identity(self) -> dict[str, str]:
        return {"schedule_sha256": self.identity}


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 12)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class DropoutLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 12)
        self.dropout = nn.Dropout(0.25)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(token_ids))


def _config(
    tmp_path: Path, checkpoint_name: str, *, epochs: int = 1
) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=1,
        gradient_accumulation_steps=2,
        grad_clip=1.0,
        epochs=epochs,
        use_amp=False,
        seed=314159,
        save_every_steps=0,
        checkpoint_path=str(tmp_path / checkpoint_name),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        num_workers=0,
        persistent_workers=False,
    )


def _build_trainer(
    tmp_path: Path,
    checkpoint_name: str,
    seen: list[int],
    monkeypatch,
    *,
    epochs: int = 1,
) -> Trainer:
    monkeypatch.setattr(
        trainer_module,
        "build_optimizer",
        lambda model, config: torch.optim.SGD(model.parameters(), lr=0.05),
    )
    dataset = RecordingLanguageDataset(seen)
    return Trainer(
        TinyLanguageModel(),
        tokenizer=None,
        train_dataset=dataset,
        val_dataset=dataset,
        config=_config(tmp_path, checkpoint_name, epochs=epochs),
        device=torch.device("cpu"),
    )


def _parameters(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def test_changed_dataset_identity_blocks_exact_resume(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "build_optimizer",
        lambda model, config: torch.optim.SGD(model.parameters(), lr=0.05),
    )
    config = _config(tmp_path, "identity.pt")
    first = Trainer(
        TinyLanguageModel(),
        tokenizer=None,
        train_dataset=IdentityRecordingDataset([], "schedule-a"),
        val_dataset=RecordingLanguageDataset([]),
        config=config,
        device=torch.device("cpu"),
    )
    first.train_epoch(max_microbatches=1)
    first.save_training_checkpoint(config.checkpoint_path)

    with pytest.raises(ValueError, match="dataset/schedule identity"):
        Trainer(
            TinyLanguageModel(),
            tokenizer=None,
            train_dataset=IdentityRecordingDataset([], "schedule-b"),
            val_dataset=RecordingLanguageDataset([]),
            config=config,
            device=torch.device("cpu"),
        )


def test_opt_in_optimizer_step_scheduler_preserves_legacy_default(
    tmp_path: Path, monkeypatch
) -> None:
    legacy = _build_trainer(tmp_path, "legacy-scheduler.pt", [], monkeypatch)
    assert legacy._scheduler_step_unit == "epoch"

    config = _config(tmp_path, "step-scheduler.pt")
    config.scheduler_total_steps = 4
    config.warmup_steps = 1
    config.minimum_learning_rate = 0.005
    config.learning_rate = 0.05
    step_based = Trainer(
        TinyLanguageModel(),
        tokenizer=None,
        train_dataset=RecordingLanguageDataset([]),
        val_dataset=RecordingLanguageDataset([]),
        config=config,
        device=torch.device("cpu"),
    )
    assert step_based._scheduler_step_unit == "optimizer"
    step_based.train_epoch(max_microbatches=2)
    assert step_based.global_step == 1
    assert step_based.scheduler.last_epoch == 1


def test_uninterrupted_and_mid_accumulation_resume_match(
    tmp_path: Path, monkeypatch
) -> None:
    original_loss = trainer_module.language_model_loss
    loss_sequences: dict[str, list[float]] = {"uninterrupted": [], "resumed": []}
    active_sequence = "uninterrupted"

    def recording_loss(logits, targets, mask=None):
        value = original_loss(logits, targets, mask)
        loss_sequences[active_sequence].append(float(value.detach()))
        return value

    monkeypatch.setattr(trainer_module, "language_model_loss", recording_loss)
    torch.manual_seed(17)
    uninterrupted_seen: list[int] = []
    uninterrupted = _build_trainer(
        tmp_path, "uninterrupted.pt", uninterrupted_seen, monkeypatch
    )
    uninterrupted.train_epoch()
    expected_parameters = _parameters(uninterrupted.model)

    active_sequence = "resumed"
    torch.manual_seed(17)
    resumed_seen: list[int] = []
    interrupted = _build_trainer(
        tmp_path, "interrupted.pt", resumed_seen, monkeypatch
    )
    interrupted.train_epoch(max_microbatches=3)
    assert interrupted.global_step == 1
    assert interrupted._accumulated_microbatches == 1
    interrupted.save_training_checkpoint(interrupted.config.checkpoint_path)

    resumed = _build_trainer(
        tmp_path, "interrupted.pt", resumed_seen, monkeypatch
    )
    assert resumed.exact_resume_available
    assert resumed.global_step == 1
    assert resumed._accumulated_microbatches == 1
    resumed.train_epoch()

    assert resumed_seen == uninterrupted_seen
    assert loss_sequences["resumed"] == loss_sequences["uninterrupted"]
    assert resumed.global_step == uninterrupted.global_step == 4
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    for expected, actual in zip(expected_parameters, resumed.model.parameters(), strict=True):
        assert torch.equal(expected, actual.detach())


def test_resume_iterator_does_not_advance_model_rng(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "build_optimizer",
        lambda model, config: torch.optim.SGD(model.parameters(), lr=0.05),
    )
    config = _config(tmp_path, "dropout-resume.pt")

    torch.manual_seed(73)
    uninterrupted = Trainer(
        DropoutLanguageModel(),
        tokenizer=None,
        train_dataset=RecordingLanguageDataset([]),
        val_dataset=RecordingLanguageDataset([]),
        config=config,
        device=torch.device("cpu"),
    )
    uninterrupted.train_epoch()
    expected_parameters = _parameters(uninterrupted.model)
    expected_rng = torch.get_rng_state().clone()

    torch.manual_seed(73)
    interrupted = Trainer(
        DropoutLanguageModel(),
        tokenizer=None,
        train_dataset=RecordingLanguageDataset([]),
        val_dataset=RecordingLanguageDataset([]),
        config=config,
        device=torch.device("cpu"),
    )
    interrupted.train_epoch(max_microbatches=3)
    interrupted.save_training_checkpoint(config.checkpoint_path)
    resumed = Trainer(
        DropoutLanguageModel(),
        tokenizer=None,
        train_dataset=RecordingLanguageDataset([]),
        val_dataset=RecordingLanguageDataset([]),
        config=config,
        device=torch.device("cpu"),
    )
    resumed.train_epoch()

    assert torch.equal(torch.get_rng_state(), expected_rng)
    for expected, actual in zip(
        expected_parameters,
        resumed.model.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual.detach())


def test_checkpoint_positions_cover_optimizer_and_epoch_boundaries(
    tmp_path: Path, monkeypatch
) -> None:
    torch.manual_seed(3)
    seen: list[int] = []
    trainer = _build_trainer(tmp_path, "positions.pt", seen, monkeypatch)
    trainer.train_epoch(max_microbatches=1)
    assert trainer._accumulated_microbatches == 1  # Immediately before a step.
    trainer.save_training_checkpoint(trainer.config.checkpoint_path)
    before_step = _build_trainer(tmp_path, "positions.pt", seen, monkeypatch)
    before_step.train_epoch(max_microbatches=1)
    assert before_step.global_step == 1  # Immediately after a step.
    before_step.train_epoch()
    assert before_step.train_sampler.epoch_complete  # Final batch of epoch.


def test_optimizer_boundary_checkpoint_omits_empty_gradient_payload(
    tmp_path: Path, monkeypatch
) -> None:
    """Exact-resume metadata must not serialize gradients after an update."""

    trainer = _build_trainer(tmp_path, "boundary.pt", [], monkeypatch)
    trainer.train_epoch(max_microbatches=2)
    assert trainer._accumulated_microbatches == 0
    trainer.save_training_checkpoint(trainer.config.checkpoint_path)

    payload = torch.load(
        trainer.config.checkpoint_path, map_location="cpu", weights_only=False
    )
    assert payload["training_progress"]["gradients"] == {}


def test_legacy_checkpoint_warns_and_uses_legacy_next_epoch_resume(
    tmp_path: Path, monkeypatch
) -> None:
    torch.manual_seed(8)
    path = tmp_path / "legacy.pt"
    model = TinyLanguageModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    torch.save(
        {
            "epoch": 2,
            "global_step": 9,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = _build_trainer(tmp_path, "legacy.pt", [], monkeypatch)
    assert not trainer.exact_resume_available
    assert trainer.start_epoch == 3
    assert any(
        "may skip its unprocessed remaining samples" in str(item.message)
        for item in caught
    )


def test_resume_from_best_checkpoint_does_not_repeat_validation_or_scheduler(
    tmp_path: Path, monkeypatch
) -> None:
    torch.manual_seed(21)
    uninterrupted = _build_trainer(
        tmp_path, "uninterrupted.pt", [], monkeypatch
    )
    uninterrupted.fit()
    expected_parameters = _parameters(uninterrupted.model)
    expected_scheduler = uninterrupted.scheduler.state_dict()

    torch.manual_seed(21)
    resumed = _build_trainer(
        tmp_path, "checkpoints/best.pt", [], monkeypatch
    )
    validation_calls = 0
    original_validate = resumed.validate_epoch

    def counted_validate() -> float:
        nonlocal validation_calls
        validation_calls += 1
        return original_validate()

    monkeypatch.setattr(resumed, "validate_epoch", counted_validate)
    resumed.fit()

    assert validation_calls == 0
    assert resumed.scheduler.state_dict() == expected_scheduler
    for expected, actual in zip(expected_parameters, resumed.model.parameters(), strict=True):
        assert torch.equal(expected, actual.detach())


def test_final_batch_step_checkpoint_runs_validation_and_scheduler_once(
    tmp_path: Path, monkeypatch
) -> None:
    torch.manual_seed(23)
    uninterrupted = _build_trainer(
        tmp_path, "uninterrupted.pt", [], monkeypatch
    )
    uninterrupted.fit()
    expected_parameters = _parameters(uninterrupted.model)
    expected_scheduler = uninterrupted.scheduler.state_dict()

    torch.manual_seed(23)
    interrupted = _build_trainer(
        tmp_path, "final-step.pt", [], monkeypatch
    )
    interrupted.train_epoch()
    assert interrupted.train_sampler.epoch_complete
    assert interrupted._resume_phase == "post_train_pre_validation"
    interrupted.save_training_checkpoint(interrupted.config.checkpoint_path)

    resumed = _build_trainer(tmp_path, "final-step.pt", [], monkeypatch)
    validation_calls = 0
    original_validate = resumed.validate_epoch

    def counted_validate() -> float:
        nonlocal validation_calls
        validation_calls += 1
        return original_validate()

    monkeypatch.setattr(resumed, "validate_epoch", counted_validate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resumed.fit()

    assert validation_calls == 1
    assert not any(
        "lr_scheduler.step() before optimizer.step()" in str(item.message)
        for item in caught
    )
    assert resumed.scheduler.state_dict() == expected_scheduler
    for expected, actual in zip(expected_parameters, resumed.model.parameters(), strict=True):
        assert torch.equal(expected, actual.detach())


def test_non_finite_gradients_skip_optimizer_and_scheduler_steps(
    tmp_path: Path, monkeypatch
) -> None:
    torch.manual_seed(29)
    trainer = _build_trainer(tmp_path, "skipped.pt", [], monkeypatch)
    initial_scheduler = trainer.scheduler.state_dict()
    monkeypatch.setattr(trainer_module, "finite_gradients", lambda _: False)

    with pytest.warns(RuntimeWarning, match="gradients were non-finite"):
        trainer.fit()

    assert trainer.global_step == 0
    assert trainer.scheduler.state_dict() == initial_scheduler
    assert trainer._optimizer_steps_in_epoch == 0


def test_changed_epoch_target_retains_checkpoint_scheduler_horizon(
    tmp_path: Path, monkeypatch
) -> None:
    torch.manual_seed(31)
    original = _build_trainer(tmp_path, "source.pt", [], monkeypatch, epochs=2)
    original.fit()
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resumed = _build_trainer(
            tmp_path, "checkpoints/best.pt", [], monkeypatch, epochs=3
        )

    assert resumed.scheduler.T_max == 2
    assert any("retains checkpoint T_max=2" in str(item.message) for item in caught)
    assert checkpoint_path.exists()


def test_sampler_state_is_compact_serializable_and_never_repeats() -> None:
    sampler = ResumableBatchSampler(
        num_samples=11,
        batch_size=3,
        shuffle=True,
        seed=7,
        drop_last=False,
    )
    first = next(iter(sampler))
    sampler.mark_batch_consumed()
    state = sampler.state_dict()
    assert "permutation" not in state
    assert len(json.dumps(state)) < 300

    restored = ResumableBatchSampler(
        num_samples=11,
        batch_size=3,
        shuffle=True,
        seed=7,
        drop_last=False,
    )
    restored.load_state_dict(state)
    remaining = [index for batch in restored for index in batch]
    assert set(first).isdisjoint(remaining)
    assert sorted(first + remaining) == list(range(11))


def test_sampler_rejects_drop_last_contract_mismatch() -> None:
    sampler = ResumableBatchSampler(
        num_samples=8,
        batch_size=2,
        shuffle=True,
        seed=7,
        drop_last=True,
    )
    restored = ResumableBatchSampler(
        num_samples=8,
        batch_size=2,
        shuffle=True,
        seed=7,
        drop_last=False,
    )
    with pytest.raises(ValueError, match="drop_last"):
        restored.load_state_dict(sampler.state_dict())


def test_trainer_rejects_dataset_that_produces_zero_batches(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "build_optimizer",
        lambda model, config: torch.optim.SGD(model.parameters(), lr=0.05),
    )
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[1, 2]]), torch.tensor([[2, 3]])
    )
    config = _config(tmp_path, "empty.pt")
    config.batch_size = 2
    config.drop_last = True
    with pytest.raises(ValueError, match="zero batches"):
        Trainer(
            TinyLanguageModel(),
            tokenizer=None,
            train_dataset=dataset,
            val_dataset=dataset,
            config=config,
            device=torch.device("cpu"),
        )


def test_non_finite_validation_does_not_promote_or_advance_scheduler(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _build_trainer(tmp_path, "nan-validation.pt", [], monkeypatch)
    initial_scheduler = trainer.scheduler.state_dict()
    monkeypatch.setattr(trainer, "validate_epoch", lambda: float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.fit()

    assert trainer.scheduler.state_dict() == initial_scheduler
    assert not (Path(trainer.config.checkpoint_dir) / "best.pt").exists()


def test_multiworker_loader_keeps_sampler_order_for_deterministic_dataset() -> None:
    sampler = ResumableBatchSampler(
        num_samples=12,
        batch_size=2,
        shuffle=True,
        seed=99,
        drop_last=True,
    )
    dataset = torch.utils.data.TensorDataset(torch.arange(12))
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=2)
    observed: list[int] = []
    for (batch,) in loader:
        observed.extend(batch.tolist())
        sampler.mark_batch_consumed()
    expected = torch.randperm(12, generator=torch.Generator().manual_seed(99)).tolist()
    assert observed == expected
