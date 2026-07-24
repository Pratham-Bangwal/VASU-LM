from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import vasu.training.capability_runtime as runtime
from train_vasu_60m_capability_cpt import parse_args
from vasu.training.capability_runtime import (
    CapabilityTrainer,
    ThermalMonitor,
    apply_checkpoint_retention,
    build_capability_identity,
    checkpoint_disk_requirement_bytes,
    inspect_capability_checkpoint,
    require_disk_space,
    validation_configuration_hash,
    validation_loss,
)
from vasu.training.trainer import Trainer


class TinyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, count: int = 8) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor([index % 8, (index + 1) % 8])
        return x, torch.roll(x, -1)

    def resume_identity(self) -> dict[str, str]:
        return {"schedule_sha256": "schedule"}


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 8)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


def _capability_config(tmp_path: Path) -> dict[str, object]:
    return {
        "experiment_id": "test",
        "parent_checkpoint": {"path": "parent", "sha256": "a" * 64},
        "model_configuration": "vasu_60m",
        "tokenizer": {"path": "tokenizer", "sha256": "b" * 64},
        "resolved_mixture_manifest_sha256": "c" * 64,
        "expected_schedule_sha256": "d" * 64,
        "expected_source_hashes": {},
        "optimizer_backend": "standard",
        "scheduler": {"name": "cosine", "total_steps": 4, "warmup_steps": 1},
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "sequence_length": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "validation_interval": 1,
        "checkpoint_interval": 2,
        "validation": {
            "fineweb": {},
            "wikimedia": {},
            "arithmetic_proxy": {
                "selection": "test",
                "max_new_tokens": 2,
            },
        },
        "best_checkpoint_policy": {"mixed_score": False},
        "abort_policy": {
            "maximum_optimizer_skips_total": 3,
            "maximum_optimizer_skips_consecutive": 2,
            "fineweb_baseline_loss": 100.0,
            "fineweb_max_relative_regression": 0.03,
            "wikimedia_baseline_loss": 100.0,
            "wikimedia_catastrophic_relative_regression": 0.10,
        },
        "checkpoint_retention": {
            "periodic_keep": 2,
            "milestone_steps": [],
        },
        "disk_safety": {
            "estimated_boundary_checkpoint_bytes": 1,
            "estimated_mid_accumulation_checkpoint_bytes": 1,
            "minimum_free_space_margin_gib": 0.0,
        },
        "thermal_safety": {
            "monitor_interval_steps": 100,
            "warning_celsius": 82,
            "abort_celsius": 87,
            "critical_celsius": 90,
            "consecutive_abort_readings": 2,
            "required_for_authorized_run": False,
        },
    }


def _trainer_config(tmp_path: Path, checkpoint: Path) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=1,
        gradient_accumulation_steps=2,
        grad_clip=1.0,
        epochs=1,
        use_amp=False,
        seed=42,
        shuffle=False,
        drop_last=True,
        save_every_steps=0,
        checkpoint_path=str(checkpoint),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        num_workers=0,
        persistent_workers=False,
        optimizer_backend="standard",
        learning_rate=0.01,
        weight_decay=0.0,
        scheduler_total_steps=4,
        warmup_steps=1,
        minimum_learning_rate=0.001,
    )


class RecordingCapabilityTrainer(CapabilityTrainer):
    def _run_validation_event(self):
        existing = [
            event
            for event in self.validation_events
            if event["experiment_step"] == self.global_step
        ]
        if existing:
            return existing[0]
        event = {
            "experiment_step": self.global_step,
            "fineweb_loss": 1.0 / self.global_step,
            "wikimedia_loss": 2.0 / self.global_step,
            "arithmetic_exact_accuracy": self.global_step / 10,
            "arithmetic_malformed_rate": 0.0,
            "arithmetic_unanswered_rate": 0.0,
            "duration_seconds": {
                "fineweb": 0.0,
                "wikimedia": 0.0,
                "arithmetic": 0.0,
            },
        }
        self.validation_events.append(event)
        return event


def _trainer(tmp_path: Path, checkpoint: Path) -> RecordingCapabilityTrainer:
    config = _capability_config(tmp_path)
    identity = build_capability_identity(config)
    dataset = TinyDataset()
    return RecordingCapabilityTrainer(
        TinyModel(),
        tokenizer=None,
        train_dataset=dataset,
        val_dataset=dataset,
        config=_trainer_config(tmp_path, checkpoint),
        device=torch.device("cpu"),
        capability_config=config,
        capability_identity=identity,
        validation_loaders={
            "fineweb": DataLoader(dataset, batch_size=1),
            "wikimedia": DataLoader(dataset, batch_size=1),
        },
        arithmetic_records=[],
    )


def test_validation_hash_and_resume_identity_change_together(tmp_path: Path) -> None:
    config = _capability_config(tmp_path)
    first = validation_configuration_hash(config)
    identity = build_capability_identity(config)
    config["validation_interval"] = 2
    assert validation_configuration_hash(config) != first
    assert build_capability_identity(config) != identity


def test_interval_validation_uses_successful_optimizer_steps_and_resumes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "latest.pt"
    first = _trainer(tmp_path, checkpoint)
    first.train_epoch(max_microbatches=2)
    assert first.global_step == 1
    assert [event["experiment_step"] for event in first.validation_events] == [1]
    resumed = _trainer(tmp_path, checkpoint)
    resumed.train_epoch()
    assert resumed.global_step == 4
    assert [event["experiment_step"] for event in resumed.validation_events] == [
        1,
        2,
        3,
        4,
    ]


def test_skipped_optimizer_step_does_not_trigger_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(tmp_path, tmp_path / "none.pt")
    monkeypatch.setattr(Trainer, "_optimizer_step", lambda self: False)
    assert not trainer._optimizer_step()
    assert trainer.global_step == 0
    assert trainer.validation_events == []


def test_validation_loss_restores_callers_mode() -> None:
    model = TinyModel().train()
    loader = DataLoader(TinyDataset(2), batch_size=1)
    value = validation_loss(model, loader, torch.device("cpu"))
    assert value > 0
    # The low-level loss helper deliberately leaves mode management to the
    # enclosing event, which restores the exact previous mode.
    assert model.training


def test_multidomain_validation_event_preserves_rng_and_training_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(tmp_path, tmp_path / "none.pt")
    # Exercise the real event wrapper while replacing only expensive generation.
    monkeypatch.setattr(runtime, "evaluate_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        runtime,
        "summarize_results",
        lambda results: {
            "overall": {
                "exact_accuracy": 0.0,
                "malformed_rate": 0.0,
                "unanswered_rate": 1.0,
            }
        },
    )
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    trainer.model.train()
    event = CapabilityTrainer._run_validation_event(trainer)
    assert torch.equal(torch.get_rng_state(), before)
    assert trainer.model.training
    assert set(event["duration_seconds"]) == {
        "fineweb",
        "wikimedia",
        "arithmetic",
    }


def test_multidomain_validation_does_not_mutate_training_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(tmp_path, tmp_path / "none.pt")
    monkeypatch.setattr(runtime, "evaluate_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        runtime,
        "summarize_results",
        lambda results: {
            "overall": {
                "exact_accuracy": 0.0,
                "malformed_rate": 0.0,
                "unanswered_rate": 1.0,
            }
        },
    )
    optimizer_before = deepcopy(trainer.optimizer.state_dict())
    scheduler_before = deepcopy(trainer.scheduler.state_dict())
    scaler_before = deepcopy(trainer.scaler.state_dict())
    sampler_before = deepcopy(trainer.train_sampler.state_dict())
    accumulation_before = trainer._accumulated_microbatches
    optimizer_steps_before = trainer._optimizer_steps_in_epoch
    gradients_before = {
        name: parameter.grad.clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }

    CapabilityTrainer._run_validation_event(trainer)

    assert trainer.optimizer.state_dict() == optimizer_before
    assert trainer.scheduler.state_dict() == scheduler_before
    assert trainer.scaler.state_dict() == scaler_before
    assert trainer.train_sampler.state_dict() == sampler_before
    assert trainer._accumulated_microbatches == accumulation_before
    assert trainer._optimizer_steps_in_epoch == optimizer_steps_before
    gradients_after = {
        name: parameter.grad
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients_after.keys() == gradients_before.keys()
    for name, gradient in gradients_before.items():
        assert torch.equal(gradients_after[name], gradient)


def test_thermal_warning_sustained_abort_and_critical_abort() -> None:
    values = iter([82, 87, 87])
    monitor = ThermalMonitor(
        warning_celsius=82,
        abort_celsius=87,
        critical_celsius=90,
        consecutive_abort_readings=2,
        reader=lambda: next(values),
    )
    assert monitor.read(1)["warning"]
    assert not monitor.read(2)["abort"]
    assert monitor.read(3)["abort"]
    critical = ThermalMonitor(
        warning_celsius=82,
        abort_celsius=87,
        critical_celsius=90,
        consecutive_abort_readings=2,
        reader=lambda: 90,
    )
    assert critical.read(1)["abort"]


def test_retention_preserves_two_newest_and_milestone(tmp_path: Path) -> None:
    for step in (100, 200, 300, 400):
        path = tmp_path / f"step_{step}.pt"
        path.write_text(str(step), encoding="utf-8")
        path.with_suffix(".pt.json").write_text("{}", encoding="utf-8")
    for name in ("latest.pt", "final.pt", *runtime.DOMAIN_BEST_NAMES):
        (tmp_path / name).write_text(name, encoding="utf-8")
    removed = apply_checkpoint_retention(
        tmp_path,
        keep_periodic=2,
        milestones=[100],
    )
    assert {path.name for path in removed} == {"step_200.pt"}
    assert (tmp_path / "step_100.pt").exists()
    assert (tmp_path / "step_300.pt").exists()
    assert (tmp_path / "step_400.pt").exists()
    assert all((tmp_path / name).exists() for name in runtime.DOMAIN_BEST_NAMES)


def test_disk_estimate_and_insufficient_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _capability_config(tmp_path)
    assert checkpoint_disk_requirement_bytes(config) == 8
    usage = SimpleNamespace(total=100, used=95, free=5)
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda path: usage)
    with pytest.raises(RuntimeError, match="insufficient"):
        require_disk_space(config, tmp_path / "latest.pt")


def test_corrupt_temporary_and_identity_mismatch_rejected(tmp_path: Path) -> None:
    identity = build_capability_identity(_capability_config(tmp_path))
    temporary = tmp_path / "checkpoint.pt.tmp"
    temporary.write_bytes(b"x" * 2048)
    with pytest.raises(ValueError, match="completed"):
        inspect_capability_checkpoint(temporary, identity)
    corrupt = tmp_path / "checkpoint.pt"
    corrupt.write_bytes(b"x" * 2048)
    with pytest.raises(ValueError, match="cannot be loaded"):
        inspect_capability_checkpoint(corrupt, identity)


def test_candidate_c_authorization_remains_false() -> None:
    config = json.loads(
        Path(
            "configs/training/capability_cpt_c_control_20m_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert config["training_authorized"] is False


def test_launcher_accepts_only_explicit_named_resume_argument() -> None:
    args = parse_args(
        [
            "--config",
            "candidate.json",
            "--resume-from",
            "checkpoints/step_200.pt",
        ]
    )
    assert args.resume_from == Path("checkpoints/step_200.pt")
