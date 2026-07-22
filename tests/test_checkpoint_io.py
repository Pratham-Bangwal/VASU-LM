from pathlib import Path

import torch
import pytest

import vasu.training.checkpoint as checkpoint_module
from vasu.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


def build_training_state():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
    )
    return model, optimizer, scheduler


def test_checkpoint_round_trip_restores_training_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    model, optimizer, scheduler = build_training_state()

    inputs = torch.randn(4, 3)
    loss = model(inputs).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint_path = tmp_path / "nested" / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        loss=1.25,
        path=checkpoint_path,
        global_step=17,
        best_val_loss=0.9,
    )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert payload["epoch"] == 2
    assert payload["global_step"] == 17
    assert payload["loss"] == 1.25
    assert payload["best_val_loss"] == 0.9

    restored_model, restored_optimizer, restored_scheduler = (
        build_training_state()
    )

    next_epoch = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
    )

    assert next_epoch == 3
    assert restored_scheduler.state_dict() == scheduler.state_dict()

    for expected, actual in zip(
        model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual)


def test_legacy_checkpoint_without_scheduler_remains_loadable(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = build_training_state()
    path = tmp_path / "legacy.pt"

    torch.save(
        {
            "epoch": 4,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": 2.0,
        },
        path,
    )

    restored_model, restored_optimizer, restored_scheduler = (
        build_training_state()
    )
    original_scheduler_state = restored_scheduler.state_dict()

    next_epoch = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
    )

    assert next_epoch == 5
    assert restored_scheduler.state_dict() == original_scheduler_state


def test_original_positional_save_api_remains_supported(
    tmp_path: Path,
) -> None:
    model, optimizer, _ = build_training_state()
    path = tmp_path / "legacy-call.pt"

    save_checkpoint(
        model,
        optimizer,
        1,
        0.5,
        path,
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    assert payload["epoch"] == 1
    assert payload["global_step"] == 0
    assert "scheduler" not in payload


def test_missing_checkpoint_returns_epoch_zero(
    tmp_path: Path,
) -> None:
    model, _, _ = build_training_state()

    assert load_checkpoint(
        tmp_path / "missing.pt",
        model,
    ) == 0


def test_atomic_save_failure_preserves_existing_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    model, optimizer, _ = build_training_state()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous-valid-checkpoint")

    def failed_save(*args, **kwargs) -> None:
        raise OSError("injected serialization failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", failed_save)
    with pytest.raises(OSError, match="serialization failure"):
        save_checkpoint(model, optimizer, epoch=0, loss=1.0, path=path)

    assert path.read_bytes() == b"previous-valid-checkpoint"
    assert not path.with_suffix(".pt.tmp").exists()
