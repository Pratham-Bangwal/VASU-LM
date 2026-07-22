from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from torch import nn
from torch.utils.data import TensorDataset

import vasu.training.trainer as trainer_module
from vasu.training.accumulation import (
    normalize_partial_accumulation as real_normalize,
)
from vasu.training.losses import (
    language_model_loss as real_language_model_loss,
)
from vasu.training.trainer import Trainer


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


def build_trainer(
    tmp_path: Path,
    monkeypatch,
) -> Trainer:
    monkeypatch.setattr(
        trainer_module,
        "build_optimizer",
        lambda model, config: torch.optim.SGD(
            model.parameters(),
            lr=0.05,
        ),
    )

    inputs = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
        ]
    )
    targets = torch.tensor(
        [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        ]
    )
    masks = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    dataset = TensorDataset(inputs, targets, masks)

    config = SimpleNamespace(
        batch_size=1,
        gradient_accumulation_steps=2,
        grad_clip=1.0,
        epochs=1,
        use_amp=False,
        checkpoint_path=str(tmp_path / "missing.pt"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    return Trainer(
        model=TinyLanguageModel(),
        tokenizer=None,
        train_dataset=dataset,
        val_dataset=dataset,
        config=config,
        device=torch.device("cpu"),
    )


def test_trainer_passes_masks_to_training_and_validation_loss(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed_masks = []

    def recording_loss(logits, targets, mask=None):
        observed_masks.append(
            None if mask is None else mask.detach().cpu().clone()
        )
        return real_language_model_loss(logits, targets, mask)

    monkeypatch.setattr(
        trainer_module,
        "language_model_loss",
        recording_loss,
    )

    trainer = build_trainer(tmp_path, monkeypatch)

    trainer.train_epoch()

    assert len(observed_masks) == 3
    assert all(mask is not None for mask in observed_masks)

    observed_masks.clear()
    trainer.validate_epoch()

    assert len(observed_masks) == 3
    assert all(mask is not None for mask in observed_masks)


def test_inference_mode_validation_matches_reference_loss(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Validation's faster context must preserve the masked-loss result."""

    trainer = build_trainer(tmp_path, monkeypatch)
    trainer.model.eval()
    reference_losses = []
    with torch.no_grad():
        for inputs, targets, mask in trainer.val_loader:
            reference_losses.append(
                real_language_model_loss(trainer.model(inputs), targets, mask)
            )
    reference = torch.stack(reference_losses).mean().item()

    assert trainer.validate_epoch() == pytest.approx(reference, abs=1e-6)


def test_trainer_normalizes_partial_accumulation_tail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed_calls = []

    def recording_normalize(
        parameters,
        *,
        accumulated_microbatches,
        target_microbatches,
    ):
        observed_calls.append(
            (
                accumulated_microbatches,
                target_microbatches,
            )
        )
        return real_normalize(
            parameters,
            accumulated_microbatches=accumulated_microbatches,
            target_microbatches=target_microbatches,
        )

    monkeypatch.setattr(
        trainer_module,
        "normalize_partial_accumulation",
        recording_normalize,
    )

    trainer = build_trainer(tmp_path, monkeypatch)
    trainer.train_epoch()

    assert observed_calls == [
        (2, 2),
        (1, 2),
    ]
