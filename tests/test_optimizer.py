from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from vasu.training.checkpoint import save_checkpoint
from vasu.training.optimizer import build_optimizer
from vasu.training.trainer import Trainer


def _config(backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        learning_rate=1e-3,
        weight_decay=0.1,
        optimizer_backend=backend,
    )


def test_standard_backend_preserves_default_adamw_configuration() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, _config("standard"))
    assert optimizer.defaults["lr"] == 1e-3
    assert optimizer.defaults["weight_decay"] == 0.1
    assert optimizer.defaults["foreach"] is None
    assert optimizer.defaults["fused"] is None


def test_auto_backend_remains_a_standard_adamw_alias() -> None:
    """Keep automatic selection conservative until explicitly approved."""

    optimizer = build_optimizer(torch.nn.Linear(2, 2), _config("auto"))
    assert optimizer.defaults["foreach"] is None
    assert optimizer.defaults["fused"] is None


def test_foreach_backend_is_explicit_and_checkpoint_cross_loads() -> None:
    source_model = torch.nn.Linear(2, 2)
    source = build_optimizer(source_model, _config("foreach"))
    source_model(torch.ones(1, 2)).sum().backward()
    source.step()
    target = build_optimizer(torch.nn.Linear(2, 2), _config("standard"))
    target.load_state_dict(source.state_dict())
    assert target.state_dict()["state"]


def test_fused_backend_rejects_cpu_parameters() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        build_optimizer(torch.nn.Linear(2, 2), _config("fused"))


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="optimizer_backend"):
        build_optimizer(torch.nn.Linear(2, 2), _config("unknown"))


class _RecordingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, seen: list[int]) -> None:
        self.seen = seen
        self.tokens = torch.tensor([[index, index + 1] for index in range(8)])

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen.append(index)
        return self.tokens[index] % 16, (self.tokens[index] + 1) % 16


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


def _fused_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        model = _TinyLanguageModel().cuda()
        torch.optim.AdamW(model.parameters(), fused=True)
    except (RuntimeError, ValueError):
        return False
    return True


def _cuda_config(tmp_path: Path, checkpoint: str) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=1,
        gradient_accumulation_steps=2,
        grad_clip=1.0,
        epochs=1,
        use_amp=True,
        seed=123,
        save_every_steps=0,
        checkpoint_path=str(tmp_path / checkpoint),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        num_workers=0,
        persistent_workers=False,
        optimizer_backend="fused",
        learning_rate=1e-3,
        weight_decay=0.01,
    )


def _fused_trainer(
    tmp_path: Path, checkpoint: str, seen: list[int]
) -> Trainer:
    dataset = _RecordingDataset(seen)
    return Trainer(
        _TinyLanguageModel(),
        tokenizer=None,
        train_dataset=dataset,
        val_dataset=dataset,
        config=_cuda_config(tmp_path, checkpoint),
        device=torch.device("cuda"),
    )


def _assert_nested_close(expected, actual) -> None:
    assert type(expected) is type(actual)
    if torch.is_tensor(expected):
        torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_close(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_nested_close(left, right)
    else:
        assert expected == actual


@pytest.mark.skipif(not _fused_available(), reason="CUDA fused AdamW unavailable")
@pytest.mark.parametrize("interrupt_after", [2, 3])
def test_fused_cuda_exact_resume_matches_uninterrupted(
    tmp_path: Path, interrupt_after: int
) -> None:
    """Check fused AdamW at an optimizer boundary and mid-accumulation."""

    torch.manual_seed(77)
    torch.cuda.manual_seed_all(77)
    uninterrupted_seen: list[int] = []
    uninterrupted = _fused_trainer(tmp_path, "uninterrupted.pt", uninterrupted_seen)
    uninterrupted.train_epoch()
    expected_parameters = [item.detach().clone() for item in uninterrupted.model.parameters()]
    expected_optimizer = uninterrupted.optimizer.state_dict()
    expected_scaler = uninterrupted.scaler.state_dict()

    torch.manual_seed(77)
    torch.cuda.manual_seed_all(77)
    resumed_seen: list[int] = []
    interrupted = _fused_trainer(tmp_path, "interrupted.pt", resumed_seen)
    interrupted.train_epoch(max_microbatches=interrupt_after)
    interrupted.save_training_checkpoint(interrupted.config.checkpoint_path)
    resumed = _fused_trainer(tmp_path, "interrupted.pt", resumed_seen)
    resumed.train_epoch()

    assert resumed_seen == uninterrupted_seen
    assert resumed.global_step == uninterrupted.global_step == 4
    assert resumed._optimizer_steps_in_epoch == uninterrupted._optimizer_steps_in_epoch == 4
    _assert_nested_close(uninterrupted.scheduler.state_dict(), resumed.scheduler.state_dict())
    _assert_nested_close(expected_scaler, resumed.scaler.state_dict())
    _assert_nested_close(expected_optimizer, resumed.optimizer.state_dict())
    for expected, actual in zip(expected_parameters, resumed.model.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not _fused_available(), reason="CUDA fused AdamW unavailable")
def test_fused_and_standard_optimizer_checkpoint_cross_load(tmp_path: Path) -> None:
    """AdamW backend choice must not add checkpoint payload or state keys."""

    device = torch.device("cuda")
    config = _cuda_config(tmp_path, "fused.pt")
    fused_model = _TinyLanguageModel().to(device)
    fused = build_optimizer(fused_model, config)
    fused_model(torch.tensor([[1, 2]], device=device)).sum().backward()
    fused.step()
    scheduler = torch.optim.lr_scheduler.StepLR(fused, step_size=1)
    fused_path = tmp_path / "fused.pt"
    save_checkpoint(fused_model, fused, scheduler, 0, 1.0, fused_path)
    payload = torch.load(fused_path, map_location="cpu", weights_only=False)
    assert "optimizer_backend" not in payload

    standard_model = _TinyLanguageModel().to(device)
    standard = build_optimizer(standard_model, _config("standard"))
    standard.load_state_dict(payload["optimizer"])
    restored_fused_model = _TinyLanguageModel().to(device)
    restored_fused = build_optimizer(restored_fused_model, config)
    restored_fused.load_state_dict(payload["optimizer"])

    standard_path = tmp_path / "standard.pt"
    standard_model(torch.tensor([[1, 2]], device=device)).sum().backward()
    standard.step()
    standard_scheduler = torch.optim.lr_scheduler.StepLR(standard, step_size=1)
    save_checkpoint(
        standard_model,
        standard,
        standard_scheduler,
        0,
        1.0,
        standard_path,
    )
    standard_payload = torch.load(standard_path, map_location="cpu", weights_only=False)
    assert "optimizer_backend" not in standard_payload
    restored_fused.load_state_dict(standard_payload["optimizer"])
