from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from evaluation import evaluate_factual_cpt_baseline as baseline
import train_vasu_60m_factual_cpt as factual_training
from vasu.training.accumulation import (
    build_accumulation_plan,
    normalize_partial_accumulation,
)


def test_factual_cpt_partial_accumulation_accounting() -> None:
    plan = build_accumulation_plan(
        record_count=39_062,
        batch_size=2,
        accumulation_steps=16,
        sequence_length=256,
    )
    assert plan.microbatches == 19_531
    assert plan.optimizer_steps == 1_221
    assert plan.final_microbatches == 11
    assert plan.final_records == 22
    assert plan.final_tokens == 5_632
    assert plan.trained_tokens == 9_999_872


def test_partial_gradient_normalization_uses_actual_count() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    for _ in range(11):
        (parameter / 16).backward()
    factor = normalize_partial_accumulation(
        (parameter,), accumulated_microbatches=11, target_microbatches=16
    )
    assert factor == pytest.approx(16 / 11)
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(1.0)


def test_exact_accumulation_boundary_is_unchanged() -> None:
    plan = build_accumulation_plan(
        record_count=64,
        batch_size=2,
        accumulation_steps=16,
        sequence_length=256,
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(3.0)
    factor = normalize_partial_accumulation(
        (parameter,), accumulated_microbatches=16, target_microbatches=16
    )
    assert plan.optimizer_steps == 2
    assert plan.final_microbatches == 16
    assert plan.final_tokens == 8_192
    assert factor == 1.0
    assert parameter.grad.item() == 3.0


def test_parent_baseline_loader_strictly_loads_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self, _config: object) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))

    path = tmp_path / "parent.pt"
    torch.save(
        {
            "epoch": 0,
            "global_step": 200_000,
            "model": {"weight": torch.tensor([2.0])},
            "optimizer": {},
            "scheduler": {},
            "loss": 1.0,
        },
        path,
    )
    monkeypatch.setattr(baseline, "VASUModel", TinyModel)
    model, step = baseline._load_parent_model(path, torch.device("cpu"))
    assert step == 200_000
    assert model.weight.item() == 2.0
    assert not model.training


def test_baseline_validation_performs_no_optimizer_update() -> None:
    class TinyLanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.head = torch.nn.Linear(4, 8)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.head(self.embedding(tokens))

    model = TinyLanguageModel()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    x = torch.tensor([[0, 1, 2], [2, 3, 4]])
    y = torch.tensor([[1, 2, 3], [3, 4, 5]])
    result = baseline.validation_loss(
        model, TensorDataset(x, y), batch_size=1, device=torch.device("cpu")
    )
    assert result["tokens"] == 6
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_completed_experiment_is_closed_and_intervals_are_preserved() -> None:
    path = Path("configs/training/vasu_60m_factual_cpt_wikimedia_15pct.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["training_authorized"] is False
    assert config["validation_interval_steps"] == 100
    assert config["checkpoint_interval_steps"] == 200
    assert config["sample_generation_interval_steps"] == 100
    assert config["preserve_checkpoint_steps"] == [200]
    assert config["keep_best_checkpoint"] is True
    assert config["keep_latest_checkpoint"] is True
    loaded = factual_training.load_experiment_config(path)
    assert loaded["training_authorized"] is False


def test_response_aggregates_include_requested_metrics() -> None:
    outputs = [
        {"metrics": baseline.response_metrics("Earth rotates and follows an orbit.", 9)},
        {"metrics": baseline.response_metrics("References\n[1] A source", 6)},
    ]
    result = baseline.aggregate_generation_metrics(outputs)
    assert "repetition" in result
    assert "response_lengths" in result
    assert result["encyclopedic_style_indicators"]["any_indicator"]["count"] == 1
