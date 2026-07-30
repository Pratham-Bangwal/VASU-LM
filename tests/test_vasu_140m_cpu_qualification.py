import json
from pathlib import Path

import pytest
import torch

from scripts.qualify_vasu_140m_cpu import (
    report_sha256,
    run_cache_parity,
    run_forward_backward,
    write_immutable_report,
)
from vasu.config import ModelConfig
from vasu.model import VASUModel


def _tiny_model() -> VASUModel:
    torch.manual_seed(12)
    return VASUModel(
        ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            dim=16,
            n_heads=4,
            n_layers=2,
            hidden_dim=32,
            dropout=0.0,
        )
    )


def test_forward_backward_is_finite_and_performs_no_update() -> None:
    model = _tiny_model()
    parameter_versions = [
        parameter._version for parameter in model.parameters()
    ]
    inputs = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])

    report = run_forward_backward(model, inputs, targets)

    assert report["finite_logits"] is True
    assert report["finite_loss"] is True
    assert report["finite_gradients"] is True
    assert report["all_parameters_received_gradients"] is True
    assert report["gradients_cleared"] is True
    assert report["optimizer_created"] is False
    assert report["optimizer_step_performed"] is False
    assert parameter_versions == [
        parameter._version for parameter in model.parameters()
    ]


def test_dynamic_and_preallocated_cache_match_uncached_model() -> None:
    report = run_cache_parity(
        _tiny_model(),
        torch.tensor([[1, 2, 3, 4]]),
        (5, 6, 7),
    )

    assert report["dynamic_cache"]["passed"] is True
    assert report["preallocated_cache"]["passed"] is True
    assert report["dynamic_cache"]["final_cache_length"] == 7
    assert report["preallocated_cache"]["final_cache_length"] == 7


def test_report_hash_is_canonical() -> None:
    assert report_sha256({"b": 2, "a": 1}) == report_sha256(
        {"a": 1, "b": 2}
    )
    assert report_sha256({"a": 1, "report_sha256": "ignored"}) == (
        report_sha256({"a": 1})
    )


def test_immutable_report_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    report = {"passed": True}

    write_immutable_report(output, report)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_immutable_report(output, report)
