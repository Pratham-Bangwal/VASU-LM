from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from evaluation.build_factual_cpt_v2_benchmark import benchmark_bytes, build_benchmark
from evaluation.evaluate_factual_cpt_v2 import (
    _bootstrap_ci,
    aggregate_generated_rows,
    conditional_log_likelihood,
    rank_option_scores,
    validate_checkpoint_hash,
    validate_result_provenance,
)


BENCHMARK = Path("evaluation/benchmarks/factual_cpt_v2.json")


def test_benchmark_is_frozen_and_has_required_counts() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert BENCHMARK.read_bytes() == benchmark_bytes()
    assert len(payload["cloze"]) == 100
    assert len(payload["multiple_choice"]) == 100
    assert len(payload["continuations"]) == 50
    assert len(payload["open_ended"]) == 50
    assert payload == build_benchmark()


def test_all_benchmark_examples_are_unique_and_shared_by_configuration() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    ids = [item["id"] for section in ("cloze", "multiple_choice", "continuations", "open_ended") for item in payload[section]]
    prompts = [item["prompt"] for section in ("cloze", "multiple_choice", "continuations", "open_ended") for item in payload[section]]
    assert len(ids) == len(set(ids)) == 300
    assert len(prompts) == len(set(prompts)) == 300
    config = json.loads(Path("configs/evaluation/vasu_60m_factual_cpt_v2.json").read_text())
    assert len(config["checkpoints"]) == 3
    assert config["benchmark_path"] == str(BENCHMARK).replace("\\", "/")


def test_required_categories_are_represented() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    categories = {item["category"] for section in ("cloze", "multiple_choice", "continuations", "open_ended") for item in payload[section]}
    assert {"science", "history", "geography", "technology", "biography", "definitions", "general language"}.issubset(categories)


def test_prompts_are_not_verbatim_wikimedia_release_sentences() -> None:
    source = Path(
        "data/processed/pretrain/factual/wikimedia_pilot_release/documents.jsonl"
    ).read_text(encoding="utf-8").casefold()
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    prompts = [item["prompt"].casefold() for section in ("cloze", "multiple_choice", "continuations", "open_ended") for item in payload[section]]
    assert all(prompt not in source for prompt in prompts)


def test_checkpoint_hash_validation(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    assert validate_checkpoint_hash(path, digest) == digest
    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint_hash(path, "0" * 64)


class _ConstantModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(tokens.size(0), tokens.size(1), -1)


def test_conditional_likelihood_is_correct_and_has_no_updates() -> None:
    model = _ConstantModel()
    before = model.bias.detach().clone()
    result = conditional_log_likelihood(model, [1, 2], [7, 6], torch.device("cpu"))
    expected = torch.log_softmax(torch.arange(8, dtype=torch.float32), dim=0)[[7, 6]]
    assert result["total_log_likelihood"] == pytest.approx(expected.sum().item())
    assert result["mean_log_likelihood"] == pytest.approx(expected.mean().item())
    assert torch.equal(model.bias, before)
    assert model.bias.grad is None


def test_raw_and_length_normalized_multiple_choice_ranking() -> None:
    scores = [
        {"option": "short", "total_log_likelihood": -2.0, "mean_log_likelihood": -2.0},
        {"option": "long", "total_log_likelihood": -3.0, "mean_log_likelihood": -1.0},
        {"option": "bad", "total_log_likelihood": -8.0, "mean_log_likelihood": -4.0},
    ]
    result = rank_option_scores(scores)
    assert result["predicted_index"] == 0
    assert result["length_normalized_predicted_index"] == 1
    assert result["winning_margin"] == pytest.approx(1.0)
    assert result["length_normalized_winning_margin"] == pytest.approx(1.0)


def test_bootstrap_seed_is_deterministic() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    assert _bootstrap_ci(values, seed=42, samples=200) == _bootstrap_ci(
        values, seed=42, samples=200
    )


def test_empty_and_repetition_metrics_are_consistent() -> None:
    rows = [
        {"response": "", "repetition_ratio": 0.0, "premature_eos": True, "empty_output": True, "generated_tokens": 1, "words": 0},
        {"response": "word word word", "repetition_ratio": 2 / 3, "premature_eos": False, "empty_output": False, "generated_tokens": 3, "words": 3},
    ]
    result = aggregate_generated_rows(rows)
    assert result["empty_output_rate"] == 0.5
    assert result["premature_eos_rate"] == 0.5
    assert result["mean_repetition_ratio"] == pytest.approx(1 / 3)
    assert result["distinct_1"] == pytest.approx(1 / 3)


def test_distinct_ngrams_do_not_cross_response_boundaries() -> None:
    rows = [
        {"response": "alpha beta", "repetition_ratio": 0.0, "premature_eos": False, "empty_output": False, "generated_tokens": 2, "words": 2},
        {"response": "gamma delta", "repetition_ratio": 0.0, "premature_eos": False, "empty_output": False, "generated_tokens": 2, "words": 2},
    ]
    result = aggregate_generated_rows(rows)
    assert result["distinct_2"] == 1.0
    assert result["distinct_3"] == 0.0


def test_result_requires_all_reproducibility_hashes() -> None:
    valid = {name: {"sha256": "a" * 64} for name in ("checkpoint", "benchmark", "evaluation_config", "tokenizer")}
    validate_result_provenance(valid)
    del valid["tokenizer"]
    with pytest.raises(ValueError, match="tokenizer"):
        validate_result_provenance(valid)
