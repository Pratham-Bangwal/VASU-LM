from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation import build_ultrachat_promotion_benchmark as builder
from evaluation import evaluate_ultrachat_promotion as evaluation


def test_benchmark_is_deterministic_and_large() -> None:
    first = builder.build_payload()
    second = builder.build_payload()
    assert first == second
    assert first["prompt_count"] == 216
    assert len(first["categories"]) == 12
    assert len({item["id"] for item in first["prompts"]}) == 216


def test_all_checkpoints_use_identical_prompts() -> None:
    prompts = builder.build_prompts()
    prompt_ids = [item["id"] for item in prompts]
    assert len(prompt_ids) == len(set(prompt_ids))
    assert set(evaluation.CHECKPOINTS) == {
        "alpaca_parent", "ultrachat_best", "ultrachat_latest"
    }


def test_checkpoint_hashes_are_pinned() -> None:
    assert set(evaluation.EXPECTED_HASHES) == set(evaluation.CHECKPOINTS)
    assert all(len(value) == 64 for value in evaluation.EXPECTED_HASHES.values())


def test_generation_modes_are_fixed() -> None:
    greedy = evaluation.MODES["greedy"]
    sampled = evaluation.MODES["controlled_sampling"]
    assert greedy["do_sample"] is False
    assert greedy["max_new_tokens"] == 128
    assert sampled == {
        "do_sample": True,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 128,
        "seed": 42,
    }


def test_list_sentence_and_word_compliance() -> None:
    bullet = {"constraints": {"bullet_count": 3}}
    assert evaluation.format_compliance(bullet, "- A\n- B\n- C")["strict"]
    sentence = {"constraints": {"sentence_count": 2}}
    assert evaluation.format_compliance(sentence, "One. Two.")["strict"]
    one_word = {"constraints": {"exact_words": 1}}
    assert evaluation.format_compliance(one_word, "45")["strict"]


def test_json_and_no_list_compliance() -> None:
    item = {"constraints": {"json_keys": ["topic", "summary"]}}
    assert evaluation.format_compliance(
        item, '{"topic": "rain", "summary": "water"}'
    )["strict"]
    assert not evaluation.format_compliance(item, "topic: rain")["relaxed"]
    no_list = {"constraints": {"forbid_list": True}}
    assert evaluation.format_compliance(no_list, "Plain prose.")["strict"]
    assert not evaluation.format_compliance(no_list, "- item")["strict"]


def test_empty_output_and_repetition_metrics() -> None:
    item = {"constraints": {"max_words": 30}}
    empty = evaluation.response_metrics(item, "", [3], 3)
    assert empty["empty"] is True
    assert empty["premature_eos"] is True
    repeated = evaluation.response_metrics(
        item, "word word word word", [4, 4, 4, 4], 3
    )
    assert repeated["repetition_ratio"] == 0.75
    assert repeated["distinct_1"] == 0.25
    assert repeated["repeated_ngram_rate"] > 0


def test_seed_reset_is_deterministic() -> None:
    evaluation.set_seed(42)
    first = torch.rand(4)
    evaluation.set_seed(42)
    assert torch.equal(first, torch.rand(4))


def test_atomic_result_contains_provenance_hashes(tmp_path: Path) -> None:
    payload = {
        "checkpoint_sha256": "a" * 64,
        "benchmark_sha256": "b" * 64,
        "tokenizer_sha256": evaluation.TOKENIZER_SHA,
        "ultrachat_dataset_sha256": evaluation.ULTRACHAT_TOKEN_SHA,
        "ultrachat_mask_sha256": evaluation.ULTRACHAT_MASK_SHA,
        "config_sha256": "c" * 64,
        "optimizer_updates": 0,
    }
    output = tmp_path / "result.json"
    evaluation.atomic_json(output, payload)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == payload
    assert loaded["optimizer_updates"] == 0


def test_evaluator_has_no_optimizer_step_call() -> None:
    source = Path("evaluation/evaluate_ultrachat_promotion.py").read_text(
        encoding="utf-8"
    )
    assert "optimizer.step(" not in source
    assert "scaler.step(" not in source


def test_promotion_requires_every_gate() -> None:
    result = json.loads(
        Path("evaluation/results/ultrachat_promotion_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assessment = evaluation.promotion_assessment(result["results"])
    assert assessment["gates"]["ultrachat_validation_improved"] is True
    assert assessment["gates"]["instruction_format_stable_or_improved"] is False
    assert assessment["promotion_recommended"] is False
    assert assessment["main_checkpoint_recommendation"] == "alpaca_parent"
