import pytest
import torch

import vasu.inference.generate as generate_module
from evaluation.compare_checkpoints import (
    build_generation_result,
    build_score_summary,
    effective_generation_config,
    format_text_report,
    generation_kwargs,
    resolve_checkpoint_entry,
    resolve_model_config,
    resolve_report_paths,
    select_checkpoints,
    validate_prompt_items,
)
from vasu.inference.generate import generate


def test_checkpoint_cli_filtering_selects_only_requested_ids():
    checkpoints = {"first": "first.pt", "second": "second.pt"}

    selected = select_checkpoints(checkpoints, ["second"])

    assert selected == {"second": "second.pt"}


def test_unknown_checkpoint_id_lists_valid_names():
    checkpoints = {"first": "first.pt", "second": "second.pt"}

    with pytest.raises(ValueError, match="Valid checkpoint IDs: first, second"):
        select_checkpoints(checkpoints, ["missing"])


def test_vasu_31m_configuration_remains_unchanged():
    config = resolve_model_config("vasu_31m")

    assert (
        config.vocab_size,
        config.max_seq_len,
        config.dim,
        config.n_heads,
        config.n_layers,
        config.hidden_dim,
    ) == (32000, 256, 384, 6, 8, 1536)


def test_vasu_60m_configuration_loads_correctly():
    config = resolve_model_config("vasu_60m")

    assert (
        config.vocab_size,
        config.max_seq_len,
        config.dim,
        config.n_heads,
        config.n_layers,
        config.hidden_dim,
    ) == (32000, 256, 512, 8, 10, 2048)


def test_deterministic_mode_disables_sampling_arguments():
    effective = effective_generation_config(
        {
            "max_new_tokens": 60,
            "temperature": 0.45,
            "top_k": 20,
            "top_p": 0.8,
        },
        deterministic=True,
    )

    assert effective["do_sample"] is False
    assert effective["temperature"] is None
    assert effective["top_k"] is None
    assert effective["top_p"] is None
    assert generation_kwargs(effective) == {
        "max_new_tokens": 60,
        "do_sample": False,
    }


def test_legacy_checkpoint_entry_remains_backward_compatible():
    assert resolve_checkpoint_entry("legacy.pt") == (
        "legacy.pt",
        "vasu_31m",
        "alpaca",
    )


def test_named_reports_preserve_existing_history():
    text_path, json_path, summary_path = resolve_report_paths(
        "alpaca_masked_v2"
    )
    assert text_path.name == "checkpoint_comparison_alpaca_masked_v2.txt"
    assert json_path.name == "checkpoint_comparison_alpaca_masked_v2.json"
    assert summary_path.name == (
        "checkpoint_score_summary_alpaca_masked_v2.json"
    )


def test_generation_entry_includes_automatic_evaluation():
    item = {
        "id": "answer",
        "category": "reasoning",
        "prompt": "What is 17 plus 25?",
        "expected_behavior": "Answers 42.",
        "checks": {"exact_answer": "42"},
    }
    result = build_generation_result(item, "42", 0.125, None)

    assert result["manual_score"] is None
    assert result["automatic_evaluation"]["score"] == 1.0
    assert result["automatic_evaluation"]["passed_checks"] == 1


def test_score_summary_adds_automatic_fields_without_changing_manual_fields():
    results = [
        {
            "name": "model",
            "generations": [
                {
                    "manual_score_average": 2.5,
                    "automatic_evaluation": {
                        "score": 0.5,
                        "passed_checks": 1,
                        "total_checks": 2,
                    },
                },
                {
                    "manual_score_average": None,
                    "automatic_evaluation": {
                        "score": None,
                        "passed_checks": 0,
                        "total_checks": 0,
                    },
                },
            ],
        }
    ]

    assert build_score_summary(results)["model"] == {
        "scored_prompts": 1,
        "average_score": 2.5,
        "automatically_evaluated_prompts": 1,
        "automatic_average_score": 0.5,
        "automatic_passed_checks": 1,
        "automatic_total_checks": 2,
    }


def test_prompt_without_checks_remains_valid_and_unscored():
    validate_prompt_items([{"id": "legacy"}])
    result = build_generation_result(
        {
            "id": "legacy",
            "category": "creative",
            "prompt": "Tell a story.",
            "expected_behavior": "A story.",
        },
        "Once upon a time.",
        0.0,
        None,
    )
    assert result["automatic_evaluation"]["score"] is None


def test_invalid_prompt_check_fails_before_generation():
    with pytest.raises(ValueError, match="prompt broken"):
        validate_prompt_items(
            [{"id": "broken", "checks": {"made_up": True}}]
        )


def test_text_report_prints_automatic_score_and_failed_checks():
    generation = {
        "prompt_id": "answer",
        "category": "reasoning",
        "prompt": "Question",
        "expected_behavior": "Answer",
        "response": "41",
        "time_seconds": 0.1,
        "stats": {},
        "manual_score": None,
        "manual_score_average": None,
        "automatic_evaluation": {
            "checks": {
                "exact_answer": {
                    "passed": False,
                    "score": 0.0,
                    "details": "does not match",
                }
            },
            "passed_checks": 0,
            "total_checks": 1,
            "score": 0.0,
        },
    }
    result = {
        "name": "model",
        "path": "model.pt",
        "model_config": "vasu_60m",
        "prompt_format": "alpaca",
        "generations": [generation],
    }
    summary = build_score_summary([result])
    report = format_text_report([result], summary, {"mode": "greedy"})

    assert "Automatic score: 0.000 (0/1 checks)" in report
    assert "- exact_answer: does not match" in report


class _InnerTokenizer:
    def token_to_id(self, token):
        return 7 if token == "[EOS]" else None


class _Tokenizer:
    tokenizer = _InnerTokenizer()

    def __init__(self):
        self.encoded_text = ""

    def encode(self, text):
        self.encoded_text = text
        return [1, 2, 3]

    def decode(self, ids, **kwargs):
        return ",".join(str(token) for token in ids)


class _GreedyModel:
    def eval(self):
        return self

    def __call__(self, input_ids, kv_cache=None):
        logits = torch.zeros((1, input_ids.size(1), 8))
        logits[:, -1, 5] = 10.0
        return logits


def test_generated_answer_excludes_prompt_and_greedy_skips_sampler(
    monkeypatch,
):
    def fail_if_sampled(*args, **kwargs):
        raise AssertionError("sampling helper must not run in greedy mode")

    monkeypatch.setattr(
        generate_module,
        "sample_next_token",
        fail_if_sampled,
    )
    tokenizer = _Tokenizer()

    answer = generate(
        model=_GreedyModel(),
        tokenizer=tokenizer,
        prompt="Explain gravity.",
        device="cpu",
        max_new_tokens=1,
        do_sample=False,
        prompt_format="alpaca",
    )

    assert tokenizer.encoded_text == "User: Explain gravity.\nAssistant:"
    assert answer == "5"
