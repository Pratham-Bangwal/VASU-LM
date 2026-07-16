import random

import pytest
import torch

import vasu.inference.generate as generate_module
from evaluation.compare_checkpoints import (
    aggregate_samples,
    build_generation_result,
    build_multi_seed_result,
    build_sample_result,
    build_score_summary,
    effective_generation_config,
    format_text_report,
    generation_kwargs,
    parse_args,
    resolve_checkpoint_entry,
    resolve_model_config,
    resolve_report_paths,
    resolve_sampling_seeds,
    select_checkpoints,
    set_generation_seed,
    validate_prompt_items,
)
from vasu.inference.generate import generate, generate_token_ids


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


def test_default_sampled_behavior_remains_unseeded_and_backward_compatible():
    args = parse_args([])
    assert args.seed is None
    assert args.num_samples is None
    assert args.seeds is None
    assert resolve_sampling_seeds(
        deterministic=False,
        seed=args.seed,
        num_samples=args.num_samples,
        seeds=args.seeds,
    ) is None


def test_num_samples_derives_consecutive_seeds_from_default_and_base():
    assert resolve_sampling_seeds(
        deterministic=False, seed=None, num_samples=5, seeds=None
    ) == [42, 43, 44, 45, 46]
    assert resolve_sampling_seeds(
        deterministic=False, seed=10, num_samples=3, seeds=None
    ) == [10, 11, 12]


def test_explicit_seeds_preserve_order_and_different_values_are_accepted():
    assert resolve_sampling_seeds(
        deterministic=False, seed=None, num_samples=None, seeds=[9, 2, 7]
    ) == [9, 2, 7]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {
                "deterministic": False,
                "seed": None,
                "num_samples": None,
                "seeds": [1, 1],
            },
            "duplicates",
        ),
        (
            {
                "deterministic": False,
                "seed": None,
                "num_samples": 0,
                "seeds": None,
            },
            "at least 1",
        ),
        (
            {
                "deterministic": False,
                "seed": 4,
                "num_samples": None,
                "seeds": [4],
            },
            "cannot be combined",
        ),
        (
            {
                "deterministic": True,
                "seed": 4,
                "num_samples": None,
                "seeds": None,
            },
            "greedy mode",
        ),
        (
            {
                "deterministic": False,
                "seed": -1,
                "num_samples": None,
                "seeds": None,
            },
            "integers from 0",
        ),
        (
            {
                "deterministic": False,
                "seed": 2**63,
                "num_samples": None,
                "seeds": None,
            },
            "integers from 0",
        ),
    ],
)
def test_invalid_seed_configurations_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_sampling_seeds(**kwargs)


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


def test_set_generation_seed_repeats_python_and_torch_randomness():
    set_generation_seed(123)
    first = (random.random(), torch.rand(3))
    set_generation_seed(123)
    second = (random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_multi_seed_result_uses_samples_and_aggregate_schema():
    prompt = {
        "id": "repeat",
        "category": "formatting",
        "prompt": "Answer.",
        "expected_behavior": "An answer.",
        "checks": {"non_empty": True, "max_repetition_ratio": 0.2},
    }
    samples = [
        build_sample_result(prompt, "a b c", 0.1, 42),
        build_sample_result(prompt, "a a", 0.2, 43),
    ]
    result = build_multi_seed_result(prompt, samples)

    assert "response" not in result
    assert [sample["seed"] for sample in result["samples"]] == [42, 43]
    assert result["aggregate"] == {
        "sample_count": 2,
        "automatic_score_mean": 0.75,
        "automatic_score_min": 0.5,
        "automatic_score_max": 1.0,
        "automatic_score_std": 0.25,
        "mean_repetition_ratio": 0.25,
        "min_repetition_ratio": 0.0,
        "max_repetition_ratio": 0.5,
        "passed_checks": 3,
        "total_checks": 4,
        "check_pass_rates": {
            "non_empty": 1.0,
            "max_repetition_ratio": 0.5,
        },
    }
    assert all(sample["manual_score"] is None for sample in samples)


def test_single_seed_generation_result_keeps_flat_schema_and_manual_score():
    prompt = {
        "id": "answer",
        "category": "reasoning",
        "prompt": "Question",
        "expected_behavior": "Answer",
    }
    manual = {
        "relevance": 3,
        "factuality": 3,
        "instruction_following": 3,
        "fluency": 3,
        "repetition_control": 3,
    }
    result = build_generation_result(prompt, "Answer", 0.1, manual)

    assert "samples" not in result
    assert result["response"] == "Answer"
    assert result["manual_score"] == manual
    assert result["manual_score_average"] == 3.0


def test_prompt_without_checks_aggregates_repetition_safely():
    prompt = {
        "id": "creative",
        "category": "creative",
        "prompt": "Story",
        "expected_behavior": "Story",
    }
    aggregate = aggregate_samples(
        [
            build_sample_result(prompt, "one two", 0.1, 1),
            build_sample_result(prompt, "one one", 0.1, 2),
        ]
    )

    assert aggregate["automatic_score_mean"] is None
    assert aggregate["automatic_score_std"] is None
    assert aggregate["passed_checks"] == 0
    assert aggregate["total_checks"] == 0
    assert aggregate["mean_repetition_ratio"] == 0.25


def test_multi_seed_checkpoint_summary_contains_seed_and_categories():
    checked_prompt = {
        "id": "checked",
        "category": "reasoning",
        "prompt": "Question",
        "expected_behavior": "Answer",
        "checks": {"non_empty": True, "max_repetition_ratio": 0.2},
    }
    unchecked_prompt = {
        "id": "unchecked",
        "category": "creative",
        "prompt": "Story",
        "expected_behavior": "Story",
    }
    generations = [
        build_multi_seed_result(
            checked_prompt,
            [
                build_sample_result(checked_prompt, "a b", 0.1, 42),
                build_sample_result(checked_prompt, "a a", 0.1, 43),
            ],
        ),
        build_multi_seed_result(
            unchecked_prompt,
            [
                build_sample_result(unchecked_prompt, "x y", 0.1, 42),
                build_sample_result(unchecked_prompt, "x x", 0.1, 43),
            ],
        ),
    ]
    summary = build_score_summary(
        [{"name": "model", "seed_list": [42, 43], "generations": generations}]
    )["model"]

    assert summary["automatically_evaluated_prompts"] == 1
    assert summary["automatic_average_score"] == 0.75
    assert summary["automatic_score_std_across_samples"] == 0.25
    assert summary["sample_count_per_prompt"] == 2
    assert summary["total_sample_generations"] == 4
    assert summary["mean_repetition_ratio"] == 0.25
    assert summary["seed_list"] == [42, 43]
    assert summary["category_summary"]["reasoning"] == {
        "prompt_count": 1,
        "sample_generations": 2,
        "automatic_average_score": 0.75,
        "mean_repetition_ratio": 0.25,
    }
    assert summary["category_summary"]["creative"][
        "automatic_average_score"
    ] is None


def test_multi_seed_text_report_groups_samples_by_seed():
    prompt = {
        "id": "checked",
        "category": "reasoning",
        "prompt": "Question",
        "expected_behavior": "Answer",
        "checks": {"non_empty": True},
    }
    generation = build_multi_seed_result(
        prompt,
        [
            build_sample_result(prompt, "first", 0.1, 42),
            build_sample_result(prompt, "second", 0.1, 43),
        ],
    )
    result = {
        "name": "model",
        "path": "model.pt",
        "model_config": "vasu_60m",
        "prompt_format": "alpaca",
        "seed_list": [42, 43],
        "generations": [generation],
    }
    summary = build_score_summary([result])
    report = format_text_report(
        [result], summary, {"mode": "sampling", "seed_list": [42, 43]}
    )

    assert "Seed list: [42, 43]" in report
    assert "Aggregate automatic score: 1.000" in report
    assert "Seed 42" in report
    assert "Seed 43" in report


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


class _SamplingModel:
    def eval(self):
        return self

    def __call__(self, input_ids, kv_cache=None):
        logits = torch.zeros((1, input_ids.size(1), 8))
        logits[:, -1, :7] = torch.tensor(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        )
        logits[:, -1, 7] = -100.0
        return logits


def test_same_seed_produces_identical_sampled_token_sequences():
    tokenizer = _Tokenizer()
    kwargs = {
        "model": _SamplingModel(),
        "tokenizer": tokenizer,
        "prompt": "Test",
        "device": "cpu",
        "max_new_tokens": 5,
        "temperature": 0.8,
        "top_k": 7,
        "top_p": 0.9,
        "do_sample": True,
        "prompt_format": "plain",
    }

    set_generation_seed(314)
    first = generate_token_ids(**kwargs)
    set_generation_seed(314)
    second = generate_token_ids(**kwargs)

    assert first == second


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
    parse_args,
