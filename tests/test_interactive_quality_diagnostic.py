from __future__ import annotations

import torch

from evaluation import diagnose_interactive_quality as diagnostic
from evaluation.evaluate_ultrachat_promotion import format_compliance
from vasu.inference.generate import generate_token_ids
from vasu.tokenizer.tokenizer import VASUTokenizer


def tokenizer() -> VASUTokenizer:
    value = VASUTokenizer()
    value.load("assets/tokenizer.json")
    return value


def test_tokenizer_round_trip_preserves_internal_spacing() -> None:
    result = diagnostic.tokenizer_audit(tokenizer())
    assert result["sentences"] == 200
    assert result["expected_normalized_exact_rate"] == 1.0
    assert result["spacing_mismatch_rate"] == 0.0
    assert all(row["normalized_match"] for row in result["direct_cases"])


def test_prompt_alignment_matches_training_encoder() -> None:
    result = diagnostic.prompt_alignment_audit(tokenizer())
    assert result["examples"] == 20
    assert result["all_prompt_ids_match"] is True
    assert result["all_response_headers_match"] is True
    assert result["bos_used"] is False
    assert result["eos_id"] == 3


def test_format_checks_cover_json_lists_and_sentences() -> None:
    assert format_compliance(
        {"constraints": {"json_keys": ["name", "purpose"]}},
        '{"name": "VASU", "purpose": "assistant"}',
    )["strict"]
    assert format_compliance(
        {"constraints": {"bullet_count": 3}}, "- one\n- two\n- three"
    )["strict"]
    assert format_compliance(
        {"constraints": {"sentence_count": 2}}, "One. Two."
    )["strict"]


class _InnerTokenizer:
    def token_to_id(self, token: str) -> int | None:
        return 3 if token == "[EOS]" else None


class _Tokenizer:
    tokenizer = _InnerTokenizer()

    def encode(self, text: str) -> list[int]:
        return [7, 8, 9]


class _Model:
    def __init__(self) -> None:
        self.config = type("Config", (), {"max_seq_len": 6})()

    def eval(self) -> "_Model":
        return self

    def __call__(self, token_ids: torch.Tensor, kv_cache=None) -> torch.Tensor:
        logits = torch.full((1, token_ids.shape[1], 10), -100.0)
        logits[:, -1, 4] = 100.0
        return logits


def test_generation_returns_only_new_tokens_and_stops_at_context_limit() -> None:
    output = generate_token_ids(
        model=_Model(),
        tokenizer=_Tokenizer(),
        prompt="hello",
        device="cpu",
        max_new_tokens=10,
        do_sample=False,
        prompt_format="plain",
    )
    assert output == [4, 4, 4]
    assert len(output) + 3 == 6


def test_malformed_spacing_detector_is_narrow() -> None:
    assert diagnostic.malformed_spacing("such asweb tools") == ["such asweb"]
    assert diagnostic.malformed_spacing("such as web tools") == []
