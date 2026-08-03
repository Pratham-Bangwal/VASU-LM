"""Emit deterministic evidence for the public generation request contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vasu.inference.generate import generate_token_ids  # noqa: E402
from vasu.inference.sampling import sample_next_token  # noqa: E402


def _lf_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _InnerTokenizer:
    @staticmethod
    def token_to_id(token: str) -> int | None:
        return 2 if token == "[EOS]" else None


class _Tokenizer:
    tokenizer = _InnerTokenizer()

    @staticmethod
    def encode(text: str) -> list[int]:
        return [0, 1]


class _Model:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.forward_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, input_ids, kv_cache=None):
        self.forward_calls += 1
        logits = torch.full((1, input_ids.size(1), 8), -100.0)
        logits[:, -1, 3] = 100.0
        return logits


def _raises(expected: type[Exception], action) -> bool:
    try:
        action()
    except expected:
        return True
    return False


def qualify() -> dict[str, object]:
    tokenizer = _Tokenizer()
    invalid_limit_model = _Model()
    invalid_limit_rejected = _raises(
        ValueError,
        lambda: generate_token_ids(
            invalid_limit_model,
            tokenizer,
            "prompt",
            "cpu",
            max_new_tokens=-1,
        ),
    )

    invalid_cache_model = _Model()
    invalid_cache_rejected = _raises(
        ValueError,
        lambda: generate_token_ids(
            invalid_cache_model,
            tokenizer,
            "prompt",
            "cpu",
            max_new_tokens=0,
            do_sample=False,
            kv_cache_implementation="unknown",
        ),
    )

    zero_model = _Model()
    zero_result = generate_token_ids(
        zero_model,
        tokenizer,
        "prompt",
        "cpu",
        max_new_tokens=0,
        do_sample=False,
    )

    greedy_model = _Model()
    greedy_result = generate_token_ids(
        greedy_model,
        tokenizer,
        "prompt",
        "cpu",
        max_new_tokens=1,
        do_sample=False,
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        repetition_penalty=0.0,
    )

    logits = torch.zeros(1, 1, 4)
    history = torch.zeros(1, 1, dtype=torch.long)
    invalid_sampling_cases = (
        {"temperature": 0.0},
        {"temperature": float("inf")},
        {"top_k": 0},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"repetition_penalty": 0.0},
    )
    sampling_rejections = sum(
        _raises(
            ValueError,
            lambda overrides=overrides: sample_next_token(
                logits,
                history,
                **overrides,
            ),
        )
        for overrides in invalid_sampling_cases
    )

    return {
        "schema_id": "vasu_generation_request_contract_qualification_v1",
        "date": "2026-08-03",
        "device": "cpu",
        "generation_implementation_sha256": _lf_sha256(
            ROOT / "vasu/inference/generate.py"
        ),
        "sampling_implementation_sha256": _lf_sha256(
            ROOT / "vasu/inference/sampling.py"
        ),
        "invalid_limit_rejected_before_eval": (
            invalid_limit_rejected and invalid_limit_model.eval_calls == 0
        ),
        "invalid_cache_rejected_for_zero_token_request": (
            invalid_cache_rejected and invalid_cache_model.eval_calls == 0
        ),
        "zero_token_forward_calls": zero_model.forward_calls,
        "zero_token_result_empty": zero_result == [],
        "greedy_sampling_sentinels_preserved": greedy_result == [3],
        "invalid_sampling_cases": len(invalid_sampling_cases),
        "invalid_sampling_rejections": sampling_rejections,
        "checkpoint_loaded": False,
        "dataset_opened": False,
        "kv_cache_enabled_by_default": False,
        "training_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(qualify(), indent=2, sort_keys=True))
