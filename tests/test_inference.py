import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from vasu.inference.generate import generate, generate_token_ids
from vasu.inference.sampling import sample_next_token


ROOT = Path(__file__).resolve().parents[1]


def _lf_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_top_k_sampling_keeps_highest_token():
    logits = torch.tensor([[[0.0, 1.0, 5.0, 2.0]]])
    input_ids = torch.tensor([[0]])

    token = sample_next_token(
        logits,
        input_ids,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    assert token.item() == 2


def test_top_k_is_clamped_to_vocab_size():
    logits = torch.tensor([[[0.0, 1.0, 5.0, 2.0]]])
    input_ids = torch.tensor([[0]])

    token = sample_next_token(
        logits,
        input_ids,
        top_k=100,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    assert 0 <= token.item() < logits.size(-1)


def test_top_p_sampling_keeps_nucleus_token():
    logits = torch.tensor([[[10.0, 8.0, 1.0, 0.0]]])
    input_ids = torch.tensor([[3]])

    token = sample_next_token(
        logits,
        input_ids,
        top_k=None,
        top_p=0.7,
        repetition_penalty=1.0,
    )

    assert token.item() == 0


def test_repetition_penalty_changes_token_choice():
    logits = torch.tensor([[[2.0, 1.5, 0.0]]])
    input_ids = torch.tensor([[0]])

    token = sample_next_token(
        logits,
        input_ids,
        top_k=1,
        top_p=1.0,
        repetition_penalty=2.0,
    )

    assert token.item() == 1


@pytest.mark.parametrize("temperature", [0, -1.0, float("inf"), float("nan")])
def test_sampling_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(torch.zeros(1, 1, 4), torch.zeros(1, 1), temperature)


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_sampling_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        sample_next_token(
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1),
            top_k=top_k,
        )


@pytest.mark.parametrize("top_p", [0, -0.1, 1.1, float("inf"), float("nan")])
def test_sampling_rejects_invalid_top_p(top_p):
    with pytest.raises(ValueError, match="top_p"):
        sample_next_token(
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1),
            top_p=top_p,
        )


@pytest.mark.parametrize(
    "repetition_penalty", [0, -1.0, float("inf"), float("nan")]
)
def test_sampling_rejects_invalid_repetition_penalty(repetition_penalty):
    with pytest.raises(ValueError, match="repetition_penalty"):
        sample_next_token(
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1),
            repetition_penalty=repetition_penalty,
        )


class _FakeInnerTokenizer:
    def __init__(self, eos_id):
        self.eos_id = eos_id

    def token_to_id(self, token):
        if token == "[EOS]":
            return self.eos_id
        return None


class _FakeTokenizer:
    def __init__(self, eos_id=2):
        self.tokenizer = _FakeInnerTokenizer(eos_id)

    def encode(self, text):
        return [0, 1]

    def decode(self, ids, **kwargs):
        return " ".join(str(token_id) for token_id in ids)


class _FakeModel:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, input_ids, kv_cache=None):
        vocab_size = 8
        logits = torch.full(
            (1, input_ids.size(1), vocab_size),
            -100.0,
            device=input_ids.device,
        )
        index = min(self.calls, len(self.tokens) - 1)
        logits[:, -1, self.tokens[index]] = 100.0
        self.calls += 1
        return logits


@pytest.mark.parametrize("max_new_tokens", [-1, 1.5, True])
def test_generate_rejects_invalid_token_limit_before_model_execution(
    max_new_tokens,
):
    model = _FakeModel(tokens=[3])
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_token_ids(
            model,
            _FakeTokenizer(),
            "hello",
            "cpu",
            max_new_tokens=max_new_tokens,
        )
    assert model.calls == 0


def test_generate_zero_token_request_returns_without_model_forward():
    model = _FakeModel(tokens=[3])
    generated = generate_token_ids(
        model,
        _FakeTokenizer(),
        "hello",
        "cpu",
        max_new_tokens=0,
        do_sample=False,
    )
    assert generated == []
    assert model.calls == 0


def test_generate_rejects_invalid_cache_selection_for_zero_token_request():
    model = _FakeModel(tokens=[3])
    with pytest.raises(ValueError, match="kv_cache_implementation"):
        generate_token_ids(
            model,
            _FakeTokenizer(),
            "hello",
            "cpu",
            max_new_tokens=0,
            do_sample=False,
            kv_cache_implementation="unknown",
        )
    assert model.calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("do_sample", 1), ("use_kv_cache", 0)],
)
def test_generate_rejects_non_boolean_mode_flags(field, value):
    model = _FakeModel(tokens=[3])
    kwargs = {field: value}
    with pytest.raises(TypeError, match=field):
        generate_token_ids(
            model,
            _FakeTokenizer(),
            "hello",
            "cpu",
            max_new_tokens=0,
            **kwargs,
        )
    assert model.calls == 0


def test_uncached_generation_rejects_empty_encoded_prompt():
    model = _FakeModel(tokens=[3])
    tokenizer = _FakeTokenizer()
    tokenizer.encode = lambda text: []
    with pytest.raises(ValueError, match="non-empty prompt"):
        generate_token_ids(
            model,
            tokenizer,
            "",
            "cpu",
            max_new_tokens=0,
            do_sample=False,
        )
    assert model.calls == 0


def test_uncached_generation_rejects_prompt_beyond_declared_context():
    model = _FakeModel(tokens=[3])
    model.config = SimpleNamespace(max_seq_len=1)
    with pytest.raises(ValueError, match="maximum sequence length"):
        generate_token_ids(
            model,
            _FakeTokenizer(),
            "hello",
            "cpu",
            max_new_tokens=0,
            do_sample=False,
        )
    assert model.calls == 0


def test_greedy_generation_preserves_ignored_sampling_sentinels():
    model = _FakeModel(tokens=[3])
    generated = generate_token_ids(
        model,
        _FakeTokenizer(),
        "hello",
        "cpu",
        max_new_tokens=1,
        do_sample=False,
        temperature=0.0,
        top_k=0,
        top_p=0.0,
        repetition_penalty=0.0,
    )
    assert generated == [3]


def test_generation_request_contract_fixture_matches_current_implementation():
    report = json.loads(
        (
            ROOT
            / "evaluation/fixtures/"
            "generation_request_contract_qualification_20260803.json"
        ).read_text(encoding="utf-8")
    )
    assert report["generation_implementation_sha256"] == _lf_sha256(
        ROOT / "vasu/inference/generate.py"
    )
    assert report["sampling_implementation_sha256"] == _lf_sha256(
        ROOT / "vasu/inference/sampling.py"
    )
    assert report["invalid_limit_rejected_before_eval"] is True
    assert report["invalid_cache_rejected_for_zero_token_request"] is True
    assert report["zero_token_forward_calls"] == 0
    assert report["zero_token_result_empty"] is True
    assert report["greedy_sampling_sentinels_preserved"] is True
    assert report["invalid_sampling_rejections"] == report[
        "invalid_sampling_cases"
    ]
    assert report["kv_cache_enabled_by_default"] is False
    assert report["training_authorized"] is False


def test_generate_stops_on_eos():
    model = _FakeModel(tokens=[3, 2, 4])
    tokenizer = _FakeTokenizer(eos_id=2)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        device="cpu",
        max_new_tokens=5,
        top_k=1,
        top_p=1.0,
    )

    assert output == "3 2"
    assert model.calls == 2


def test_generate_stops_at_max_new_tokens():
    model = _FakeModel(tokens=[3, 4, 5])
    tokenizer = _FakeTokenizer(eos_id=2)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        device="cpu",
        max_new_tokens=3,
        top_k=1,
        top_p=1.0,
    )

    assert output == "3 4 5"
    assert model.calls == 3


def test_generate_does_not_crash():
    model = _FakeModel(tokens=[3])
    tokenizer = _FakeTokenizer(eos_id=2)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        device="cpu",
        max_new_tokens=1,
        top_k=1,
        top_p=1.0,
    )

    assert output == "3"
