import torch

from vasu.inference.generate import generate
from vasu.inference.sampling import sample_next_token


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
