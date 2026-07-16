from __future__ import annotations

import pytest
import torch

import vasu.inference.generate as generate_module
from vasu.cache import KVCache
from vasu.config import ModelConfig
from vasu.inference.generate import generate_token_ids
from vasu.model.model import VASUModel


class _InnerTokenizer:
    def __init__(self, eos_id: int = 31) -> None:
        self.eos_id = eos_id

    def token_to_id(self, token: str) -> int | None:
        return self.eos_id if token == "[EOS]" else None


class _Tokenizer:
    def __init__(self, prompt_ids=None, eos_id: int = 31) -> None:
        self.prompt_ids = list(prompt_ids or [1, 2, 3])
        self.tokenizer = _InnerTokenizer(eos_id)

    def encode(self, text: str) -> list[int]:
        return self.prompt_ids.copy()

    def decode(self, ids, **kwargs) -> str:
        return " ".join(str(value) for value in ids)


def _config(max_seq_len: int = 16) -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=max_seq_len,
        dim=16,
        n_heads=4,
        n_layers=2,
        hidden_dim=32,
        dropout=0.0,
        bias=False,
    )


def _model(max_seq_len: int = 16, device: str = "cpu") -> VASUModel:
    torch.manual_seed(1234)
    return VASUModel(_config(max_seq_len)).to(device).eval()


def _kv(length: int = 2, *, dtype=torch.float32, device="cpu"):
    return torch.randn(1, 4, length, 4, dtype=dtype, device=device)


def test_empty_cache_reports_zero_sequence_length():
    cache = KVCache(n_layers=2, max_seq_len=8)
    assert cache.sequence_length == 0


def test_cache_update_and_retrieval_preserve_tensor_identity():
    cache = KVCache(n_layers=1, max_seq_len=8)
    key = _kv()
    value = _kv()
    cache.update(0, key, value)
    cached_key, cached_value = cache.get(0)
    assert cached_key is key
    assert cached_value is value
    assert cache.sequence_length == 2


def test_cache_reset_clears_every_layer():
    cache = KVCache(n_layers=2, max_seq_len=8)
    cache.update(0, _kv(), _kv())
    cache.update(1, _kv(), _kv())
    cache.reset()
    assert cache.sequence_length == 0
    assert cache.get(0) == (None, None)
    assert cache.get(1) == (None, None)


@pytest.mark.parametrize("layer_idx", [-1, 2])
def test_invalid_layer_index_is_rejected(layer_idx):
    cache = KVCache(n_layers=2, max_seq_len=8)
    with pytest.raises(IndexError, match="layer_idx"):
        cache.get(layer_idx)
    with pytest.raises(IndexError, match="layer_idx"):
        cache.update(layer_idx, _kv(), _kv())


def test_key_value_shape_mismatch_is_rejected():
    cache = KVCache(n_layers=1, max_seq_len=8)
    with pytest.raises(ValueError, match="shapes must match"):
        cache.update(0, _kv(2), _kv(3))


def test_device_and_dtype_mismatch_are_rejected():
    cache = KVCache(n_layers=1, max_seq_len=8)
    with pytest.raises(ValueError, match="dtypes must match"):
        cache.update(0, _kv(dtype=torch.float32), _kv(dtype=torch.float64))

    key = torch.empty((1, 4, 2, 4), device="cpu")
    value = torch.empty((1, 4, 2, 4), device="meta")
    with pytest.raises(ValueError, match="devices must match"):
        cache.update(0, key, value)


def test_initialized_signature_mismatch_is_rejected():
    cache = KVCache(n_layers=2, max_seq_len=8)
    cache.update(0, _kv(), _kv())
    with pytest.raises(ValueError, match="cache batch/head"):
        cache.update(
            1,
            torch.randn(2, 4, 2, 4),
            torch.randn(2, 4, 2, 4),
        )


def test_maximum_sequence_length_is_enforced():
    cache = KVCache(n_layers=1, max_seq_len=2)
    with pytest.raises(ValueError, match="exceeds maximum"):
        cache.update(0, _kv(3), _kv(3))


def test_inconsistent_layer_lengths_are_rejected():
    cache = KVCache(n_layers=2, max_seq_len=8)
    cache.update(0, _kv(2), _kv(2))
    with pytest.raises(ValueError, match="inconsistent per-layer"):
        cache.update(1, _kv(3), _kv(3))


def test_cache_is_not_part_of_model_state_dict():
    model = _model()
    before = tuple(model.state_dict())
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    model(torch.tensor([[1, 2, 3]]), kv_cache=cache, cache_mode="prefill")
    assert tuple(model.state_dict()) == before
    assert not any("cache" in key for key in model.state_dict())


def test_cached_prefill_logits_match_uncached_logits_cpu():
    model = _model()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    uncached = model(tokens)
    cached = model(tokens, kv_cache=cache, cache_mode="prefill")
    torch.testing.assert_close(cached, uncached, rtol=1e-4, atol=1e-5)
    assert cache.sequence_length == tokens.size(1)


def test_cached_decode_matches_uncached_final_logits_at_every_step_cpu():
    model = _model()
    history = torch.tensor([[1, 2, 3]])
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    prefill = model(history, kv_cache=cache, cache_mode="prefill")
    torch.testing.assert_close(
        prefill, model(history), rtol=1e-4, atol=1e-5
    )
    for token_id in (4, 5, 6):
        newest = torch.tensor([[token_id]])
        cached = model(newest, kv_cache=cache, cache_mode="decode")
        history = torch.cat((history, newest), dim=1)
        uncached = model(history)
        torch.testing.assert_close(
            cached[:, -1, :],
            uncached[:, -1, :],
            rtol=1e-4,
            atol=1e-5,
        )
        assert cache.sequence_length == history.size(1)


def test_greedy_cached_and_uncached_token_ids_are_identical_cpu():
    model = _model()
    tokenizer = _Tokenizer()
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        prompt="test",
        device="cpu",
        max_new_tokens=6,
        do_sample=False,
        prompt_format="plain",
    )
    uncached = generate_token_ids(**kwargs, use_kv_cache=False)
    cached = generate_token_ids(**kwargs, use_kv_cache=True)
    assert cached == uncached


def test_eos_stopping_is_identical():
    model = _model()
    for parameter in model.parameters():
        parameter.data.zero_()
    tokenizer = _Tokenizer(eos_id=0)
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        prompt="test",
        device="cpu",
        max_new_tokens=5,
        do_sample=False,
        prompt_format="plain",
    )
    assert generate_token_ids(**kwargs, use_kv_cache=False) == [0]
    assert generate_token_ids(**kwargs, use_kv_cache=True) == [0]


def test_maximum_context_stopping_is_identical():
    model = _model(max_seq_len=5)
    tokenizer = _Tokenizer(prompt_ids=[1, 2, 3])
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        prompt="test",
        device="cpu",
        max_new_tokens=10,
        do_sample=False,
        prompt_format="plain",
    )
    uncached = generate_token_ids(**kwargs, use_kv_cache=False)
    cached = generate_token_ids(**kwargs, use_kv_cache=True)
    assert cached == uncached
    assert len(cached) == 2


def test_separate_generate_calls_create_fresh_caches(monkeypatch):
    created: list[KVCache] = []

    class TrackingCache(KVCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(generate_module, "KVCache", TrackingCache)
    model = _model()
    tokenizer = _Tokenizer()
    for _ in range(2):
        generate_token_ids(
            model,
            tokenizer,
            "test",
            "cpu",
            max_new_tokens=1,
            do_sample=False,
            prompt_format="plain",
            use_kv_cache=True,
        )
    assert len(created) == 2
    assert created[0] is not created[1]


def test_sampling_receives_complete_history(monkeypatch):
    history_lengths: list[int] = []

    def deterministic_sample(logits, input_ids, *args, **kwargs):
        history_lengths.append(input_ids.size(1))
        return torch.tensor([[7]], device=input_ids.device)

    monkeypatch.setattr(
        generate_module, "sample_next_token", deterministic_sample
    )
    model = _model(max_seq_len=8)
    tokenizer = _Tokenizer(prompt_ids=[1, 2, 3], eos_id=31)
    generated = generate_token_ids(
        model,
        tokenizer,
        "test",
        "cpu",
        max_new_tokens=3,
        do_sample=True,
        prompt_format="plain",
        use_kv_cache=True,
    )
    assert generated == [7, 7, 7]
    assert history_lengths == [3, 4, 5]


def test_decode_rejects_multi_token_query():
    model = _model()
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    model(torch.tensor([[1, 2]]), kv_cache=cache, cache_mode="prefill")
    with pytest.raises(ValueError, match="query length exactly 1"):
        model(torch.tensor([[3, 4]]), kv_cache=cache, cache_mode="decode")


def test_prefill_rejects_non_empty_cache():
    model = _model()
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    model(torch.tensor([[1, 2]]), kv_cache=cache, cache_mode="prefill")
    with pytest.raises(ValueError, match="empty cache"):
        model(torch.tensor([[3]]), kv_cache=cache, cache_mode="prefill")


def test_decode_rejects_empty_cache():
    model = _model()
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    with pytest.raises(ValueError, match="populated cache"):
        model(torch.tensor([[1]]), kv_cache=cache, cache_mode="decode")


def test_cached_generation_rejects_empty_and_overlong_prompts():
    model = _model(max_seq_len=4)
    empty = _Tokenizer(prompt_ids=[])
    # The helper's default fallback is non-empty, so override encode directly.
    empty.encode = lambda text: []
    with pytest.raises(ValueError, match="non-empty prompt"):
        generate_token_ids(
            model,
            empty,
            "",
            "cpu",
            do_sample=False,
            prompt_format="plain",
            use_kv_cache=True,
        )
    overlong = _Tokenizer(prompt_ids=[1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="exceeds model maximum"):
        generate_token_ids(
            model,
            overlong,
            "test",
            "cpu",
            do_sample=False,
            prompt_format="plain",
            use_kv_cache=True,
        )


def test_cache_mode_enabled_without_layer_index_is_rejected():
    model = _model()
    attention = model.blocks[0].attention
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    hidden = torch.randn(1, 1, model.config.dim)
    with pytest.raises(ValueError, match="requires kv_cache and layer_idx"):
        attention(
            hidden,
            kv_cache=cache,
            layer_idx=None,
            cache_mode="prefill",
        )


def test_none_mode_rejects_ambiguous_cache_argument():
    model = _model()
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    with pytest.raises(ValueError, match="does not accept a cache"):
        model(torch.tensor([[1]]), kv_cache=cache, cache_mode="none")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_prefill_decode_and_greedy_parity():
    model = _model(device="cuda")
    tokens = torch.tensor([[1, 2, 3]], device="cuda")
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    cached = model(tokens, kv_cache=cache, cache_mode="prefill")
    torch.testing.assert_close(cached, model(tokens), rtol=1e-4, atol=1e-5)
    newest = torch.tensor([[4]], device="cuda")
    decoded = model(newest, kv_cache=cache, cache_mode="decode")
    full = model(torch.cat((tokens, newest), dim=1))
    torch.testing.assert_close(
        decoded[:, -1, :], full[:, -1, :], rtol=1e-4, atol=1e-5
    )
    tokenizer = _Tokenizer()
    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        prompt="test",
        device="cuda",
        max_new_tokens=5,
        do_sample=False,
        prompt_format="plain",
    )
    assert generate_token_ids(**kwargs, use_kv_cache=True) == (
        generate_token_ids(**kwargs, use_kv_cache=False)
    )
