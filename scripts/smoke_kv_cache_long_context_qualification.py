"""Emit deterministic long-context KV-cache qualification evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vasu.cache import PreallocatedKVCache  # noqa: E402
from vasu.config import ModelConfig  # noqa: E402
from vasu.inference.generate import generate_token_ids  # noqa: E402
from vasu.model.model import VASUModel  # noqa: E402


def _lf_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _InnerTokenizer:
    def token_to_id(self, token: str) -> int | None:
        return 31 if token == "[EOS]" else None


class _Tokenizer:
    tokenizer = _InnerTokenizer()

    @staticmethod
    def encode(text: str) -> list[int]:
        return [1]


def _model() -> VASUModel:
    torch.manual_seed(20260803)
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=64,
        dim=16,
        n_heads=4,
        n_layers=2,
        hidden_dim=32,
        dropout=0.0,
        bias=False,
    )
    return VASUModel(config).eval()


def qualify() -> dict[str, object]:
    model = _model()
    tokenizer = _Tokenizer()
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": "qualification",
        "device": "cpu",
        "max_new_tokens": 128,
        "do_sample": False,
        "prompt_format": "plain",
    }
    uncached = generate_token_ids(**kwargs, use_kv_cache=False)
    dynamic = generate_token_ids(
        **kwargs,
        use_kv_cache=True,
        kv_cache_implementation="dynamic",
    )
    preallocated = generate_token_ids(
        **kwargs,
        use_kv_cache=True,
        kv_cache_implementation="preallocated",
    )

    config = model.config
    cache = PreallocatedKVCache(
        config.n_layers,
        config.max_seq_len,
        1,
        config.n_heads,
        config.dim // config.n_heads,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    storage_pointer = cache.storage.untyped_storage().data_ptr()
    full = torch.zeros(1, config.n_heads, config.max_seq_len, 4)
    for layer_idx in range(config.n_layers):
        cache.update(layer_idx, full, full)
    full_capacity_reached = cache.sequence_length == config.max_seq_len
    cache.reset()
    storage_reused = (
        cache.storage.untyped_storage().data_ptr() == storage_pointer
        and cache.sequence_length == 0
    )

    malformed_rank_rejected = False
    malformed = torch.empty(1, config.n_heads, 1)
    try:
        cache.update(0, malformed, malformed)
    except ValueError:
        malformed_rank_rejected = True

    return {
        "schema_id": "vasu_kv_cache_long_context_qualification_v1",
        "date": "2026-08-03",
        "device": "cpu",
        "seed": 20260803,
        "max_sequence_length": config.max_seq_len,
        "prompt_tokens": 1,
        "generated_tokens": len(preallocated),
        "cache_implementation_sha256": _lf_sha256(
            ROOT / "vasu/cache/preallocated_kv_cache.py"
        ),
        "generation_implementation_sha256": _lf_sha256(
            ROOT / "vasu/inference/generate.py"
        ),
        "attention_implementation_sha256": _lf_sha256(
            ROOT / "vasu/model/attention.py"
        ),
        "full_context_parity": preallocated == dynamic == uncached,
        "exact_context_stop": len(preallocated) == config.max_seq_len - 1,
        "full_capacity_reached": full_capacity_reached,
        "storage_reused_after_reset": storage_reused,
        "malformed_rank_rejected_with_value_error": malformed_rank_rejected,
        "checkpoint_loaded": False,
        "dataset_opened": False,
        "kv_cache_enabled_by_default": False,
        "training_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(qualify(), indent=2, sort_keys=True))
