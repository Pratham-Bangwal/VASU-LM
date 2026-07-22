"""
VASU-1 Language Model
"""

import torch
import torch.nn as nn
from typing import Literal

from vasu.cache import KVCache, PreallocatedKVCache
from vasu.config import ModelConfig

from .block import TransformerBlock
from .embedding import TokenEmbedding
from .rmsnorm import RMSNorm


class VASUModel(nn.Module):
    """
    Decoder-only Transformer Language Model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embedding = TokenEmbedding(config)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.norm = RMSNorm(config.dim)

        self.lm_head = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = self.embedding.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | PreallocatedKVCache | None = None,
        cache_mode: Literal["none", "prefill", "decode"] = "none",
    ) -> torch.Tensor:

        """
        Forward pass.
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)

        Returns:
            Logits of shape (batch_size, sequence_len th, vocab_size)
        """

        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("input_ids must have shape (B, T) with T >= 1")
        if cache_mode not in ("none", "prefill", "decode"):
            raise ValueError(f"unsupported cache_mode: {cache_mode}")
        if cache_mode == "none":
            if kv_cache is not None:
                raise ValueError("cache_mode='none' does not accept a cache")
            if input_ids.size(1) > self.config.max_seq_len:
                raise ValueError("input exceeds model maximum sequence length")
        else:
            if kv_cache is None:
                raise ValueError("cache_mode requires a KVCache")
            if kv_cache.n_layers != self.config.n_layers:
                raise ValueError("cache layer count does not match model")
            if kv_cache.max_seq_len != self.config.max_seq_len:
                raise ValueError("cache maximum length does not match model")
            if cache_mode == "prefill":
                if kv_cache.sequence_length != 0:
                    raise ValueError("prefill requires an empty cache")
                total_length = input_ids.size(1)
            else:
                if input_ids.size(1) != 1:
                    raise ValueError("decode requires query length exactly 1")
                cached_length = kv_cache.sequence_length
                if cached_length == 0:
                    raise ValueError("decode requires a populated cache")
                total_length = cached_length + 1
            if total_length > self.config.max_seq_len:
                raise ValueError("cached sequence exceeds model maximum length")

        x = self.embedding(input_ids)

        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                cache_mode=cache_mode,
            )

        x = self.norm(x)

        x = x / (self.config.dim ** 0.5)

        logits = self.lm_head(x)

        return logits
