import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from vasu.cache import KVCache
from vasu.config import ModelConfig
from .rope import RotaryEmbedding
from .utils import apply_rotary


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.dim % config.n_heads != 0:
            raise ValueError(
                f"dim ({config.dim}) must be divisible by "
                f"n_heads ({config.n_heads})"
            )

        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.q_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=config.bias)

        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.bias)

        self.dropout_p = config.dropout

        self.rope = RotaryEmbedding(
        head_dim=self.head_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        cache_mode: Literal["none", "prefill", "decode"] = "none",
    ) -> torch.Tensor:

        """
        Multi-head self-attention.

        Args:
            x: (B,T,C)

        Returns:
            Tensor (B,T,C)
        """

        if cache_mode not in ("none", "prefill", "decode"):
            raise ValueError(f"unsupported cache_mode: {cache_mode}")
        if cache_mode == "none":
            if kv_cache is not None:
                raise ValueError("cache_mode='none' does not accept a cache")
        elif kv_cache is None or layer_idx is None:
            raise ValueError("cache mode requires kv_cache and layer_idx")

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        past_len = 0
        if cache_mode != "none":
            assert kv_cache is not None and layer_idx is not None
            past_k, past_v = kv_cache.get(layer_idx)
            if cache_mode == "prefill" and past_k is not None:
                raise ValueError("prefill requires an empty layer cache")
            if cache_mode == "decode":
                if T != 1:
                    raise ValueError("decode requires query length exactly 1")
                if past_k is None or past_v is None:
                    raise ValueError("decode requires a populated cache")
                past_len = past_k.size(2)

        if past_len + T > self.rope.cos.size(0):
            raise ValueError("attention sequence exceeds model maximum length")

        cos = self.rope.cos[
            past_len: past_len + T
        ]

        sin = self.rope.sin[
            past_len: past_len + T
        ]

        q, k = apply_rotary(q, k, cos, sin)

        if cache_mode == "decode":
            assert past_k is not None and past_v is not None
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        if cache_mode != "none":
            assert kv_cache is not None and layer_idx is not None
            kv_cache.update(layer_idx, k, v)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            # Prefill must remain causal. A one-token decode query has no
            # future key positions, so it may attend to the full cache.
            is_causal=cache_mode != "decode",
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.out_proj(y)
