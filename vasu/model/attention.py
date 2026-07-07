import torch
import torch.nn as nn
import torch.nn.functional as F

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
            kv_cache=None,
            layer_idx=None,
        )-> torch.Tensor:

        """
        Multi-head self-attention.

        Args:
            x: (B,T,C)

        Returns:
            Tensor (B,T,C)
        """

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        past_len = 0

        if (
            kv_cache is not None
            and layer_idx is not None
        ):
            past_k, _ = kv_cache.get(layer_idx)

            if past_k is not None:
                past_len = past_k.size(2)

        cos = self.rope.cos[
            past_len: past_len + T
        ]

        sin = self.rope.sin[
            past_len: past_len + T
        ]

        q, k = apply_rotary(q, k, cos, sin)

        if kv_cache is not None:

            past_k, past_v = kv_cache.get(
                layer_idx
            )

            if past_k is not None:

                k = torch.cat(
                    [past_k, k],
                    dim=2,
                )

                v = torch.cat(
                    [past_v, v],
                    dim=2,
                )

            kv_cache.update(
                layer_idx,
                k,
                v,
            )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.out_proj(y)