"""
Transformer Block for VASU.
"""

import torch
import torch.nn as nn
from typing import Literal

from vasu.cache import KVCache
from vasu.config import ModelConfig

from .attention import MultiHeadAttention
from .mlp import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Modern Pre-Norm Transformer Block.

    Architecture:

        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm1 = RMSNorm(config.dim)

        self.attention = MultiHeadAttention(config)

        self.norm2 = RMSNorm(config.dim)

        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        cache_mode: Literal["none", "prefill", "decode"] = "none",
    ) -> torch.Tensor:

        x = x + self.attention(
            self.norm1(x),
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            cache_mode=cache_mode,
        )

        x = x + self.mlp(self.norm2(x))

        return x
