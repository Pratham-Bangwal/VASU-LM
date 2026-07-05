import torch
import torch.nn as nn
import torch.nn.functional as F

from vasu.config import ModelConfig
from .rope import RotaryEmbedding
from .utils import apply_rotary


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.q_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=config.bias)

        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.bias)

        self.dropout = config.dropout

        self.rope = RotaryEmbedding(
        head_dim=self.head_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        )

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T)

        q, k = apply_rotary(q, k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.out_proj(y)