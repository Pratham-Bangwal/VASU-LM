"""
Transformer Block for VASU.
"""

import torch
import torch.nn as nn

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

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)

        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = RMSNorm(dim)

        self.mlp = SwiGLU(
            dim=dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x + self.attention(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        return x