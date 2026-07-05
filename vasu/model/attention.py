"""
Multi-Head Causal Self-Attention for VASU.

Architecture:
Input
   │
QKV Projection
   │
Split into Heads
   │
Scaled Dot-Product Attention
   │
Causal Mask
   │
Softmax
   │
Weighted Sum
   │
Merge Heads
   │
Output Projection
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vasu.config import ModelConfig


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Args:
        dim:
            Embedding dimension.

        num_heads:
            Number of attention heads.

        dropout:
            Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # One projection produces Q, K and V
        self.qkv = nn.Linear(dim, dim * 3)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        # (B, T, 3 * D)
        qkv = self.qkv(x)

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape into heads
        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Attention scores
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)

        attention = self.dropout(attention)

        output = attention @ v

        # Merge heads
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.dim,
            )
        )

        output = self.out_proj(output)

        return output