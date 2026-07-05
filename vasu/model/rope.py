import torch
import torch.nn as nn
from vasu.config import ModelConfig


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
    ):
        super().__init__()

        dim = head_dim
        base = rope_theta

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        t = torch.arange(max_seq_len).float()

        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int):

        return (
            self.cos[:seq_len],
            self.sin[:seq_len],
        )