import torch
import torch.nn as nn
from vasu.config import ModelConfig


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()

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