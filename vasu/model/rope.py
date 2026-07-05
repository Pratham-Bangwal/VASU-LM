import torch

from vasu.config import ModelConfig


class RotaryEmbedding:
    """
    Rotary Positional Embeddings (RoPE)

    This class precomputes sine and cosine tables
    used by the attention mechanism.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
    ):
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        positions = torch.arange(max_seq_len).float()

        freqs = torch.outer(positions, inv_freq)

        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)