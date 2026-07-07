import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Precomputes Rotary Positional Embeddings (RoPE).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
    ):
        super().__init__()

        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        t = torch.arange(max_seq_len).float()

        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

        self.cos: torch.Tensor
        self.sin: torch.Tensor

    def forward(
        self,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cosine and sine rotary embeddings
        for the requested sequence length.
        """

        return (
            self.cos[:seq_len],
            self.sin[:seq_len],
        )