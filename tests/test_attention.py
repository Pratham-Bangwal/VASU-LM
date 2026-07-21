import torch

from vasu.config import ModelConfig
from vasu.model import MultiHeadAttention


def test_attention_preserves_shape():
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=8,
        dim=24,
        n_heads=4,
        n_layers=2,
        hidden_dim=48,
    )

    attention = MultiHeadAttention(config)
    x = torch.randn(2, config.max_seq_len, config.dim)

    y = attention(x)

    assert y.shape == x.shape
